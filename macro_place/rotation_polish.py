"""Lever C — Macro orientation polish (slow + fast paths).

Operates as the *final* stage of the placer. Positions are already fixed by
the upstream pipeline (CD/LNS + SA escape + hessian + ORFS spacing). This
stage explores the orientation DoF that the rest of the placer ignores.

Approach:
  - For each hard macro, consider only orientations in its initial class
    (NS_ORIENTATIONS or EW_ORIENTATIONS). Cross-class flips would swap
    macro w/h and require re-legalization, which is out of MVP scope.
  - Score each alternative via ``compute_proxy_cost`` (the official
    evaluator). Keep the orientation that strictly minimizes proxy.
  - To bound wall time, only search the top-K macros by net degree
    (rotation of a heavily-connected macro is most likely to move WL).

Returns:
  - chosen_orientations: dict {macro_node_idx -> orientation string}
  - improved_count: int — number of macros that flipped to a better orientation
  - delta_proxy: float — proxy improvement (initial - final), negative = win
  - final_metrics: dict from compute_proxy_cost at the chosen state
"""
from __future__ import annotations

from typing import Any

import torch


# Orientation classes (matches Plc_client/coordinate_descent_placer.py).
NS_ORIENTATIONS = ("N", "FN", "S", "FS")
EW_ORIENTATIONS = ("E", "FE", "W", "FW")


def _net_degree_per_hard_macro(plc: Any, benchmark: Any) -> dict[int, int]:
    """Count net incidences per hard macro (via macro pins -> nets).

    Returns a dict keyed by plc node index for hard macros only.
    """
    # The plc exposes nets via its module structure; we approximate degree
    # using benchmark net data when available, else fall back to 0.
    out: dict[int, int] = {}
    hard_idx = list(getattr(benchmark, "hard_macro_indices", []))
    # Try benchmark-side net adjacency first.
    macro_to_nets = getattr(benchmark, "macro_to_nets", None)
    if macro_to_nets is not None:
        for i, idx in enumerate(hard_idx):
            nets = macro_to_nets[i] if i < len(macro_to_nets) else []
            out[int(idx)] = int(len(nets))
        return out
    # Fall back: use macro area as a tie-breaker proxy for "importance".
    sizes = getattr(benchmark, "macro_sizes", None)
    if sizes is not None:
        areas = (sizes[:, 0] * sizes[:, 1]).cpu().numpy()
        for i, idx in enumerate(hard_idx):
            out[int(idx)] = -int(areas[i] * -1)  # higher area = higher score
        return out
    for idx in hard_idx:
        out[int(idx)] = 0
    return out


def _orientation_class(ori: str) -> tuple[str, ...]:
    if ori in NS_ORIENTATIONS:
        return NS_ORIENTATIONS
    if ori in EW_ORIENTATIONS:
        return EW_ORIENTATIONS
    # Default to NS class for unknown orientation (matches plc default "N").
    return NS_ORIENTATIONS


def polish_orientations(
    *,
    positions: torch.Tensor,
    benchmark: Any,
    plc: Any,
    top_k: int = 64,
    proxy_cost_fn: Any = None,
) -> dict[str, Any]:
    """Iterate hard macros (top-K by degree) and pick best same-class orientation.

    Args:
      positions: [num_macros, 2] tensor — passed unchanged to the cost fn.
      benchmark: Benchmark object with hard_macro_indices.
      plc: PlacementCost object with update_macro_orientation method.
      top_k: cap on number of macros to search (sorted descending by degree).
      proxy_cost_fn: callable(positions, benchmark=, plc=) -> dict with proxy_cost.
        Defaults to ``macro_place.objective.compute_proxy_cost`` when None.

    Returns: dict with chosen_orientations, improved_count, delta_proxy, final_metrics.
    """
    if proxy_cost_fn is None:
        from macro_place.objective import compute_proxy_cost
        proxy_cost_fn = compute_proxy_cost

    # Baseline proxy.
    initial = proxy_cost_fn(positions, benchmark=benchmark, plc=plc)
    initial_proxy = float(initial["proxy_cost"])

    hard_idx = list(getattr(benchmark, "hard_macro_indices", []))
    if not hard_idx:
        return {
            "chosen_orientations": {},
            "improved_count": 0,
            "delta_proxy": 0.0,
            "final_metrics": initial,
        }

    # Pick top-K by degree (most-connected first).
    degrees = _net_degree_per_hard_macro(plc, benchmark)
    ranked = sorted(
        [int(i) for i in hard_idx],
        key=lambda i: -int(degrees.get(int(i), 0)),
    )[: max(0, int(top_k))]

    chosen: dict[int, str] = {}
    improved_count = 0
    current_proxy = initial_proxy

    for node_idx in ranked:
        cur_ori = plc.get_macro_orientation(node_idx) or "N"
        ori_class = _orientation_class(cur_ori)
        best_ori = cur_ori
        best_proxy = current_proxy
        for ori in ori_class:
            if ori == cur_ori:
                continue
            plc.update_macro_orientation(node_idx, ori)
            scored = proxy_cost_fn(positions, benchmark=benchmark, plc=plc)
            p = float(scored["proxy_cost"])
            if p + 1e-9 < best_proxy:
                best_proxy = p
                best_ori = ori
        # Apply the winning orientation (could be the original).
        plc.update_macro_orientation(node_idx, best_ori)
        if best_ori != cur_ori:
            improved_count += 1
            chosen[int(node_idx)] = best_ori
            current_proxy = best_proxy

    final = proxy_cost_fn(positions, benchmark=benchmark, plc=plc)
    return {
        "chosen_orientations": chosen,
        "improved_count": int(improved_count),
        "delta_proxy": float(current_proxy - initial_proxy),
        "initial_proxy": float(initial_proxy),
        "final_proxy": float(current_proxy),
        "final_metrics": final,
    }


def polish_orientations_fast(
    *,
    positions_np,
    ctx,
    state,
    benchmark,
    plc,
    top_k: int = 64,
) -> dict[str, Any]:
    """Fast orientation polish using fast_proxy + in-place ctx mutation.

    Search loop uses ``fast_proxy`` (μs per eval) and mutates ctx pin offsets
    via ``apply_orientation``. After the loop, plc orientations are synced to
    ``state.macro_orientation`` so that ``compute_proxy_cost`` reflects the
    chosen orientations.

    Returns a dict with the same shape as ``polish_orientations`` plus a
    ``final_official_proxy`` from ``compute_proxy_cost`` after sync.
    """
    from macro_place.fast_proxy import fast_proxy
    from macro_place.objective import compute_proxy_cost
    from macro_place.orientation import (
        apply_orientation,
        orientation_class_indices,
        orientation_name,
    )
    import numpy as np

    pos_np = np.asarray(positions_np, dtype=np.float64)
    initial_fast = fast_proxy(pos_np, ctx)
    initial_proxy = float(initial_fast.proxy_cost)

    hard_idx_plc = list(getattr(benchmark, "hard_macro_indices", []))
    # The state.macro_orientation array is indexed by local macro index
    # (matches FastProxyContext.macro_w/h indexing). Hard macros occupy the
    # leading slots in that ordering (see build_fast_proxy_context).
    n_hard = len(hard_idx_plc)
    if n_hard == 0:
        return {
            "chosen_orientations": {},
            "improved_count": 0,
            "delta_proxy": 0.0,
            "initial_proxy": initial_proxy,
            "final_proxy": initial_proxy,
            "final_official_proxy": initial_proxy,
        }

    # Rank by net degree (most-connected first); fall back to area.
    degrees = _net_degree_per_hard_macro(plc, benchmark)
    ranked_local = sorted(
        range(n_hard),
        key=lambda i: -int(degrees.get(int(hard_idx_plc[i]), 0)),
    )[: max(0, int(top_k))]

    improved_count = 0
    current_proxy = initial_proxy
    chosen: dict[int, str] = {}

    for local_i in ranked_local:
        cur_ori_idx = int(state.macro_orientation[local_i])
        ori_class = orientation_class_indices(cur_ori_idx)
        best_ori_idx = cur_ori_idx
        best_proxy = current_proxy
        for ori_idx in ori_class:
            if ori_idx == cur_ori_idx:
                continue
            apply_orientation(ctx, state, local_i, ori_idx)
            res = fast_proxy(pos_np, ctx)
            p = float(res.proxy_cost)
            if p + 1e-9 < best_proxy:
                best_proxy = p
                best_ori_idx = ori_idx
        # Apply the winner (could be the original).
        apply_orientation(ctx, state, local_i, best_ori_idx)
        if best_ori_idx != cur_ori_idx:
            improved_count += 1
            current_proxy = best_proxy
            chosen[int(hard_idx_plc[local_i])] = orientation_name(best_ori_idx)

    # Sync plc orientations from state so compute_proxy_cost reflects choices.
    for local_i in range(n_hard):
        plc_idx = int(hard_idx_plc[local_i])
        ori_name = orientation_name(int(state.macro_orientation[local_i]))
        try:
            plc.update_macro_orientation(plc_idx, ori_name)
        except Exception:
            pass

    import torch
    pos_t = torch.as_tensor(pos_np, dtype=torch.float32)
    final_official = compute_proxy_cost(pos_t, benchmark=benchmark, plc=plc)
    return {
        "chosen_orientations": chosen,
        "improved_count": int(improved_count),
        "delta_proxy": float(current_proxy - initial_proxy),
        "initial_proxy": float(initial_proxy),
        "final_proxy": float(current_proxy),
        "final_official_proxy": float(final_official["proxy_cost"]),
        "final_official_metrics": final_official,
    }
