"""Coordinate-descent core for the CD+LNS placer (Bet 7 restart).

For each node, holds all others fixed and grid-searches a candidate
window for the lowest-proxy position. Calls into ``fast_proxy`` for
every candidate evaluation; the surrogate's calibrated <1% error is
sufficient for inner-loop guidance, with the official ``compute_proxy_cost``
used only for final validation in the placer entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from macro_place.fast_proxy import FastProxyContext, fast_proxy
from macro_place.fast_proxy_incremental import (
    FastProxyCache,
    apply_move,
    build_cache,
    cache_result,
    revert_move,
)


@dataclass(frozen=True)
class CDState:
    positions: np.ndarray
    sweep_idx: int
    radius: float
    current_cost: float


def cd_grid_search(
    node_idx: int,
    positions: np.ndarray,
    ctx: FastProxyContext,
    radius: float,
    k_per_axis: int = 8,
    cache: FastProxyCache | None = None,
    tiebreak_enabled: bool = False,
    tie_epsilon_rel: float = 1e-3,
    orientation_state: Any | None = None,
    search_orientations: bool = False,
) -> Tuple[np.ndarray, float]:
    """Grid-search the best position for one node, holding others fixed.

    Builds a ``k_per_axis * k_per_axis`` grid of candidate positions
    centered on the node's current location, snaps each to canvas bounds,
    evaluates the surrogate proxy for each, and returns the (position,
    cost) pair with the lowest cost.

    Args:
        node_idx: index into ``positions`` whose row to vary.
        positions: float64 array shape ``[num_nodes, 2]`` of all node positions.
        ctx: pre-built FastProxyContext for this benchmark.
        radius: half-width of the candidate window in canvas units.
        k_per_axis: candidates per axis (total = ``k_per_axis ** 2``).

    Returns:
        (best_position, best_cost) — best_position shape ``[2]`` float64,
        best_cost float. If no candidate beats the current position the
        current position is returned.
    """
    cx = float(positions[node_idx, 0])
    cy = float(positions[node_idx, 1])

    # Linear grid in [-radius, +radius] around current position
    offsets = np.linspace(-radius, +radius, k_per_axis, dtype=np.float64)
    grid_x = np.clip(cx + offsets, 0.0, ctx.canvas_w)
    grid_y = np.clip(cy + offsets, 0.0, ctx.canvas_h)

    best_pos = np.array([cx, cy], dtype=np.float64)
    owned_cache = cache is None
    if cache is None:
        cache = build_cache(positions, ctx)
    base_cost = float(cache_result(cache).proxy_cost)
    base_congestion = float(cache_result(cache).congestion)
    best_cost = base_cost
    best_congestion = base_congestion
    old_xy = np.array([cx, cy], dtype=np.float64)

    # Lever C — orientation search outer loop. Default behavior (None state
    # or search_orientations=False) keeps the original single-orientation
    # search. With both provided, iterate the macro's same-class orientations.
    do_ori_search = bool(search_orientations) and orientation_state is not None
    if do_ori_search:
        from macro_place.orientation import orientation_class_indices
        from macro_place.orientation_cache import apply_rotation_to_cache

        cur_ori = int(orientation_state.macro_orientation[node_idx])
        orientations_to_try: Tuple[int, ...] = tuple(
            orientation_class_indices(cur_ori)
        )
    else:
        cur_ori = 0  # unused
        orientations_to_try = (0,)
    best_ori = cur_ori

    for ori_idx in orientations_to_try:
        if do_ori_search and ori_idx != int(orientation_state.macro_orientation[node_idx]):
            # Switch orientation: mutates ctx.pin_offset_x/y + cache HPWL.
            apply_rotation_to_cache(cache, ctx, orientation_state, node_idx, ori_idx)
        # Re-read base cost for tiebreak comparison at this orientation.
        # (Not used for "best" tracking — that uses the original base_cost so
        # all orientations compete fairly on absolute proxy.)
        for gy in grid_y:
            for gx in grid_x:
                result = apply_move(
                    cache,
                    ctx,
                    node_idx,
                    np.array([gx, gy], dtype=np.float64),
                    exact_hpwl=owned_cache,
                )
                cost = float(result.proxy_cost)
                congestion = float(result.congestion)
                if cost < best_cost:
                    best_cost = cost
                    best_congestion = congestion
                    best_pos = np.array([gx, gy], dtype=np.float64)
                    if do_ori_search:
                        best_ori = ori_idx
                elif (
                    bool(tiebreak_enabled)
                    and best_cost + 1e-9 < base_cost
                    and _relative_diff(cost, best_cost) <= max(0.0, float(tie_epsilon_rel))
                    and congestion + 1e-9 < best_congestion
                ):
                    best_cost = cost
                    best_congestion = congestion
                    best_pos = np.array([gx, gy], dtype=np.float64)
                    if do_ori_search:
                        best_ori = ori_idx
                revert_move(cache, ctx, node_idx, old_xy)

    # Restore orientation to initial before applying the winner (or for owned_cache).
    if do_ori_search:
        cur_state_ori = int(orientation_state.macro_orientation[node_idx])
        if cur_state_ori != cur_ori:
            apply_rotation_to_cache(cache, ctx, orientation_state, node_idx, cur_ori)

    if not owned_cache and best_cost + 1e-9 < base_cost:
        # Apply winning orientation (if changed) before the move.
        if do_ori_search and best_ori != cur_ori:
            apply_rotation_to_cache(cache, ctx, orientation_state, node_idx, best_ori)
        apply_move(cache, ctx, node_idx, best_pos, exact_hpwl=False)

    return best_pos, best_cost


def _relative_diff(candidate_cost: float, best_cost: float) -> float:
    return (float(candidate_cost) - float(best_cost)) / max(abs(float(best_cost)), 1e-12)


def cd_sweep(
    positions: np.ndarray,
    ctx: FastProxyContext,
    radius: float,
    k_per_axis: int = 8,
    seed: int = 0,
    cache: FastProxyCache | None = None,
    freeze_mask: np.ndarray | None = None,
    tiebreak_enabled: bool = False,
    tie_epsilon_rel: float = 1e-3,
    orientation_state: Any | None = None,
    search_orientations: bool = False,
) -> Tuple[np.ndarray, bool, int]:
    """One full coordinate-descent sweep over all nodes.

    Visits each node in a randomized order (deterministic given ``seed``),
    grid-searches its best position via ``cd_grid_search``, and accepts
    if the search returns lower cost. Other nodes' positions are read from
    the same ``positions`` array, so accepted moves take effect for
    subsequent nodes within the sweep.

    Args:
        positions: float64 array shape ``[num_nodes, 2]``.
        ctx: pre-built FastProxyContext.
        radius: half-width of the candidate window per node.
        k_per_axis: candidates per axis (total per node = ``k_per_axis ** 2``).
        seed: RNG seed for the sweep order.
        freeze_mask: optional bool array shape ``[num_nodes]``. Macros where
            ``freeze_mask[i]`` is True are skipped in the sweep — used by
            Lever M (big-macro-first / two-stage placement).

    Returns:
        (new_positions, improved, num_evals)
        - new_positions: copy of input with accepted moves applied.
        - improved: True iff at least one move was accepted.
        - num_evals: total surrogate evaluations performed in this sweep.
    """
    work = positions.copy()
    n_nodes = work.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_nodes)
    if freeze_mask is not None:
        frozen = np.asarray(freeze_mask, dtype=bool)
        if frozen.shape != (n_nodes,):
            raise ValueError(
                f"freeze_mask shape {frozen.shape} != ({n_nodes},)"
            )
        order = order[~frozen[order]]

    improved = False
    num_evals = 0
    owned_cache = cache is None
    if cache is None:
        cache = build_cache(work, ctx)
    current_cost = float(cache_result(cache).proxy_cost)
    num_evals += 1

    for idx in order:
        node_idx = int(idx)
        best_pos, best_cost = cd_grid_search(
            node_idx=node_idx,
            positions=work,
            ctx=ctx,
            radius=radius,
            k_per_axis=k_per_axis,
            cache=cache,
            tiebreak_enabled=tiebreak_enabled,
            tie_epsilon_rel=tie_epsilon_rel,
            orientation_state=orientation_state,
            search_orientations=search_orientations,
        )
        # cd_grid_search itself runs k_per_axis**2 + 1 evaluations
        num_evals += k_per_axis * k_per_axis + 1
        if best_cost + 1e-9 < current_cost:
            work[node_idx, 0] = best_pos[0]
            work[node_idx, 1] = best_pos[1]
            current_cost = best_cost
            improved = True

    final = fast_proxy(work, ctx)
    cache.total_hpwl = float(final.wirelength)
    cache.total_density = float(final.density)
    cache.total_congestion = float(final.congestion)
    cache.total_overlap_count = int(final.overlap_count)
    if owned_cache:
        # Keep the local cache alive through the sweep for speed, but callers
        # that did not pass it do not observe it.
        del cache

    return work, improved, num_evals


import time as _time


@dataclass(frozen=True)
class CDResult:
    positions: np.ndarray
    final_cost: float
    sweeps_completed: int
    total_evals: int
    plateaued: bool


def cd_loop(
    initial_positions: np.ndarray,
    ctx: FastProxyContext,
    canvas_w: float,
    canvas_h: float,
    max_sweeps: int = 20,
    k_per_axis: int = 8,
    radius_init_ratio: float = 0.25,
    radius_min_ratio: float = 1.0 / 64.0,
    time_budget_s: float = 60.0,
    seed: int = 0,
    freeze_mask: np.ndarray | None = None,
    tiebreak_enabled: bool = False,
    tie_epsilon_rel: float = 1e-3,
    orientation_state: Any | None = None,
    search_orientations: bool = False,
) -> CDResult:
    """Run coordinate-descent sweeps with shrinking search radius.

    Halves the radius every 4 sweeps until ``radius_min_ratio * canvas``,
    then continues at minimum until either a sweep produces no improvement
    (plateau) or ``max_sweeps`` / ``time_budget_s`` is exhausted.

    Returns a CDResult holding the final positions, surrogate cost, and
    bookkeeping.
    """
    canvas_max = max(float(canvas_w), float(canvas_h))
    radius_init = canvas_max * radius_init_ratio
    radius_min = canvas_max * radius_min_ratio

    work = initial_positions.copy()
    total_evals = 0
    sweeps_completed = 0
    plateaued = False
    start = _time.perf_counter()

    for sweep_idx in range(max_sweeps):
        if _time.perf_counter() - start > time_budget_s:
            break

        # Radius schedule: halve every 4 sweeps, floor at radius_min.
        decay_steps = sweep_idx // 4
        radius = max(radius_min, radius_init * (0.5 ** decay_steps))

        new_pos, improved, evals = cd_sweep(
            positions=work,
            ctx=ctx,
            radius=radius,
            k_per_axis=k_per_axis,
            seed=seed + sweep_idx,
            freeze_mask=freeze_mask,
            tiebreak_enabled=tiebreak_enabled,
            tie_epsilon_rel=tie_epsilon_rel,
            orientation_state=orientation_state,
            search_orientations=search_orientations,
        )
        total_evals += evals
        sweeps_completed += 1
        work = new_pos
        if not improved:
            plateaued = True
            break

    final_cost = float(fast_proxy(work, ctx).proxy_cost)
    return CDResult(
        positions=work,
        final_cost=final_cost,
        sweeps_completed=sweeps_completed,
        total_evals=total_evals,
        plateaued=plateaued,
    )
