"""Destroy-K-rebuild local-neighborhood search (Bet 7 restart).

When CD plateaus, LNS picks K random nodes, treats them as floating,
and runs small-scale CD over just those K nodes (others frozen).
Accepts the new layout iff its surrogate proxy is strictly lower than
the pre-destroy cost.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from macro_place.cd import cd_grid_search
from macro_place.fast_proxy import FastProxyContext, fast_proxy
from macro_place.fast_proxy_incremental import build_cache, cache_result
from macro_place.orientation import OrientationState, orientation_class_indices
from macro_place.orientation_cache import apply_rotation_to_cache


def lns_destroy_rebuild(
    positions: np.ndarray,
    ctx: FastProxyContext,
    canvas_w: float,
    canvas_h: float,
    num_destroy: int = 8,
    max_lns_iters: int = 20,
    k_per_axis: int = 8,
    seed: int = 0,
    destroy_seed_indices: np.ndarray | None = None,
    orientation_state: OrientationState | None = None,
    rotation_probability: float = 0.0,
) -> Tuple[np.ndarray, bool, int]:
    """Destroy ``num_destroy`` nodes and CD-rebuild on that subset.

    Args:
        positions: float64 array shape ``[num_nodes, 2]``.
        ctx: pre-built FastProxyContext.
        canvas_w, canvas_h: canvas dimensions for radius bound.
        num_destroy: number of nodes to re-place this iteration.
        max_lns_iters: max rebuild sweeps over the destroyed subset.
        k_per_axis: candidates per axis in each grid search.
        seed: RNG seed for which nodes to destroy and sweep order.
        destroy_seed_indices: optional explicit destroy set. When given,
            its (deduplicated, clipped-to-node-count) entries are used
            up to ``num_destroy``; any shortfall is filled with random
            picks from the remaining nodes. When ``None``, the destroy
            set is fully random — backward-compatible with the original
            signature.

    Returns:
        (new_positions, accepted, total_evals)
        - new_positions: a copy of positions; either the rebuilt layout
          (if it strictly beats the input) or the original layout (otherwise).
        - accepted: whether the rebuilt layout was accepted.
        - total_evals: surrogate evaluations performed.
    """
    rng = np.random.default_rng(seed)
    n_nodes = positions.shape[0]
    if num_destroy >= n_nodes:
        num_destroy = max(1, n_nodes - 1)

    if destroy_seed_indices is not None and len(destroy_seed_indices) > 0:
        seeds = np.asarray(destroy_seed_indices, dtype=np.int64).ravel()
        # Drop out-of-range entries and dedupe while preserving order.
        seeds = seeds[(seeds >= 0) & (seeds < n_nodes)]
        _, first_pos = np.unique(seeds, return_index=True)
        seeds = seeds[np.sort(first_pos)]
        seeds = seeds[:num_destroy]
        if seeds.shape[0] < num_destroy:
            # Pad with random picks from the remaining nodes.
            remaining = np.setdiff1d(
                np.arange(n_nodes, dtype=np.int64), seeds, assume_unique=True
            )
            fill_n = int(num_destroy - seeds.shape[0])
            if fill_n > 0 and remaining.shape[0] > 0:
                fill = rng.choice(
                    remaining,
                    size=min(fill_n, remaining.shape[0]),
                    replace=False,
                )
                seeds = np.concatenate([seeds, fill.astype(np.int64)])
        destroyed = seeds
    else:
        destroyed = rng.choice(n_nodes, size=num_destroy, replace=False)
    canvas_max = max(float(canvas_w), float(canvas_h))
    radius = canvas_max * 0.25  # broad initial rebuild radius

    work = positions.copy()
    cache = build_cache(work, ctx)
    base_cost = float(cache_result(cache).proxy_cost)
    total_evals = 1

    rotation_enabled = (
        orientation_state is not None and float(rotation_probability) > 0.0
    )
    rot_prob = float(rotation_probability) if rotation_enabled else 0.0
    initial_orientations = (
        orientation_state.macro_orientation.copy()
        if rotation_enabled and orientation_state is not None
        else None
    )

    current_cost = base_cost
    for it in range(max_lns_iters):
        improved_this_iter = False
        order = rng.permutation(destroyed)
        for node_idx in order:
            best_pos, best_cost = cd_grid_search(
                node_idx=int(node_idx),
                positions=work,
                ctx=ctx,
                radius=radius,
                k_per_axis=k_per_axis,
                cache=cache,
            )
            total_evals += k_per_axis * k_per_axis + 1
            if best_cost + 1e-9 < current_cost:
                work[int(node_idx), 0] = best_pos[0]
                work[int(node_idx), 1] = best_pos[1]
                current_cost = best_cost
                improved_this_iter = True

            if (
                rotation_enabled
                and orientation_state is not None
                and rng.random() < rot_prob
            ):
                node_i = int(node_idx)
                ori_state = orientation_state
                cur_ori = int(ori_state.macro_orientation[node_i])
                class_oris = orientation_class_indices(cur_ori)
                best_alt_cost = current_cost
                best_alt_ori = cur_ori
                for alt_ori in class_oris:
                    if int(alt_ori) == cur_ori:
                        continue
                    prev = apply_rotation_to_cache(
                        cache, ctx, ori_state, node_i, int(alt_ori)
                    )
                    alt_cost = float(cache_result(cache).proxy_cost)
                    total_evals += 1
                    if alt_cost + 1e-9 < best_alt_cost:
                        best_alt_cost = alt_cost
                        best_alt_ori = int(alt_ori)
                    apply_rotation_to_cache(cache, ctx, ori_state, node_i, prev)
                if best_alt_ori != cur_ori:
                    apply_rotation_to_cache(
                        cache, ctx, ori_state, node_i, best_alt_ori
                    )
                    current_cost = best_alt_cost
                    improved_this_iter = True
        if not improved_this_iter:
            break

    final = fast_proxy(work, ctx)
    current_cost = float(final.proxy_cost)
    cache.total_hpwl = float(final.wirelength)
    cache.total_density = float(final.density)
    cache.total_congestion = float(final.congestion)
    cache.total_overlap_count = int(final.overlap_count)

    if current_cost + 1e-9 < base_cost:
        return work, True, total_evals

    # Reject: restore initial orientations so caller-visible state is unchanged.
    if (
        rotation_enabled
        and orientation_state is not None
        and initial_orientations is not None
    ):
        ori_state = orientation_state
        for macro_i in range(initial_orientations.shape[0]):
            target = int(initial_orientations[macro_i])
            current = int(ori_state.macro_orientation[macro_i])
            if target != current:
                apply_rotation_to_cache(cache, ctx, ori_state, macro_i, target)
    return positions.copy(), False, total_evals
