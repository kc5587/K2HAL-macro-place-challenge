"""Coordinate-descent core for the CD+LNS placer (Bet 7 restart).

For each node, holds all others fixed and grid-searches a candidate
window for the lowest-proxy position. Calls into ``fast_proxy`` for
every candidate evaluation; the surrogate's calibrated <1% error is
sufficient for inner-loop guidance, with the official ``compute_proxy_cost``
used only for final validation in the placer entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from macro_place.fast_proxy import FastProxyContext, fast_proxy


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

    work = positions.copy()
    best_pos = np.array([cx, cy], dtype=np.float64)
    best_cost = float(fast_proxy(positions, ctx).proxy_cost)

    for gy in grid_y:
        for gx in grid_x:
            work[node_idx, 0] = gx
            work[node_idx, 1] = gy
            cost = float(fast_proxy(work, ctx).proxy_cost)
            if cost < best_cost:
                best_cost = cost
                best_pos = np.array([gx, gy], dtype=np.float64)

    return best_pos, best_cost


def cd_sweep(
    positions: np.ndarray,
    ctx: FastProxyContext,
    radius: float,
    k_per_axis: int = 8,
    seed: int = 0,
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

    improved = False
    num_evals = 0
    current_cost = float(fast_proxy(work, ctx).proxy_cost)
    num_evals += 1

    for idx in order:
        node_idx = int(idx)
        best_pos, best_cost = cd_grid_search(
            node_idx=node_idx,
            positions=work,
            ctx=ctx,
            radius=radius,
            k_per_axis=k_per_axis,
        )
        # cd_grid_search itself runs k_per_axis**2 + 1 evaluations
        num_evals += k_per_axis * k_per_axis + 1
        if best_cost + 1e-9 < current_cost:
            work[node_idx, 0] = best_pos[0]
            work[node_idx, 1] = best_pos[1]
            current_cost = best_cost
            improved = True

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
