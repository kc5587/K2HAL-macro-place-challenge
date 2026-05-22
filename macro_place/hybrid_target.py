"""Lever 3 — Hybrid target macro scoring for targeted-SA escape.

The default targeted-SA target selection uses ``worst_congestion_bin_destroy_seeds``
(picks macros near hottest congestion bins). Lever 3 adds an optional hybrid
score that combines four factors:

  hybrid_score = w_c * congestion_hotness
               + w_a * macro_area
               + w_d * net_degree
               + w_l * local_density_pressure

Each factor is min-max normalized across hard macros to [0, 1]. The function is
deterministic (no RNG) so the same inputs produce the same target set across
runs.
"""
from __future__ import annotations

import numpy as np

from macro_place.fast_proxy import (
    FastProxyContext,
    fast_congestion_per_bin,
)


def _normalize_to_unit(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Zero vector when all values are equal."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= 1e-30:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _net_degree_per_macro(ctx: FastProxyContext) -> np.ndarray:
    """Count the number of nets each macro participates in (via its pins).

    Uses ``pin_macro_idx`` (per-pin owner macro) and the net CSR layout.
    """
    num_macros = int(ctx.macro_is_hard.shape[0])
    degree = np.zeros(num_macros, dtype=np.float64)
    pin_macro_idx = np.asarray(ctx.pin_macro_idx, dtype=np.int64)
    net_pin_starts = np.asarray(ctx.net_pin_starts, dtype=np.int64)
    net_pin_indices = np.asarray(ctx.net_pin_indices, dtype=np.int64)
    num_nets = net_pin_starts.shape[0] - 1
    for net_idx in range(num_nets):
        start = int(net_pin_starts[net_idx])
        end = int(net_pin_starts[net_idx + 1])
        if end <= start:
            continue
        pins = net_pin_indices[start:end]
        macros_in_net = pin_macro_idx[pins]
        unique_macros = np.unique(macros_in_net[macros_in_net >= 0])
        for m in unique_macros:
            if 0 <= m < num_macros:
                degree[int(m)] += 1.0
    return degree


def _local_macro_crowding(positions: np.ndarray, ctx: FastProxyContext) -> np.ndarray:
    """Per-macro count of OTHER macro centers in the same bin (local crowding).

    Used as a substitute for "local density pressure" since fast_proxy exposes
    a scalar density helper but no per-bin grid. Counting macro centers per
    bin captures the same intuition (a macro in a crowded region scores higher).
    """
    grid_row = int(ctx.grid_row)
    grid_col = int(ctx.grid_col)
    if grid_row == 0 or grid_col == 0:
        return np.zeros(positions.shape[0], dtype=np.float64)
    cell_w = float(ctx.canvas_w) / float(grid_col)
    cell_h = float(ctx.canvas_h) / float(grid_row)
    centers = _macro_centers(positions, ctx)
    col = np.clip(np.floor(centers[:, 0] / max(cell_w, 1e-30)).astype(int), 0, grid_col - 1)
    row = np.clip(np.floor(centers[:, 1] / max(cell_h, 1e-30)).astype(int), 0, grid_row - 1)
    # Build per-bin macro-count grid, then read each macro's bin value (minus
    # self) so a macro in a bin alone scores 0.
    counts = np.zeros((grid_row, grid_col), dtype=np.float64)
    for r, c in zip(row, col):
        counts[int(r), int(c)] += 1.0
    per_macro = counts[row, col] - 1.0
    return np.clip(per_macro, 0.0, None)


def _macro_centers(positions: np.ndarray, ctx: FastProxyContext) -> np.ndarray:
    """Return macro centers [num_macros, 2]."""
    pos = np.asarray(positions, dtype=np.float64)
    half_w = np.asarray(ctx.macro_w, dtype=np.float64) * 0.5
    half_h = np.asarray(ctx.macro_h, dtype=np.float64) * 0.5
    centers = pos.copy()
    centers[:, 0] += half_w
    centers[:, 1] += half_h
    return centers


def _per_macro_bin_value(
    positions: np.ndarray, ctx: FastProxyContext, per_bin: np.ndarray
) -> np.ndarray:
    """Read each macro's center bin value from a per-bin grid."""
    grid_row, grid_col = per_bin.shape
    if grid_row == 0 or grid_col == 0:
        return np.zeros(positions.shape[0], dtype=np.float64)
    cell_w = float(ctx.canvas_w) / float(grid_col)
    cell_h = float(ctx.canvas_h) / float(grid_row)
    centers = _macro_centers(positions, ctx)
    col = np.clip(np.floor(centers[:, 0] / max(cell_w, 1e-30)).astype(int), 0, grid_col - 1)
    row = np.clip(np.floor(centers[:, 1] / max(cell_h, 1e-30)).astype(int), 0, grid_row - 1)
    return per_bin[row, col].astype(np.float64)


def hybrid_target_hard_macros(
    positions: np.ndarray,
    ctx: FastProxyContext,
    *,
    num_seeds: int,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Select top hard-macro indices by hybrid score (congestion + area + degree + density).

    All four per-macro factors are min-max normalized to [0,1] over hard macros,
    then summed with ``weights``. Deterministic; identical inputs return
    identical outputs.

    Returns up to ``num_seeds`` ``int64`` indices, ranked descending by score.
    Ties broken by ascending macro index for determinism.
    """
    if num_seeds <= 0:
        return np.empty(0, dtype=np.int64)
    is_hard = np.asarray(ctx.macro_is_hard, dtype=bool)
    hard_idx = np.flatnonzero(is_hard)
    if hard_idx.size == 0:
        return np.empty(0, dtype=np.int64)

    # Per-macro raw factors (over ALL macros; we'll slice hard only at the end).
    per_bin_cong = fast_congestion_per_bin(np.asarray(positions, dtype=np.float64), ctx)
    f_cong = _per_macro_bin_value(positions, ctx, per_bin_cong)
    f_area = np.asarray(ctx.macro_w, dtype=np.float64) * np.asarray(
        ctx.macro_h, dtype=np.float64
    )
    f_deg = _net_degree_per_macro(ctx)
    f_dens = _local_macro_crowding(np.asarray(positions, dtype=np.float64), ctx)

    # Slice to hard macros and normalize per-axis.
    n_cong = _normalize_to_unit(f_cong[hard_idx])
    n_area = _normalize_to_unit(f_area[hard_idx])
    n_deg = _normalize_to_unit(f_deg[hard_idx])
    n_dens = _normalize_to_unit(f_dens[hard_idx])

    w_c, w_a, w_d, w_l = (float(w) for w in weights)
    score = w_c * n_cong + w_a * n_area + w_d * n_deg + w_l * n_dens

    # Stable sort: primary descending by score, tie-break ascending by hard_idx.
    order = np.lexsort((hard_idx, -score))
    take = min(int(num_seeds), hard_idx.size)
    return hard_idx[order[:take]].astype(np.int64)
