"""Worst-congestion-bin destroy seed selection for LNS (Lever L).

Picks hard-macro indices whose footprints overlap the most congested bins,
where congestion = the per-bin demand grid from ``fast_congestion_per_bin``.

Motivation: EDA in docs/superpowers/specs/2026-05-19-cost-decomp-eda.md
shows congestion = 65.5% of proxy averaged over 17 IBM benches. Existing
LNS destroy sources (random, spatial-window, Hessian) target HPWL/density
or geometric clustering, not congestion directly. L picks macros sitting
in the most congested bins, which is the dominant cost on every bench
where proxy > 1.0.
"""

from __future__ import annotations

import numpy as np

from macro_place.fast_proxy import FastProxyContext, fast_congestion_per_bin


def worst_congestion_bin_destroy_seeds(
    positions: np.ndarray,
    ctx: FastProxyContext,
    *,
    num_seeds: int,
    top_n_bins: int = 8,
    macros_per_bin: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Return hard-macro indices nearest to the worst-congestion bins.

    Key insight from EDA: congestion peaks happen in routing CHANNELS
    between macros, not inside macro footprints. A bbox-overlap heuristic
    finds nothing because hot bins are empty. Instead we pick macros
    nearest to each hot bin's center — those macros pin the nets routing
    through that hot channel.

    Algorithm:
      1. Compute per-bin congestion via ``fast_congestion_per_bin``.
      2. argsort descending; take the top ``top_n_bins`` bins.
      3. For each top bin: rank hard macros by Euclidean distance from
         macro center to bin center; take the closest ``macros_per_bin``.
      4. Tie-break across bins by congestion magnitude × inverse-distance:
         macros near hotter bins are picked first.
      5. Dedupe; return up to ``num_seeds`` indices as int64.

    ``seed`` is currently unused but accepted for API symmetry with the
    other destroy-seed sources; future revisions may sample within ties.
    """
    del seed  # reserved for future stochastic tie-breaking
    if num_seeds <= 0 or top_n_bins <= 0:
        return np.empty(0, dtype=np.int64)

    pos = np.asarray(positions, dtype=np.float64)
    num_hard = int(ctx.macro_is_hard.sum())
    if num_hard < 1:
        return np.empty(0, dtype=np.int64)

    per_bin = fast_congestion_per_bin(pos, ctx)  # [grid_row, grid_col]
    grid_row, grid_col = per_bin.shape
    flat = per_bin.ravel()
    if flat.size == 0:
        return np.empty(0, dtype=np.int64)

    k = min(int(top_n_bins), flat.size)
    top_idx_unsorted = np.argpartition(flat, flat.size - k)[flat.size - k:]
    top_idx = top_idx_unsorted[np.argsort(-flat[top_idx_unsorted])]

    cell_w = ctx.canvas_w / grid_col
    cell_h = ctx.canvas_h / grid_row
    hard_pos = pos[:num_hard]
    per_macros = max(1, int(macros_per_bin))

    # Score each candidate macro by sum over hot bins of (congestion / (dist + epsilon)).
    # Macros near multiple hot bins get amplified; far-away macros score near zero.
    epsilon = 0.25 * min(cell_w, cell_h)
    scores = np.zeros(num_hard, dtype=np.float64)
    for flat_bin in top_idx:
        row = int(flat_bin // grid_col)
        col = int(flat_bin % grid_col)
        bin_cx = (col + 0.5) * cell_w
        bin_cy = (row + 0.5) * cell_h
        bin_cong = float(flat[flat_bin])
        d = np.sqrt(
            (hard_pos[:, 0] - bin_cx) ** 2 + (hard_pos[:, 1] - bin_cy) ** 2
        )
        # Only the closest `per_macros` macros to this bin contribute to its score
        nearest = np.argpartition(d, per_macros - 1)[:per_macros]
        contribution = bin_cong / (d[nearest] + epsilon)
        scores[nearest] += contribution

    # Pick the top-scoring macros across all hot bins
    if scores.max() <= 0.0:
        return np.empty(0, dtype=np.int64)
    candidate_count = int(min(num_seeds, num_hard))
    order = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
    order = order[np.argsort(-scores[order])]
    # Drop macros that scored zero (no hot-bin neighborhood)
    order = order[scores[order] > 0.0]
    return order.astype(np.int64)[:num_seeds]
