"""Spatial-window LNS destroy seeds.

Selects hard macros inside the most macro-dense region(s) of the canvas, so the
LNS destroy/rebuild step relocates a coherent cluster instead of N isolated
macros. Density correlates strongly with congestion on the ICCAD04 benchmarks
(~65% of proxy cost), so attacking the densest region tends to dissolve a real
routing bottleneck that the per-macro Hessian view cannot see.

The selector is a 2D bucketed search:

  1. Build an ``grid_size x grid_size`` coarse grid over the canvas.
  2. For each hard macro, mark every grid cell its bounding box overlaps.
  3. Iterate grid cells in descending order of marked-macro count. Add every
     macro that touches the current cell to the destroy set, until ``num_select``
     macros are gathered.

The function is intentionally cheap (O(num_hard * grid_size^2 worst case, but
typically O(num_hard) since each bbox touches a constant number of cells).
"""

from __future__ import annotations

import numpy as np


def spatial_window_destroy_seeds(
    positions: np.ndarray,
    macro_w: np.ndarray,
    macro_h: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    num_select: int,
    num_hard_macros: int,
    *,
    grid_size: int = 16,
) -> np.ndarray:
    """Return indices of hard macros inside the densest region(s).

    Args:
        positions: ``[N, 2]`` macro centers (float). Hard macros must be first.
        macro_w: ``[N]`` macro widths.
        macro_h: ``[N]`` macro heights.
        canvas_w, canvas_h: canvas extent in placement units.
        num_select: target number of macros to return.
        num_hard_macros: how many leading rows of ``positions`` are hard macros
            (the only movable kind, the only ones legally destructible).
        grid_size: coarse-grid resolution per axis.

    Returns:
        ``int64`` array of unique hard-macro indices, length ``<= num_select``.
        Empty array if no hard macros are eligible.
    """
    if num_hard_macros <= 0 or num_select <= 0 or grid_size <= 0:
        return np.empty(0, dtype=np.int64)
    if canvas_w <= 0.0 or canvas_h <= 0.0:
        return np.empty(0, dtype=np.int64)

    num_select = int(min(num_select, num_hard_macros))
    cell_w = float(canvas_w) / grid_size
    cell_h = float(canvas_h) / grid_size

    # Per-cell macro count.
    density = np.zeros((grid_size, grid_size), dtype=np.int64)
    # Per-macro list of cells it overlaps (stored as flat (r * grid_size + c)).
    macro_cells: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * num_hard_macros

    for i in range(num_hard_macros):
        cx = float(positions[i, 0])
        cy = float(positions[i, 1])
        half_w = float(macro_w[i]) * 0.5
        half_h = float(macro_h[i]) * 0.5
        c_lo = max(0, int((cx - half_w) / cell_w))
        c_hi = min(grid_size - 1, int((cx + half_w) / cell_w))
        r_lo = max(0, int((cy - half_h) / cell_h))
        r_hi = min(grid_size - 1, int((cy + half_h) / cell_h))
        if r_hi < r_lo or c_hi < c_lo:
            continue

        rows = np.arange(r_lo, r_hi + 1, dtype=np.int64)
        cols = np.arange(c_lo, c_hi + 1, dtype=np.int64)
        # Flat cell ids for this macro's footprint.
        cells = (rows[:, None] * grid_size + cols[None, :]).ravel()
        macro_cells[i] = cells
        density[rows[:, None], cols[None, :]] += 1

    # Reverse-map cell -> list of macros that touch it.
    cell_macros: list[list[int]] = [[] for _ in range(grid_size * grid_size)]
    for i in range(num_hard_macros):
        for cell_flat in macro_cells[i]:
            cell_macros[int(cell_flat)].append(i)

    # Walk cells in descending density order; accumulate macros.
    flat_density = density.ravel()
    order = np.argsort(-flat_density, kind="stable")

    selected: list[int] = []
    seen: set[int] = set()
    for cell_flat in order:
        if flat_density[cell_flat] == 0:
            break
        for macro_idx in cell_macros[int(cell_flat)]:
            if macro_idx in seen:
                continue
            seen.add(macro_idx)
            selected.append(macro_idx)
            if len(selected) >= num_select:
                break
        if len(selected) >= num_select:
            break

    return np.asarray(selected, dtype=np.int64)
