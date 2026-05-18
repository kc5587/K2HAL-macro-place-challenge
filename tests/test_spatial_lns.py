"""Unit tests for macro_place.spatial_lns.spatial_window_destroy_seeds."""

from __future__ import annotations

import numpy as np

from macro_place.spatial_lns import spatial_window_destroy_seeds


def _uniform_macros(n: int, *, size: float = 1.0, canvas: float = 100.0) -> tuple:
    """Equally-spaced macros on a grid. Useful for predictable cell counts."""
    side = int(np.ceil(np.sqrt(n)))
    step = canvas / (side + 1)
    positions = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        r = i // side
        c = i % side
        positions[i, 0] = (c + 1) * step
        positions[i, 1] = (r + 1) * step
    macro_w = np.full(n, size, dtype=np.float64)
    macro_h = np.full(n, size, dtype=np.float64)
    return positions, macro_w, macro_h


def test_empty_when_no_hard_macros():
    pos, mw, mh = _uniform_macros(10)
    out = spatial_window_destroy_seeds(pos, mw, mh, 100.0, 100.0, 5, num_hard_macros=0)
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_empty_when_num_select_zero():
    pos, mw, mh = _uniform_macros(10)
    out = spatial_window_destroy_seeds(pos, mw, mh, 100.0, 100.0, 0, num_hard_macros=10)
    assert out.shape == (0,)


def test_returns_indices_within_hard_macros():
    pos, mw, mh = _uniform_macros(20)
    out = spatial_window_destroy_seeds(pos, mw, mh, 100.0, 100.0, 5, num_hard_macros=15)
    assert out.dtype == np.int64
    assert len(out) <= 5
    assert len(set(out.tolist())) == len(out), "indices must be unique"
    assert (out >= 0).all() and (out < 15).all()


def test_cap_at_num_hard_macros():
    pos, mw, mh = _uniform_macros(5)
    out = spatial_window_destroy_seeds(pos, mw, mh, 100.0, 100.0, 50, num_hard_macros=5)
    assert len(out) == 5
    assert set(out.tolist()) == {0, 1, 2, 3, 4}


def test_picks_densest_region_first():
    # Place 4 macros tightly clustered in top-right, 4 isolated elsewhere.
    pos = np.array(
        [
            # Cluster at (90, 90).
            [89.0, 89.0],
            [91.0, 89.0],
            [89.0, 91.0],
            [91.0, 91.0],
            # Scattered isolates.
            [10.0, 10.0],
            [50.0, 10.0],
            [10.0, 50.0],
            [50.0, 50.0],
        ],
        dtype=np.float64,
    )
    mw = np.full(8, 4.0, dtype=np.float64)
    mh = np.full(8, 4.0, dtype=np.float64)
    out = spatial_window_destroy_seeds(
        pos, mw, mh, canvas_w=100.0, canvas_h=100.0,
        num_select=4, num_hard_macros=8, grid_size=16,
    )
    # All 4 of the cluster (indices 0..3) should be selected first.
    assert set(out.tolist()) == {0, 1, 2, 3}


def test_falls_back_to_lower_density_cells_when_needed():
    # 2 clustered macros + 6 isolates; we ask for 5 picks → must include some isolates.
    pos = np.array(
        [
            [50.0, 50.0],
            [51.0, 51.0],
            [10.0, 10.0],
            [10.0, 90.0],
            [90.0, 10.0],
            [90.0, 90.0],
            [25.0, 25.0],
            [75.0, 75.0],
        ],
        dtype=np.float64,
    )
    mw = np.full(8, 2.0, dtype=np.float64)
    mh = np.full(8, 2.0, dtype=np.float64)
    out = spatial_window_destroy_seeds(
        pos, mw, mh, 100.0, 100.0, num_select=5, num_hard_macros=8, grid_size=16,
    )
    assert len(out) == 5
    # The cluster (0, 1) must be in the selection since they share a cell.
    assert 0 in set(out.tolist()) and 1 in set(out.tolist())


def test_ignores_soft_macros_even_if_in_dense_cell():
    # Hard macros 0..3 scattered; soft macros 4..7 clustered. Selector must only
    # ever return hard-macro indices, never the soft ones.
    pos = np.array(
        [
            [10.0, 10.0],
            [30.0, 30.0],
            [70.0, 70.0],
            [90.0, 90.0],
            # Soft-macro cluster — must be ignored.
            [50.0, 50.0],
            [50.5, 50.0],
            [50.0, 50.5],
            [50.5, 50.5],
        ],
        dtype=np.float64,
    )
    mw = np.full(8, 1.0, dtype=np.float64)
    mh = np.full(8, 1.0, dtype=np.float64)
    out = spatial_window_destroy_seeds(
        pos, mw, mh, 100.0, 100.0, num_select=4, num_hard_macros=4, grid_size=16,
    )
    assert (out < 4).all()
    assert len(set(out.tolist())) == len(out)


def test_handles_zero_canvas_gracefully():
    pos, mw, mh = _uniform_macros(4)
    out = spatial_window_destroy_seeds(pos, mw, mh, 0.0, 100.0, 2, num_hard_macros=4)
    assert out.shape == (0,)


def test_deterministic_across_calls():
    pos, mw, mh = _uniform_macros(20)
    out1 = spatial_window_destroy_seeds(pos, mw, mh, 100.0, 100.0, 8, num_hard_macros=20)
    out2 = spatial_window_destroy_seeds(pos, mw, mh, 100.0, 100.0, 8, num_hard_macros=20)
    np.testing.assert_array_equal(out1, out2)
