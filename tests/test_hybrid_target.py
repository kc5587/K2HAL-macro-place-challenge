"""TDD for Lever 3 — Hybrid target macro scoring.

Tests rely on a tiny synthetic FastProxyContext-like stub so they don't depend
on benchmark loading. The hybrid scoring function only reads a small set of
fields from the context, so duck typing works.
"""
from __future__ import annotations

import numpy as np
import pytest


def _stub_ctx(num_macros: int = 6, grid: tuple = (4, 4)):
    """Build a minimal FastProxyContext-like object.

    Provides only the attributes ``hybrid_target_hard_macros`` actually reads:
    macro_is_hard, macro_w, macro_h, canvas_w/h, grid_row/col, net CSR, pins.
    """
    grid_row, grid_col = grid

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.macro_is_hard = np.ones(num_macros, dtype=bool)
    ctx.macro_w = np.linspace(10.0, 60.0, num_macros, dtype=np.float64)  # varied areas
    ctx.macro_h = np.linspace(10.0, 60.0, num_macros, dtype=np.float64)
    ctx.canvas_w = 100.0
    ctx.canvas_h = 100.0
    ctx.grid_row = grid_row
    ctx.grid_col = grid_col
    # No pins / no nets so net_degree is zero (we'll test degree separately).
    ctx.pin_macro_idx = np.empty(0, dtype=np.int64)
    ctx.net_pin_starts = np.zeros(1, dtype=np.int64)
    ctx.net_pin_indices = np.empty(0, dtype=np.int64)
    ctx.net_weights = np.empty(0, dtype=np.float64)
    ctx.net_source_pin_local = np.empty(0, dtype=np.int64)
    ctx.pin_offset_x = np.empty(0, dtype=np.float64)
    ctx.pin_offset_y = np.empty(0, dtype=np.float64)
    ctx.h_routes_per_micron = 1.0
    ctx.v_routes_per_micron = 1.0
    ctx.hrouting_alloc = 0.5
    ctx.vrouting_alloc = 0.5
    ctx.smooth_range = 1
    ctx.overlap_threshold = 0.0
    ctx.net_cnt = 1.0
    return ctx


@pytest.mark.unit
def test_hybrid_target_deterministic() -> None:
    """Same inputs must always return the same target indices."""
    from macro_place.hybrid_target import hybrid_target_hard_macros

    ctx = _stub_ctx(num_macros=6)
    pos = np.array(
        [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]],
        dtype=np.float64,
    )
    a = hybrid_target_hard_macros(pos, ctx, num_seeds=3)
    b = hybrid_target_hard_macros(pos, ctx, num_seeds=3)
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_hybrid_target_respects_num_seeds() -> None:
    from macro_place.hybrid_target import hybrid_target_hard_macros

    ctx = _stub_ctx(num_macros=6)
    pos = np.array(
        [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]],
        dtype=np.float64,
    )
    assert hybrid_target_hard_macros(pos, ctx, num_seeds=2).size == 2
    assert hybrid_target_hard_macros(pos, ctx, num_seeds=100).size == 6
    assert hybrid_target_hard_macros(pos, ctx, num_seeds=0).size == 0


@pytest.mark.unit
def test_hybrid_target_prefers_larger_area_when_only_area_weighted() -> None:
    """With weights (0,1,0,0), only area matters → largest-area macro picked."""
    from macro_place.hybrid_target import hybrid_target_hard_macros

    ctx = _stub_ctx(num_macros=4)
    pos = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float64)
    # macro_w/h are linspaced ascending → macro 3 has largest area.
    out = hybrid_target_hard_macros(pos, ctx, num_seeds=1, weights=(0.0, 1.0, 0.0, 0.0))
    assert out.tolist() == [3]


@pytest.mark.unit
def test_hybrid_target_changes_with_weights() -> None:
    """Different weight vectors must produce different orderings."""
    from macro_place.hybrid_target import hybrid_target_hard_macros

    ctx = _stub_ctx(num_macros=6)
    pos = np.array(
        [[10, 10], [10, 10], [50, 50], [50, 50], [80, 80], [80, 80]],
        dtype=np.float64,
    )
    out_a = hybrid_target_hard_macros(
        pos, ctx, num_seeds=3, weights=(1.0, 0.0, 0.0, 0.0)
    )
    out_b = hybrid_target_hard_macros(
        pos, ctx, num_seeds=3, weights=(0.0, 1.0, 0.0, 0.0)
    )
    # Not necessarily disjoint, but at least one element must differ between
    # the congestion-only and area-only orderings on this synthetic case.
    assert not np.array_equal(out_a, out_b)


@pytest.mark.unit
def test_hybrid_target_returns_int64_hard_indices() -> None:
    from macro_place.hybrid_target import hybrid_target_hard_macros

    ctx = _stub_ctx(num_macros=5)
    ctx.macro_is_hard = np.array([True, False, True, False, True], dtype=bool)
    pos = np.zeros((5, 2), dtype=np.float64)
    out = hybrid_target_hard_macros(pos, ctx, num_seeds=10)
    # Only hard macros (indices 0,2,4) should appear.
    assert out.dtype == np.int64
    assert set(out.tolist()).issubset({0, 2, 4})


@pytest.mark.unit
def test_default_config_uses_congestion_strategy() -> None:
    """Default targeted_sa_target_strategy must preserve current behavior."""
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    cfg = CDLNSPlacer()._config
    assert "targeted_sa_target_strategy" in cfg
    assert cfg["targeted_sa_target_strategy"] == "congestion"
