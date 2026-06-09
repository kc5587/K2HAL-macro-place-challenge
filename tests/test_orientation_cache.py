"""TDD for Lever C Step E — Cache-aware orientation switching.

Tests use a tiny synthetic context + cache via fast_proxy_incremental on a
2-macro 2-pin layout so the rotation effect on HPWL is hand-computable.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_tiny_ctx():
    """Build a tiny FastProxyContext with 2 hard macros, 2 macro pins, 1 net.

    Layout:
      macro 0 at (10, 10), pin offset (1, 0)  → pin abs (11, 10)
      macro 1 at (20, 20), pin offset (0, 0)  → pin abs (20, 20)
      net 0 connects both pins; HPWL = |11-20| + |10-20| = 19 (weight=1)

    After rotating macro 0 by FN (x→-x), pin offset becomes (-1, 0):
      pin abs (9, 10), HPWL = |9-20| + |10-20| = 21
    """
    from macro_place.fast_proxy import FastProxyContext

    ctx = FastProxyContext(
        pin_macro_idx=np.array([0, 1], dtype=np.int32),
        pin_offset_x=np.array([1.0, 0.0], dtype=np.float32),
        pin_offset_y=np.array([0.0, 0.0], dtype=np.float32),
        net_pin_starts=np.array([0, 2], dtype=np.int32),
        net_pin_indices=np.array([0, 1], dtype=np.int32),
        net_weights=np.array([1.0], dtype=np.float32),
        net_source_pin_local=np.array([0], dtype=np.int32),
        macro_w=np.array([5.0, 5.0], dtype=np.float32),
        macro_h=np.array([5.0, 5.0], dtype=np.float32),
        macro_is_hard=np.array([True, True], dtype=bool),
        grid_col=10,
        grid_row=10,
        canvas_w=100.0,
        canvas_h=100.0,
        h_routes_per_micron=1.0,
        v_routes_per_micron=1.0,
        hrouting_alloc=0.5,
        vrouting_alloc=0.5,
        smooth_range=1,
        overlap_threshold=0.0,
        net_cnt=1.0,
    )
    return ctx


def _build_tiny_state(ctx):
    """Build an OrientationState matching the tiny ctx (initial orientations N)."""
    from macro_place.orientation import OrientationState

    n_pins = ctx.pin_macro_idx.shape[0]
    by_ori_x = np.tile(ctx.pin_offset_x.reshape(-1, 1), (1, 8)).astype(np.float32)
    by_ori_y = np.tile(ctx.pin_offset_y.reshape(-1, 1), (1, 8)).astype(np.float32)
    # FN: x→-x
    by_ori_x[:, 1] = -ctx.pin_offset_x
    # S: x→-x, y→-y
    by_ori_x[:, 2] = -ctx.pin_offset_x
    by_ori_y[:, 2] = -ctx.pin_offset_y
    # FS: y→-y
    by_ori_y[:, 3] = -ctx.pin_offset_y
    # E: x = y, y = -x
    by_ori_x[:, 4] = ctx.pin_offset_y
    by_ori_y[:, 4] = -ctx.pin_offset_x
    # FE: x = -y, y = -x
    by_ori_x[:, 5] = -ctx.pin_offset_y
    by_ori_y[:, 5] = -ctx.pin_offset_x
    # W: x = -y, y = x
    by_ori_x[:, 6] = -ctx.pin_offset_y
    by_ori_y[:, 6] = ctx.pin_offset_x
    # FW: x = y, y = x
    by_ori_x[:, 7] = ctx.pin_offset_y
    by_ori_y[:, 7] = ctx.pin_offset_x
    return OrientationState(
        pin_offset_x_by_ori=by_ori_x,
        pin_offset_y_by_ori=by_ori_y,
        macro_orientation=np.zeros(2, dtype=np.int8),
        macro_pin_starts=np.array([0, 1, 2], dtype=np.int32),
        macro_pin_indices=np.array([0, 1], dtype=np.int32),
    )


@pytest.mark.unit
def test_apply_rotation_updates_cache_hpwl_for_affected_nets() -> None:
    """Rotating macro 0 from N to FN must update HPWL of nets touching macro 0."""
    from macro_place.orientation_cache import apply_rotation_to_cache
    from macro_place.fast_proxy_incremental import build_cache, cache_result

    ctx = _build_tiny_ctx()
    state = _build_tiny_state(ctx)
    positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    cache = build_cache(positions, ctx)
    initial_wl = float(cache_result(cache).wirelength)

    prev = apply_rotation_to_cache(cache, ctx, state, macro_idx=0, new_ori_idx=1)  # N → FN
    assert prev == 0

    # After FN, pin offset on macro 0 flipped from (1,0) to (-1,0).
    # Net 0 HPWL: was |11-20| + |10-20| = 19. Now |9-20| + |10-20| = 21.
    # cache_result.wirelength is HPWL normalized by (canvas_w + canvas_h) = 200.
    new_wl = float(cache_result(cache).wirelength)
    assert new_wl == pytest.approx(21.0 / 200.0, abs=1e-5)
    assert initial_wl == pytest.approx(19.0 / 200.0, abs=1e-5)


@pytest.mark.unit
def test_apply_rotation_revert_via_returned_prev_ori() -> None:
    """Calling apply_rotation_to_cache with the returned prev_ori must revert exactly."""
    from macro_place.orientation_cache import apply_rotation_to_cache
    from macro_place.fast_proxy_incremental import build_cache, cache_result

    ctx = _build_tiny_ctx()
    state = _build_tiny_state(ctx)
    positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    cache = build_cache(positions, ctx)
    wl0 = float(cache_result(cache).wirelength)
    pin_x_before = ctx.pin_offset_x.copy()

    prev = apply_rotation_to_cache(cache, ctx, state, macro_idx=0, new_ori_idx=1)
    apply_rotation_to_cache(cache, ctx, state, macro_idx=0, new_ori_idx=prev)
    wl_after = float(cache_result(cache).wirelength)

    assert wl_after == pytest.approx(wl0, abs=1e-5)
    np.testing.assert_allclose(ctx.pin_offset_x, pin_x_before, atol=1e-6)
    assert int(state.macro_orientation[0]) == 0


@pytest.mark.unit
def test_apply_rotation_same_orientation_is_noop() -> None:
    from macro_place.orientation_cache import apply_rotation_to_cache
    from macro_place.fast_proxy_incremental import build_cache, cache_result

    ctx = _build_tiny_ctx()
    state = _build_tiny_state(ctx)
    positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    cache = build_cache(positions, ctx)
    wl0 = float(cache_result(cache).wirelength)
    prev = apply_rotation_to_cache(cache, ctx, state, macro_idx=0, new_ori_idx=0)
    assert prev == 0
    assert float(cache_result(cache).wirelength) == pytest.approx(wl0, abs=1e-12)


@pytest.mark.unit
def test_apply_rotation_cross_class_raises() -> None:
    """N (NS) → E (EW) must be rejected — w/h would swap."""
    from macro_place.orientation_cache import apply_rotation_to_cache
    from macro_place.fast_proxy_incremental import build_cache

    ctx = _build_tiny_ctx()
    state = _build_tiny_state(ctx)
    positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    cache = build_cache(positions, ctx)
    with pytest.raises(ValueError):
        apply_rotation_to_cache(cache, ctx, state, macro_idx=0, new_ori_idx=4)  # N → E


@pytest.mark.unit
def test_apply_rotation_does_not_affect_unrelated_nets() -> None:
    """Rotating macro 1 must not change HPWL of nets that don't touch macro 1."""
    from macro_place.orientation_cache import apply_rotation_to_cache
    from macro_place.fast_proxy_incremental import build_cache, cache_result

    ctx = _build_tiny_ctx()
    state = _build_tiny_state(ctx)
    positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    cache = build_cache(positions, ctx)
    # In this tiny ctx, both macros share net 0, so this is a degenerate case.
    # The point of the test is that the API ONLY recomputes nets in
    # cache.macro_to_nets[macro_idx]. For a 2-macro 1-net layout, that's all
    # nets — so we just sanity-check it runs and yields a sensible HPWL.
    apply_rotation_to_cache(cache, ctx, state, macro_idx=1, new_ori_idx=1)
    wl = float(cache_result(cache).wirelength)
    # macro 1 had pin offset (0,0); FN flips x → still (0,0). HPWL unchanged.
    # Normalized: 19 / 200 = 0.095.
    assert wl == pytest.approx(19.0 / 200.0, abs=1e-5)
