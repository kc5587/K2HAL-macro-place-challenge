"""TDD for Lever C Step F — SA rotation moves.

``generate_sa_candidates`` gains two optional params: ``orientation_state``
and ``rotation_probability``. With probability 0 (default) the function must
be bit-exact identical to the prior implementation. With probability > 0,
each step has that probability of being a rotation proposal instead of a
translation; rotations stay same-class.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_ctx_two_macros():
    from macro_place.fast_proxy import FastProxyContext

    return FastProxyContext(
        pin_macro_idx=np.array([0, 1], dtype=np.int32),
        pin_offset_x=np.array([3.0, -2.0], dtype=np.float32),
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


def _build_state(ctx):
    from macro_place.orientation import OrientationState

    n_pins = ctx.pin_macro_idx.shape[0]
    by_ori_x = np.zeros((n_pins, 8), dtype=np.float32)
    by_ori_y = np.zeros((n_pins, 8), dtype=np.float32)
    bx = ctx.pin_offset_x
    by = ctx.pin_offset_y
    by_ori_x[:, 0], by_ori_y[:, 0] = bx, by
    by_ori_x[:, 1], by_ori_y[:, 1] = -bx, by
    by_ori_x[:, 2], by_ori_y[:, 2] = -bx, -by
    by_ori_x[:, 3], by_ori_y[:, 3] = bx, -by
    by_ori_x[:, 4], by_ori_y[:, 4] = by, -bx
    by_ori_x[:, 5], by_ori_y[:, 5] = -by, -bx
    by_ori_x[:, 6], by_ori_y[:, 6] = -by, bx
    by_ori_x[:, 7], by_ori_y[:, 7] = by, bx
    return OrientationState(
        pin_offset_x_by_ori=by_ori_x,
        pin_offset_y_by_ori=by_ori_y,
        macro_orientation=np.zeros(2, dtype=np.int8),
        macro_pin_starts=np.array([0, 1, 2], dtype=np.int32),
        macro_pin_indices=np.array([0, 1], dtype=np.int32),
    )


@pytest.mark.unit
def test_sa_rotation_off_is_bit_exact() -> None:
    """rotation_probability=0.0 (default) must match the prior call signature."""
    from macro_place.sa_generator import generate_sa_candidates

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)

    baseline = generate_sa_candidates(
        initial_positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        seed=11,
        steps=12,
        num_candidates=1,
    )
    same = generate_sa_candidates(
        initial_positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        seed=11,
        steps=12,
        num_candidates=1,
        orientation_state=None,
        rotation_probability=0.0,
    )
    assert len(baseline) == len(same)
    for a, b in zip(baseline, same):
        assert a.proxy_cost == pytest.approx(b.proxy_cost, rel=0, abs=0)
        np.testing.assert_array_equal(a.positions, b.positions)


@pytest.mark.unit
def test_sa_rotation_state_passed_zero_prob_no_mutation() -> None:
    """Passing a state but probability=0 must leave state and ctx untouched."""
    from macro_place.sa_generator import generate_sa_candidates

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)
    state = _build_state(ctx)
    initial_ori = state.macro_orientation.copy()
    initial_pin_x = ctx.pin_offset_x.copy()

    generate_sa_candidates(
        initial_positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        seed=3,
        steps=8,
        num_candidates=1,
        orientation_state=state,
        rotation_probability=0.0,
    )
    np.testing.assert_array_equal(state.macro_orientation, initial_ori)
    np.testing.assert_allclose(ctx.pin_offset_x, initial_pin_x, atol=1e-6)


@pytest.mark.unit
def test_sa_rotation_on_stays_same_class() -> None:
    """rotation_probability=1.0 must keep every macro in its starting class."""
    from macro_place.sa_generator import generate_sa_candidates
    from macro_place.orientation import NS_CLASS_INDICES

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)
    state = _build_state(ctx)  # both start N (NS class)

    generate_sa_candidates(
        initial_positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        seed=5,
        steps=20,
        num_candidates=1,
        orientation_state=state,
        rotation_probability=1.0,
    )
    for ori in state.macro_orientation:
        assert int(ori) in NS_CLASS_INDICES


@pytest.mark.unit
def test_sa_rotation_default_args_unchanged() -> None:
    """Existing callers without rotation kwargs must still work."""
    from macro_place.sa_generator import generate_sa_candidates

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)
    out = generate_sa_candidates(
        initial_positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        seed=0,
        steps=5,
        num_candidates=1,
    )
    assert isinstance(out, list)
