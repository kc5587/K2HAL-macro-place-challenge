"""TDD for Lever C Step F — LNS rotation moves.

`lns_destroy_rebuild` gains two optional params: ``orientation_state`` and
``rotation_probability``. When the probability is 0.0 (default) the function
must be bit-exact identical to the prior implementation. When > 0, the
rebuild loop additionally proposes same-class rotations for destroyed
macros and keeps the rotation iff it lowers the cached proxy cost.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_ctx_two_macros():
    """Two hard macros + 1 net, both pins owned by the macros (non-zero offsets).

    Pin layout:
      pin 0 -> macro 0, offset (3, 0)
      pin 1 -> macro 1, offset (-2, 0)
      net 0 connects both pins.
    """
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
    """OrientationState with all 8 forward transforms applied to pin offsets."""
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
    # CSR macro -> pins: macro 0 owns pin 0, macro 1 owns pin 1.
    return OrientationState(
        pin_offset_x_by_ori=by_ori_x,
        pin_offset_y_by_ori=by_ori_y,
        macro_orientation=np.zeros(2, dtype=np.int8),
        macro_pin_starts=np.array([0, 1, 2], dtype=np.int32),
        macro_pin_indices=np.array([0, 1], dtype=np.int32),
    )


@pytest.mark.unit
def test_lns_rotation_off_is_bit_exact() -> None:
    """rotation_probability=0.0 with orientation_state=None must equal prior path."""
    from macro_place.lns_v2 import lns_destroy_rebuild

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)

    baseline = lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        num_destroy=1,
        max_lns_iters=2,
        k_per_axis=4,
        seed=0,
    )
    new_pos_b, accepted_b, evals_b = baseline

    # With rotation params explicitly off — must produce identical output.
    same = lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        num_destroy=1,
        max_lns_iters=2,
        k_per_axis=4,
        seed=0,
        orientation_state=None,
        rotation_probability=0.0,
    )
    new_pos_s, accepted_s, evals_s = same

    np.testing.assert_array_equal(new_pos_b, new_pos_s)
    assert accepted_b == accepted_s
    assert evals_b == evals_s


@pytest.mark.unit
def test_lns_rotation_state_passed_zero_prob_no_mutation() -> None:
    """rotation_probability=0.0 with a state passed in must not mutate state."""
    from macro_place.lns_v2 import lns_destroy_rebuild

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)
    state = _build_state(ctx)
    initial_ori = state.macro_orientation.copy()
    initial_pin_x = ctx.pin_offset_x.copy()

    lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        num_destroy=1,
        max_lns_iters=1,
        k_per_axis=3,
        seed=0,
        orientation_state=state,
        rotation_probability=0.0,
    )
    np.testing.assert_array_equal(state.macro_orientation, initial_ori)
    np.testing.assert_allclose(ctx.pin_offset_x, initial_pin_x, atol=1e-6)


@pytest.mark.unit
def test_lns_rotation_on_attempts_rotations_when_prob_one() -> None:
    """rotation_probability=1.0 with state must attempt rotations and stay same-class."""
    from macro_place.lns_v2 import lns_destroy_rebuild
    from macro_place.orientation import NS_CLASS_INDICES

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)
    state = _build_state(ctx)
    # Both macros start at N (index 0, NS class).

    new_pos, _accepted, _evals = lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        num_destroy=2,
        max_lns_iters=2,
        k_per_axis=3,
        seed=7,
        orientation_state=state,
        rotation_probability=1.0,
    )

    # Same-class invariant: every macro should remain in NS_CLASS_INDICES.
    for ori in state.macro_orientation:
        assert int(ori) in NS_CLASS_INDICES


@pytest.mark.unit
def test_lns_rotation_default_args_match_existing_callers() -> None:
    """Existing callers that don't pass rotation kwargs must keep working."""
    from macro_place.lns_v2 import lns_destroy_rebuild

    ctx = _build_ctx_two_macros()
    pos = np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float64)
    out = lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=100.0,
        canvas_h=100.0,
        num_destroy=1,
        max_lns_iters=1,
        k_per_axis=3,
        seed=0,
    )
    assert isinstance(out, tuple)
    assert len(out) == 3
