"""TDD for Lever C Step 1 — orientation-aware pin offsets.

Tests cover:
  - Forward/inverse transform round-trip
  - Class membership (NS vs EW)
  - apply_orientation mutates ctx pin offsets and updates macro_orientation
  - Same orientation = no-op
  - All 8 orientations precomputed correctly from a 2-macro stub
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Round-trip: forward then inverse should equal identity.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("ori_idx", range(8))
def test_forward_inverse_roundtrip(ori_idx: int) -> None:
    from macro_place.orientation import _apply_forward_transform, _apply_inverse_transform

    base_x = np.array([1.0, 2.5, -3.0], dtype=np.float32)
    base_y = np.array([4.0, -1.5, 0.0], dtype=np.float32)
    fx, fy = _apply_forward_transform(base_x, base_y, ori_idx)
    # Invert each element back and check we recover base.
    for k in range(base_x.shape[0]):
        rx, ry = _apply_inverse_transform(float(fx[k]), float(fy[k]), ori_idx)
        assert rx == pytest.approx(base_x[k], abs=1e-5)
        assert ry == pytest.approx(base_y[k], abs=1e-5)


@pytest.mark.unit
def test_orientation_index_and_name_roundtrip() -> None:
    from macro_place.orientation import orientation_index, orientation_name, ORIENTATIONS

    for i, name in enumerate(ORIENTATIONS):
        assert orientation_index(name) == i
        assert orientation_name(i) == name
    # Unknown defaults.
    assert orientation_index("?") == 0
    assert orientation_name(99) == "N"


@pytest.mark.unit
def test_class_membership() -> None:
    from macro_place.orientation import orientation_class_indices, NS_CLASS_INDICES, EW_CLASS_INDICES

    for ns in NS_CLASS_INDICES:
        assert orientation_class_indices(ns) == NS_CLASS_INDICES
    for ew in EW_CLASS_INDICES:
        assert orientation_class_indices(ew) == EW_CLASS_INDICES


# ---------------------------------------------------------------------------
# apply_orientation: in-place mutation of ctx pin offsets.
# ---------------------------------------------------------------------------

class _StubCtx:
    def __init__(self, pin_macro_idx: np.ndarray, pin_offset_x: np.ndarray, pin_offset_y: np.ndarray):
        self.pin_macro_idx = pin_macro_idx
        self.pin_offset_x = pin_offset_x  # mutable!
        self.pin_offset_y = pin_offset_y


def _build_stub_with_state(
    pin_macro_idx: list[int],
    pin_offset_x: list[float],
    pin_offset_y: list[float],
    macro_orientation: list[int],
    n_macros: int,
):
    """Build a stub ctx + a precomputed OrientationState.

    by_ori tables are computed by applying forward transforms to base offsets
    that we derive via inverse transform from the given current offsets.
    """
    from macro_place.orientation import OrientationState, _apply_forward_transform, _apply_inverse_transform

    cur_x = np.asarray(pin_offset_x, dtype=np.float32).copy()
    cur_y = np.asarray(pin_offset_y, dtype=np.float32).copy()
    pmi = np.asarray(pin_macro_idx, dtype=np.int32)
    n_pins = pmi.shape[0]
    base_x = np.zeros(n_pins, dtype=np.float32)
    base_y = np.zeros(n_pins, dtype=np.float32)
    for p in range(n_pins):
        owner = int(pmi[p])
        if owner < 0:
            base_x[p] = cur_x[p]
            base_y[p] = cur_y[p]
        else:
            bx, by = _apply_inverse_transform(cur_x[p], cur_y[p], macro_orientation[owner])
            base_x[p] = bx
            base_y[p] = by
    by_ori_x = np.zeros((n_pins, 8), dtype=np.float32)
    by_ori_y = np.zeros((n_pins, 8), dtype=np.float32)
    for ori in range(8):
        fx, fy = _apply_forward_transform(base_x, base_y, ori)
        by_ori_x[:, ori] = fx
        by_ori_y[:, ori] = fy
    # Macro→pin CSR.
    starts = np.zeros(n_macros + 1, dtype=np.int32)
    pin_lists = [[] for _ in range(n_macros)]
    for p in range(n_pins):
        owner = int(pmi[p])
        if 0 <= owner < n_macros:
            pin_lists[owner].append(p)
    flat: list[int] = []
    for i in range(n_macros):
        flat.extend(pin_lists[i])
        starts[i + 1] = len(flat)
    indices = np.asarray(flat, dtype=np.int32)
    state = OrientationState(
        pin_offset_x_by_ori=by_ori_x,
        pin_offset_y_by_ori=by_ori_y,
        macro_orientation=np.asarray(macro_orientation, dtype=np.int8),
        macro_pin_starts=starts,
        macro_pin_indices=indices,
    )
    ctx = _StubCtx(pmi, cur_x, cur_y)
    return ctx, state


@pytest.mark.unit
def test_apply_orientation_updates_ctx_pin_offsets() -> None:
    from macro_place.orientation import apply_orientation, orientation_index

    # 1 macro (idx 0) with 2 pins at (1.0, 2.0) and (-1.0, 0.5), currently at N.
    ctx, state = _build_stub_with_state(
        pin_macro_idx=[0, 0],
        pin_offset_x=[1.0, -1.0],
        pin_offset_y=[2.0, 0.5],
        macro_orientation=[0],
        n_macros=1,
    )
    # Switch macro 0 to FN: forward FN(x,y) = (-x, y).
    prev = apply_orientation(ctx, state, macro_idx=0, new_ori_idx=orientation_index("FN"))
    assert prev == 0  # was N
    np.testing.assert_allclose(ctx.pin_offset_x, [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(ctx.pin_offset_y, [2.0, 0.5], atol=1e-6)
    assert int(state.macro_orientation[0]) == orientation_index("FN")


@pytest.mark.unit
def test_apply_orientation_idempotent_for_same_index() -> None:
    from macro_place.orientation import apply_orientation, orientation_index

    ctx, state = _build_stub_with_state(
        pin_macro_idx=[0, 0],
        pin_offset_x=[1.0, -1.0],
        pin_offset_y=[2.0, 0.5],
        macro_orientation=[1],  # currently FN
        n_macros=1,
    )
    before_x = ctx.pin_offset_x.copy()
    before_y = ctx.pin_offset_y.copy()
    prev = apply_orientation(ctx, state, macro_idx=0, new_ori_idx=1)  # FN → FN
    np.testing.assert_array_equal(ctx.pin_offset_x, before_x)
    np.testing.assert_array_equal(ctx.pin_offset_y, before_y)
    assert prev == 1


@pytest.mark.unit
def test_apply_orientation_revert_via_prev_index() -> None:
    """apply_orientation returns prev_idx so caller can revert exactly."""
    from macro_place.orientation import apply_orientation, orientation_index

    ctx, state = _build_stub_with_state(
        pin_macro_idx=[0, 0],
        pin_offset_x=[3.0, 1.0],
        pin_offset_y=[1.0, -2.0],
        macro_orientation=[0],
        n_macros=1,
    )
    saved_x = ctx.pin_offset_x.copy()
    saved_y = ctx.pin_offset_y.copy()
    # Switch N → S
    prev = apply_orientation(ctx, state, macro_idx=0, new_ori_idx=orientation_index("S"))
    assert prev == 0
    # Switch back S → N (the returned prev).
    apply_orientation(ctx, state, macro_idx=0, new_ori_idx=prev)
    np.testing.assert_allclose(ctx.pin_offset_x, saved_x, atol=1e-6)
    np.testing.assert_allclose(ctx.pin_offset_y, saved_y, atol=1e-6)
    assert int(state.macro_orientation[0]) == 0


@pytest.mark.unit
def test_apply_orientation_does_not_affect_other_macro_pins() -> None:
    """Pins of macro 1 must be unchanged when we rotate macro 0."""
    from macro_place.orientation import apply_orientation, orientation_index

    ctx, state = _build_stub_with_state(
        pin_macro_idx=[0, 1, 0, 1],
        pin_offset_x=[1.0, 5.0, 2.0, -5.0],
        pin_offset_y=[1.0, 6.0, -1.0, -6.0],
        macro_orientation=[0, 0],
        n_macros=2,
    )
    m1_x_before = ctx.pin_offset_x[[1, 3]].copy()
    m1_y_before = ctx.pin_offset_y[[1, 3]].copy()
    apply_orientation(ctx, state, macro_idx=0, new_ori_idx=orientation_index("S"))
    np.testing.assert_allclose(ctx.pin_offset_x[[1, 3]], m1_x_before, atol=1e-6)
    np.testing.assert_allclose(ctx.pin_offset_y[[1, 3]], m1_y_before, atol=1e-6)


@pytest.mark.unit
def test_apply_orientation_port_pins_invariant() -> None:
    """PORT pins (macro_idx == -1) must not be touched even when traversing macros."""
    from macro_place.orientation import apply_orientation, orientation_index

    ctx, state = _build_stub_with_state(
        pin_macro_idx=[0, -1, 0],
        pin_offset_x=[1.0, 99.0, 2.0],
        pin_offset_y=[1.0, 88.0, -1.0],
        macro_orientation=[0],
        n_macros=1,
    )
    apply_orientation(ctx, state, macro_idx=0, new_ori_idx=orientation_index("S"))
    assert ctx.pin_offset_x[1] == pytest.approx(99.0)
    assert ctx.pin_offset_y[1] == pytest.approx(88.0)
