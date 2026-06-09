"""Lever C Step 1 — Orientation-aware pin offsets for fast_proxy.

Adds an ``OrientationState`` companion to ``FastProxyContext`` that tracks the
current orientation of every macro and precomputes pin offsets for all 8
orientations. Applying a new orientation mutates the context's
``pin_offset_x``/``pin_offset_y`` in place for that macro's pins; existing
``fast_proxy`` kernels see updated offsets without any kernel changes.

Orientation indices (match Plc_client constants):
  0=N, 1=FN, 2=S, 3=FS, 4=E, 5=FE, 6=W, 7=FW

Transform from N base (x, y) to orientation ``i``:
  N      (x,  y)
  FN    (-x,  y)
  S     (-x, -y)
  FS     (x, -y)
  E      (y, -x)
  FE    (-y, -x)
  W     (-y,  x)
  FW     (y,  x)

Class membership:
  NS class = {N, FN, S, FS} (indices 0-3) — bbox w/h unchanged
  EW class = {E, FE, W, FW} (indices 4-7) — bbox w/h swap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ORIENTATIONS: tuple[str, ...] = ("N", "FN", "S", "FS", "E", "FE", "W", "FW")
NS_CLASS_INDICES: tuple[int, ...] = (0, 1, 2, 3)
EW_CLASS_INDICES: tuple[int, ...] = (4, 5, 6, 7)


def orientation_index(name: str) -> int:
    """Map orientation name to integer index. Unknown → 0 (N)."""
    try:
        return ORIENTATIONS.index(str(name))
    except ValueError:
        return 0


def orientation_name(idx: int) -> str:
    """Map orientation index to name. Out-of-range → 'N'."""
    i = int(idx)
    if 0 <= i < len(ORIENTATIONS):
        return ORIENTATIONS[i]
    return "N"


def orientation_class_indices(ori_idx: int) -> tuple[int, ...]:
    """Return the 4 orientation indices in the same class as ``ori_idx``."""
    return NS_CLASS_INDICES if int(ori_idx) in NS_CLASS_INDICES else EW_CLASS_INDICES


@dataclass
class OrientationState:
    """Mutable orientation tracking + precomputed pin offset tables.

    ``pin_offset_x_by_ori[pin, ori]`` is the x-offset for that pin when its
    parent macro is at orientation index ``ori``. Same for y. PORT pins
    (``pin_macro_idx == -1``) are unaffected — their offsets are fixed
    absolute positions and the same row is used for every orientation.

    ``macro_orientation[macro_idx]`` is the *current* orientation index for
    that macro. Updated via ``apply_orientation`` when ctx pin offsets are
    mutated.

    ``macro_pin_starts``/``macro_pin_indices`` are a CSR-style reverse map
    from macro to its owned pin indices, used by ``apply_orientation`` to
    update pin offsets in O(pins-per-macro) rather than O(total pins).
    """
    pin_offset_x_by_ori: np.ndarray  # [N_pins, 8] float32
    pin_offset_y_by_ori: np.ndarray  # [N_pins, 8] float32
    macro_orientation: np.ndarray    # [N_macros] int8
    macro_pin_starts: np.ndarray     # [N_macros + 1] int32
    macro_pin_indices: np.ndarray    # [total_macro_pins] int32


def _apply_forward_transform(
    base_x: np.ndarray, base_y: np.ndarray, ori_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the forward transform for ``ori_idx`` to N-base offsets."""
    i = int(ori_idx)
    if i == 0:  # N
        return base_x.copy(), base_y.copy()
    if i == 1:  # FN
        return -base_x, base_y.copy()
    if i == 2:  # S
        return -base_x, -base_y
    if i == 3:  # FS
        return base_x.copy(), -base_y
    if i == 4:  # E:  x = y_base, y = -x_base
        return base_y.copy(), -base_x
    if i == 5:  # FE: x = -y_base, y = -x_base
        return -base_y, -base_x
    if i == 6:  # W:  x = -y_base, y = x_base
        return -base_y, base_x.copy()
    if i == 7:  # FW: x = y_base, y = x_base
        return base_y.copy(), base_x.copy()
    return base_x.copy(), base_y.copy()


def _apply_inverse_transform(
    cur_x: float, cur_y: float, cur_ori_idx: int
) -> tuple[float, float]:
    """Recover N-base offset from current offset given the current orientation."""
    i = int(cur_ori_idx)
    if i == 0:  # N → identity
        return float(cur_x), float(cur_y)
    if i == 1:  # FN: cur_x = -base_x, cur_y = base_y → base = (-cur_x, cur_y)
        return -float(cur_x), float(cur_y)
    if i == 2:  # S:  cur = (-base_x, -base_y) → base = (-cur_x, -cur_y)
        return -float(cur_x), -float(cur_y)
    if i == 3:  # FS: cur = (base_x, -base_y) → base = (cur_x, -cur_y)
        return float(cur_x), -float(cur_y)
    if i == 4:  # E:  cur = (base_y, -base_x) → base_x = -cur_y, base_y = cur_x
        return -float(cur_y), float(cur_x)
    if i == 5:  # FE: cur = (-base_y, -base_x) → base_x = -cur_y, base_y = -cur_x
        return -float(cur_y), -float(cur_x)
    if i == 6:  # W:  cur = (-base_y, base_x) → base_x = cur_y, base_y = -cur_x
        return float(cur_y), -float(cur_x)
    if i == 7:  # FW: cur = (base_y, base_x) → base_x = cur_y, base_y = cur_x
        return float(cur_y), float(cur_x)
    return float(cur_x), float(cur_y)


def build_orientation_state(ctx: Any, plc: Any, benchmark: Any) -> OrientationState:
    """Construct orientation state from a built FastProxyContext + plc.

    Captures current macro orientations from the plc, derives N-base pin
    offsets via the inverse transform, then computes pin offsets for all 8
    orientations. PORT pins are passed through unchanged across all 8 slots.

    Side effect: none. The returned state's tables are independent from
    ``ctx.pin_offset_x``/``ctx.pin_offset_y``; the caller can later mutate
    those via ``apply_orientation`` to switch a macro's pin offsets.
    """
    pin_macro_idx = np.asarray(ctx.pin_macro_idx, dtype=np.int32)
    cur_x = np.asarray(ctx.pin_offset_x, dtype=np.float32).copy()
    cur_y = np.asarray(ctx.pin_offset_y, dtype=np.float32).copy()
    n_pins = int(pin_macro_idx.shape[0])
    n_macros = int(np.asarray(ctx.macro_is_hard).shape[0])

    # Per-macro current orientation index (default N for macros we can't read).
    macro_ori = np.zeros(n_macros, dtype=np.int8)
    # benchmark.hard_macro_indices + soft_macro_indices give us plc node indices
    # in the macro-array order.
    macro_plc_indices: list[int] = []
    for i in range(n_macros):
        macro_plc_indices.append(-1)
    hard_idx = list(getattr(benchmark, "hard_macro_indices", []))
    soft_idx = list(getattr(benchmark, "soft_macro_indices", []))
    for i, plc_idx in enumerate(hard_idx):
        if 0 <= i < n_macros:
            macro_plc_indices[i] = int(plc_idx)
    offset = len(hard_idx)
    for j, plc_idx in enumerate(soft_idx):
        i = offset + j
        if 0 <= i < n_macros:
            macro_plc_indices[i] = int(plc_idx)
    for i in range(n_macros):
        plc_idx = macro_plc_indices[i]
        if plc_idx < 0:
            continue
        try:
            ori_name = plc.get_macro_orientation(plc_idx) or "N"
        except Exception:
            ori_name = "N"
        macro_ori[i] = orientation_index(ori_name)

    # Derive N-base pin offsets via inverse transform of current offsets.
    base_x = np.zeros(n_pins, dtype=np.float32)
    base_y = np.zeros(n_pins, dtype=np.float32)
    for p in range(n_pins):
        owner = int(pin_macro_idx[p])
        if owner < 0:
            # PORT pin: fixed absolute, no transform; treat current as base.
            base_x[p] = cur_x[p]
            base_y[p] = cur_y[p]
            continue
        cur_ori = int(macro_ori[owner])
        bx, by = _apply_inverse_transform(cur_x[p], cur_y[p], cur_ori)
        base_x[p] = bx
        base_y[p] = by

    # Precompute the 8-orientation pin offset table.
    by_ori_x = np.zeros((n_pins, 8), dtype=np.float32)
    by_ori_y = np.zeros((n_pins, 8), dtype=np.float32)
    for ori in range(8):
        fx, fy = _apply_forward_transform(base_x, base_y, ori)
        by_ori_x[:, ori] = fx
        by_ori_y[:, ori] = fy
    # PORT pins: keep current absolute pos in every slot.
    is_port = pin_macro_idx < 0
    if np.any(is_port):
        port_x = cur_x[is_port].reshape(-1, 1)
        port_y = cur_y[is_port].reshape(-1, 1)
        by_ori_x[is_port, :] = port_x
        by_ori_y[is_port, :] = port_y

    # Build reverse CSR: macro → pin local indices.
    macro_pin_lists: list[list[int]] = [[] for _ in range(n_macros)]
    for p in range(n_pins):
        owner = int(pin_macro_idx[p])
        if 0 <= owner < n_macros:
            macro_pin_lists[owner].append(p)
    starts = np.zeros(n_macros + 1, dtype=np.int32)
    flat: list[int] = []
    for i in range(n_macros):
        flat.extend(macro_pin_lists[i])
        starts[i + 1] = len(flat)
    indices = np.asarray(flat, dtype=np.int32)

    return OrientationState(
        pin_offset_x_by_ori=by_ori_x,
        pin_offset_y_by_ori=by_ori_y,
        macro_orientation=macro_ori,
        macro_pin_starts=starts,
        macro_pin_indices=indices,
    )


def apply_orientation(
    ctx: Any, state: OrientationState, macro_idx: int, new_ori_idx: int
) -> int:
    """Mutate ctx.pin_offset_x/y for ``macro_idx``'s pins to ``new_ori_idx``.

    Returns the previous orientation index for ``macro_idx`` so callers can
    revert by calling ``apply_orientation`` with that value.
    """
    new_ori = int(new_ori_idx)
    prev_ori = int(state.macro_orientation[int(macro_idx)])
    if new_ori == prev_ori:
        return prev_ori
    starts = state.macro_pin_starts
    indices = state.macro_pin_indices
    lo = int(starts[int(macro_idx)])
    hi = int(starts[int(macro_idx) + 1])
    pins = indices[lo:hi]
    if pins.size > 0:
        ctx.pin_offset_x[pins] = state.pin_offset_x_by_ori[pins, new_ori]
        ctx.pin_offset_y[pins] = state.pin_offset_y_by_ori[pins, new_ori]
    state.macro_orientation[int(macro_idx)] = np.int8(new_ori)
    return prev_ori
