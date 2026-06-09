"""Fast Numba/numpy proxy-cost surrogate calibrated to the TILOS evaluator.

This module produces the same proxy-cost components as
``macro_place.objective.compute_proxy_cost`` (formula
``1.0 * wirelength + 0.5 * density + 0.5 * congestion``) without going
through the per-pin Python loop in ``plc_client_os.py``. It is intended
for the SA/LNS/DR inner loop only; final scoring always uses the
official TILOS evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark

try:
    from numba import njit  # type: ignore[import-not-found]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        # No-op fallback: if called as @njit(...), return the inner decorator
        # that returns the function unchanged. If called as @njit (no parens),
        # the first arg is the function itself.
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator


@dataclass(frozen=True)
class FastProxyContext:
    pin_macro_idx: np.ndarray
    pin_offset_x: np.ndarray
    pin_offset_y: np.ndarray
    net_pin_starts: np.ndarray
    net_pin_indices: np.ndarray
    net_weights: np.ndarray
    net_source_pin_local: np.ndarray  # local pin index of source pin for each net (-1 if none)
    macro_w: np.ndarray
    macro_h: np.ndarray
    macro_is_hard: np.ndarray
    grid_col: int
    grid_row: int
    canvas_w: float
    canvas_h: float
    h_routes_per_micron: float
    v_routes_per_micron: float
    hrouting_alloc: float  # fraction of macro area blocking H routing
    vrouting_alloc: float  # fraction of macro area blocking V routing
    smooth_range: int
    overlap_threshold: float
    net_cnt: float  # sum of net weights, matches plc.net_cnt used in get_cost()


@dataclass(frozen=True)
class FastProxyResult:
    proxy_cost: float
    wirelength: float
    density: float
    congestion: float
    overlap_count: int


def build_fast_proxy_context(plc: Any, benchmark: "Benchmark") -> FastProxyContext:
    """Walk the plc once and pack pin/net/macro arrays into a CSR layout.

    ``plc.modules_w_pins`` mixes MACRO, MACRO_PIN, PORT, and STDCELL nodes.
    We index macros by their position in ``benchmark.hard_macro_indices`` +
    ``benchmark.soft_macro_indices`` so that ``positions[i]`` corresponds
    to ``ctx.macro_*[i]``.

    The net CSR is built from ``plc.nets`` (driver → list[sinks] strings of
    the form ``"macro_name/pin_name"``), mapping each pin string to the local
    pin index built during the MACRO_PIN walk.  Nets with no locally-indexed
    pins are still emitted (with an empty pin slice) to keep net count stable;
    the starts array therefore has length ``num_nets + 1``.
    """
    num_macros = int(benchmark.num_macros)
    macro_w = np.zeros(num_macros, dtype=np.float32)
    macro_h = np.zeros(num_macros, dtype=np.float32)
    macro_is_hard = np.zeros(num_macros, dtype=bool)

    # Map plc node index → benchmark macro index (0-based).
    node_to_macro: dict[int, int] = {}

    for i, idx in enumerate(benchmark.hard_macro_indices):
        node_to_macro[int(idx)] = i
        node = plc.modules_w_pins[idx]
        macro_w[i] = float(getattr(node, "width", None) or node.get_width())
        macro_h[i] = float(getattr(node, "height", None) or node.get_height())
        macro_is_hard[i] = True

    offset = int(benchmark.num_hard_macros)
    for i, idx in enumerate(benchmark.soft_macro_indices):
        node_to_macro[int(idx)] = offset + i
        node = plc.modules_w_pins[idx]
        macro_w[offset + i] = float(getattr(node, "width", None) or node.get_width())
        macro_h[offset + i] = float(getattr(node, "height", None) or node.get_height())
        macro_is_hard[offset + i] = False

    # Build name → plc index for MACRO nodes (hard + soft).
    name_to_plc_idx: dict[str, int] = {}
    for plc_idx, mod in enumerate(plc.modules_w_pins):
        node_type = mod.get_type()
        if node_type in ("MACRO", "SOFT_MACRO", "STDCELL") or plc_idx in node_to_macro:
            name_to_plc_idx[mod.get_name()] = plc_idx

    # Walk MACRO_PIN and PORT nodes to build pin arrays and a name→local index map.
    #
    # For MACRO_PINs: pin_macro_idx[i] = macro index (>=0), pin_offset_x/y = offset.
    # For PORT pins:  pin_macro_idx[i] = -1 (sentinel), pin_offset_x/y = fixed position.
    # In fast_hpwl, positions[macro_idx] + offset is used for MACRO_PINs,
    # and offset alone is used for PORT pins (since their position is fixed).
    pin_macro_idx_list: list[int] = []
    pin_offset_x_list: list[float] = []
    pin_offset_y_list: list[float] = []
    # "macro_name/pin_name" or "port_name" → local pin index (matches plc.nets key format).
    pin_full_name_to_local: dict[str, int] = {}

    for plc_idx, mod in enumerate(plc.modules_w_pins):
        mod_type = mod.get_type()
        if mod_type == "MACRO_PIN":
            parent_name = mod.get_macro_name()
            parent_plc_idx = name_to_plc_idx.get(parent_name)
            if parent_plc_idx is None or parent_plc_idx not in node_to_macro:
                continue
            local_pin = len(pin_macro_idx_list)
            full_name = mod.get_name()  # typically "macro_name/pin_name"
            pin_full_name_to_local[full_name] = local_pin
            pin_macro_idx_list.append(node_to_macro[parent_plc_idx])
            pin_offset_x_list.append(float(getattr(mod, "x_offset", 0.0)))
            pin_offset_y_list.append(float(getattr(mod, "y_offset", 0.0)))
        elif mod_type == "PORT":
            # PORT nodes have fixed absolute positions; use macro_idx=-1 sentinel.
            local_pin = len(pin_macro_idx_list)
            pin_full_name_to_local[mod.get_name()] = local_pin
            pin_macro_idx_list.append(-1)
            px_val, py_val = mod.get_pos()
            pin_offset_x_list.append(float(px_val))
            pin_offset_y_list.append(float(py_val))

    pin_macro_idx = np.asarray(pin_macro_idx_list, dtype=np.int32)
    pin_offset_x = np.asarray(pin_offset_x_list, dtype=np.float32)
    pin_offset_y = np.asarray(pin_offset_y_list, dtype=np.float32)

    # Build CSR net arrays from plc.nets (driver → list[sinks] pin-name strings).
    starts: list[int] = [0]
    indices: list[int] = []
    weights: list[float] = []

    # Build a lookup from driver pin name → plc node index for weight retrieval.
    driver_pin_name_to_plc_idx: dict[str, int] = {}
    for plc_idx, mod in enumerate(plc.modules_w_pins):
        node_type = mod.get_type()
        if node_type == "MACRO_PIN":
            driver_pin_name_to_plc_idx[mod.get_name()] = plc_idx

    net_cnt_accum = 0.0
    net_source_pin_local_list: list[int] = []
    for driver, sinks in plc.nets.items():
        net_indices: list[int] = []
        for pin_name in [driver] + list(sinks):
            local = pin_full_name_to_local.get(pin_name)
            if local is not None:
                net_indices.append(local)
        # Deduplicate while preserving order.
        seen: set[int] = set()
        dedup_indices: list[int] = []
        for idx in net_indices:
            if idx not in seen:
                seen.add(idx)
                dedup_indices.append(idx)
                indices.append(idx)
        starts.append(len(indices))
        # Track source pin: first pin in deduplicated list (matches TILOS driver-first ordering).
        net_source_pin_local_list.append(dedup_indices[0] if dedup_indices else -1)
        # Retrieve driver pin weight from plc node (matches get_wirelength logic).
        driver_plc_idx = driver_pin_name_to_plc_idx.get(driver)
        if driver_plc_idx is not None:
            w = float(plc.modules_w_pins[driver_plc_idx].get_weight())
        else:
            w = 1.0
        weights.append(w)
        net_cnt_accum += w

    if net_cnt_accum == 0.0:
        net_cnt_accum = 1.0

    net_pin_starts = np.asarray(starts, dtype=np.int32)
    net_pin_indices = np.asarray(indices, dtype=np.int32)
    net_weights = np.asarray(weights, dtype=np.float32)
    net_source_pin_local = np.asarray(net_source_pin_local_list, dtype=np.int32)

    # TILOS plc uses ``hroutes_per_micron`` (no underscore) internally; some wrappers
    # expose ``h_routes_per_micron``.  Try both spellings before defaulting to 0.
    def _get_float(obj: Any, *names: str, default: float = 0.0) -> float:
        for name in names:
            val = getattr(obj, name, None)
            if val is not None:
                return float(val)
        return default

    h_rpm = _get_float(plc, "h_routes_per_micron", "hroutes_per_micron")
    v_rpm = _get_float(plc, "v_routes_per_micron", "vroutes_per_micron")
    h_alloc = _get_float(plc, "hrouting_alloc", "macro_horizontal_routing_allocation")
    v_alloc = _get_float(plc, "vrouting_alloc", "macro_vertical_routing_allocation")

    return FastProxyContext(
        pin_macro_idx=pin_macro_idx,
        pin_offset_x=pin_offset_x,
        pin_offset_y=pin_offset_y,
        net_pin_starts=net_pin_starts,
        net_pin_indices=net_pin_indices,
        net_weights=net_weights,
        net_source_pin_local=net_source_pin_local,
        macro_w=macro_w,
        macro_h=macro_h,
        macro_is_hard=macro_is_hard,
        grid_col=int(plc.grid_col),
        grid_row=int(plc.grid_row),
        canvas_w=float(plc.width),
        canvas_h=float(plc.height),
        h_routes_per_micron=h_rpm,
        v_routes_per_micron=v_rpm,
        hrouting_alloc=h_alloc,
        vrouting_alloc=v_alloc,
        smooth_range=int(getattr(plc, "smooth_range", 2)),
        overlap_threshold=float(getattr(plc, "overlap_threshold", 0.004)),
        net_cnt=net_cnt_accum,
    )


def fast_proxy(positions: np.ndarray, ctx: FastProxyContext) -> FastProxyResult:
    """Combined surrogate matching ``compute_proxy_cost``'s formula:
    1.0 * wirelength + 0.5 * density + 0.5 * congestion.
    """
    wl = fast_hpwl(positions, ctx)
    den = fast_density(positions, ctx)
    cong = fast_congestion(positions, ctx)
    proxy = 1.0 * wl + 0.5 * den + 0.5 * cong
    overlap = fast_overlap_count(positions, ctx)
    return FastProxyResult(
        proxy_cost=float(proxy),
        wirelength=float(wl),
        density=float(den),
        congestion=float(cong),
        overlap_count=int(overlap),
    )


def fast_overlap_count(positions: np.ndarray, ctx: FastProxyContext) -> int:
    """Vectorized O(H^2) hard-macro overlap detection.

    Exact match for ``macro_place.objective.compute_overlap_metrics``: count
    of unordered hard-macro pairs whose bboxes overlap (strict ``> 0`` in
    both dims, not ``>=``).
    """
    hard_mask = ctx.macro_is_hard
    if not hard_mask.any():
        return 0
    pos = positions[hard_mask]
    w = ctx.macro_w[hard_mask]
    h = ctx.macro_h[hard_mask]
    n = pos.shape[0]
    if n < 2:
        return 0

    dx = np.abs(pos[:, None, 0] - pos[None, :, 0])
    dy = np.abs(pos[:, None, 1] - pos[None, :, 1])
    min_x = (w[:, None] + w[None, :]) * 0.5
    min_y = (h[:, None] + h[None, :]) * 0.5

    # Strict less-than (overlap_x > 0) → distance must be STRICTLY less than min_sep
    overlap = (dx < min_x) & (dy < min_y)
    np.fill_diagonal(overlap, False)
    return int(overlap.sum() // 2)


@njit(cache=True)
def _hpwl_kernel(
    px: np.ndarray,
    py: np.ndarray,
    starts: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> float:
    """JIT-compiled inner loop for HPWL accumulation."""
    total = 0.0
    for net_id in range(starts.shape[0] - 1):
        s = starts[net_id]
        e = starts[net_id + 1]
        if e <= s:
            continue
        x_min = px[indices[s]]
        x_max = x_min
        y_min = py[indices[s]]
        y_max = y_min
        for k in range(s + 1, e):
            p = indices[k]
            x = px[p]
            y = py[p]
            if x < x_min:
                x_min = x
            elif x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            elif y > y_max:
                y_max = y
        total += weights[net_id] * ((x_max - x_min) + (y_max - y_min))
    return total


def fast_hpwl(positions: np.ndarray, ctx: FastProxyContext) -> float:
    """Vectorized HPWL: bbox per net, weighted sum, normalized to match plc.get_cost().

    TILOS reference: ``plc_client_os.py:get_cost`` (line 672) and
    ``plc_client_os.py:get_wirelength`` (line 745).
    Normalization matches the reference exactly:
    ``HPWL_total / ((canvas_w + canvas_h) * net_cnt)``
    where ``net_cnt`` is the sum of net weights (same accumulation as plc.net_cnt).

    PORT pins (``pin_macro_idx == -1``) have fixed absolute positions stored in
    ``pin_offset_x/y``; MACRO_PINs use ``positions[macro_idx] + offset``.

    Args:
        positions: float32 array of shape ``[num_macros, 2]`` with (x, y) macro centers.
        ctx: ``FastProxyContext`` built by ``build_fast_proxy_context``.

    Returns:
        Wirelength cost calibrated to ``plc.get_cost()``.
    """
    macro_idx = ctx.pin_macro_idx  # shape [num_pins], -1 for PORT pins

    # Build absolute pin coordinates: start from offsets (fixed for PORT, offsets for MACRO_PINs).
    px = ctx.pin_offset_x.copy()
    py = ctx.pin_offset_y.copy()

    # For MACRO_PINs (macro_idx >= 0), add macro center position.
    movable = macro_idx >= 0
    if movable.any():
        valid_idx = macro_idx[movable]
        px[movable] += positions[valid_idx, 0]
        py[movable] += positions[valid_idx, 1]

    starts = ctx.net_pin_starts
    indices = ctx.net_pin_indices
    weights = ctx.net_weights

    total = _hpwl_kernel(px, py, starts, indices, weights)

    norm = (ctx.canvas_w + ctx.canvas_h) * ctx.net_cnt
    return total / norm if norm > 0 else 0.0


@njit(cache=True)
def _density_kernel(
    positions: np.ndarray,
    macro_w: np.ndarray,
    macro_h: np.ndarray,
    grid_row: int,
    grid_col: int,
    cell_w: float,
    cell_h: float,
    occ: np.ndarray,
) -> None:
    """JIT-compiled inner loop for per-cell macro occupancy accumulation."""
    for m in range(positions.shape[0]):
        hw = macro_w[m] * 0.5
        hh = macro_h[m] * 0.5
        x_lo = positions[m, 0] - hw
        x_hi = positions[m, 0] + hw
        y_lo = positions[m, 1] - hh
        y_hi = positions[m, 1] + hh

        c0 = int(x_lo / cell_w)
        if c0 < 0:
            c0 = 0
        c1 = int(x_hi / cell_w)
        if c1 > grid_col - 1:
            c1 = grid_col - 1
        r0 = int(y_lo / cell_h)
        if r0 < 0:
            r0 = 0
        r1 = int(y_hi / cell_h)
        if r1 > grid_row - 1:
            r1 = grid_row - 1

        for r in range(r0, r1 + 1):
            cell_y_lo = r * cell_h
            cell_y_hi = cell_y_lo + cell_h
            oy = (y_hi if y_hi < cell_y_hi else cell_y_hi) - (y_lo if y_lo > cell_y_lo else cell_y_lo)
            if oy <= 0.0:
                continue
            for c in range(c0, c1 + 1):
                cell_x_lo = c * cell_w
                cell_x_hi = cell_x_lo + cell_w
                ox = (x_hi if x_hi < cell_x_hi else cell_x_hi) - (x_lo if x_lo > cell_x_lo else cell_x_lo)
                if ox <= 0.0:
                    continue
                occ[r * grid_col + c] += ox * oy


def fast_density(positions: np.ndarray, ctx: FastProxyContext) -> float:
    """Per-grid-cell macro occupancy density, calibrated to TILOS ``get_density_cost``.

    TILOS reference: ``plc_client_os.py:get_density_cost`` (line 1083),
    ``get_grid_cells_density`` (line 1047), ``__add_module_to_grid_cells`` (line 991).

    Algorithm (exactly matching TILOS):
    1. Accumulate overlap area of each macro (hard + soft, per ``ctx.macro_w/h``)
       into the grid cell it intersects.
    2. Normalize each cell by ``cell_area`` → density in [0, 1+].
    3. Filter to non-zero cells only (TILOS sorts ``[gc for gc in grid_cells if gc != 0.0]``).
    4. ``density_cnt = math.floor(num_grid_cells * 0.1)``; if ``< 10`` total cells,
       average over all occupied cells.
    5. Sum the top-``density_cnt`` densities, divide by ``density_cnt``, multiply by 0.5.

    Important TILOS details confirmed from source:
    - ``density_cnt`` uses ``math.floor``, not integer division.
    - Reduction iterates while ``idx < density_cnt`` AND ``idx < len(occupied_cells)``
      (handles cases where occupied < density_cnt).
    - Final result is ``0.5 * (sum_top / density_cnt)``.
    - Ports and MACRO_PINs are excluded; only MACRO and SOFT_MACRO nodes contribute.

    Args:
        positions: float64-or-float32 array of shape ``[num_macros, 2]`` with (x, y) centers.
                   Ordering: hard macros first, then soft macros (matches ``ctx.macro_w/h``).
        ctx: ``FastProxyContext`` built by ``build_fast_proxy_context``.

    Returns:
        Density cost calibrated to ``plc.get_density_cost()``, relative error < 0.5%.
    """
    import math

    grid_col = ctx.grid_col
    grid_row = ctx.grid_row
    cell_w = ctx.canvas_w / grid_col
    cell_h = ctx.canvas_h / grid_row
    cell_area = cell_w * cell_h
    if cell_area <= 0.0:
        return 0.0

    num_cells = grid_row * grid_col
    occ = np.zeros(num_cells, dtype=np.float64)

    pos_f64 = positions.astype(np.float64)
    macro_w_f64 = ctx.macro_w.astype(np.float64)
    macro_h_f64 = ctx.macro_h.astype(np.float64)
    _density_kernel(pos_f64, macro_w_f64, macro_h_f64, grid_row, grid_col, cell_w, cell_h, occ)

    # Normalize to density (occupation fraction).
    densities = occ / cell_area  # shape [num_cells]

    # Filter to occupied cells only (mirrors TILOS: [gc for gc in grid_cells if gc != 0.0]).
    occupied = [d for d in densities if d != 0.0]

    if not occupied:
        return 0.0

    # Sort descending.
    occupied_sorted = sorted(occupied, reverse=True)

    density_cnt = math.floor(num_cells * 0.1)

    if num_cells < 10:
        # TILOS fallback: average over all occupied cells.
        avg = sum(occupied_sorted) / len(occupied_sorted)
        return 0.5 * avg

    # Sum top density_cnt occupied cells (stop early if fewer occupied than density_cnt).
    sum_density = 0.0
    idx = 0
    while idx < density_cnt and idx < len(occupied_sorted):
        sum_density += occupied_sorted[idx]
        idx += 1

    return 0.5 * float(sum_density / density_cnt)


@njit(cache=True)
def _congestion_net_routing_kernel(
    px: np.ndarray,
    py: np.ndarray,
    starts: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    source_locals: np.ndarray,
    grid_row: int,
    grid_col: int,
    cell_w: float,
    cell_h: float,
    H_cong: np.ndarray,
    V_cong: np.ndarray,
) -> None:
    """JIT-compiled net routing loop for congestion accumulation.

    Uses integer-encoded gcell ids (row * grid_col + col) and a fixed-size
    dedup buffer to avoid Python sets/tuples.
    """
    # Fixed-size dedup buffer: max distinct gcells per net is bounded by the
    # number of pins; we allocate a generous cap (512 should cover any real net).
    buf = np.empty(512, dtype=np.int32)

    for net_id in range(starts.shape[0] - 1):
        s = starts[net_id]
        e = starts[net_id + 1]
        if e - s < 2:
            continue

        weight = weights[net_id]

        # --- collect distinct encoded gcells ---
        n_distinct = 0
        for k in range(s, e):
            p = indices[k]
            x = px[p]
            y = py[p]
            row = int(y / cell_h)
            if row < 0:
                row = 0
            elif row > grid_row - 1:
                row = grid_row - 1
            col = int(x / cell_w)
            if col < 0:
                col = 0
            elif col > grid_col - 1:
                col = grid_col - 1
            enc = row * grid_col + col
            # dedup check
            found = False
            for j in range(n_distinct):
                if buf[j] == enc:
                    found = True
                    break
            if not found:
                if n_distinct < 512:
                    buf[n_distinct] = enc
                    n_distinct += 1

        if n_distinct < 2:
            continue

        # --- source gcell ---
        src_local = source_locals[net_id]
        if src_local >= 0:
            sx = px[src_local]
            sy = py[src_local]
            sr = int(sy / cell_h)
            if sr < 0:
                sr = 0
            elif sr > grid_row - 1:
                sr = grid_row - 1
            sc = int(sx / cell_w)
            if sc < 0:
                sc = 0
            elif sc > grid_col - 1:
                sc = grid_col - 1
            src_enc = sr * grid_col + sc
        else:
            src_enc = buf[0]
            sr = src_enc // grid_col
            sc = src_enc % grid_col

        src_row = src_enc // grid_col
        src_col = src_enc % grid_col

        if n_distinct == 2:
            # --- 2-pin L-route ---
            enc0 = buf[0]
            enc1 = buf[1]
            if enc0 == src_enc:
                sink_enc = enc1
            else:
                sink_enc = enc0
            sink_row = sink_enc // grid_col
            sink_col = sink_enc % grid_col

            col_min = src_col if src_col < sink_col else sink_col
            col_max = src_col if src_col > sink_col else sink_col
            row_min = src_row if src_row < sink_row else sink_row
            row_max = src_row if src_row > sink_row else sink_row

            for col in range(col_min, col_max):
                H_cong[src_row * grid_col + col] += weight
            for r in range(row_min, row_max):
                V_cong[r * grid_col + sink_col] += weight

        elif n_distinct == 3:
            # --- 3-pin: sort by (col, row) = (enc%grid_col, enc//grid_col) ---
            g0 = buf[0]
            g1 = buf[1]
            g2 = buf[2]
            # bubble sort 3 elements by (col, row)
            c0 = g0 % grid_col; r0 = g0 // grid_col
            c1 = g1 % grid_col; r1 = g1 // grid_col
            c2 = g2 % grid_col; r2 = g2 // grid_col
            # sort by (col, row)
            if (c0, r0) > (c1, r1):
                g0, g1 = g1, g0
                c0, c1 = c1, c0
                r0, r1 = r1, r0
            if (c1, r1) > (c2, r2):
                g1, g2 = g2, g1
                c1, c2 = c2, c1
                r1, r2 = r2, r1
            if (c0, r0) > (c1, r1):
                g0, g1 = g1, g0
                c0, c1 = c1, c0
                r0, r1 = r1, r0
            # Now (r0,c0), (r1,c1), (r2,c2) sorted by (col, row) = (x, y)
            y1, x1 = r0, c0
            y2, x2 = r1, c1
            y3, x3 = r2, c2

            if x1 < x2 and x2 < x3 and (y1 if y1 < y3 else y3) < y2 and (y1 if y1 > y3 else y3) > y2:
                # _l_routing
                for col in range(x1, x2):
                    H_cong[y1 * grid_col + col] += weight
                for col in range(x2, x3):
                    H_cong[y2 * grid_col + col] += weight
                vr1_lo = y1 if y1 < y2 else y2
                vr1_hi = y1 if y1 > y2 else y2
                for r in range(vr1_lo, vr1_hi):
                    V_cong[r * grid_col + x2] += weight
                vr2_lo = y2 if y2 < y3 else y3
                vr2_hi = y2 if y2 > y3 else y3
                for r in range(vr2_lo, vr2_hi):
                    V_cong[r * grid_col + x3] += weight
            elif x2 == x3 and x1 < x2 and y1 < (y2 if y2 < y3 else y3):
                for col in range(x1, x2):
                    H_cong[y1 * grid_col + col] += weight
                vmax23 = y2 if y2 > y3 else y3
                for r in range(y1, vmax23):
                    V_cong[r * grid_col + x2] += weight
            elif y2 == y3:
                for col in range(x1, x2):
                    H_cong[y1 * grid_col + col] += weight
                for col in range(x2, x3):
                    H_cong[y2 * grid_col + col] += weight
                vlo = y2 if y2 < y1 else y1
                vhi = y2 if y2 > y1 else y1
                for r in range(vlo, vhi):
                    V_cong[r * grid_col + x2] += weight
            else:
                # _t_routing: sort by (row, col) for t-route
                # we already have sorted-by-col list; re-sort by (row, col)
                t0 = g0; t1 = g1; t2 = g2
                tr0 = r0; tc0 = c0
                tr1 = r1; tc1 = c1
                tr2 = r2; tc2 = c2
                # bubble sort by (row, col)
                if (tr0, tc0) > (tr1, tc1):
                    tr0, tr1 = tr1, tr0
                    tc0, tc1 = tc1, tc0
                if (tr1, tc1) > (tr2, tc2):
                    tr1, tr2 = tr2, tr1
                    tc1, tc2 = tc2, tc1
                if (tr0, tc0) > (tr1, tc1):
                    tr0, tr1 = tr1, tr0
                    tc0, tc1 = tc1, tc0
                ty1 = tr0; tx1 = tc0
                ty2 = tr1; tx2 = tc1
                ty3 = tr2; tx3 = tc2
                txmin = tx1 if tx1 < tx2 else tx2
                txmin = txmin if txmin < tx3 else tx3
                txmax = tx1 if tx1 > tx2 else tx2
                txmax = txmax if txmax > tx3 else tx3
                for col in range(txmin, txmax):
                    H_cong[ty2 * grid_col + col] += weight
                vlo12 = ty1 if ty1 < ty2 else ty2
                vhi12 = ty1 if ty1 > ty2 else ty2
                for r in range(vlo12, vhi12):
                    V_cong[r * grid_col + tx1] += weight
                vlo23 = ty2 if ty2 < ty3 else ty3
                vhi23 = ty2 if ty2 > ty3 else ty3
                for r in range(vlo23, vhi23):
                    V_cong[r * grid_col + tx3] += weight

        else:
            # >3 pins: split into source–sink_i 2-pin pairs
            for k in range(n_distinct):
                sink_enc = buf[k]
                if sink_enc == src_enc:
                    continue
                sink_row = sink_enc // grid_col
                sink_col = sink_enc % grid_col

                col_min = src_col if src_col < sink_col else sink_col
                col_max = src_col if src_col > sink_col else sink_col
                row_min = src_row if src_row < sink_row else sink_row
                row_max = src_row if src_row > sink_row else sink_row

                for col in range(col_min, col_max):
                    H_cong[src_row * grid_col + col] += weight
                for r in range(row_min, row_max):
                    V_cong[r * grid_col + sink_col] += weight


def fast_congestion(positions: np.ndarray, ctx: FastProxyContext) -> float:
    """Routing congestion cost calibrated to TILOS ``get_congestion_cost``.

    TILOS reference: ``plc_client_os.py:get_congestion_cost`` (line 905),
    ``get_routing`` (line 1514), ``__two_pin_net_routing`` (line 1269),
    ``__three_pin_net_routing`` (line 1354), ``__l_routing`` (line 1295),
    ``__t_routing`` (line 1332), ``__macro_route_over_grid_cell`` (line 1392),
    ``__smooth_routing_cong`` (line 1608), ``abu`` (line 850).

    Routing model (exactly matching TILOS):

    **Net routing:**
    - Nets driven by PORT or MACRO_PIN with ≥2 distinct grid cells are routed.
    - 2-pin nets: L-shaped route — H demand at source row along [col_min, col_max),
      V demand at sink col along [row_min, row_max).
    - 3-pin nets: dispatched to __l_routing or __t_routing or a special-case
      (two H edges + one V edge), sorted by (col, row).
    - >3-pin nets: split into (source, sink_i) 2-pin pairs.

    **Macro blockage:**
    - Hard macros only (not soft macros).
    - Per grid cell: overlap_x * vrouting_alloc added to V_macro, overlap_y * hrouting_alloc to H_macro.
    - Partial-overlap correction on boundary rows/cols (subtract over-counted strips).

    **Normalization:**
    - ``grid_v_routes = cell_w * v_routes_per_micron``
    - ``grid_h_routes = cell_h * h_routes_per_micron``
    - Divide raw demand by capacity to get utilization.

    **Smoothing (applied to wire-routing cong only, before adding macro cong):**
    - V: horizontally averaged in window [col-smooth_range, col+smooth_range] per row.
    - H: vertically averaged in window [row-smooth_range, row+smooth_range] per col.

    **Final reduction:**
    - ``abu(V_routing_cong + H_routing_cong, n=0.05)`` — top-5% mean, no 0.5× factor.
    - ``abu`` returns mean of top ``floor(len * 0.05)`` elements; if that count is 0,
      returns max.

    Args:
        positions: float32 array shape ``[num_macros, 2]`` (x, y) macro centers.
                   Ordering: hard macros first, then soft macros.
        ctx: ``FastProxyContext`` built by ``build_fast_proxy_context``.

    Returns:
        Congestion cost calibrated to ``plc.get_congestion_cost()``, relative error < 1%.
    """
    import math

    grid_col = ctx.grid_col
    grid_row = ctx.grid_row
    cell_w = ctx.canvas_w / grid_col
    cell_h = ctx.canvas_h / grid_row

    # Capacity per grid edge (routes per unit length * edge length).
    grid_v_routes = cell_w * ctx.v_routes_per_micron
    grid_h_routes = cell_h * ctx.h_routes_per_micron

    n_cells = grid_row * grid_col
    H_cong = np.zeros(n_cells, dtype=np.float64)
    V_cong = np.zeros(n_cells, dtype=np.float64)
    H_macro = np.zeros(n_cells, dtype=np.float64)
    V_macro = np.zeros(n_cells, dtype=np.float64)

    # --- Build absolute pin positions ---
    macro_idx = ctx.pin_macro_idx  # -1 for PORT
    px = ctx.pin_offset_x.astype(np.float64)
    py = ctx.pin_offset_y.astype(np.float64)
    movable = macro_idx >= 0
    if movable.any():
        valid_idx = macro_idx[movable]
        px[movable] += positions[valid_idx, 0].astype(np.float64)
        py[movable] += positions[valid_idx, 1].astype(np.float64)

    def _gcell(x: float, y: float) -> tuple[int, int]:
        """Return (row, col) grid cell for world coordinate (x, y), clamped to valid bounds."""
        row = max(0, min(grid_row - 1, int(math.floor(y / cell_h))))
        col = max(0, min(grid_col - 1, int(math.floor(x / cell_w))))
        return row, col

    # --- Net routing (JIT) ---
    _congestion_net_routing_kernel(
        px,
        py,
        ctx.net_pin_starts,
        ctx.net_pin_indices,
        ctx.net_weights,
        ctx.net_source_pin_local,
        grid_row,
        grid_col,
        cell_w,
        cell_h,
        H_cong,
        V_cong,
    )

    # --- Normalize wire routing demand by capacity ---
    if grid_v_routes > 0.0:
        V_cong /= grid_v_routes
    if grid_h_routes > 0.0:
        H_cong /= grid_h_routes

    # --- Smoothing (JIT, applied before adding macro blockage) ---
    _smooth_v(V_cong, grid_row, grid_col, ctx.smooth_range)
    _smooth_h(H_cong, grid_row, grid_col, ctx.smooth_range)

    # --- Macro blockage (hard macros only, JIT batch) ---
    num_hard = int(ctx.macro_is_hard.sum())
    if num_hard > 0:
        pos_x = positions[:num_hard, 0].astype(np.float64)
        pos_y = positions[:num_hard, 1].astype(np.float64)
        mw_f64 = ctx.macro_w[:num_hard].astype(np.float64)
        mh_f64 = ctx.macro_h[:num_hard].astype(np.float64)
        _macro_blockage_all_kernel(
            pos_x,
            pos_y,
            mw_f64,
            mh_f64,
            num_hard,
            grid_row,
            grid_col,
            cell_w,
            cell_h,
            float(ctx.hrouting_alloc),
            float(ctx.vrouting_alloc),
            H_macro,
            V_macro,
        )

    if grid_v_routes > 0.0:
        V_macro /= grid_v_routes
    if grid_h_routes > 0.0:
        H_macro /= grid_h_routes

    # TILOS ``get_congestion_cost`` does list concatenation of V_routing_cong and
    # H_routing_cong, then abu(concat, 0.05) = top-5% mean over 2*n_cells entries.
    concat = np.concatenate([V_cong + V_macro, H_cong + H_macro])
    n_total = concat.size
    cnt = int(0.05 * n_total)  # math.floor for positive floats
    if cnt == 0:
        return float(concat.max()) if n_total > 0 else 0.0
    # Top-cnt mean via partition (fastest correct approach)
    top_vals = np.partition(concat, n_total - cnt)[n_total - cnt:]
    return float(top_vals.mean())


def fast_congestion_per_bin(
    positions: np.ndarray, ctx: FastProxyContext
) -> np.ndarray:
    """Per-bin routing demand as a [grid_row, grid_col] grid.

    Uses the same kernels as ``fast_congestion`` (net routing + smoothing +
    macro blockage) but returns the per-cell demand BEFORE the top-5% abu
    reduction. Each cell's value is ``(V_cong + V_macro) + (H_cong + H_macro)``
    after capacity normalization and V-smoothing of wire routes — i.e., the
    same numbers the abu reduction operates on, summed by direction so a
    single scalar per bin ranks "how congested is this bin".

    Used by Lever L (worst-congestion-bin destroy).
    """
    import math

    grid_col = ctx.grid_col
    grid_row = ctx.grid_row
    cell_w = ctx.canvas_w / grid_col
    cell_h = ctx.canvas_h / grid_row

    grid_v_routes = cell_w * ctx.v_routes_per_micron
    grid_h_routes = cell_h * ctx.h_routes_per_micron

    n_cells = grid_row * grid_col
    H_cong = np.zeros(n_cells, dtype=np.float64)
    V_cong = np.zeros(n_cells, dtype=np.float64)
    H_macro = np.zeros(n_cells, dtype=np.float64)
    V_macro = np.zeros(n_cells, dtype=np.float64)

    macro_idx = ctx.pin_macro_idx
    px = ctx.pin_offset_x.astype(np.float64)
    py = ctx.pin_offset_y.astype(np.float64)
    movable = macro_idx >= 0
    if movable.any():
        valid_idx = macro_idx[movable]
        px[movable] += positions[valid_idx, 0].astype(np.float64)
        py[movable] += positions[valid_idx, 1].astype(np.float64)

    _congestion_net_routing_kernel(
        px, py, ctx.net_pin_starts, ctx.net_pin_indices, ctx.net_weights,
        ctx.net_source_pin_local, grid_row, grid_col, cell_w, cell_h,
        H_cong, V_cong,
    )
    if grid_v_routes > 0.0:
        V_cong /= grid_v_routes
    if grid_h_routes > 0.0:
        H_cong /= grid_h_routes

    _smooth_v(V_cong, grid_row, grid_col, ctx.smooth_range)
    _smooth_h(H_cong, grid_row, grid_col, ctx.smooth_range)

    num_hard = int(ctx.macro_is_hard.sum())
    if num_hard > 0:
        pos_x = positions[:num_hard, 0].astype(np.float64)
        pos_y = positions[:num_hard, 1].astype(np.float64)
        mw_f64 = ctx.macro_w[:num_hard].astype(np.float64)
        mh_f64 = ctx.macro_h[:num_hard].astype(np.float64)
        _macro_blockage_all_kernel(
            pos_x, pos_y, mw_f64, mh_f64, num_hard,
            grid_row, grid_col, cell_w, cell_h,
            float(ctx.hrouting_alloc), float(ctx.vrouting_alloc),
            H_macro, V_macro,
        )
    if grid_v_routes > 0.0:
        V_macro /= grid_v_routes
    if grid_h_routes > 0.0:
        H_macro /= grid_h_routes

    per_bin = (V_cong + V_macro) + (H_cong + H_macro)
    return per_bin.reshape(grid_row, grid_col)


def _two_pin_route(
    node_gcells: list[tuple[int, int]],
    source_gcell: tuple[int, int],
    weight: float,
    grid_col: int,
    H_cong: np.ndarray,
    V_cong: np.ndarray,
) -> None:
    """L-shaped 2-pin routing matching TILOS ``__two_pin_net_routing``."""
    if node_gcells[0] == source_gcell:
        sink_gcell = node_gcells[1]
    else:
        sink_gcell = node_gcells[0]

    row_min = min(sink_gcell[0], source_gcell[0])
    row_max = max(sink_gcell[0], source_gcell[0])
    col_min = min(sink_gcell[1], source_gcell[1])
    col_max = max(sink_gcell[1], source_gcell[1])

    # H routing along source row.
    row = source_gcell[0]
    for col in range(col_min, col_max):
        H_cong[row * grid_col + col] += weight

    # V routing along sink col.
    col = sink_gcell[1]
    for r in range(row_min, row_max):
        V_cong[r * grid_col + col] += weight


def _l_routing(
    node_gcells: list[tuple[int, int]],
    weight: float,
    grid_col: int,
    H_cong: np.ndarray,
    V_cong: np.ndarray,
) -> None:
    """TILOS ``__l_routing`` for 3-pin nets (sorted by (col, row))."""
    node_gcells_sorted = sorted(node_gcells, key=lambda x: (x[1], x[0]))
    y1, x1 = node_gcells_sorted[0]
    y2, x2 = node_gcells_sorted[1]
    y3, x3 = node_gcells_sorted[2]

    for col in range(x1, x2):
        H_cong[y1 * grid_col + col] += weight
    for col in range(x2, x3):
        H_cong[y2 * grid_col + col] += weight
    for r in range(min(y1, y2), max(y1, y2)):
        V_cong[r * grid_col + x2] += weight
    for r in range(min(y2, y3), max(y2, y3)):
        V_cong[r * grid_col + x3] += weight


def _t_routing(
    node_gcells: list[tuple[int, int]],
    weight: float,
    grid_col: int,
    H_cong: np.ndarray,
    V_cong: np.ndarray,
) -> None:
    """TILOS ``__t_routing`` for 3-pin nets (sorted by (row, col))."""
    node_gcells_sorted = sorted(node_gcells)
    y1, x1 = node_gcells_sorted[0]
    y2, x2 = node_gcells_sorted[1]
    y3, x3 = node_gcells_sorted[2]

    xmin = min(x1, x2, x3)
    xmax = max(x1, x2, x3)
    for col in range(xmin, xmax):
        H_cong[y2 * grid_col + col] += weight
    for r in range(min(y1, y2), max(y1, y2)):
        V_cong[r * grid_col + x1] += weight
    for r in range(min(y2, y3), max(y2, y3)):
        V_cong[r * grid_col + x3] += weight


def _three_pin_route(
    node_gcells: list[tuple[int, int]],
    weight: float,
    grid_col: int,
    H_cong: np.ndarray,
    V_cong: np.ndarray,
) -> None:
    """TILOS ``__three_pin_net_routing`` dispatcher."""
    temp = sorted(node_gcells, key=lambda x: (x[1], x[0]))
    y1, x1 = temp[0]
    y2, x2 = temp[1]
    y3, x3 = temp[2]

    if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
        _l_routing(temp, weight, grid_col, H_cong, V_cong)
    elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
        for col in range(x1, x2):
            H_cong[y1 * grid_col + col] += weight
        for r in range(y1, max(y2, y3)):
            V_cong[r * grid_col + x2] += weight
    elif y2 == y3:
        for col in range(x1, x2):
            H_cong[y1 * grid_col + col] += weight
        for col in range(x2, x3):
            H_cong[y2 * grid_col + col] += weight
        for r in range(min(y2, y1), max(y2, y1)):
            V_cong[r * grid_col + x2] += weight
    else:
        _t_routing(temp, weight, grid_col, H_cong, V_cong)


@njit(cache=True)
def _smooth_v(
    V_cong: np.ndarray,
    grid_row: int,
    grid_col: int,
    smooth_range: int,
) -> None:
    """Horizontal smoothing of V routing congestion (TILOS ``__smooth_routing_cong`` V part)."""
    temp = np.zeros(grid_row * grid_col, dtype=np.float64)
    for row in range(grid_row):
        for col in range(grid_col):
            lp = col - smooth_range
            if lp < 0:
                lp = 0
            rp = col + smooth_range
            if rp > grid_col - 1:
                rp = grid_col - 1
            gcell_cnt = rp - lp + 1
            val = V_cong[row * grid_col + col] / gcell_cnt
            for ptr in range(lp, rp + 1):
                temp[row * grid_col + ptr] += val
    for i in range(grid_row * grid_col):
        V_cong[i] = temp[i]


@njit(cache=True)
def _smooth_h(
    H_cong: np.ndarray,
    grid_row: int,
    grid_col: int,
    smooth_range: int,
) -> None:
    """Vertical smoothing of H routing congestion (TILOS ``__smooth_routing_cong`` H part)."""
    temp = np.zeros(grid_row * grid_col, dtype=np.float64)
    for row in range(grid_row):
        for col in range(grid_col):
            lp = row - smooth_range
            if lp < 0:
                lp = 0
            up = row + smooth_range
            if up > grid_row - 1:
                up = grid_row - 1
            gcell_cnt = up - lp + 1
            val = H_cong[row * grid_col + col] / gcell_cnt
            for ptr in range(lp, up + 1):
                temp[ptr * grid_col + col] += val
    for i in range(grid_row * grid_col):
        H_cong[i] = temp[i]


@njit(cache=True)
def _macro_blockage_all_kernel(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    macro_w: np.ndarray,
    macro_h: np.ndarray,
    num_hard: int,
    grid_row: int,
    grid_col: int,
    cell_w: float,
    cell_h: float,
    hrouting_alloc: float,
    vrouting_alloc: float,
    H_macro: np.ndarray,
    V_macro: np.ndarray,
) -> None:
    """Batched macro blockage — deposits all hard-macro footprints in one JIT call.

    Mirrors TILOS ``__macro_route_over_grid_cell`` semantics including the partial-
    overlap boundary correction on the upper-right row and column.
    """
    for m in range(num_hard):
        mod_x = pos_x[m]
        mod_y = pos_y[m]
        mod_w = macro_w[m]
        mod_h = macro_h[m]

        x_lo = mod_x - mod_w / 2.0
        x_hi = mod_x + mod_w / 2.0
        y_lo = mod_y - mod_h / 2.0
        y_hi = mod_y + mod_h / 2.0

        ur_row = int(y_hi // cell_h)
        ur_col = int(x_hi // cell_w)
        bl_row = int(y_lo // cell_h)
        bl_col = int(x_lo // cell_w)

        if ur_row < 0 or ur_col < 0:
            continue
        if bl_row < 0:
            bl_row = 0
        if bl_col < 0:
            bl_col = 0
        if ur_row > grid_row - 1:
            ur_row = grid_row - 1
        if ur_col > grid_col - 1:
            ur_col = grid_col - 1

        partial_v = False
        partial_h = False

        for r_i in range(bl_row, ur_row + 1):
            cell_y_lo = r_i * cell_h
            cell_y_hi = cell_y_lo + cell_h
            y_top = y_hi if y_hi < cell_y_hi else cell_y_hi
            y_bot = y_lo if y_lo > cell_y_lo else cell_y_lo
            y_dist = y_top - y_bot
            if y_dist < 0.0:
                y_dist = 0.0

            for c_i in range(bl_col, ur_col + 1):
                cell_x_lo = c_i * cell_w
                cell_x_hi = cell_x_lo + cell_w
                x_right = x_hi if x_hi < cell_x_hi else cell_x_hi
                x_left = x_lo if x_lo > cell_x_lo else cell_x_lo
                x_dist = x_right - x_left
                if x_dist < 0.0:
                    x_dist = 0.0

                if ur_row != bl_row:
                    diff = y_dist - cell_h
                    if diff < 0.0:
                        diff = -diff
                    if (r_i == bl_row and diff > 1e-5) or (r_i == ur_row and diff > 1e-5):
                        partial_v = True
                if ur_col != bl_col:
                    diff = x_dist - cell_w
                    if diff < 0.0:
                        diff = -diff
                    if (c_i == bl_col and diff > 1e-5) or (c_i == ur_col and diff > 1e-5):
                        partial_h = True

                V_macro[r_i * grid_col + c_i] += x_dist * vrouting_alloc
                H_macro[r_i * grid_col + c_i] += y_dist * hrouting_alloc

        if partial_v:
            r_i = ur_row
            cell_y_lo = r_i * cell_h
            cell_y_hi = cell_y_lo + cell_h
            for c_i in range(bl_col, ur_col + 1):
                cell_x_lo = c_i * cell_w
                cell_x_hi = cell_x_lo + cell_w
                x_right = x_hi if x_hi < cell_x_hi else cell_x_hi
                x_left = x_lo if x_lo > cell_x_lo else cell_x_lo
                x_dist = x_right - x_left
                if x_dist <= 0.0:
                    continue
                y_top = y_hi if y_hi < cell_y_hi else cell_y_hi
                y_bot = y_lo if y_lo > cell_y_lo else cell_y_lo
                if y_top - y_bot <= 0.0:
                    continue
                V_macro[r_i * grid_col + c_i] -= x_dist * vrouting_alloc

        if partial_h:
            c_i = ur_col
            for r_i in range(bl_row, ur_row + 1):
                cell_y_lo = r_i * cell_h
                cell_y_hi = cell_y_lo + cell_h
                y_top = y_hi if y_hi < cell_y_hi else cell_y_hi
                y_bot = y_lo if y_lo > cell_y_lo else cell_y_lo
                y_dist_row = y_top - y_bot
                if y_dist_row <= 0.0:
                    continue
                cell_x_lo = c_i * cell_w
                cell_x_hi = cell_x_lo + cell_w
                x_right = x_hi if x_hi < cell_x_hi else cell_x_hi
                x_left = x_lo if x_lo > cell_x_lo else cell_x_lo
                if x_right - x_left <= 0.0:
                    continue
                H_macro[r_i * grid_col + c_i] -= y_dist_row * hrouting_alloc


def _macro_blockage(
    mod_x: float,
    mod_y: float,
    mod_w: float,
    mod_h: float,
    grid_row: int,
    grid_col: int,
    cell_w: float,
    cell_h: float,
    hrouting_alloc: float,
    vrouting_alloc: float,
    H_macro: np.ndarray,
    V_macro: np.ndarray,
) -> None:
    """TILOS ``__macro_route_over_grid_cell`` — deposit overlap-based blockage.

    Kept for backward compatibility; new code should use
    ``_macro_blockage_all_kernel`` to amortize JIT-call overhead.
    """
    import math

    x_lo = mod_x - mod_w / 2.0
    x_hi = mod_x + mod_w / 2.0
    y_lo = mod_y - mod_h / 2.0
    y_hi = mod_y + mod_h / 2.0

    ur_row = math.floor(y_hi / cell_h)
    ur_col = math.floor(x_hi / cell_w)
    bl_row = math.floor(y_lo / cell_h)
    bl_col = math.floor(x_lo / cell_w)

    # OOB check: skip if upper-right corner is out of bounds.
    if ur_row < 0 or ur_col < 0:
        return
    if bl_row < 0:
        bl_row = 0
    if bl_col < 0:
        bl_col = 0
    if ur_row > grid_row - 1:
        ur_row = grid_row - 1
    if ur_col > grid_col - 1:
        ur_col = grid_col - 1

    if_PARTIAL_OVERLAP_VERTICAL = False
    if_PARTIAL_OVERLAP_HORIZONTAL = False

    for r_i in range(bl_row, ur_row + 1):
        cell_y_lo = r_i * cell_h
        cell_y_hi = cell_y_lo + cell_h
        y_dist = min(y_hi, cell_y_hi) - max(y_lo, cell_y_lo)
        if y_dist <= 0.0:
            y_dist = 0.0

        for c_i in range(bl_col, ur_col + 1):
            cell_x_lo = c_i * cell_w
            cell_x_hi = cell_x_lo + cell_w
            x_dist = min(x_hi, cell_x_hi) - max(x_lo, cell_x_lo)
            if x_dist <= 0.0:
                x_dist = 0.0

            if ur_row != bl_row:
                if (r_i == bl_row and abs(y_dist - cell_h) > 1e-5) or \
                   (r_i == ur_row and abs(y_dist - cell_h) > 1e-5):
                    if_PARTIAL_OVERLAP_VERTICAL = True

            if ur_col != bl_col:
                if (c_i == bl_col and abs(x_dist - cell_w) > 1e-5) or \
                   (c_i == ur_col and abs(x_dist - cell_w) > 1e-5):
                    if_PARTIAL_OVERLAP_HORIZONTAL = True

            V_macro[r_i * grid_col + c_i] += x_dist * vrouting_alloc
            H_macro[r_i * grid_col + c_i] += y_dist * hrouting_alloc

    if if_PARTIAL_OVERLAP_VERTICAL:
        for r_i in range(ur_row, ur_row + 1):
            cell_y_lo = r_i * cell_h
            cell_y_hi = cell_y_lo + cell_h
            for c_i in range(bl_col, ur_col + 1):
                cell_x_lo = c_i * cell_w
                cell_x_hi = cell_x_lo + cell_w
                x_dist = min(x_hi, cell_x_hi) - max(x_lo, cell_x_lo)
                if x_dist <= 0.0:
                    continue
                y_dist_top = min(y_hi, cell_y_hi) - max(y_lo, cell_y_lo)
                if y_dist_top <= 0.0:
                    continue
                V_macro[r_i * grid_col + c_i] -= x_dist * vrouting_alloc

    if if_PARTIAL_OVERLAP_HORIZONTAL:
        for r_i in range(bl_row, ur_row + 1):
            cell_y_lo = r_i * cell_h
            cell_y_hi = cell_y_lo + cell_h
            y_dist_row = min(y_hi, cell_y_hi) - max(y_lo, cell_y_lo)
            if y_dist_row <= 0.0:
                continue
            for c_i in range(ur_col, ur_col + 1):
                cell_x_lo = c_i * cell_w
                cell_x_hi = cell_x_lo + cell_w
                x_dist = min(x_hi, cell_x_hi) - max(x_lo, cell_x_lo)
                if x_dist <= 0.0:
                    continue
                H_macro[r_i * grid_col + c_i] -= y_dist_row * hrouting_alloc
