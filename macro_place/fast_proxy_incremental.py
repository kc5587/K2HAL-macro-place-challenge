"""Incremental fast-proxy updates for single-macro coordinate moves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from macro_place.fast_proxy import (
    FastProxyContext,
    FastProxyResult,
    fast_congestion,
    fast_hpwl,
)


@dataclass
class FastProxyCache:
    """Mutable per-component caches keyed by macro index."""

    positions: np.ndarray
    per_net_hpwl: np.ndarray
    per_net_bbox: np.ndarray
    macro_to_nets: list[np.ndarray]
    density_bins: np.ndarray
    macro_to_bins: list[np.ndarray]
    congestion_bins: np.ndarray
    overlap_pairs: set[tuple[int, int]]
    total_hpwl: float
    total_hpwl_raw: float
    total_density: float
    total_congestion: float
    total_overlap_count: int
    _last_snapshot: dict[str, Any] | None = field(default=None, repr=False)


def build_cache(positions: np.ndarray, ctx: FastProxyContext) -> FastProxyCache:
    """Build an exact cache for ``positions``."""
    pos = np.asarray(positions, dtype=np.float64).copy()
    macro_to_nets = _build_macro_to_nets(ctx, pos.shape[0])
    per_net_hpwl, per_net_bbox = _build_net_hpwl(pos, ctx)
    density_bins, macro_to_bins = _build_density(pos, ctx)
    overlap_pairs = _build_overlap_pairs(pos, ctx)
    total_hpwl_raw = _sum_in_order(per_net_hpwl)
    return FastProxyCache(
        positions=pos,
        per_net_hpwl=per_net_hpwl,
        per_net_bbox=per_net_bbox,
        macro_to_nets=macro_to_nets,
        density_bins=density_bins,
        macro_to_bins=macro_to_bins,
        congestion_bins=np.zeros((ctx.grid_row, ctx.grid_col), dtype=np.float64),
        overlap_pairs=overlap_pairs,
        total_hpwl=float(fast_hpwl(pos, ctx)),
        total_hpwl_raw=total_hpwl_raw,
        total_density=_density_cost(density_bins, ctx),
        total_congestion=_congestion_cost(pos, ctx),
        total_overlap_count=len(overlap_pairs),
    )


def apply_move(
    cache: FastProxyCache,
    ctx: FastProxyContext,
    macro_idx: int,
    new_xy: np.ndarray | tuple[float, float],
    *,
    exact_hpwl: bool = True,
    update_congestion: bool = True,
) -> FastProxyResult:
    """Apply one macro move in place and return exact cached totals."""
    idx = int(macro_idx)
    xy = np.asarray(new_xy, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError("new_xy must have shape (2,)")
    if idx < 0 or idx >= cache.positions.shape[0]:
        raise IndexError(f"macro_idx out of range: {idx}")

    nets = cache.macro_to_nets[idx]
    cache._last_snapshot = {
        "macro_idx": idx,
        "position": cache.positions[idx].copy(),
        "nets": nets.copy(),
        "per_net_hpwl": cache.per_net_hpwl[nets].copy(),
        "per_net_bbox": cache.per_net_bbox[nets].copy(),
        "density_bins": cache.density_bins.copy(),
        "macro_bins": cache.macro_to_bins[idx].copy(),
        "overlap_pairs": set(cache.overlap_pairs),
        "total_hpwl": cache.total_hpwl,
        "total_hpwl_raw": cache.total_hpwl_raw,
        "total_density": cache.total_density,
        "total_congestion": cache.total_congestion,
        "total_overlap_count": cache.total_overlap_count,
    }

    _remove_density(cache, idx)
    _remove_overlap_pairs(cache, idx)
    cache.positions[idx] = xy
    _add_density(cache, ctx, idx)
    _add_overlap_pairs(cache, ctx, idx)
    _update_hpwl(cache, ctx, nets)
    if exact_hpwl:
        cache.total_hpwl = float(fast_hpwl(cache.positions, ctx))
    else:
        cache.total_hpwl = _normalize_hpwl(cache.total_hpwl_raw, ctx)
    cache.total_density = _density_cost(cache.density_bins, ctx)
    if update_congestion:
        cache.total_congestion = _congestion_cost(cache.positions, ctx)
    cache.total_overlap_count = len(cache.overlap_pairs)
    return cache_result(cache)


def revert_move(
    cache: FastProxyCache,
    ctx: FastProxyContext,
    macro_idx: int,
    old_xy: np.ndarray | tuple[float, float],
) -> None:
    """Undo the most recent ``apply_move`` exactly from its snapshot."""
    del ctx, old_xy
    snapshot = cache._last_snapshot
    if snapshot is None or int(snapshot["macro_idx"]) != int(macro_idx):
        raise ValueError("revert_move must match the most recent apply_move")
    idx = int(snapshot["macro_idx"])
    nets = snapshot["nets"]
    cache.positions[idx] = snapshot["position"]
    cache.per_net_hpwl[nets] = snapshot["per_net_hpwl"]
    cache.per_net_bbox[nets] = snapshot["per_net_bbox"]
    cache.density_bins[:, :] = snapshot["density_bins"]
    cache.macro_to_bins[idx] = snapshot["macro_bins"]
    cache.overlap_pairs = set(snapshot["overlap_pairs"])
    cache.total_hpwl = float(snapshot["total_hpwl"])
    cache.total_hpwl_raw = float(snapshot["total_hpwl_raw"])
    cache.total_density = float(snapshot["total_density"])
    cache.total_congestion = float(snapshot["total_congestion"])
    cache.total_overlap_count = int(snapshot["total_overlap_count"])
    cache._last_snapshot = None


def cache_result(cache: FastProxyCache) -> FastProxyResult:
    """Return a result object from the cache's current component totals."""
    proxy = cache.total_hpwl + 0.5 * cache.total_density + 0.5 * cache.total_congestion
    return FastProxyResult(
        proxy_cost=float(proxy),
        wirelength=float(cache.total_hpwl),
        density=float(cache.total_density),
        congestion=float(cache.total_congestion),
        overlap_count=int(cache.total_overlap_count),
    )


def can_update_incrementally(ctx: FastProxyContext) -> bool:
    """Compatibility helper; cd.py deliberately does not gate on this."""
    return _congestion_is_static(ctx)


def _build_macro_to_nets(ctx: FastProxyContext, num_macros: int) -> list[np.ndarray]:
    nets_by_macro: list[list[int]] = [[] for _ in range(num_macros)]
    for net_id in range(ctx.net_pin_starts.shape[0] - 1):
        start = int(ctx.net_pin_starts[net_id])
        end = int(ctx.net_pin_starts[net_id + 1])
        seen: set[int] = set()
        for pin_local in ctx.net_pin_indices[start:end]:
            macro_idx = int(ctx.pin_macro_idx[int(pin_local)])
            if macro_idx < 0 or macro_idx in seen or macro_idx >= num_macros:
                continue
            nets_by_macro[macro_idx].append(net_id)
            seen.add(macro_idx)
    return [np.asarray(nets, dtype=np.int32) for nets in nets_by_macro]


def _build_net_hpwl(
    positions: np.ndarray,
    ctx: FastProxyContext,
) -> tuple[np.ndarray, np.ndarray]:
    num_nets = ctx.net_pin_starts.shape[0] - 1
    per_net = np.zeros(num_nets, dtype=np.float64)
    bbox = np.zeros((num_nets, 4), dtype=np.float64)
    for net_id in range(num_nets):
        per_net[net_id], bbox[net_id] = _net_hpwl_and_bbox(positions, ctx, net_id)
    return per_net, bbox


def _net_hpwl_and_bbox(
    positions: np.ndarray,
    ctx: FastProxyContext,
    net_id: int,
) -> tuple[float, np.ndarray]:
    start = int(ctx.net_pin_starts[net_id])
    end = int(ctx.net_pin_starts[net_id + 1])
    if end <= start:
        return 0.0, np.zeros(4, dtype=np.float64)

    xs: list[float] = []
    ys: list[float] = []
    for pin_local in ctx.net_pin_indices[start:end]:
        pin = int(pin_local)
        macro_idx = int(ctx.pin_macro_idx[pin])
        x = np.float32(ctx.pin_offset_x[pin])
        y = np.float32(ctx.pin_offset_y[pin])
        if macro_idx >= 0:
            x = np.float32(x + positions[macro_idx, 0])
            y = np.float32(y + positions[macro_idx, 1])
        xs.append(float(x))
        ys.append(float(y))
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    hpwl = float(ctx.net_weights[net_id]) * ((x_max - x_min) + (y_max - y_min))
    return hpwl, np.asarray([x_min, y_min, x_max, y_max], dtype=np.float64)


def _update_hpwl(
    cache: FastProxyCache,
    ctx: FastProxyContext,
    nets: np.ndarray,
) -> None:
    for net_id_raw in nets:
        net_id = int(net_id_raw)
        old = float(cache.per_net_hpwl[net_id])
        new, bbox = _net_hpwl_and_bbox(cache.positions, ctx, net_id)
        cache.per_net_hpwl[net_id] = new
        cache.per_net_bbox[net_id] = bbox
        cache.total_hpwl_raw += new - old


def _sum_in_order(values: np.ndarray) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total


def _normalize_hpwl(total_raw_hpwl: float, ctx: FastProxyContext) -> float:
    norm = (ctx.canvas_w + ctx.canvas_h) * ctx.net_cnt
    return total_raw_hpwl / norm if norm > 0.0 else 0.0


def _build_density(
    positions: np.ndarray,
    ctx: FastProxyContext,
) -> tuple[np.ndarray, list[np.ndarray]]:
    bins = np.zeros((ctx.grid_row, ctx.grid_col), dtype=np.float64)
    macro_to_bins: list[np.ndarray] = []
    for macro_idx in range(positions.shape[0]):
        entries = _density_entries(positions, ctx, macro_idx)
        macro_to_bins.append(entries)
        for flat, contribution in entries:
            row = int(flat) // ctx.grid_col
            col = int(flat) % ctx.grid_col
            bins[row, col] += float(contribution)
    return bins, macro_to_bins


def _density_entries(
    positions: np.ndarray,
    ctx: FastProxyContext,
    macro_idx: int,
) -> np.ndarray:
    cell_w = ctx.canvas_w / ctx.grid_col
    cell_h = ctx.canvas_h / ctx.grid_row
    cell_area = cell_w * cell_h
    if cell_area <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    half_w = float(ctx.macro_w[macro_idx]) * 0.5
    half_h = float(ctx.macro_h[macro_idx]) * 0.5
    x_lo = float(positions[macro_idx, 0]) - half_w
    x_hi = float(positions[macro_idx, 0]) + half_w
    y_lo = float(positions[macro_idx, 1]) - half_h
    y_hi = float(positions[macro_idx, 1]) + half_h

    c0 = max(0, int(x_lo / cell_w))
    c1 = min(ctx.grid_col - 1, int(x_hi / cell_w))
    r0 = max(0, int(y_lo / cell_h))
    r1 = min(ctx.grid_row - 1, int(y_hi / cell_h))
    entries: list[tuple[float, float]] = []
    for row in range(r0, r1 + 1):
        cell_y_lo = row * cell_h
        cell_y_hi = cell_y_lo + cell_h
        oy = min(y_hi, cell_y_hi) - max(y_lo, cell_y_lo)
        if oy <= 0.0:
            continue
        for col in range(c0, c1 + 1):
            cell_x_lo = col * cell_w
            cell_x_hi = cell_x_lo + cell_w
            ox = min(x_hi, cell_x_hi) - max(x_lo, cell_x_lo)
            if ox <= 0.0:
                continue
            entries.append((float(row * ctx.grid_col + col), (ox * oy) / cell_area))
    return np.asarray(entries, dtype=np.float64)


def _remove_density(cache: FastProxyCache, macro_idx: int) -> None:
    for flat, contribution in cache.macro_to_bins[macro_idx]:
        row = int(flat) // cache.density_bins.shape[1]
        col = int(flat) % cache.density_bins.shape[1]
        cache.density_bins[row, col] -= float(contribution)


def _add_density(cache: FastProxyCache, ctx: FastProxyContext, macro_idx: int) -> None:
    entries = _density_entries(cache.positions, ctx, macro_idx)
    cache.macro_to_bins[macro_idx] = entries
    for flat, contribution in entries:
        row = int(flat) // ctx.grid_col
        col = int(flat) % ctx.grid_col
        cache.density_bins[row, col] += float(contribution)


def _density_cost(density_bins: np.ndarray, ctx: FastProxyContext) -> float:
    import math

    num_cells = ctx.grid_row * ctx.grid_col
    occupied = [float(d) for d in density_bins.ravel() if float(d) != 0.0]
    if not occupied:
        return 0.0
    occupied.sort(reverse=True)
    density_cnt = math.floor(num_cells * 0.1)
    if num_cells < 10:
        return 0.5 * float(sum(occupied) / len(occupied))
    sum_density = 0.0
    idx = 0
    while idx < density_cnt and idx < len(occupied):
        sum_density += occupied[idx]
        idx += 1
    return 0.5 * float(sum_density / density_cnt)


def _build_overlap_pairs(
    positions: np.ndarray,
    ctx: FastProxyContext,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    hard_indices = np.flatnonzero(ctx.macro_is_hard)
    for offset, left_raw in enumerate(hard_indices):
        left = int(left_raw)
        for right_raw in hard_indices[offset + 1:]:
            right = int(right_raw)
            if _overlaps(positions, ctx, left, right):
                pairs.add((left, right))
    return pairs


def _remove_overlap_pairs(cache: FastProxyCache, macro_idx: int) -> None:
    idx = int(macro_idx)
    cache.overlap_pairs = {
        pair for pair in cache.overlap_pairs if pair[0] != idx and pair[1] != idx
    }


def _add_overlap_pairs(
    cache: FastProxyCache,
    ctx: FastProxyContext,
    macro_idx: int,
) -> None:
    idx = int(macro_idx)
    if not bool(ctx.macro_is_hard[idx]):
        return
    for other_raw in np.flatnonzero(ctx.macro_is_hard):
        other = int(other_raw)
        if other == idx:
            continue
        left, right = (idx, other) if idx < other else (other, idx)
        if _overlaps(cache.positions, ctx, left, right):
            cache.overlap_pairs.add((left, right))


def _overlaps(
    positions: np.ndarray,
    ctx: FastProxyContext,
    left: int,
    right: int,
) -> bool:
    dx = abs(float(positions[left, 0]) - float(positions[right, 0]))
    dy = abs(float(positions[left, 1]) - float(positions[right, 1]))
    min_x = float(np.float32(ctx.macro_w[left] + ctx.macro_w[right]) * np.float32(0.5))
    min_y = float(np.float32(ctx.macro_h[left] + ctx.macro_h[right]) * np.float32(0.5))
    return dx < min_x and dy < min_y


def _congestion_cost(positions: np.ndarray, ctx: FastProxyContext) -> float:
    if _congestion_is_static(ctx):
        return 0.0
    return float(fast_congestion(positions, ctx))


def _congestion_is_static(ctx: FastProxyContext) -> bool:
    has_routed_nets = bool((ctx.net_source_pin_local >= 0).any())
    has_hard_blockage = bool(ctx.macro_is_hard.any()) and (
        float(ctx.hrouting_alloc) != 0.0 or float(ctx.vrouting_alloc) != 0.0
    )
    return not has_routed_nets and not has_hard_blockage
