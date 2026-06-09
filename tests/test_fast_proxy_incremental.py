from __future__ import annotations

import numpy as np
import pytest

from macro_place.fast_proxy import fast_proxy
from macro_place.fast_proxy_incremental import (
    apply_move,
    build_cache,
    cache_result,
    revert_move,
)


def _initial_positions(benchmark, plc) -> np.ndarray:
    pos = np.zeros((benchmark.num_macros, 2), dtype=np.float64)
    for i, idx in enumerate(benchmark.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[i, 0] = x
        pos[i, 1] = y
    for i, idx in enumerate(benchmark.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[benchmark.num_hard_macros + i, 0] = x
        pos[benchmark.num_hard_macros + i, 1] = y
    return pos


def _load_ctx(name: str):
    from pathlib import Path

    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context

    path = Path(f"benchmarks/processed/public/{name}.pt")
    if not path.exists():
        pytest.skip(f"{path} is not available in this checkout")
    benchmark = Benchmark.load(str(path))
    plc = resolve_plc(benchmark)
    assert plc is not None
    return benchmark, plc, build_fast_proxy_context(plc, benchmark)


@pytest.mark.integration
@pytest.mark.parametrize("bench", ["ibm01", "ibm02", "ibm03", "ibm04", "ibm05"])
def test_incremental_matches_full(bench: str) -> None:
    benchmark, plc, ctx = _load_ctx(bench)
    rng = np.random.default_rng(123)
    positions = _initial_positions(benchmark, plc)
    cache = build_cache(positions, ctx)

    for _ in range(100):
        macro_idx = int(rng.integers(0, benchmark.num_macros))
        old_xy = cache.positions[macro_idx].copy()
        new_xy = np.asarray(
            [
                rng.uniform(0.0, benchmark.canvas_width),
                rng.uniform(0.0, benchmark.canvas_height),
            ],
            dtype=np.float64,
        )
        got = apply_move(cache, ctx, macro_idx, new_xy)
        expected = fast_proxy(cache.positions, ctx)
        assert got.proxy_cost == pytest.approx(expected.proxy_cost, rel=1e-9)
        assert got.wirelength == pytest.approx(expected.wirelength, rel=1e-9)
        assert got.density == pytest.approx(expected.density, rel=1e-9)
        assert got.congestion == pytest.approx(expected.congestion, rel=1e-9)
        assert got.overlap_count == expected.overlap_count
        revert_move(cache, ctx, macro_idx, old_xy)


@pytest.mark.integration
def test_revert_restores_state() -> None:
    benchmark, plc, ctx = _load_ctx("ibm01")
    rng = np.random.default_rng(456)
    positions = _initial_positions(benchmark, plc)
    cache = build_cache(positions, ctx)
    baseline = build_cache(positions, ctx)

    for _ in range(100):
        macro_idx = int(rng.integers(0, benchmark.num_macros))
        old_xy = cache.positions[macro_idx].copy()
        new_xy = np.asarray(
            [
                rng.uniform(0.0, benchmark.canvas_width),
                rng.uniform(0.0, benchmark.canvas_height),
            ],
            dtype=np.float64,
        )
        apply_move(cache, ctx, macro_idx, new_xy)
        revert_move(cache, ctx, macro_idx, old_xy)

    np.testing.assert_array_equal(cache.positions, baseline.positions)
    np.testing.assert_array_equal(cache.per_net_hpwl, baseline.per_net_hpwl)
    np.testing.assert_array_equal(cache.per_net_bbox, baseline.per_net_bbox)
    np.testing.assert_array_equal(cache.density_bins, baseline.density_bins)
    assert cache.overlap_pairs == baseline.overlap_pairs
    assert cache.total_hpwl_raw == baseline.total_hpwl_raw
    assert cache.total_hpwl == baseline.total_hpwl
    assert cache.total_density == baseline.total_density
    assert cache.total_congestion == baseline.total_congestion
    assert cache.total_overlap_count == baseline.total_overlap_count


@pytest.mark.integration
def test_apply_revert_apply_loop() -> None:
    benchmark, plc, ctx = _load_ctx("ibm01")
    rng = np.random.default_rng(789)
    positions = _initial_positions(benchmark, plc)
    cache = build_cache(positions, ctx)

    for _ in range(500):
        macro_idx = int(rng.integers(0, benchmark.num_macros))
        old_xy = cache.positions[macro_idx].copy()
        new_xy = np.asarray(
            [
                rng.uniform(0.0, benchmark.canvas_width),
                rng.uniform(0.0, benchmark.canvas_height),
            ],
            dtype=np.float64,
        )
        apply_move(cache, ctx, macro_idx, new_xy)
        revert_move(cache, ctx, macro_idx, old_xy)
        got = cache_result(cache)
        expected = fast_proxy(cache.positions, ctx)
        assert got.proxy_cost == pytest.approx(expected.proxy_cost, rel=1e-9)
        assert got.overlap_count == expected.overlap_count


@pytest.mark.integration
def test_cache_shared_across_sweep() -> None:
    from macro_place.cd import cd_sweep

    benchmark, plc, ctx = _load_ctx("ibm01")
    positions = _initial_positions(benchmark, plc)
    cache = build_cache(positions, ctx)
    new_pos, _improved, _evals = cd_sweep(
        positions=positions,
        ctx=ctx,
        radius=max(benchmark.canvas_width, benchmark.canvas_height) / 8.0,
        k_per_axis=2,
        seed=0,
        cache=cache,
    )
    expected = fast_proxy(new_pos, ctx)
    got = cache_result(cache)
    assert got.proxy_cost == pytest.approx(expected.proxy_cost, rel=1e-9)
    assert got.overlap_count == expected.overlap_count
