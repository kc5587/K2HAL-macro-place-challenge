"""TDD for warm-start init helper."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_warm_start_seed_0_is_deterministic_initial_plc() -> None:
    """seed=0 must return the unperturbed initial.plc positions."""
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from submissions.macro_placer.cd_lns_placer import (
        CDLNSPlacer,
        _warm_start_positions,
    )

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None

    placer = CDLNSPlacer()
    initial = placer._initial_positions(b, plc)
    warm = _warm_start_positions(b, plc, seed=0, sigma=0.05)
    np.testing.assert_array_equal(warm, initial)


@pytest.mark.unit
def test_warm_start_seed_positive_perturbs_within_canvas() -> None:
    """seed>0 must perturb but keep all positions in canvas."""
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from submissions.macro_placer.cd_lns_placer import (
        CDLNSPlacer,
        _warm_start_positions,
    )

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None

    placer = CDLNSPlacer()
    initial = placer._initial_positions(b, plc)
    warm = _warm_start_positions(b, plc, seed=1, sigma=0.05)

    # In canvas
    assert (warm[:, 0] >= 0.0).all() and (warm[:, 0] <= b.canvas_width).all()
    assert (warm[:, 1] >= 0.0).all() and (warm[:, 1] <= b.canvas_height).all()
    # At least 90% perturbed (some near-edge positions may not move much after clip)
    diff = np.abs(warm - initial).sum(axis=1)
    assert (diff > 0.0).mean() >= 0.9, (
        f"only {(diff > 0).mean():.0%} of nodes perturbed"
    )


@pytest.mark.integration
def test_warm_start_beats_random_on_ibm01_in_60s() -> None:
    """Calibration gate: warm-start must reach surrogate cost <= 80% of
    random-init's surrogate cost in a 60-second cd_loop run on ibm01."""
    import time
    import numpy as np
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.cd import cd_loop
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy
    from submissions.macro_placer.cd_lns_placer import _warm_start_positions

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    # Random init baseline
    rng = np.random.default_rng(seed=42)
    rand_pos = np.zeros((b.num_macros, 2), dtype=np.float64)
    rand_pos[:, 0] = rng.uniform(0.0, b.canvas_width, size=b.num_macros)
    rand_pos[:, 1] = rng.uniform(0.0, b.canvas_height, size=b.num_macros)

    rand_result = cd_loop(
        initial_positions=rand_pos, ctx=ctx,
        canvas_w=b.canvas_width, canvas_h=b.canvas_height,
        max_sweeps=20, k_per_axis=8,
        radius_init_ratio=0.25, radius_min_ratio=1.0/64.0,
        time_budget_s=60.0, seed=0,
    )
    rand_final = float(fast_proxy(rand_result.positions, ctx).proxy_cost)

    # Warm-start
    warm_pos = _warm_start_positions(b, plc, seed=0, sigma=0.05)
    warm_result = cd_loop(
        initial_positions=warm_pos, ctx=ctx,
        canvas_w=b.canvas_width, canvas_h=b.canvas_height,
        max_sweeps=20, k_per_axis=8,
        radius_init_ratio=0.25, radius_min_ratio=1.0/64.0,
        time_budget_s=60.0, seed=0,
    )
    warm_final = float(fast_proxy(warm_result.positions, ctx).proxy_cost)

    print(f"\nibm01 60s cd_loop: random={rand_final:.3f} warm={warm_final:.3f} "
          f"ratio={warm_final/rand_final:.2%}")
    assert warm_final <= rand_final * 0.80, (
        f"warm-start did not beat random by 20%: "
        f"random={rand_final} warm={warm_final}"
    )
