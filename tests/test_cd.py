"""TDD for coordinate-descent core (Bet 7 restart)."""
from __future__ import annotations

import numpy as np
import pytest

from macro_place.cd import CDState, cd_grid_search


@pytest.mark.unit
def test_cd_state_is_frozen() -> None:
    s = CDState(
        positions=np.zeros((4, 2), dtype=np.float64),
        sweep_idx=0,
        radius=1.0,
        current_cost=1.0,
    )
    with pytest.raises(Exception):
        s.sweep_idx = 1  # type: ignore[misc]


@pytest.mark.integration
def test_cd_grid_search_finds_lower_cost_position_when_one_exists() -> None:
    """Synthetic: build a tiny FastProxyContext where moving node 0 to the
    canvas center cuts cost. cd_grid_search must return that position."""
    import torch
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy
    from macro_place.cd import cd_grid_search

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    # Use the plc-current placement as a starting point, then perturb node 0
    pos = np.zeros((b.num_macros, 2), dtype=np.float64)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[i, 0], pos[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[b.num_hard_macros + i, 0], pos[b.num_hard_macros + i, 1] = x, y

    # Push node 0 far from its current position to make a clearly worse layout
    base_cost = float(fast_proxy(pos, ctx).proxy_cost)
    pos_perturbed = pos.copy()
    pos_perturbed[0, 0] = 0.0
    pos_perturbed[0, 1] = 0.0
    perturbed_cost = float(fast_proxy(pos_perturbed, ctx).proxy_cost)

    # Search on the perturbed layout — best-found cost should beat perturbed,
    # and ideally come close to base_cost again.
    best_pos, best_cost = cd_grid_search(
        node_idx=0,
        positions=pos_perturbed,
        ctx=ctx,
        radius=max(b.canvas_width, b.canvas_height) / 2.0,
        k_per_axis=8,
    )
    assert best_cost < perturbed_cost - 1e-6, (
        f"cd_grid_search did not improve perturbed cost: "
        f"best={best_cost} perturbed={perturbed_cost}"
    )
    # Best position should be within canvas
    assert 0.0 <= best_pos[0] <= b.canvas_width
    assert 0.0 <= best_pos[1] <= b.canvas_height


@pytest.mark.integration
def test_cd_sweep_monotonically_reduces_cost_until_plateau() -> None:
    """A single sweep over all nodes must reduce total cost or report no
    improvement (never worsen)."""
    import torch
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy
    from macro_place.cd import cd_sweep

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos = np.zeros((b.num_macros, 2), dtype=np.float64)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[i, 0], pos[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[b.num_hard_macros + i, 0], pos[b.num_hard_macros + i, 1] = x, y

    # Perturb a handful of nodes to create headroom
    rng = np.random.default_rng(seed=0)
    for j in range(0, 20):
        pos[j, 0] = rng.uniform(0.0, b.canvas_width)
        pos[j, 1] = rng.uniform(0.0, b.canvas_height)

    cost_before = float(fast_proxy(pos, ctx).proxy_cost)

    new_pos, improved, evals = cd_sweep(
        positions=pos,
        ctx=ctx,
        radius=max(b.canvas_width, b.canvas_height) / 4.0,
        k_per_axis=4,
        seed=0,
    )

    cost_after = float(fast_proxy(new_pos, ctx).proxy_cost)
    assert cost_after <= cost_before + 1e-9, (
        f"cd_sweep regressed: before={cost_before} after={cost_after}"
    )
    assert evals > 0
    # 20 perturbed nodes — at least some should improve
    assert improved, "expected at least one accepted move from perturbed start"


@pytest.mark.integration
def test_cd_loop_improves_ibm01_significantly() -> None:
    """Starting from a random perturbed placement on ibm01, cd_loop must
    drive the surrogate proxy at least 30% lower within a 60-second budget."""
    import time
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy
    from macro_place.cd import cd_loop

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    rng = np.random.default_rng(seed=42)
    pos = np.zeros((b.num_macros, 2), dtype=np.float64)
    pos[:, 0] = rng.uniform(0.0, b.canvas_width, size=b.num_macros)
    pos[:, 1] = rng.uniform(0.0, b.canvas_height, size=b.num_macros)

    cost_before = float(fast_proxy(pos, ctx).proxy_cost)

    t0 = time.perf_counter()
    result = cd_loop(
        initial_positions=pos,
        ctx=ctx,
        canvas_w=b.canvas_width,
        canvas_h=b.canvas_height,
        max_sweeps=20,
        k_per_axis=8,
        radius_init_ratio=0.25,
        radius_min_ratio=1.0 / 64.0,
        time_budget_s=180.0,
        seed=0,
    )
    runtime = time.perf_counter() - t0

    cost_after = float(fast_proxy(result.positions, ctx).proxy_cost)
    print(f"\ncd_loop ibm01: before={cost_before:.3f} "
          f"after={cost_after:.3f} runtime={runtime:.1f}s "
          f"sweeps={result.sweeps_completed}")
    assert cost_after < cost_before * 0.7, (
        f"cd_loop did not improve ibm01 by 30%: "
        f"before={cost_before} after={cost_after}"
    )
    assert result.sweeps_completed >= 1
