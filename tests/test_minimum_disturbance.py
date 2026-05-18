"""TDD for Bet 8: minimum-disturbance restart variants + always-improving guard."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_restart_modes_default_has_three_modes() -> None:
    """Default _DEFAULT_CONFIG must include restart_modes covering all 3 modes."""
    from submissions.macro_placer.cd_lns_placer import _DEFAULT_CONFIG

    modes = _DEFAULT_CONFIG["restart_modes"]
    assert isinstance(modes, tuple)
    assert "conservative" in modes
    assert "light" in modes
    assert "aggressive" in modes
    assert len(modes) >= 1


@pytest.mark.integration
def test_conservative_mode_stays_close_to_initial_plc_on_ibm01() -> None:
    """Conservative restart's output must stay within 5% canvas L2 of initial.plc."""
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    placer = CDLNSPlacer()
    initial = placer._initial_positions(b, plc)

    placer._config["restart_modes"] = ("conservative",)
    pos, _ = placer._run_one_restart(
        benchmark=b,
        ctx=ctx,
        plc=plc,
        seed=0,
        time_budget_s=30.0,
        restart_idx=0,
    )

    canvas_max = max(b.canvas_width, b.canvas_height)
    drift = float(np.linalg.norm(pos - initial)) / float(b.num_macros)
    assert drift < canvas_max * 0.05, (
        f"conservative drifted {drift:.3f} > 5% canvas "
        f"({canvas_max * 0.05:.3f})"
    )


@pytest.mark.unit
def test_place_returns_legalized_initial_when_restarts_disabled(monkeypatch) -> None:
    """With num_restarts=0, place() should return the legalized initial.plc guard."""
    import torch
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None

    def fail_restart(*args: object, **kwargs: object) -> None:
        raise AssertionError("num_restarts=0 must not run restart search")

    monkeypatch.setattr(CDLNSPlacer, "_run_one_restart", fail_restart)

    placer = CDLNSPlacer()
    placer._config["num_restarts"] = 0
    placer._config["restart_modes"] = ()
    positions = placer.place(b)

    pos_t = positions.detach().cpu().to(torch.float32)
    cost = compute_proxy_cost(pos_t, b, plc)
    assert int(cost["overlap_count"]) == 0
    assert float(cost["proxy_cost"]) > 0.0
    assert float(cost["proxy_cost"]) < 2.0
