"""TDD for Lever R — Restart-mode parameter dispatch.

The new ``exploratory`` mode must produce a parameter set that is genuinely
distinct from the existing 3 modes (conservative / light / aggressive).
``aggressive`` semantics must match the prior implicit ``else``-branch logic
in ``cd_lns_placer._run_restart``.
"""
from __future__ import annotations

import pytest


_CFG = {
    "radius_init_ratio": 0.25,
    "max_sweeps": 30,
    "warm_start_sigma": 0.05,
}


@pytest.mark.unit
def test_conservative_mode_unchanged() -> None:
    """Must match the prior ``if mode == "conservative":`` branch exactly."""
    from macro_place.restart_modes import mode_params

    p = mode_params("conservative", _CFG)
    assert p["radius_init_ratio"] == pytest.approx(1.0 / 32.0)
    assert p["max_sweeps"] == 5
    assert p["do_lns"] is False
    assert p["warm_sigma"] == 0.0


@pytest.mark.unit
def test_light_mode_unchanged() -> None:
    from macro_place.restart_modes import mode_params

    p = mode_params("light", _CFG)
    assert p["radius_init_ratio"] == pytest.approx(1.0 / 16.0)
    assert p["max_sweeps"] == 10
    assert p["do_lns"] is False
    assert p["warm_sigma"] == pytest.approx(0.02)


@pytest.mark.unit
def test_aggressive_mode_uses_cfg_values() -> None:
    """Default path must pull from config keys exactly like the placer used to."""
    from macro_place.restart_modes import mode_params

    cfg = dict(_CFG, radius_init_ratio=0.33, max_sweeps=42, warm_start_sigma=0.08)
    p = mode_params("aggressive", cfg)
    assert p["radius_init_ratio"] == pytest.approx(0.33)
    assert p["max_sweeps"] == 42
    assert p["do_lns"] is True
    assert p["warm_sigma"] == pytest.approx(0.08)


@pytest.mark.unit
def test_unknown_mode_falls_back_to_aggressive() -> None:
    """Backward compat: any non-matching string lands in the aggressive branch."""
    from macro_place.restart_modes import mode_params

    a = mode_params("aggressive", _CFG)
    u = mode_params("totally_unknown_mode", _CFG)
    assert a == u


@pytest.mark.unit
def test_exploratory_mode_is_distinct_from_others() -> None:
    """New mode must NOT collide with any existing mode's parameter signature."""
    from macro_place.restart_modes import mode_params

    p = mode_params("exploratory", _CFG)
    others = [mode_params(m, _CFG) for m in ("conservative", "light", "aggressive")]
    # Genuine distinctness on at least 2 of the 4 keys vs every existing mode.
    for other in others:
        diffs = sum(1 for k in ("radius_init_ratio", "max_sweeps", "do_lns", "warm_sigma") if p[k] != other[k])
        assert diffs >= 2, f"exploratory too close to {other}: only {diffs} diffs"
    # Specific assertions for the design intent.
    assert p["warm_sigma"] > 0.10, "exploratory should perturb the warm start aggressively"
    assert p["radius_init_ratio"] >= 0.30, "exploratory should sweep a wide initial radius"
    assert p["do_lns"] is True, "exploratory benefits from LNS to escape its high-sigma start"


@pytest.mark.unit
def test_returned_dict_has_required_keys() -> None:
    """Downstream callers expect exactly these keys."""
    from macro_place.restart_modes import mode_params

    expected = {"radius_init_ratio", "max_sweeps", "do_lns", "warm_sigma"}
    for mode in ("conservative", "light", "aggressive", "exploratory", "unknown"):
        p = mode_params(mode, _CFG)
        assert expected.issubset(set(p.keys())), f"missing keys for {mode}"
