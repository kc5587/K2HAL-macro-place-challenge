"""TDD for Lever K' — Adaptive config from bench metrics.

The rule layer must:
- Be pure (no I/O, no benchmark/plc deps).
- Preserve the prior inline congestion rule's behavior exactly.
- Compose Rules A–D so multi-rule cases produce the union of overrides.
- Return ``{}`` for empty metrics so the placer's existing defaults apply.
"""
from __future__ import annotations

import pytest


_BASE_CFG = {
    "cd_phase_time_budget_s": 60.0,
    "lns_num_destroy": 10,
    "max_consecutive_lns_failures": 3,
}


# ---------------------------------------------------------------------------
# extract_bench_metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_metrics_shares_sum_to_one_at_unit_weights() -> None:
    from macro_place.adaptive_config import extract_bench_metrics

    # When proxy_cost == wl + density + 0.5*cong (matches 1*WL + 0.5*D + 0.5*C
    # at proxy weights of (1, 0.5, 0.5)), shares should add to 1.0.
    m = extract_bench_metrics(
        initial_proxy_cost=1.0,
        initial_wirelength=0.10,
        initial_density=0.40,  # density share = 0.40
        initial_congestion=1.00,  # cong share at 0.5 weight = 0.5
        num_macros=200,
    )
    assert m["wl_share"] == pytest.approx(0.10)
    assert m["density_share"] == pytest.approx(0.40)
    assert m["cong_share"] == pytest.approx(0.50)
    assert m["num_macros"] == 200.0


@pytest.mark.unit
def test_extract_metrics_handles_zero_proxy() -> None:
    from macro_place.adaptive_config import extract_bench_metrics

    m = extract_bench_metrics(
        initial_proxy_cost=0.0,
        initial_wirelength=0.0,
        initial_density=0.0,
        initial_congestion=0.0,
        num_macros=10,
    )
    # No NaN/Inf; shares default to 0.
    assert m["wl_share"] == 0.0
    assert m["cong_share"] == 0.0


# ---------------------------------------------------------------------------
# adaptive_overrides_from_metrics — empty / no-op
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_metrics_no_overrides() -> None:
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    assert adaptive_overrides_from_metrics({}, _BASE_CFG) == {}


@pytest.mark.unit
def test_neutral_metrics_no_overrides() -> None:
    """Mid-range bench (no rule trips) should produce zero overrides."""
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {
        "wl_share": 0.10,
        "density_share": 0.40,
        "cong_share": 0.40,  # below 0.6 threshold
        "num_macros": 200,   # between 100 and 400
    }
    assert adaptive_overrides_from_metrics(metrics, _BASE_CFG) == {}


# ---------------------------------------------------------------------------
# Rule A: congestion-dominated
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rule_a_high_cong_matches_prior_inline_rule() -> None:
    """Must match the prior inline rule from cd_lns_placer._compute_dynamic_overrides."""
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {"wl_share": 0.05, "cong_share": 0.70, "num_macros": 200}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    assert o["cd_phase_time_budget_s"] == pytest.approx(60.0 * 0.5)
    assert o["max_consecutive_lns_failures"] == 3 * 2
    assert o["lns_num_destroy"] == max(10, int(round(10 * 1.5)))


# ---------------------------------------------------------------------------
# Rule B: WL-dominated
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rule_b_wl_dominated_extends_cd_when_no_rule_a() -> None:
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {"wl_share": 0.45, "cong_share": 0.20, "num_macros": 200}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    assert o["cd_phase_time_budget_s"] == pytest.approx(60.0 * 1.25)


@pytest.mark.unit
def test_rule_b_yields_to_rule_a_on_cd_budget() -> None:
    """When both A and B would fire, A's CD shrink wins (congestion is dominant)."""
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {"wl_share": 0.45, "cong_share": 0.70, "num_macros": 200}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    assert o["cd_phase_time_budget_s"] == pytest.approx(60.0 * 0.5)


# ---------------------------------------------------------------------------
# Rule C: large macro count
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rule_c_large_macro_count_bumps_destroy() -> None:
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {"wl_share": 0.10, "cong_share": 0.20, "num_macros": 500}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    assert o["lns_num_destroy"] >= 16


@pytest.mark.unit
def test_rule_c_only_bumps_does_not_shrink() -> None:
    """Rule C must not REDUCE a higher destroy already set by Rule A."""
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    # Rule A bumps destroy to 15. Rule C floors at 16 → should land at 16.
    metrics = {"wl_share": 0.05, "cong_share": 0.70, "num_macros": 500}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    assert o["lns_num_destroy"] == 16


# ---------------------------------------------------------------------------
# Rule D: small macro count
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rule_d_small_bench_caps_destroy() -> None:
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {"wl_share": 0.10, "cong_share": 0.20, "num_macros": 50}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    # Rule D caps destroy; no other rules fired, so base_destroy=10 → capped at 6.
    assert o["lns_num_destroy"] == 6


@pytest.mark.unit
def test_rule_d_does_not_fire_at_zero_macros() -> None:
    """num_macros == 0 (unknown) must NOT trigger Rule D — we have no info."""
    from macro_place.adaptive_config import adaptive_overrides_from_metrics

    metrics = {"wl_share": 0.10, "cong_share": 0.20, "num_macros": 0}
    o = adaptive_overrides_from_metrics(metrics, _BASE_CFG)
    assert "lns_num_destroy" not in o
