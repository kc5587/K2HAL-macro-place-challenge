"""Lever K' — Adaptive config from measured bench properties.

Pure-function rule layer that consumes a tiny ``bench_metrics`` dict (computed
once at restart entry from ``fast_proxy`` + bench shape) and returns
config-override fragments that the placer applies on top of its defaults.

This is the *allowed re-framing* of K (per-bench hardcoded gating) flagged in
``docs/superpowers/specs/2026-05-19-cost-decomp-eda.md``: rules derive from
runtime measurements, never from the bench name.

Rules included:
  A. **Congestion-dominated** (cong_share > 0.6): shrink CD phase, expand
     LNS destroy. (Same as the existing inline rule in the placer; extracted
     here so it's testable and additive with the new rules.)
  B. **Wirelength-dominated** (wl_share > 0.30): spend more time on CD
     because per-macro moves directly improve WL. Cap LNS destroy growth.
  C. **Large macro count** (>= 400 macros): bump ``lns_num_destroy`` so each
     LNS iteration touches a meaningful fraction of macros.
  D. **Low macro count** (< 100 macros): shrink ``lns_num_destroy`` floor so
     each LNS iteration doesn't destroy too large a fraction.

All thresholds are bench-agnostic (derived from measurements). No bench-name
lookups. Designed so empty input metrics produce empty overrides.
"""
from __future__ import annotations

from typing import Any, Mapping


def extract_bench_metrics(
    *,
    initial_proxy_cost: float,
    initial_wirelength: float,
    initial_density: float,
    initial_congestion: float,
    num_macros: int,
) -> dict[str, float]:
    """Pure: turn raw fast_proxy outputs into a normalized metrics dict.

    Shares are computed as ``component_cost / total_proxy_cost`` so they sum
    to ~1.0 (modulo proxy weighting). Total guard against zero proxy.
    """
    total = max(float(initial_proxy_cost), 1e-12)
    wl = max(0.0, float(initial_wirelength))
    den = max(0.0, float(initial_density))
    cong = max(0.0, float(initial_congestion))
    n = max(0, int(num_macros))
    return {
        "wl_share": wl / total,
        "density_share": den / total,
        # Match the inline rule's convention: cong contribution to proxy at
        # the 0.5 weight (so the 0.6 threshold is comparable to the prior code).
        "cong_share": (0.5 * cong) / total,
        "num_macros": float(n),
    }


def adaptive_overrides_from_metrics(
    metrics: Mapping[str, float],
    base_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply Rules A–D in priority order. Returns a dict of overrides to merge.

    Empty metrics or missing keys yield ``{}`` (no overrides). The returned
    dict is always a fresh object; callers may freely mutate it.
    """
    if not metrics:
        return {}
    overrides: dict[str, Any] = {}

    cong_share = float(metrics.get("cong_share", 0.0))
    wl_share = float(metrics.get("wl_share", 0.0))
    num_macros = int(metrics.get("num_macros", 0))

    base_cd = float(base_cfg.get("cd_phase_time_budget_s", 60.0))
    base_destroy = int(base_cfg.get("lns_num_destroy", 10))
    base_fails = int(base_cfg.get("max_consecutive_lns_failures", 3))

    # ---- Rule A: congestion-dominated (preserves prior placer behavior)
    if cong_share > 0.6:
        overrides["cd_phase_time_budget_s"] = base_cd * 0.5
        overrides["max_consecutive_lns_failures"] = base_fails * 2
        overrides["lns_num_destroy"] = max(
            base_destroy, int(round(base_destroy * 1.5))
        )

    # ---- Rule B: wirelength-dominated → more CD, less LNS destroy growth
    if wl_share > 0.30:
        # Only extend CD if we did NOT shrink it via Rule A.
        if "cd_phase_time_budget_s" not in overrides:
            overrides["cd_phase_time_budget_s"] = base_cd * 1.25
        # Keep destroy smaller — too-large destroy hurts WL.
        overrides.setdefault("lns_num_destroy", base_destroy)

    # ---- Rule C: large bench → bigger destroy windows
    if num_macros >= 400:
        bumped = max(int(overrides.get("lns_num_destroy", base_destroy)), 16)
        overrides["lns_num_destroy"] = bumped

    # ---- Rule D: small bench → don't over-destroy
    if 0 < num_macros < 100:
        capped = min(int(overrides.get("lns_num_destroy", base_destroy)), 6)
        overrides["lns_num_destroy"] = capped

    return overrides
