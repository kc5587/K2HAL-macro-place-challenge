"""Unit test for scripts/probe_budget_cd_lns.py — CDLNSPlacer wall-clock probe
used to size num_restarts / time_budget_s for the 55-min cap (Tier 1 lever #2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from probe_budget_cd_lns import _probe_one  # noqa: E402


REQUIRED_KEYS = {
    "bench",
    "configured_time_budget_s",
    "configured_num_restarts",
    "per_restart_budget_s",
    "wall_s",
    "wall_over_budget_ratio",
    "restart_runtimes_s",
    "completed_restarts",
}


def _ibm01_present() -> bool:
    return (_REPO / "benchmarks" / "processed" / "public" / "ibm01.pt").exists()


@pytest.mark.unit
@pytest.mark.skipif(not _ibm01_present(), reason="ibm01.pt benchmark not present")
def test_probe_one_returns_complete_telemetry_for_short_run() -> None:
    """Tiny budget so the test stays under ~60s even on a cold cache."""
    result = _probe_one("ibm01", time_budget_s=10.0, num_restarts=2)

    assert set(result.keys()) >= REQUIRED_KEYS, (
        f"missing keys: {REQUIRED_KEYS - set(result.keys())}"
    )
    assert result["bench"] == "ibm01"
    assert result["configured_time_budget_s"] == pytest.approx(10.0)
    assert result["configured_num_restarts"] == 2
    assert result["per_restart_budget_s"] == pytest.approx(5.0)
    assert result["wall_s"] > 0.0
    assert isinstance(result["restart_runtimes_s"], list)
    assert result["completed_restarts"] >= 1
    # Sanity: per_restart_budget_s is a positive number derived correctly.
    assert result["wall_over_budget_ratio"] > 0.0
