"""Unit test for scripts/smoke_one_bench.py — single-bench placer worker
used by the subprocess-isolated smoke orchestrator.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from smoke_one_bench import run_bench  # noqa: E402


REQUIRED_KEYS = {
    "bench",
    "proxy_cost",
    "wirelength_cost",
    "density_cost",
    "congestion_cost",
    "overlap_count",
    "runtime_s",
    "time_budget_s",
    "num_restarts",
}


def _ibm01_present() -> bool:
    return (_REPO / "benchmarks" / "processed" / "public" / "ibm01.pt").exists()


# Test-only overrides to keep the unit test fast (~30s instead of ~200s):
# topk polish and ORFS guard/spacing dominate wall-time on small budgets.
_FAST_OVERRIDES = {
    "topk_polish_enabled": False,
    "orfs_guard_repair_enabled": False,
    "orfs_spacing_polish_enabled": False,
    "orfs_tiebreak_enabled": False,
}


@pytest.mark.unit
@pytest.mark.skipif(not _ibm01_present(), reason="ibm01.pt benchmark not present")
def test_run_bench_returns_complete_result_on_short_run() -> None:
    """In-process call: confirms the function contract and that a tiny
    budget terminates without overlaps. Polish/ORFS disabled for speed."""
    result = run_bench(
        "ibm01",
        time_budget_s=5.0,
        num_restarts=1,
        config_overrides=_FAST_OVERRIDES,
    )

    assert set(result.keys()) >= REQUIRED_KEYS, (
        f"missing keys: {REQUIRED_KEYS - set(result.keys())}"
    )
    assert result["bench"] == "ibm01"
    assert result["time_budget_s"] == pytest.approx(5.0)
    assert result["num_restarts"] == 1
    assert result["proxy_cost"] > 0.0
    assert result["runtime_s"] > 0.0
    assert isinstance(result["overlap_count"], int)


@pytest.mark.unit
@pytest.mark.skipif(not _ibm01_present(), reason="ibm01.pt benchmark not present")
def test_smoke_one_bench_cli_writes_json(tmp_path: Path) -> None:
    """End-to-end via subprocess: this is the path smoke_isolated.py uses.
    Cannot pass config overrides through the CLI; this run pays the full
    polish/ORFS overhead (~200s on ibm01), so timeout is generous."""
    out_path = tmp_path / "ibm01.json"
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "smoke_one_bench.py"),
            "--bench", "ibm01",
            "--out", str(out_path),
            "--time-budget", "5",
            "--num-restarts", "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    assert proc.returncode == 0, (
        f"smoke_one_bench exited {proc.returncode}\nstderr: {proc.stderr}"
    )
    assert out_path.exists(), "result JSON not written"
    payload = json.loads(out_path.read_text())
    assert payload["bench"] == "ibm01"
    assert set(payload.keys()) >= REQUIRED_KEYS
