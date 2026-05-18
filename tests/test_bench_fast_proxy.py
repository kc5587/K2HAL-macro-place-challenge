"""Unit test for scripts/bench_fast_proxy.py — fast_proxy vs compute_proxy_cost
microbenchmark harness (Tier 1 lever #1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from bench_fast_proxy import _bench_one  # noqa: E402


REQUIRED_KEYS = {
    "bench",
    "num_macros",
    "n_eval",
    "ms_per_fast",
    "ms_per_baseline",
    "ratio",
    "fast_cost",
    "baseline_cost",
    "cost_rel_err",
}


def _ibm01_present() -> bool:
    return (_REPO / "benchmarks" / "processed" / "public" / "ibm01.pt").exists()


@pytest.mark.unit
@pytest.mark.skipif(not _ibm01_present(), reason="ibm01.pt benchmark not present")
def test_bench_one_returns_complete_result_for_ibm01() -> None:
    result = _bench_one("ibm01", n_eval=5, warmup=2, seed=0)
    assert set(result.keys()) >= REQUIRED_KEYS, f"missing keys: {REQUIRED_KEYS - set(result.keys())}"
    assert result["bench"] == "ibm01"
    assert result["n_eval"] == 5
    assert result["num_macros"] > 0


@pytest.mark.unit
@pytest.mark.skipif(not _ibm01_present(), reason="ibm01.pt benchmark not present")
def test_bench_one_fast_path_is_faster_than_baseline() -> None:
    result = _bench_one("ibm01", n_eval=5, warmup=2, seed=0)
    assert result["ms_per_fast"] > 0.0
    assert result["ms_per_baseline"] > 0.0
    assert result["ratio"] > 1.0, (
        f"fast_proxy ({result['ms_per_fast']:.3f} ms) not faster than baseline "
        f"({result['ms_per_baseline']:.3f} ms); ratio={result['ratio']:.2f}x"
    )


@pytest.mark.unit
@pytest.mark.skipif(not _ibm01_present(), reason="ibm01.pt benchmark not present")
def test_bench_one_fast_proxy_calibrated_to_baseline() -> None:
    """Sanity: fast_proxy targets the same wirelength+0.5*density+0.5*congestion
    formula. They should agree within 10% on the seeded placement.
    """
    result = _bench_one("ibm01", n_eval=5, warmup=2, seed=0)
    assert result["fast_cost"] > 0.0
    assert result["baseline_cost"] > 0.0
    assert result["cost_rel_err"] < 0.10, (
        f"fast vs baseline proxy_cost diverged: fast={result['fast_cost']:.4f} "
        f"baseline={result['baseline_cost']:.4f} rel_err={result['cost_rel_err']:.4f}"
    )
