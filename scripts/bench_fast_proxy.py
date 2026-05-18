"""Microbenchmark: fast_proxy vs compute_proxy_cost (Tier 1 lever #1).

Measures the actual speedup of macro_place.fast_proxy.fast_proxy over the
baseline macro_place.objective.compute_proxy_cost on a stream of
single-coordinate perturbations — the realistic CD/LNS per-call workload.

Usage:
    PYTHONPATH=. python3 scripts/bench_fast_proxy.py
    PYTHONPATH=. python3 scripts/bench_fast_proxy.py --benches ibm01 ibm07 --n-eval 100

Per memory project_tier1_proxy_levers.md: ratio<100x → optimize fast_proxy;
ratio>=200x → speed is fine, move to lever 2 (budget pin).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from macro_place.adapter import resolve_plc  # noqa: E402
from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402


def _bench_path(name: str) -> Path:
    return _REPO / "benchmarks" / "processed" / "public" / f"{name}.pt"


def _bench_one(
    bench_name: str,
    n_eval: int = 100,
    warmup: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a paired fast_proxy / compute_proxy_cost timing on one benchmark.

    Both code paths see the *same* stream of single-coordinate perturbations
    starting from the same randomized placement. Warmup calls are excluded
    from timing so that Numba JIT compilation and the plc cost-cache do not
    skew the measurement.

    Returns a dict with keys:
        bench, num_macros, n_eval, ms_per_fast, ms_per_baseline, ratio,
        fast_cost, baseline_cost, cost_rel_err.
    """
    bench_path = _bench_path(bench_name)
    if not bench_path.exists():
        raise FileNotFoundError(f"benchmark not found: {bench_path}")

    bench = Benchmark.load(str(bench_path))
    plc = resolve_plc(bench)
    if plc is None:
        raise RuntimeError(f"resolve_plc returned None for {bench_name}")
    ctx = build_fast_proxy_context(plc, bench)

    num_macros = int(bench.num_macros)
    canvas_w = float(bench.canvas_width)
    canvas_h = float(bench.canvas_height)

    rng = np.random.default_rng(seed)
    pos_np = np.zeros((num_macros, 2), dtype=np.float64)
    pos_np[:, 0] = rng.uniform(0.0, canvas_w, size=num_macros)
    pos_np[:, 1] = rng.uniform(0.0, canvas_h, size=num_macros)

    # Single-shot numerical sanity check: both kernels target the same
    # 1.0*wirelength + 0.5*density + 0.5*congestion formula.
    pos_t = torch.as_tensor(pos_np, dtype=torch.float32)
    baseline_result = compute_proxy_cost(pos_t, bench, plc)
    baseline_cost = float(baseline_result["proxy_cost"])
    fast_result = fast_proxy(pos_np, ctx)
    fast_cost = float(fast_result.proxy_cost)
    cost_rel_err = (
        abs(fast_cost - baseline_cost) / abs(baseline_cost)
        if baseline_cost != 0.0
        else 0.0
    )

    # Warmup: prime Numba JIT and plc internals so neither side eats a
    # first-call tax inside the timed loop.
    for _ in range(warmup):
        fast_proxy(pos_np, ctx)
        compute_proxy_cost(pos_t, bench, plc)

    # Build the perturbation stream once so both kernels see identical inputs.
    # Per iteration: bump macro (i % N) along x by +1 µm mod canvas_w.
    perturb_macros = np.arange(n_eval) % num_macros

    # --- Time fast_proxy ---
    pos_f = pos_np.copy()
    t0 = time.perf_counter()
    for i in range(n_eval):
        idx = int(perturb_macros[i])
        pos_f[idx, 0] = (pos_f[idx, 0] + 1.0) % canvas_w
        fast_proxy(pos_f, ctx)
    dt_fast = time.perf_counter() - t0

    # --- Time compute_proxy_cost (baseline) ---
    pos_b_np = pos_np.copy()
    pos_b = torch.as_tensor(pos_b_np, dtype=torch.float32)
    t0 = time.perf_counter()
    for i in range(n_eval):
        idx = int(perturb_macros[i])
        new_x = (float(pos_b[idx, 0].item()) + 1.0) % canvas_w
        pos_b[idx, 0] = new_x
        compute_proxy_cost(pos_b, bench, plc)
    dt_baseline = time.perf_counter() - t0

    ms_per_fast = dt_fast / n_eval * 1e3
    ms_per_baseline = dt_baseline / n_eval * 1e3
    ratio = ms_per_baseline / ms_per_fast if ms_per_fast > 0.0 else float("inf")

    return {
        "bench": bench_name,
        "num_macros": num_macros,
        "n_eval": int(n_eval),
        "ms_per_fast": ms_per_fast,
        "ms_per_baseline": ms_per_baseline,
        "ratio": ratio,
        "fast_cost": fast_cost,
        "baseline_cost": baseline_cost,
        "cost_rel_err": cost_rel_err,
    }


def _format_summary(results: list[dict[str, Any]]) -> str:
    lines = ["fast_proxy vs compute_proxy_cost (N per row = n_eval, ms/call):"]
    for r in results:
        lines.append(
            f"  {r['bench']:>8}  macros={r['num_macros']:>4}  "
            f"fast={r['ms_per_fast']:8.3f} ms  "
            f"baseline={r['ms_per_baseline']:9.3f} ms  "
            f"speedup={r['ratio']:7.1f}x  "
            f"calib_rel_err={r['cost_rel_err']:.3%}"
        )
    if results:
        avg_ratio = sum(r["ratio"] for r in results) / len(results)
        lines.append(f"  mean speedup across {len(results)} bench(es): {avg_ratio:.1f}x")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benches",
        nargs="+",
        default=["ibm01", "ibm07"],
        help="Benchmark names (default: ibm01 ibm07).",
    )
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output dir for summary.json (default: output/bench_fast_proxy_<ts>/).",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for name in args.benches:
        print(f"=== {name} ===", flush=True)
        r = _bench_one(name, n_eval=args.n_eval, warmup=args.warmup, seed=args.seed)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    summary_text = _format_summary(results)
    print()
    print(summary_text, flush=True)

    out_dir = args.out_dir or (_REPO / "output" / f"bench_fast_proxy_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"per_bench": results, "summary": summary_text}, indent=2)
    )
    print(f"\nWrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
