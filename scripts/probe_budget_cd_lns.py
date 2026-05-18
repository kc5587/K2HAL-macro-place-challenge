"""Wall-clock probe for CDLNSPlacer (Tier 1 lever #2 — pin search budget).

Runs the placer on a single IBM benchmark with caller-supplied
(time_budget_s, num_restarts) and reads ``placer._last_run_stats`` so we can
see how the configured budget maps to actual wall-clock. Critical for sizing
the 55-min/benchmark cap: restarts run in parallel via ProcessPoolExecutor
(see cd_lns_placer.py:_run_one_restart wiring), so wall-clock is bounded by
the slowest parallel restart plus serial overhead (legalization, topk polish,
ORFS guard/spacing), not by the sum of restart runtimes.

Usage:
    PYTHONPATH=. python3 scripts/probe_budget_cd_lns.py \\
        --bench ibm01 --time-budget 120 --num-restarts 4

For the 55-min pin probe:
    PYTHONPATH=. python3 scripts/probe_budget_cd_lns.py \\
        --bench ibm01 --time-budget 3300 --num-restarts 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from macro_place.benchmark import Benchmark  # noqa: E402
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer  # noqa: E402


def _bench_path(name: str) -> Path:
    return _REPO / "benchmarks" / "processed" / "public" / f"{name}.pt"


def _probe_one(
    bench_name: str,
    time_budget_s: float,
    num_restarts: int,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run CDLNSPlacer on one benchmark and return a wall-clock telemetry dict.

    Returns:
        {
            bench, configured_time_budget_s, configured_num_restarts,
            per_restart_budget_s, wall_s, wall_over_budget_ratio,
            restart_runtimes_s: list[float],   # per-restart runtime_s from telemetry
            restart_modes: list[str],
            completed_restarts: int,
            slowest_restart_s, fastest_restart_s, mean_restart_s,
            wall_minus_max_restart_s,   # serial overhead (legalize + topk polish + ORFS)
            run_stats: <full _last_run_stats dict, lightly stringified>,
        }
    """
    bench_path = _bench_path(bench_name)
    if not bench_path.exists():
        raise FileNotFoundError(f"benchmark not found: {bench_path}")

    bench = Benchmark.load(str(bench_path))

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = float(time_budget_s)
    placer._config["num_restarts"] = int(num_restarts)
    if config_overrides:
        for k, v in config_overrides.items():
            placer._config[k] = v

    per_restart_budget_s = float(time_budget_s) / max(int(num_restarts), 1)

    t0 = time.perf_counter()
    placer.place(bench)
    wall_s = time.perf_counter() - t0

    run_stats = getattr(placer, "_last_run_stats", {}) or {}
    restarts = run_stats.get("restarts", [])
    restart_runtimes_s = [float(r.get("runtime_s", 0.0)) for r in restarts]
    restart_modes = [str(r.get("mode", "?")) for r in restarts]

    slowest = max(restart_runtimes_s) if restart_runtimes_s else 0.0
    fastest = min(restart_runtimes_s) if restart_runtimes_s else 0.0
    mean = (
        sum(restart_runtimes_s) / len(restart_runtimes_s)
        if restart_runtimes_s
        else 0.0
    )

    return {
        "bench": bench_name,
        "configured_time_budget_s": float(time_budget_s),
        "configured_num_restarts": int(num_restarts),
        "per_restart_budget_s": per_restart_budget_s,
        "wall_s": wall_s,
        "wall_over_budget_ratio": wall_s / float(time_budget_s) if time_budget_s > 0 else 0.0,
        "restart_runtimes_s": restart_runtimes_s,
        "restart_modes": restart_modes,
        "completed_restarts": len(restart_runtimes_s),
        "slowest_restart_s": slowest,
        "fastest_restart_s": fastest,
        "mean_restart_s": mean,
        "wall_minus_max_restart_s": wall_s - slowest,
        "topk_polish_attempts": int(run_stats.get("topk_polish_attempts", 0)),
        "topk_polish_accepts": int(run_stats.get("topk_polish_accepts", 0)),
    }


def _format_summary(r: dict[str, Any]) -> str:
    lines = [
        f"=== probe: {r['bench']} ===",
        f"  configured: time_budget_s={r['configured_time_budget_s']:.1f}  "
        f"num_restarts={r['configured_num_restarts']}  "
        f"per_restart={r['per_restart_budget_s']:.1f}s",
        f"  wall_s={r['wall_s']:.1f}s  "
        f"(wall/budget={r['wall_over_budget_ratio']:.2f})",
        f"  restarts completed={r['completed_restarts']}  "
        f"slowest={r['slowest_restart_s']:.1f}s  "
        f"fastest={r['fastest_restart_s']:.1f}s  "
        f"mean={r['mean_restart_s']:.1f}s",
        f"  serial_overhead (wall - max_restart) = {r['wall_minus_max_restart_s']:.1f}s",
        f"  topk_polish: attempts={r['topk_polish_attempts']}  "
        f"accepts={r['topk_polish_accepts']}",
        f"  per-restart runtimes: {[round(x, 1) for x in r['restart_runtimes_s']]}",
        f"  per-restart modes: {r['restart_modes']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="ibm01")
    parser.add_argument("--time-budget", type=float, default=120.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output dir for telemetry.json (default: output/probe_budget_<ts>/).",
    )
    args = parser.parse_args()

    print(
        f"Probing {args.bench}: time_budget_s={args.time_budget}, "
        f"num_restarts={args.num_restarts}",
        flush=True,
    )
    r = _probe_one(args.bench, args.time_budget, args.num_restarts)
    print(_format_summary(r), flush=True)

    out_dir = args.out_dir or (_REPO / "output" / f"probe_budget_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "telemetry.json").write_text(json.dumps(r, indent=2))
    print(f"\nWrote {out_dir / 'telemetry.json'}", flush=True)


if __name__ == "__main__":
    main()
