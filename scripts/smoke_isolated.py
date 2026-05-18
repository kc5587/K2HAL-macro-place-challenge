"""Subprocess-isolated smoke runner (5-bench CD+LNS gate, swap-safe).

Same acceptance contract as scripts/smoke_cd_lns.py — 5 benches
(ibm01–04, ibm06), avg proxy ≤ 1.30, no overlaps — but runs each benchmark
in a fresh python subprocess so plc/Numba/ProcessPoolExecutor state cannot
leak between benches. Solves the swap-thrash we observed when sustained
runs accumulated RAM until the system paged.

Usage:
    PYTHONPATH=. python3 scripts/smoke_isolated.py
    PYTHONPATH=. python3 scripts/smoke_isolated.py --time-budget 3300 --num-restarts 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

BENCHES = ("ibm01", "ibm02", "ibm03", "ibm04", "ibm06")
ACCEPTANCE_AVG = 1.30
WORKER = _REPO / "scripts" / "smoke_one_bench.py"


def _run_bench_subprocess(
    bench: str,
    out_path: Path,
    time_budget_s: float,
    num_restarts: int,
    log_path: Path,
    config_overrides: list[str] | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(WORKER),
        "--bench", bench,
        "--out", str(out_path),
        "--time-budget", str(time_budget_s),
        "--num-restarts", str(num_restarts),
    ]
    for s in config_overrides or []:
        cmd += ["--config-override", s]
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"{bench} worker exited {proc.returncode}; see {log_path}"
        )
    if not out_path.exists():
        raise RuntimeError(f"{bench} worker did not write {out_path}")
    result = json.loads(out_path.read_text())
    result["wall_s"] = wall_s
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget", type=float, default=1800.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument(
        "--benches",
        nargs="+",
        default=list(BENCHES),
        help="Benchmark subset (default: ibm01 ibm02 ibm03 ibm04 ibm06).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: output/smoke_isolated_<ts>/).",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Repeatable key=value override forwarded to each worker. "
        "Value is JSON-parsed. Example: "
        "--config-override hessian_lns_destroy_enabled=false",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (_REPO / "output" / f"smoke_isolated_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Isolated smoke → {out_dir}", flush=True)
    print(
        f"  config: time_budget_s={args.time_budget}  num_restarts={args.num_restarts}",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    total_wall_s = 0.0
    for name in args.benches:
        print(f"=== {name} (subprocess) ===", flush=True)
        out_path = out_dir / f"{name}.json"
        log_path = out_dir / f"{name}.log"
        result = _run_bench_subprocess(
            name,
            out_path,
            args.time_budget,
            args.num_restarts,
            log_path,
            config_overrides=list(args.config_override),
        )
        total_wall_s += float(result["wall_s"])
        results[name] = result
        print(
            f"  proxy={result['proxy_cost']:.4f}  overlap={result['overlap_count']}  "
            f"runtime_s={result['runtime_s']:.1f}  wall_s={result['wall_s']:.1f}",
            flush=True,
        )

    avg = sum(r["proxy_cost"] for r in results.values()) / max(len(results), 1)
    overlaps = [n for n, r in results.items() if r["overlap_count"] > 0]
    summary = {
        "config": {
            "time_budget_s": args.time_budget,
            "num_restarts": args.num_restarts,
            "benches": list(args.benches),
        },
        "per_bench": results,
        "avg_proxy_cost": avg,
        "total_wall_s": total_wall_s,
        "acceptance_avg": ACCEPTANCE_AVG,
        "overlaps": overlaps,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"avg_proxy = {avg:.4f}  (acceptance {ACCEPTANCE_AVG})", flush=True)
    print(f"total_wall_s = {total_wall_s:.1f}s ({total_wall_s/60:.1f} min)", flush=True)
    if overlaps:
        print(f"FAIL: overlaps on {overlaps}", flush=True)
        sys.exit(2)
    if avg >= ACCEPTANCE_AVG:
        print(f"FAIL: avg {avg:.4f} >= acceptance {ACCEPTANCE_AVG:.4f}", flush=True)
        sys.exit(3)
    print("PASS: isolated smoke gate met", flush=True)


if __name__ == "__main__":
    main()
