"""Phase D: Tier-1 lock-in benchmark.

Runs the kept survivor config on all 17 ibm benches at full budget
(3000s × 4 restarts). Output goes to results/submission_lock_2026-05-21/.

Only runs if a survivors.json exists and Phase C confirmed non-regression.

Estimated runtime: ~9 hours sequential (small ibms 10-25 min, large 30-50 min).

Usage:
    PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        scripts/phase_d_tier1_lockin.py --survivors path/to/survivors.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place.adapter import resolve_plc
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer


IBM_BENCHES = (
    "ibm01", "ibm02", "ibm03", "ibm04",
    "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13",
    "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
)


def _as_float_dict(cost: dict[str, Any]) -> dict[str, float | int]:
    return {
        "proxy_cost": float(cost["proxy_cost"]),
        "wirelength_cost": float(cost.get("wirelength_cost", 0.0)),
        "density_cost": float(cost.get("density_cost", 0.0)),
        "congestion_cost": float(cost.get("congestion_cost", 0.0)),
        "overlap_count": int(cost["overlap_count"]),
    }


def _safe_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in stats.items():
        try:
            json.dumps(v, default=str)
            out[k] = v
        except Exception:
            out[k] = str(v)[:300]
    return out


def run_one(bench_name: str, survivors: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    bench_path = f"benchmarks/processed/public/{bench_name}.pt"
    bench = Benchmark.load(bench_path)
    plc = resolve_plc(bench)
    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 3000.0
    placer._config["num_restarts"] = 4
    placer._config["targeted_sa_escape_enabled"] = True
    placer._config["targeted_sa_escape_polish_time_budget_s"] = 60.0
    for k, v in survivors.items():
        placer._config[k] = v
    start = time.perf_counter()
    positions = placer.place(bench)
    wall = time.perf_counter() - start
    pos_t = (
        positions.detach().cpu().to(torch.float32)
        if isinstance(positions, torch.Tensor)
        else torch.as_tensor(positions, dtype=torch.float32)
    )
    cost = compute_proxy_cost(pos_t, plc=plc, benchmark=bench)
    record = {
        "bench": bench_name,
        "config_overrides": survivors,
        "wall_s": float(wall),
        "wall_min": float(wall) / 60.0,
        "official": _as_float_dict(dict(cost)),
        "stats": _safe_stats(dict(placer._last_run_stats)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bench_name}.json"
    out_file.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return record


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--survivors",
        type=Path,
        default=REPO_ROOT / "results" / "targeted_sa_stack_2026-05-20" / "survivors.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "submission_lock_2026-05-21",
    )
    p.add_argument("--benches", default=",".join(IBM_BENCHES))
    args = p.parse_args()
    if not args.survivors.exists():
        print(f"No survivors file at {args.survivors}", flush=True)
        return 1
    survivors = json.loads(args.survivors.read_text())
    if not survivors:
        print(f"Empty survivors set", flush=True)
        return 1
    print(f"Phase D survivor config: {survivors}", flush=True)
    bench_names = [b.strip() for b in args.benches.split(",") if b.strip()]
    summary = []
    for bench_name in bench_names:
        print(f"[{time.strftime('%H:%M:%S')}] {bench_name} START", flush=True)
        try:
            rec = run_one(bench_name, survivors, args.out_dir)
            proxy = rec["official"]["proxy_cost"]
            ov = rec["official"]["overlap_count"]
            print(
                f"[{time.strftime('%H:%M:%S')}] {bench_name} DONE "
                f"wall={rec['wall_min']:.1f}min proxy={proxy:.6f} ov={ov}",
                flush=True,
            )
            summary.append(rec)
        except Exception as exc:
            print(
                f"[{time.strftime('%H:%M:%S')}] {bench_name} FAILED: "
                f"{type(exc).__name__}: {str(exc)[:300]}",
                flush=True,
            )
            summary.append({"bench": bench_name, "error": str(exc)[:300]})
    summary_path = args.out_dir / "phase_d_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Phase D summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
