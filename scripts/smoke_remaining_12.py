"""Run CD+LNS placer on the 12 IBM benches not yet measured (ibm07-ibm18).

Already measured at 5-bench smoke (output/smoke_cd_lns_1778242459/):
  ibm01=1.185, ibm02=1.330, ibm03=1.106, ibm04=1.180, ibm06=1.32-ish.
This script fills out the 17-bench picture for Tier 1 submission.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from macro_place.adapter import resolve_plc  # noqa: E402
from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer  # noqa: E402


BENCHES = [
    "ibm07", "ibm08", "ibm09", "ibm10",
    "ibm11", "ibm12", "ibm13", "ibm14",
    "ibm15", "ibm16", "ibm17", "ibm18",
]
PER_BENCH_BUDGET_S = 1800   # 30 min/bench, same as the 5-bench smoke


def _run_one(name: str) -> dict[str, float]:
    bench_path = Path(f"benchmarks/processed/public/{name}.pt")
    b = Benchmark.load(str(bench_path))
    plc = resolve_plc(b)
    if plc is None:
        raise RuntimeError(f"resolve_plc None for {name}")
    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = PER_BENCH_BUDGET_S
    placer._config["num_restarts"] = 4
    t0 = time.perf_counter()
    positions = placer.place(b)
    runtime = time.perf_counter() - t0
    pos_t = positions.detach().cpu().to(torch.float32) if isinstance(positions, torch.Tensor) else torch.as_tensor(positions, dtype=torch.float32)
    cost = compute_proxy_cost(pos_t, b, plc)
    return {
        "proxy_cost": float(cost["proxy_cost"]),
        "wirelength_cost": float(cost["wirelength_cost"]),
        "density_cost": float(cost["density_cost"]),
        "congestion_cost": float(cost["congestion_cost"]),
        "overlap_count": int(cost["overlap_count"]),
        "runtime_s": runtime,
    }


def main() -> None:
    out_dir = Path("output") / f"smoke_remaining_12_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for name in BENCHES:
        print(f"=== {name} ===", flush=True)
        results[name] = _run_one(name)
        (out_dir / f"{name}.json").write_text(json.dumps(results[name], indent=2))
        print(json.dumps(results[name], indent=2), flush=True)

    avg = sum(r["proxy_cost"] for r in results.values()) / len(results)
    summary = {
        "per_bench": results,
        "avg_proxy_cost_12": avg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    overlaps = [n for n, r in results.items() if r["overlap_count"] > 0]
    if overlaps:
        raise SystemExit(f"FAIL: overlaps on {overlaps}")
    print(f"DONE: 12-bench avg = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
