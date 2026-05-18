# scripts/smoke_cd_lns.py
"""5-bench smoke gate for the CD+LNS placer (Bet 7 restart).

Acceptance: avg_proxy_cost <= 1.30 (RePlAce avg-of-5 = 1.4156, top-7
cut today = 1.2788).
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


BENCHES = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06"]
ACCEPTANCE_AVG = 1.30
PER_BENCH_BUDGET_S = 1800


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
    out_dir = Path("output") / f"smoke_cd_lns_{int(time.time())}"
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
        "avg_proxy_cost": avg,
        "acceptance_avg": ACCEPTANCE_AVG,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    overlaps = [n for n, r in results.items() if r["overlap_count"] > 0]
    if overlaps:
        raise SystemExit(f"FAIL: overlaps on {overlaps}")
    if avg >= ACCEPTANCE_AVG:
        raise SystemExit(f"FAIL: avg {avg:.4f} >= acceptance {ACCEPTANCE_AVG:.4f}")
    print("PASS: T7 smoke gate met", flush=True)


if __name__ == "__main__":
    main()
