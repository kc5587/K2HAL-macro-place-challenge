"""A/B smoke for Bet 6 saddle escape effect.

Runs selected IBM benchmarks twice:
1. baseline with saddle escape disabled
2. Bet 6 with saddle escape enabled

This isolates whether saddle escape changes score on benchmarks that are more
likely to have dense or late-stage plateau behavior.
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

import torch  # noqa: E402

from macro_place.adapter import resolve_plc  # noqa: E402
from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer  # noqa: E402


DEFAULT_BENCHES = [
    "ibm07",
    "ibm10",
    "ibm12",
    "ibm13",
    "ibm14",
    "ibm16",
    "ibm18",
]


def _score_positions(positions: Any, benchmark: Benchmark, plc: Any) -> dict[str, Any]:
    pos_t = (
        positions.detach().cpu().to(torch.float32)
        if isinstance(positions, torch.Tensor)
        else torch.as_tensor(positions, dtype=torch.float32)
    )
    cost = compute_proxy_cost(pos_t, benchmark, plc)
    return {
        "proxy_cost": float(cost["proxy_cost"]),
        "wirelength_cost": float(cost["wirelength_cost"]),
        "density_cost": float(cost["density_cost"]),
        "congestion_cost": float(cost["congestion_cost"]),
        "overlap_count": int(cost["overlap_count"]),
    }


def _run_one(
    name: str,
    *,
    budget_s: float,
    num_restarts: int,
    escape_enabled: bool,
) -> dict[str, Any]:
    benchmark = Benchmark.load(f"benchmarks/processed/public/{name}.pt")
    plc = resolve_plc(benchmark)
    if plc is None:
        raise RuntimeError(f"resolve_plc returned None for {name}")

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = budget_s
    placer._config["num_restarts"] = num_restarts
    if not escape_enabled:
        placer._config["escape_max_per_restart"] = 0
        placer._config["escape_num_directions"] = 0

    start = time.perf_counter()
    positions = placer.place(benchmark)
    result = _score_positions(positions, benchmark, plc)
    result["runtime_s"] = time.perf_counter() - start
    result["telemetry"] = placer._last_run_stats
    return result


def _run_pair(
    name: str,
    *,
    budget_s: float,
    num_restarts: int,
) -> dict[str, Any]:
    print(f"=== {name}: baseline escape disabled ===", flush=True)
    baseline = _run_one(
        name,
        budget_s=budget_s,
        num_restarts=num_restarts,
        escape_enabled=False,
    )
    print(json.dumps(baseline, indent=2), flush=True)

    print(f"=== {name}: Bet 6 escape enabled ===", flush=True)
    bet6 = _run_one(
        name,
        budget_s=budget_s,
        num_restarts=num_restarts,
        escape_enabled=True,
    )
    print(json.dumps(bet6, indent=2), flush=True)

    delta = bet6["proxy_cost"] - baseline["proxy_cost"]
    return {
        "baseline_escape_disabled": baseline,
        "bet6_escape_enabled": bet6,
        "delta_proxy_cost": delta,
        "improved": delta < 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benches",
        nargs="*",
        default=DEFAULT_BENCHES,
        help="IBM benchmarks to run, e.g. ibm13 ibm14. Defaults to Bet 6 candidates.",
    )
    parser.add_argument("--budget-s", type=float, default=1800.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path("output") / f"smoke_bet6_effect_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for bench in args.benches:
        results[bench] = _run_pair(
            bench,
            budget_s=float(args.budget_s),
            num_restarts=int(args.num_restarts),
        )
        (out_dir / f"{bench}.json").write_text(json.dumps(results[bench], indent=2))

    improved = [name for name, result in results.items() if result["improved"]]
    worsened = [
        name
        for name, result in results.items()
        if result["delta_proxy_cost"] > 0.0
    ]
    overlaps = [
        name
        for name, result in results.items()
        if result["bet6_escape_enabled"]["overlap_count"] > 0
    ]
    summary = {
        "benches": list(results),
        "budget_s": float(args.budget_s),
        "num_restarts": int(args.num_restarts),
        "improved": improved,
        "worsened": worsened,
        "bet6_overlaps": overlaps,
        "per_bench": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
