"""A/B smoke for Top-K final polish.

Runs selected IBM benchmarks twice with Bet 6 saddle escape disabled:
1. CD+LNS with Top-K disabled
2. CD+LNS with Top-K enabled

The two arms keep identical restart budgets. Top-K is evaluated only as an
additive tail polish over the finalized CD+LNS candidate pool.
"""
from __future__ import annotations

import argparse
import copy
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
from submissions.macro_placer.cd_lns_placer import (  # noqa: E402
    CDLNSPlacer,
    _topk_final_polish,
    _topk_polish_budget,
)


DEFAULT_BENCHES = ["ibm07", "ibm10", "ibm13", "ibm14"]


def _score_positions(
    positions: Any,
    benchmark: Benchmark,
    plc: Any,
) -> dict[str, Any]:
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


def _result_from_candidate(
    *,
    candidate: Any,
    runtime_s: float,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "proxy_cost": float(candidate.cost["proxy_cost"]),
        "wirelength_cost": float(candidate.cost["wirelength_cost"]),
        "density_cost": float(candidate.cost["density_cost"]),
        "congestion_cost": float(candidate.cost["congestion_cost"]),
        "overlap_count": int(candidate.cost["overlap_count"]),
        "runtime_s": runtime_s,
        "telemetry": telemetry,
    }
    result["selected_summary"] = _selected_summary(telemetry)
    return result


def _selected_summary(telemetry: dict[str, Any]) -> dict[str, Any]:
    selected = telemetry.get("selected_restart", {})
    return {
        "selected_key": telemetry.get("selected_key"),
        "selected_restart_idx": selected.get("restart_idx"),
        "selected_mode": selected.get("mode"),
        "selected_candidate_kind": selected.get("candidate_kind", "original"),
        "topk_polish_attempts": telemetry.get("topk_polish_attempts", 0),
        "topk_polish_accepts": telemetry.get("topk_polish_accepts", 0),
        "topk_polish_events": telemetry.get("topk_polish_events", []),
    }


def _configure_common(
    placer: CDLNSPlacer,
    *,
    budget_s: float,
    num_restarts: int,
) -> None:
    placer._config["time_budget_s"] = budget_s
    placer._config["num_restarts"] = num_restarts
    placer._config["escape_max_per_restart"] = 0
    placer._config["escape_num_directions"] = 0


def _configure_topk(
    placer: CDLNSPlacer,
    *,
    enabled: bool,
    topk_k: int,
    topk_budget_s: float,
    topk_sweeps: int,
    topk_k_per_axis: int,
) -> None:
    placer._config["topk_polish_enabled"] = enabled
    placer._config["topk_polish_k"] = topk_k
    placer._config["topk_polish_time_budget_s"] = topk_budget_s
    placer._config["topk_polish_sweeps"] = topk_sweeps
    placer._config["topk_polish_k_per_axis"] = topk_k_per_axis


def _run_one(
    benchmark: Benchmark,
    plc: Any,
    *,
    budget_s: float,
    num_restarts: int,
    topk_enabled: bool,
    topk_k: int,
    topk_budget_s: float,
    topk_sweeps: int,
    topk_k_per_axis: int,
) -> dict[str, Any]:
    placer = CDLNSPlacer()
    _configure_common(
        placer,
        budget_s=budget_s,
        num_restarts=num_restarts,
    )
    _configure_topk(
        placer,
        enabled=topk_enabled,
        topk_k=topk_k,
        topk_budget_s=topk_budget_s,
        topk_sweeps=topk_sweeps,
        topk_k_per_axis=topk_k_per_axis,
    )

    start = time.perf_counter()
    placer.place(benchmark)
    runtime_s = time.perf_counter() - start
    selected = min(
        placer._last_final_candidates,
        key=lambda candidate: candidate.key,
    )
    result = _result_from_candidate(
        candidate=selected,
        runtime_s=runtime_s,
        telemetry=placer._last_run_stats,
    )
    result["_placer"] = placer
    return result


def _without_private(result: dict[str, Any]) -> dict[str, Any]:
    public = dict(result)
    public.pop("_placer", None)
    return public


def _same_pool_topk_result(
    *,
    topk_off: dict[str, Any],
    benchmark: Benchmark,
    plc: Any,
    topk_k: int,
    topk_budget_s: float,
    topk_sweeps: int,
    topk_k_per_axis: int,
) -> dict[str, Any]:
    placer = topk_off["_placer"]
    candidates = list(placer._last_final_candidates)
    if not candidates:
        raise RuntimeError("Top-K same-pool A/B has no finalized candidates")

    cfg = dict(placer._config)
    cfg["topk_polish_enabled"] = True
    cfg["topk_polish_k"] = topk_k
    cfg["topk_polish_time_budget_s"] = topk_budget_s
    cfg["topk_polish_sweeps"] = topk_sweeps
    cfg["topk_polish_k_per_axis"] = topk_k_per_axis

    telemetry = copy.deepcopy(placer._last_run_stats)
    telemetry["topk_polish_enabled"] = True
    telemetry["topk_polish_attempts"] = 0
    telemetry["topk_polish_accepts"] = 0
    telemetry["topk_polish_events"] = []

    ctx = placer._last_fast_proxy_context
    if ctx is None:
        raise RuntimeError("Top-K same-pool A/B has no fast-proxy context")
    polish_start = time.perf_counter()
    polished = _topk_final_polish(
        candidates=candidates,
        benchmark=benchmark,
        plc=plc,
        ctx=ctx,
        cfg=cfg,
        time_budget_s=_topk_polish_budget(cfg),
        run_stats=telemetry,
    )
    polish_runtime_s = time.perf_counter() - polish_start

    best_candidate = min(
        [*candidates, *polished],
        key=lambda candidate: candidate.key,
    )
    telemetry["selected_key"] = best_candidate.key
    telemetry["selected_restart"] = dict(best_candidate.stats)

    result = _result_from_candidate(
        candidate=best_candidate,
        runtime_s=float(topk_off["runtime_s"]) + polish_runtime_s,
        telemetry=telemetry,
    )
    result["topk_tail_runtime_s"] = polish_runtime_s
    return result


def _run_pair(
    name: str,
    *,
    budget_s: float,
    num_restarts: int,
    topk_k: int,
    topk_budget_s: float,
    topk_sweeps: int,
    topk_k_per_axis: int,
) -> dict[str, Any]:
    benchmark = Benchmark.load(f"benchmarks/processed/public/{name}.pt")
    plc = resolve_plc(benchmark)
    if plc is None:
        raise RuntimeError(f"resolve_plc returned None for {name}")

    print(f"=== {name}: Top-K disabled ===", flush=True)
    topk_off = _run_one(
        benchmark,
        plc,
        budget_s=budget_s,
        num_restarts=num_restarts,
        topk_enabled=False,
        topk_k=topk_k,
        topk_budget_s=topk_budget_s,
        topk_sweeps=topk_sweeps,
        topk_k_per_axis=topk_k_per_axis,
    )
    print(json.dumps(_without_private(topk_off), indent=2), flush=True)

    print(f"=== {name}: Top-K enabled on same candidate pool ===", flush=True)
    topk_on = _same_pool_topk_result(
        topk_off=topk_off,
        benchmark=benchmark,
        plc=plc,
        topk_k=topk_k,
        topk_budget_s=topk_budget_s,
        topk_sweeps=topk_sweeps,
        topk_k_per_axis=topk_k_per_axis,
    )
    print(json.dumps(topk_on, indent=2), flush=True)

    delta = float(topk_on["proxy_cost"]) - float(topk_off["proxy_cost"])
    return {
        "topk_disabled": _without_private(topk_off),
        "topk_enabled": topk_on,
        "delta_proxy_cost": delta,
        "topk_gain": -delta,
        "improved": delta < 0.0,
        "same_candidate_pool": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benches",
        nargs="*",
        default=DEFAULT_BENCHES,
        help="IBM benchmarks to run. Defaults to ibm07 ibm10 ibm13 ibm14.",
    )
    parser.add_argument("--budget-s", type=float, default=600.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--topk-k", type=int, default=2)
    parser.add_argument("--topk-budget-s", type=float, default=120.0)
    parser.add_argument("--topk-sweeps", type=int, default=4)
    parser.add_argument("--topk-k-per-axis", type=int, default=4)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit nonzero if any Top-K enabled result is worse than Top-K disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path("output") / f"smoke_topk_ab_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for bench in args.benches:
        results[bench] = _run_pair(
            bench,
            budget_s=float(args.budget_s),
            num_restarts=int(args.num_restarts),
            topk_k=int(args.topk_k),
            topk_budget_s=float(args.topk_budget_s),
            topk_sweeps=int(args.topk_sweeps),
            topk_k_per_axis=int(args.topk_k_per_axis),
        )
        (out_dir / f"{bench}.json").write_text(json.dumps(results[bench], indent=2))

    improved = [name for name, result in results.items() if result["improved"]]
    regressed = [
        name
        for name, result in results.items()
        if float(result["delta_proxy_cost"]) > 0.0
    ]
    overlaps = [
        name
        for name, result in results.items()
        if int(result["topk_enabled"]["overlap_count"]) > 0
    ]
    avg_off = sum(
        float(result["topk_disabled"]["proxy_cost"])
        for result in results.values()
    ) / max(1, len(results))
    avg_on = sum(
        float(result["topk_enabled"]["proxy_cost"])
        for result in results.values()
    ) / max(1, len(results))
    summary = {
        "benches": list(results),
        "budget_s": float(args.budget_s),
        "num_restarts": int(args.num_restarts),
        "topk_k": int(args.topk_k),
        "topk_budget_s": float(args.topk_budget_s),
        "topk_sweeps": int(args.topk_sweeps),
        "topk_k_per_axis": int(args.topk_k_per_axis),
        "avg_topk_disabled": avg_off,
        "avg_topk_enabled": avg_on,
        "avg_delta_proxy_cost": avg_on - avg_off,
        "avg_topk_gain": avg_off - avg_on,
        "improved": improved,
        "regressed": regressed,
        "topk_overlaps": overlaps,
        "per_bench": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out_dir / 'summary.json'}", flush=True)

    if overlaps:
        raise SystemExit(f"FAIL: Top-K overlaps on {overlaps}")
    if args.fail_on_regression and regressed:
        raise SystemExit(f"FAIL: Top-K regressions on {regressed}")


if __name__ == "__main__":
    main()
