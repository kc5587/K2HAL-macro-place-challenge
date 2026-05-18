"""Single-bench placer worker used by smoke_isolated.py.

Loads one IBM benchmark, runs CDLNSPlacer, rescores with the official
compute_proxy_cost, writes a JSON result, and exits — so all process state
(plc, Numba caches, ProcessPoolExecutor worker pool) is reclaimed by the OS
between benches. This is the cure for the swap-thrash we observed when
smoke_cd_lns.py ran all 5 benches in a single python process.

Usage:
    PYTHONPATH=. python3 scripts/smoke_one_bench.py \\
        --bench ibm01 --out output/foo/ibm01.json \\
        --time-budget 1800 --num-restarts 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from macro_place.adapter import resolve_plc  # noqa: E402
from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer  # noqa: E402


def _bench_path(name: str) -> Path:
    return _REPO / "benchmarks" / "processed" / "public" / f"{name}.pt"


def run_bench(
    bench_name: str,
    time_budget_s: float = 1800.0,
    num_restarts: int = 4,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Place one benchmark and return a result dict (no I/O)."""
    bench_path = _bench_path(bench_name)
    if not bench_path.exists():
        raise FileNotFoundError(f"benchmark not found: {bench_path}")

    bench = Benchmark.load(str(bench_path))
    plc = resolve_plc(bench)
    if plc is None:
        raise RuntimeError(f"resolve_plc returned None for {bench_name}")

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = float(time_budget_s)
    placer._config["num_restarts"] = int(num_restarts)
    if config_overrides:
        for k, v in config_overrides.items():
            placer._config[k] = v

    t0 = time.perf_counter()
    positions = placer.place(bench)
    runtime = time.perf_counter() - t0

    pos_t = (
        positions.detach().cpu().to(torch.float32)
        if isinstance(positions, torch.Tensor)
        else torch.as_tensor(positions, dtype=torch.float32)
    )
    cost = compute_proxy_cost(pos_t, bench, plc)

    # Snapshot polish-phase diagnostics so post-run analysis can tell whether
    # hessian_escape / softswap fired or rejected, and why. We deliberately
    # pull only JSON-safe primitives — _last_run_stats also contains per-restart
    # nested dicts that may not serialize cleanly.
    run_stats: dict[str, Any] = getattr(placer, "_last_run_stats", {}) or {}
    polish_diagnostics: dict[str, Any] = {}
    for key in (
        "softswap_accepted",
        "softswap_source_key",
        "softswap_new_key",
        "hessian_escape_accepted",
        "hessian_escape_source_key",
        "hessian_escape_new_key",
        "hessian_escape_source",
        "topk_polish_attempts",
        "topk_polish_accepts",
        "crossover_accepted",
        "crossover_source_key",
        "crossover_new_key",
    ):
        if key in run_stats:
            polish_diagnostics[key] = run_stats[key]

    # Aggregate per-restart Hessian-LNS-destroy fire counts so the smoke
    # report tells us whether the new lever actually ran.
    restarts = run_stats.get("restarts", []) or []
    polish_diagnostics["hessian_lns_destroy_computes_total"] = int(
        sum(int(r.get("hessian_lns_destroy_computes", 0)) for r in restarts)
    )
    polish_diagnostics["hessian_lns_destroy_enabled_per_restart"] = [
        bool(r.get("hessian_lns_destroy_enabled", False)) for r in restarts
    ]
    polish_diagnostics["crossover_run"] = dict(run_stats.get("crossover", {}))

    # hessian_diagnostics is always populated (block-diag + lanczos eigenvalues,
    # source, reason) — even on reject — so we can tell why E12 didn't help.
    hd = run_stats.get("hessian_diagnostics") or {}
    for key in (
        "hessian_block_diag_eigenvalue",
        "hessian_lanczos_eigenvalue",
        "hessian_source",
        "hessian_reason",
        "hessian_f0",
        "hessian_f_best",
        "hessian_best_alpha",
    ):
        if key in hd:
            polish_diagnostics[key] = hd[key]

    selected_restart = run_stats.get("selected_restart") or {}
    polish_diagnostics["selected_mode"] = str(selected_restart.get("mode", "unknown"))
    polish_diagnostics["selected_restart_idx"] = int(
        selected_restart.get("restart_idx", -1)
    )
    polish_diagnostics["selected_key"] = list(run_stats.get("selected_key", ()))
    polish_diagnostics["per_restart_modes"] = [
        str(r.get("mode", "unknown")) for r in restarts
    ]

    return {
        "bench": bench_name,
        "proxy_cost": float(cost["proxy_cost"]),
        "wirelength_cost": float(cost["wirelength_cost"]),
        "density_cost": float(cost["density_cost"]),
        "congestion_cost": float(cost["congestion_cost"]),
        "overlap_count": int(cost["overlap_count"]),
        "runtime_s": runtime,
        "time_budget_s": float(time_budget_s),
        "num_restarts": int(num_restarts),
        "polish_diagnostics": polish_diagnostics,
    }


def _parse_override(s: str) -> tuple[str, Any]:
    """Parse ``key=value`` overrides for placer config. Value is parsed as
    JSON when possible (so ``flag=true`` → bool, ``k=5`` → int, ``s=foo``
    → str). This lets us A/B levers from the CLI without editing defaults.
    """
    if "=" not in s:
        raise ValueError(f"--config-override must be key=value, got {s!r}")
    key, raw = s.split("=", 1)
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--time-budget", type=float, default=1800.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Repeatable key=value override (JSON-parsed). Example: "
        "--config-override hessian_lns_destroy_enabled=false",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    for s in args.config_override:
        k, v = _parse_override(s)
        overrides[k] = v

    result = run_bench(
        bench_name=args.bench,
        time_budget_s=args.time_budget,
        num_restarts=args.num_restarts,
        config_overrides=overrides or None,
    )
    if overrides:
        result["config_overrides"] = overrides
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
