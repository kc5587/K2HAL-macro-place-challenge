"""Multi-bench Hessian diagnostic — no escape attempted.

For each benchmark, runs the placer with hessian_escape disabled, then
iterates over EVERY restart candidate (not just proxy_best) and computes:
  - block-diagonal smallest eigenvalue per candidate
  - Lanczos smallest eigenvalue per candidate

Goal: answer "are CD+LNS-converged placements at saddles or true minima?"
across our 5-bench smoke set, by restart mode. If all eigenvalues are
non-negative, saddle-escape is the wrong lever for this problem family.

Output: ``output/hessian_diagnostic_<ts>/summary.json`` with per-bench,
per-candidate eigenvalues. Total wall ~ 5 benches × 12 min ≈ 60 min.

Usage:
    PYTHONPATH=. python3 scripts/hessian_diagnostic.py
    PYTHONPATH=. python3 scripts/hessian_diagnostic.py --benches ibm06
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
from macro_place.hessian_escape import (  # noqa: E402
    block_diag_min_eigenvalue,
    lanczos_min_eigenvalue,
)
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer  # noqa: E402


DEFAULT_BENCHES = ("ibm01", "ibm02", "ibm03", "ibm04", "ibm06")


def _bench_path(name: str) -> Path:
    return _REPO / "benchmarks" / "processed" / "public" / f"{name}.pt"


def _eigenvalues_for_candidate(
    positions_tensor: torch.Tensor,
    benchmark: Benchmark,
    ctx: Any,
    h_block: float,
    h_lanczos: float,
    lanczos_iters: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Compute block-diag + Lanczos smallest eigenvalues for one candidate's
    hard-macro placement, with soft macros frozen.
    """
    positions = positions_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    num_hard = int(benchmark.num_hard_macros)
    if num_hard < 1:
        return {"num_hard_macros": 0, "skipped": "no_hard_macros"}

    soft_positions = positions[num_hard:].copy()

    def _eval_fn(hard_pos: np.ndarray) -> float:
        full = np.empty_like(positions)
        full[:num_hard] = hard_pos
        if full.shape[0] > num_hard:
            full[num_hard:] = soft_positions
        return float(fast_proxy(full, ctx).proxy_cost)

    hard_in = positions[:num_hard]
    f0 = _eval_fn(hard_in)

    bd_eigval, bd_macro, _bd_eigvec = block_diag_min_eigenvalue(
        positions=hard_in, eval_fn=_eval_fn, h=h_block
    )
    lz_eigval, _lz_eigvec = lanczos_min_eigenvalue(
        positions=hard_in,
        eval_fn=_eval_fn,
        h=h_lanczos,
        max_iters=lanczos_iters,
        rng_seed=rng_seed,
    )
    return {
        "num_hard_macros": int(num_hard),
        "f0": f0,
        "block_diag_eigenvalue": float(bd_eigval),
        "block_diag_macro_idx": int(bd_macro),
        "lanczos_eigenvalue": float(lz_eigval),
        "lanczos_iters_K": int(lanczos_iters),
    }


def _diagnose_one(
    bench_name: str,
    time_budget_s: float,
    num_restarts: int,
    h_block: float,
    h_lanczos: float,
    lanczos_iters: int,
    rng_seed: int,
) -> dict[str, Any]:
    bench_path = _bench_path(bench_name)
    if not bench_path.exists():
        return {"bench": bench_name, "error": f"missing {bench_path}"}

    bench = Benchmark.load(str(bench_path))
    plc = resolve_plc(bench)
    if plc is None:
        return {"bench": bench_name, "error": "resolve_plc returned None"}
    ctx = build_fast_proxy_context(plc, bench)

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = float(time_budget_s)
    placer._config["num_restarts"] = int(num_restarts)
    placer._config["hessian_escape_enabled"] = False  # diagnostic — no escape

    t0 = time.perf_counter()
    _ = placer.place(bench)
    wall_s = time.perf_counter() - t0

    candidates = getattr(placer, "_last_final_candidates", []) or []
    run_stats = getattr(placer, "_last_run_stats", {}) or {}

    per_candidate: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        eig = _eigenvalues_for_candidate(
            positions_tensor=cand.legalized_positions,
            benchmark=bench,
            ctx=ctx,
            h_block=h_block,
            h_lanczos=h_lanczos,
            lanczos_iters=lanczos_iters,
            rng_seed=rng_seed,
        )
        per_candidate.append({
            "candidate_idx": idx,
            "candidate_kind": str(cand.stats.get("candidate_kind", "")),
            "restart_mode": str(cand.stats.get("mode", "")),
            "candidate_key_overlap": int(cand.key[0]),
            "candidate_key_proxy": float(cand.key[1]),
            **eig,
        })

    bd_eigs = [c["block_diag_eigenvalue"] for c in per_candidate if "block_diag_eigenvalue" in c]
    lz_eigs = [c["lanczos_eigenvalue"] for c in per_candidate if "lanczos_eigenvalue" in c]

    return {
        "bench": bench_name,
        "num_hard_macros": int(bench.num_hard_macros),
        "num_total_macros": int(bench.num_macros),
        "canvas_w": float(bench.canvas_width),
        "canvas_h": float(bench.canvas_height),
        "wall_s": wall_s,
        "num_candidates": len(candidates),
        "per_candidate": per_candidate,
        "summary": {
            "min_block_diag_eigenvalue": min(bd_eigs) if bd_eigs else None,
            "max_block_diag_eigenvalue": max(bd_eigs) if bd_eigs else None,
            "min_lanczos_eigenvalue": min(lz_eigs) if lz_eigs else None,
            "max_lanczos_eigenvalue": max(lz_eigs) if lz_eigs else None,
            "any_negative_block_diag": any(e < -1e-3 for e in bd_eigs),
            "any_negative_lanczos": any(e < -1e-3 for e in lz_eigs),
        },
        "selected_key": list(run_stats.get("selected_key", (0, 0.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benches", nargs="+", default=list(DEFAULT_BENCHES))
    parser.add_argument("--time-budget", type=float, default=1800.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--h-block", type=float, default=0.5)
    parser.add_argument("--h-lanczos", type=float, default=0.1)
    parser.add_argument("--lanczos-iters", type=int, default=16)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (_REPO / "output" / f"hessian_diagnostic_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Hessian diagnostic → {out_dir}", flush=True)

    results: list[dict[str, Any]] = []
    for name in args.benches:
        print(f"=== diagnosing {name} ===", flush=True)
        r = _diagnose_one(
            bench_name=name,
            time_budget_s=args.time_budget,
            num_restarts=args.num_restarts,
            h_block=args.h_block,
            h_lanczos=args.h_lanczos,
            lanczos_iters=args.lanczos_iters,
            rng_seed=args.rng_seed,
        )
        results.append(r)
        (out_dir / f"{name}.json").write_text(json.dumps(r, indent=2))
        if "error" not in r:
            s = r["summary"]
            print(
                f"  candidates={r['num_candidates']}  "
                f"bd_min={s['min_block_diag_eigenvalue']:.4f}  "
                f"lz_min={s['min_lanczos_eigenvalue']:.4f}  "
                f"any_neg_bd={s['any_negative_block_diag']}  "
                f"any_neg_lz={s['any_negative_lanczos']}",
                flush=True,
            )

    summary = {
        "config": {
            "benches": list(args.benches),
            "time_budget_s": args.time_budget,
            "num_restarts": args.num_restarts,
            "h_block": args.h_block,
            "h_lanczos": args.h_lanczos,
            "lanczos_iters": args.lanczos_iters,
        },
        "per_bench": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
