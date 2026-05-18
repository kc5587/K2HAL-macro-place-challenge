"""Autonomous DREAMPlace tuning loop.

Designed to run unattended while the user sleeps. Chains 1-axis sweeps
on ibm07, stacking the best value from each round onto the next.

Rounds (in order):
    1. density_weight  (already running externally as output/sweep_dw)
    2. target_density  (stacked on best dw)
    3. gamma           (stacked on best dw + td)
    4. iteration       (stacked on best dw + td + gamma)
    5. num_bins_xy     (stacked on best of all above)

Stop conditions (whichever comes first):
    - Best proxy ≤ ibm07 baseline 1.1322 for 2 consecutive rounds
      (no need to keep tuning if we've already won)
    - Total wall budget elapsed (default 6 hours)
    - Last round's improvement < 0.5% (converged)

Each round writes its results to ``output/sweep_<axis>/`` and the
master ledger to ``output/autonomous_tune/ledger.json``.
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
_PYTHON = sys.executable
_BASELINE_PROXY_IBM07 = 1.1322
_IMPROVEMENT_THRESHOLD = 0.005   # 0.5%
_BENCH = "ibm07"


# Axis sweep specs. Each entry is (axis_name, comma-separated values).
# Order matters; later rounds inherit best from earlier rounds.
_SWEEPS: list[tuple[str, str]] = [
    ("density_weight", "1e-6,1e-5,5e-5,1e-4,5e-4"),         # round 1 (already running externally)
    ("target_density", "0.4,0.6,0.8,1.0"),                   # round 2
    ("gamma", "2.0,4.0,8.0,16.0"),                           # round 3
]


def _wait_for_summary(summary_path: Path, timeout_s: float) -> bool:
    """Poll until summary.json appears or timeout."""
    start = time.time()
    while not summary_path.exists():
        if time.time() - start > timeout_s:
            return False
        time.sleep(15.0)
    # Also wait a few seconds for write to settle.
    time.sleep(2.0)
    return True


def _best_from_summary(summary_path: Path, axis: str) -> tuple[float, dict[str, Any]] | None:
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return None
    if not data.get("results"):
        return None
    best = data["results"][0]  # already ranked ascending by proxy
    if not best.get("variant"):
        return None
    if axis not in best["variant"]:
        return None
    return float(best["proxy"]), best


def _launch_sweep(
    axis: str,
    values: str,
    locked_overrides: dict[str, float],
    out_dir: Path,
    time_budget_s: float = 1800.0,
    num_restarts: int = 1,
) -> int:
    """Launch a sweep with locked overrides + one axis varied."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # The sweep script takes a single --variants string; we encode the
    # locked overrides via env passing them through to the underlying
    # smoke_one_bench --config-override JSON. Easiest path: spawn the
    # sweep, and rely on dreamplace_config_overrides being inherited.
    # But sweep_dreamplace.py builds the JSON dict from the variants
    # only. We extend its behavior here by passing locked vals via env.
    sweep_args = [
        _PYTHON,
        str(_REPO / "scripts" / "sweep_dreamplace.py"),
        "--bench", _BENCH,
        "--out-dir", str(out_dir),
        "--time-budget", str(time_budget_s),
        "--num-restarts", str(num_restarts),
        "--variants", f"{axis}:{values}",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(_REPO),
        "DREAMPLACE_LOCKED_OVERRIDES": json.dumps(locked_overrides),
    }
    log_path = out_dir / "_orchestrator.log"
    with log_path.open("w") as logf:
        proc = subprocess.run(sweep_args, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def _write_ledger(ledger_path: Path, ledger: list[dict[str, Any]]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-for-existing",
        type=Path,
        default=_REPO / "output" / "sweep_dw" / "summary.json",
        help="Wait for this summary.json (round-1 sweep launched externally) before proceeding.",
    )
    parser.add_argument(
        "--max-wall-h",
        type=float,
        default=6.0,
        help="Stop after this many hours regardless of progress.",
    )
    parser.add_argument(
        "--bench-time-budget-s",
        type=float,
        default=1800.0,
        help="Time-budget per smoke (default 1800s = 30min cap; usually finishes faster).",
    )
    args = parser.parse_args()

    ledger_path = _REPO / "output" / "autonomous_tune" / "ledger.json"
    ledger: list[dict[str, Any]] = []
    start_wall = time.time()
    locked: dict[str, float] = {}
    last_best: float = float("inf")

    # ---- Round 1: wait for the externally-launched density_weight sweep ----
    axis_r1, values_r1 = _SWEEPS[0]
    summary_r1 = args.wait_for_existing
    print(f"[orchestrator] waiting for round-1 sweep at {summary_r1}", flush=True)
    if not _wait_for_summary(summary_r1, timeout_s=args.max_wall_h * 3600):
        print(f"[orchestrator] timed out waiting for {summary_r1}", flush=True)
        return
    res = _best_from_summary(summary_r1, axis_r1)
    if res is None:
        print(f"[orchestrator] could not parse {summary_r1}", flush=True)
        return
    best_proxy, best_row = res
    best_val = best_row["variant"][axis_r1]
    locked[axis_r1] = best_val
    ledger.append({
        "round": 1,
        "axis": axis_r1,
        "best_value": best_val,
        "best_proxy": best_proxy,
        "summary_path": str(summary_r1),
        "elapsed_h": (time.time() - start_wall) / 3600.0,
        "locked": dict(locked),
    })
    _write_ledger(ledger_path, ledger)
    print(
        f"[orchestrator] round 1 done: best {axis_r1}={best_val:g} proxy={best_proxy:.4f}",
        flush=True,
    )
    last_best = best_proxy

    # Early exit if we already beat baseline by a comfortable margin.
    if best_proxy <= _BASELINE_PROXY_IBM07 - _IMPROVEMENT_THRESHOLD:
        print(
            f"[orchestrator] round 1 already beat baseline ({_BASELINE_PROXY_IBM07:.4f}); "
            "stopping early so user can validate on more benches",
            flush=True,
        )
        ledger.append({"round": "stop", "reason": "beat_baseline_r1"})
        _write_ledger(ledger_path, ledger)
        return

    # ---- Subsequent rounds: stack on previous best ----
    for r_idx, (axis, values) in enumerate(_SWEEPS[1:], start=2):
        elapsed_h = (time.time() - start_wall) / 3600.0
        if elapsed_h >= args.max_wall_h:
            print(f"[orchestrator] wall budget exhausted at round {r_idx}", flush=True)
            ledger.append({"round": "stop", "reason": "wall_budget"})
            _write_ledger(ledger_path, ledger)
            return

        out_dir = _REPO / "output" / f"sweep_{axis}_r{r_idx}"
        print(
            f"\n[orchestrator] round {r_idx}: sweeping {axis} over {values}, "
            f"locked={locked}",
            flush=True,
        )
        rc = _launch_sweep(
            axis=axis,
            values=values,
            locked_overrides=locked,
            out_dir=out_dir,
            time_budget_s=args.bench_time_budget_s,
        )
        summary_path = out_dir / "summary.json"
        if rc != 0 or not summary_path.exists():
            print(
                f"[orchestrator] round {r_idx} sweep failed (rc={rc}); aborting",
                flush=True,
            )
            ledger.append({"round": "stop", "reason": f"sweep_failed_r{r_idx}"})
            _write_ledger(ledger_path, ledger)
            return
        res = _best_from_summary(summary_path, axis)
        if res is None:
            print(
                f"[orchestrator] round {r_idx} could not parse summary",
                flush=True,
            )
            ledger.append({"round": "stop", "reason": f"parse_failed_r{r_idx}"})
            _write_ledger(ledger_path, ledger)
            return
        best_proxy, best_row = res
        best_val = best_row["variant"][axis]
        locked[axis] = best_val

        improvement = (last_best - best_proxy) / max(last_best, 1e-9)
        ledger.append({
            "round": r_idx,
            "axis": axis,
            "best_value": best_val,
            "best_proxy": best_proxy,
            "improvement_vs_prev": improvement,
            "summary_path": str(summary_path),
            "elapsed_h": (time.time() - start_wall) / 3600.0,
            "locked": dict(locked),
        })
        _write_ledger(ledger_path, ledger)
        print(
            f"[orchestrator] round {r_idx} done: best {axis}={best_val:g} "
            f"proxy={best_proxy:.4f} (vs prev {last_best:.4f}, "
            f"Δ={improvement*100:+.2f}%)",
            flush=True,
        )

        if best_proxy <= _BASELINE_PROXY_IBM07 - _IMPROVEMENT_THRESHOLD:
            print(
                f"[orchestrator] round {r_idx} beat baseline "
                f"({_BASELINE_PROXY_IBM07:.4f}); stopping",
                flush=True,
            )
            ledger.append({"round": "stop", "reason": f"beat_baseline_r{r_idx}"})
            _write_ledger(ledger_path, ledger)
            return

        if improvement < _IMPROVEMENT_THRESHOLD and best_proxy > _BASELINE_PROXY_IBM07:
            print(
                f"[orchestrator] round {r_idx} improvement <{_IMPROVEMENT_THRESHOLD*100:.1f}% "
                f"and still above baseline; stopping (converged but not winning)",
                flush=True,
            )
            ledger.append({"round": "stop", "reason": f"converged_above_baseline_r{r_idx}"})
            _write_ledger(ledger_path, ledger)
            return

        last_best = best_proxy

    print("\n[orchestrator] all planned rounds complete", flush=True)
    ledger.append({"round": "stop", "reason": "all_rounds_done"})
    _write_ledger(ledger_path, ledger)


if __name__ == "__main__":
    main()
