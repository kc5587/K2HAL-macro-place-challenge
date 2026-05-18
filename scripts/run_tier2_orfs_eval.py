#!/usr/bin/env python3
"""Generate a placement and run local ORFS Tier 2-style evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_place.benchmark import Benchmark
from submissions.macro_placer.cd_lns_placer import CDLNSPlacer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="ariane133_ng45")
    parser.add_argument("--budget-s", type=float, default=600.0)
    parser.add_argument("--num-restarts", type=int, default=1)
    parser.add_argument("--k-per-axis", type=int, default=4)
    parser.add_argument("--topk", action="store_true")
    parser.add_argument("--orfs-root", type=Path, default=Path("../OpenROAD-flow-scripts"))
    parser.add_argument("--output", type=Path, default=Path("output/orfs_evaluation"))
    parser.add_argument("--no-docker", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--placement-only", action="store_true")
    return parser.parse_args()


def _generate_placement(args: argparse.Namespace) -> Path:
    benchmark = Benchmark.load(f"benchmarks/processed/public/{args.benchmark}.pt")
    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = float(args.budget_s)
    placer._config["num_restarts"] = int(args.num_restarts)
    placer._config["k_per_axis"] = int(args.k_per_axis)
    placer._config["topk_polish_enabled"] = bool(args.topk)

    positions = placer.place(benchmark)
    placement_dir = args.output / "placements"
    placement_dir.mkdir(parents=True, exist_ok=True)
    placement_path = placement_dir / f"{args.benchmark}_placement.pt"
    torch.save(positions, placement_path)

    metrics = placer._last_run_stats.get("tier2_metrics", {})
    print("Tier 2 risk metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved placement: {placement_path}")
    return placement_path


def _run_orfs(args: argparse.Namespace, placement_path: Path) -> int:
    if not args.orfs_root.exists():
        print(f"ERROR: ORFS root not found: {args.orfs_root}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "scripts/evaluate_with_orfs.py",
        "--benchmark",
        args.benchmark,
        "--orfs-root",
        str(args.orfs_root),
        "--output",
        str(args.output),
        "--placement",
        str(placement_path),
    ]
    if args.no_docker:
        cmd.append("--no-docker")
    if args.skip_synthesis:
        cmd.append("--skip-synthesis")
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-cache")
    placement_path = _generate_placement(args)
    if args.placement_only:
        return 0
    return _run_orfs(args, placement_path)


if __name__ == "__main__":
    raise SystemExit(main())

