#!/usr/bin/env python3
"""Validate macros.tcl against an existing 2_1_floorplan.odb without re-running ORFS.

Iteration loop after the first full ORFS run:
  1. Edit scripts/generate_macro_placement_tcl.py
  2. python3 scripts/validate_macros_tcl.py --benchmark ariane136_ng45
  3. Get a 5-second summary: how many macros placed, what errors

Usage:
  PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \\
    scripts/validate_macros_tcl.py --benchmark ariane136_ng45 \\
    --orfs-root ../OpenROAD-flow-scripts
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from scripts.generate_macro_placement_tcl import write_orfs_macro_placement


_VALIDATOR_TCL_TEMPLATE = textwrap.dedent(
    """
    set _floorplan_odb {FLOORPLAN_ODB}
    set _macros_tcl {MACROS_TCL}

    read_db $_floorplan_odb
    set block [ord::get_db_block]

    # Source macros.tcl, capture any error.
    set _src_err ""
    if {{[catch {{source $_macros_tcl}} _src_err]}} {{
        puts "VALIDATOR_SOURCE_ERROR: $_src_err"
    }}

    # Inventory all block instances and their placement status.
    set _total 0
    set _placed 0
    set _unplaced [list]
    foreach inst [$block getInsts] {{
        if {{[$inst isBlock]}} {{
            incr _total
            set st [$inst getPlacementStatus]
            if {{$st == "FIRM" || $st == "LOCKED" || $st == "PLACED"}} {{
                incr _placed
            }} else {{
                lappend _unplaced [list [$inst getName] $st]
            }}
        }}
    }}
    puts "VALIDATOR_RESULT placed=$_placed total=$_total"
    if {{$_placed < $_total}} {{
        puts "VALIDATOR_UNPLACED ([llength $_unplaced]):"
        set _count 0
        foreach pair $_unplaced {{
            lassign $pair name status
            puts "  $status $name"
            incr _count
            if {{$_count >= 20}} {{
                puts "  ... ([expr {{[llength $_unplaced] - 20}}] more suppressed)"
                break
            }}
        }}
    }}
    exit
    """
).strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, help="e.g. ariane136_ng45")
    p.add_argument("--orfs-root", type=Path, default=Path("../OpenROAD-flow-scripts"))
    p.add_argument(
        "--placement",
        type=Path,
        help="Optional placement .pt; defaults to output/orfs_evaluation/placements/<bench>_placement.pt",
    )
    return p.parse_args()


def _resolve_benchmark_source_dir(bench: str) -> Path:
    """Mirror evaluate_with_orfs.py source-dir resolution."""
    source_name = bench.replace("_ng45", "").replace("_asap7", "")
    if "_ng45" in bench:
        return Path(f"external/MacroPlacement/Flows/NanGate45/{source_name}/netlist/output_CT_Grouping")
    if "_asap7" in bench:
        return Path(f"external/MacroPlacement/Flows/ASAP7/{source_name}/netlist/output_CT_Grouping")
    raise ValueError(f"Unsupported benchmark family: {bench}")


def _resolve_floorplan_odb(orfs_root: Path, tech: str, design_name: str) -> Path:
    """Return path to the 2_1_floorplan.odb produced by an earlier ORFS run."""
    return (
        orfs_root
        / "flow"
        / "results"
        / tech
        / design_name
        / "base"
        / "2_1_floorplan.odb"
    )


def _generate_macros_tcl(args: argparse.Namespace) -> tuple[Path, int]:
    """Generate macros.tcl into the existing ORFS design dir; return its path
    and the number of hard macros in the benchmark."""
    bench_pt = Path(f"benchmarks/processed/public/{args.benchmark}.pt")
    benchmark = Benchmark.load(str(bench_pt))
    source_dir = _resolve_benchmark_source_dir(args.benchmark)
    _, plc = load_benchmark_from_dir(str(source_dir))

    placement_path = args.placement or Path(
        f"output/orfs_evaluation/placements/{args.benchmark}_placement.pt"
    )
    placement = torch.load(placement_path, weights_only=True)

    tech = "nangate45" if "_ng45" in args.benchmark else "asap7"
    design_name = args.benchmark
    design_dir = args.orfs_root / "flow" / "designs" / tech / design_name
    macros_tcl = design_dir / "macros.tcl"

    # Mirror evaluate_with_orfs.py's fallback_core_area logic so the validator
    # exercises the same clamp/translate logic as the production wrapper.
    source_name = args.benchmark.replace("_ng45", "").replace("_asap7", "")
    if "_ng45" in args.benchmark and source_name == "mempool_tile":
        from evaluate_with_orfs import MEMPOOL_NG45_CORE_AREA
        core_area = MEMPOOL_NG45_CORE_AREA
    else:
        core_area = (15.0, 15.0, float(benchmark.canvas_width), float(benchmark.canvas_height))

    write_orfs_macro_placement(
        placement, benchmark, plc, str(macros_tcl), core_area=core_area
    )
    return macros_tcl, len(benchmark.hard_macro_indices)


def main() -> int:
    args = _parse_args()
    tech = "nangate45" if "_ng45" in args.benchmark else "asap7"
    design_name = args.benchmark

    floorplan_odb = _resolve_floorplan_odb(args.orfs_root, tech, design_name)
    if not floorplan_odb.exists():
        print(
            f"ERROR: floorplan ODB not found at {floorplan_odb}\n"
            f"       Run the full ORFS flow once first to produce it.",
            file=sys.stderr,
        )
        return 2

    macros_tcl, expected_macros = _generate_macros_tcl(args)
    print(f"Generated {macros_tcl} ({expected_macros} hard macros expected)")

    flow_dir = args.orfs_root / "flow"
    rel_odb = floorplan_odb.relative_to(flow_dir)
    rel_tcl = macros_tcl.relative_to(flow_dir)
    tcl_src = _VALIDATOR_TCL_TEMPLATE.format(
        FLOORPLAN_ODB=f"./{rel_odb}",
        MACROS_TCL=f"./{rel_tcl}",
    )
    tmp_tcl = Path("/tmp") / f"validate_macros_{args.benchmark}.tcl"
    tmp_tcl.write_text(tcl_src)
    print(f"Validator script: {tmp_tcl}")

    openroad = (args.orfs_root / "tools" / "install" / "OpenROAD" / "bin" / "openroad").resolve()
    cmd = [str(openroad), "-exit", "-no_init", "-no_splash", str(tmp_tcl)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=flow_dir, capture_output=True, text=True, check=False
    )

    out = result.stdout
    print("--- OpenROAD output ---")
    print(out)
    if result.stderr.strip():
        print("--- stderr ---")
        print(result.stderr)

    placed = total = None
    m = re.search(r"VALIDATOR_RESULT placed=(\d+) total=(\d+)", out)
    if m:
        placed, total = int(m.group(1)), int(m.group(2))

    print("=" * 60)
    if placed is None:
        print(f"FAIL: validator did not emit a result line")
        return 1
    if placed == total and placed >= expected_macros:
        print(f"PASS: {placed}/{total} macros placed (expected {expected_macros})")
        return 0
    print(
        f"FAIL: only {placed}/{total} macros placed "
        f"(benchmark expected {expected_macros} hard macros)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
