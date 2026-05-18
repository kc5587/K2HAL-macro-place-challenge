"""
PlacementCost resolution and cloning for the macro placer.

Handles lazy plc loading from benchmark metadata, path discovery
for IBM and NG45 benchmarks, and per-worker plc cloning.
"""

from pathlib import Path
from typing import Optional

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir


# Known benchmark roots
_REPO_ROOT = Path(__file__).resolve().parents[1]
_IBM_ROOT = _REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
_NG45_ROOT = _REPO_ROOT / "external" / "MacroPlacement" / "Flows" / "NanGate45"

# NG45 benchmark name -> directory name mapping
_NG45_MAP = {
    "ariane133": "ariane133",
    "ariane136": "ariane136",
    "nvdla": "nvdla",
    "mempool_tile": "mempool_tile",
    # Aliases used by evaluator
    "ariane133_ng45": "ariane133",
    "ariane136_ng45": "ariane136",
    "nvdla_ng45": "nvdla",
    "mempool_tile_ng45": "mempool_tile",
}


def _find_benchmark_dir(benchmark: Benchmark) -> Optional[Path]:
    """
    Discover the on-disk directory for a benchmark.

    Strategy 1: IBM -- benchmark.name matches directory name under ICCAD04/
    Strategy 2: NG45 -- benchmark.name or alias maps to NanGate45 subdir
    Strategy 3: NG45 fallback -- search all NG45 dirs, match by canvas + macro count

    Returns:
        Path to benchmark directory, or None if not found.
    """
    # Strategy 1: IBM by name
    ibm_dir = _IBM_ROOT / benchmark.name
    if ibm_dir.exists() and (ibm_dir / "netlist.pb.txt").exists():
        return ibm_dir

    # Strategy 2: NG45 by name/alias
    ng45_name = _NG45_MAP.get(benchmark.name)
    if ng45_name:
        ng45_dir = _NG45_ROOT / ng45_name / "netlist" / "output_CT_Grouping"
        if ng45_dir.exists() and (ng45_dir / "netlist.pb.txt").exists():
            return ng45_dir

    # Strategy 3: NG45 fallback -- match by canvas dimensions + macro count
    if _NG45_ROOT.exists():
        for design_dir in sorted(_NG45_ROOT.iterdir()):
            if not design_dir.is_dir():
                continue
            candidate = design_dir / "netlist" / "output_CT_Grouping"
            netlist = candidate / "netlist.pb.txt"
            if not netlist.exists():
                continue
            try:
                candidate_bench, _ = load_benchmark_from_dir(str(candidate))
                if (
                    abs(candidate_bench.canvas_width - benchmark.canvas_width) < 1e-6
                    and abs(candidate_bench.canvas_height - benchmark.canvas_height) < 1e-6
                    and candidate_bench.num_hard_macros == benchmark.num_hard_macros
                    and candidate_bench.num_soft_macros == benchmark.num_soft_macros
                ):
                    return candidate
            except Exception:
                continue

    return None


def resolve_plc(benchmark: Benchmark) -> Optional[PlacementCost]:
    """
    Resolve a PlacementCost object from a Benchmark's metadata.

    Discovers the on-disk benchmark directory and reloads plc from it.
    Returns None if the benchmark directory cannot be found.
    """
    bench_dir = _find_benchmark_dir(benchmark)
    if bench_dir is None:
        return None
    _, plc = load_benchmark_from_dir(str(bench_dir))
    return plc


def clone_plc(benchmark: Benchmark) -> PlacementCost:
    """
    Create an independent PlacementCost clone by reloading from disk.

    Each clone gets its own internal mutable state, which is safer than trying
    to deepcopy the underlying C-backed object for parallel workers.
    """
    bench_dir = _find_benchmark_dir(benchmark)
    if bench_dir is None:
        raise RuntimeError(
            f"Cannot clone plc: benchmark directory not found for '{benchmark.name}'"
        )
    _, plc = load_benchmark_from_dir(str(bench_dir))
    return plc
