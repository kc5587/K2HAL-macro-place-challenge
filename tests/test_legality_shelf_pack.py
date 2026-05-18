"""Regression tests for the final-pass hard-macro legalizer."""
from __future__ import annotations

import torch

from macro_place.benchmark import Benchmark
from macro_place.legality import check_overlaps, repair_overlaps


def _dense_benchmark() -> Benchmark:
    num_macros = 8
    return Benchmark(
        name="dense-2x4",
        canvas_width=16.2,
        canvas_height=8.2,
        num_macros=num_macros,
        num_hard_macros=num_macros,
        num_soft_macros=0,
        macro_positions=torch.full((num_macros, 2), 4.0, dtype=torch.float32),
        macro_sizes=torch.tensor([[6.0, 2.0]] * num_macros, dtype=torch.float32),
        macro_fixed=torch.zeros(num_macros, dtype=torch.bool),
        macro_names=[f"m{i}" for i in range(num_macros)],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0, dtype=torch.float32),
        grid_rows=1,
        grid_cols=1,
    )


def test_repair_overlaps_clears_dense_2x4_macro_layout() -> None:
    benchmark = _dense_benchmark()

    repaired = repair_overlaps(benchmark.macro_positions, benchmark)

    assert check_overlaps(repaired, benchmark) == 0


def test_repair_overlaps_clears_ibm13_initial_plc() -> None:
    from macro_place.adapter import resolve_plc
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    benchmark = Benchmark.load("benchmarks/processed/public/ibm13.pt")
    plc = resolve_plc(benchmark)
    assert plc is not None

    initial = CDLNSPlacer()._initial_positions(benchmark, plc)
    repaired = repair_overlaps(torch.as_tensor(initial, dtype=torch.float32), benchmark)

    assert check_overlaps(repaired, benchmark) == 0
