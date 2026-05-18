"""Score a DREAMPlace bookshelf ``.pl`` against our pipeline's proxy.

Reads a DREAMPlace-output ``.pl`` (one row per node: ``name x y : ori
[/FIXED]``), maps node names back to our Benchmark's hard+soft macro
order, descales coordinates by ``--scale``, legalizes via
``repair_overlaps``, and scores with ``compute_proxy_cost``.

Prints the resulting ``proxy_cost`` for direct comparison to our
baseline (e.g., ibm07 OFF = 1.1322).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_pl(pl_path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with pl_path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("UCLA"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            try:
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            out[name] = (x, y)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="ibm07")
    parser.add_argument("--pl", required=True, type=Path, help="DREAMPlace .gp.pl output")
    parser.add_argument("--scale", type=float, default=1000.0)
    args = parser.parse_args()

    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.legality import repair_overlaps
    from macro_place.objective import compute_proxy_cost

    bench_path = Path("benchmarks/processed/public") / f"{args.benchmark}.pt"
    b = Benchmark.load(str(bench_path))
    plc = resolve_plc(b)

    # Build name → (hard|soft macro index) using our pipeline's order.
    # In our pipeline: positions[0:num_hard] are hard macros (in
    # benchmark.hard_macro_indices order), positions[num_hard:] are soft
    # macros (in benchmark.soft_macro_indices order).
    pos = np.zeros((b.num_macros, 2), dtype=np.float64)
    parsed = parse_pl(args.pl)

    misses = 0
    for i, idx in enumerate(b.hard_macro_indices):
        name = plc.modules_w_pins[idx].get_name()
        if name not in parsed:
            misses += 1
            x, y = plc.modules_w_pins[idx].get_pos()
        else:
            x, y = parsed[name]
            x /= float(args.scale)
            y /= float(args.scale)
        pos[i, 0], pos[i, 1] = x, y

    soft_start = int(b.num_hard_macros)
    for i, idx in enumerate(b.soft_macro_indices):
        name = plc.modules_w_pins[idx].get_name()
        if name not in parsed:
            misses += 1
            x, y = plc.modules_w_pins[idx].get_pos()
        else:
            x, y = parsed[name]
            x /= float(args.scale)
            y /= float(args.scale)
        pos[soft_start + i, 0], pos[soft_start + i, 1] = x, y

    if misses:
        print(f"WARNING: {misses} macros missing in .pl — fell back to plc position")

    # Legalize through our standard repair_overlaps path, then score.
    raw_t = torch.as_tensor(pos, dtype=torch.float32)
    legalized = repair_overlaps(raw_t, b)
    cost = dict(compute_proxy_cost(legalized, b, plc))

    print(f"=== DREAMPlace result for {args.benchmark} ===")
    print(f"  proxy_cost     : {cost['proxy_cost']:.4f}")
    print(f"  overlap_count  : {int(cost['overlap_count'])}")
    print(f"  wirelength     : {cost['wirelength_cost']:.4f}")
    print(f"  density        : {cost['density_cost']:.4f}")
    print(f"  congestion     : {cost['congestion_cost']:.4f}")


if __name__ == "__main__":
    main()
