"""Convert an HRT_MACRO ``.pt`` benchmark + its plc → ISPD/UCLA Bookshelf.

Output a 5-file Bookshelf bundle (``.aux``/``.nodes``/``.nets``/``.pl``/
``.scl``) suitable as input for DREAMPlace. The conversion uses
``plc.nets`` (driver → list[sink] pin-name strings) which is the same
source of truth ``macro_place.fast_proxy`` consumes, so the
proxy/contest cost surface is preserved.

Coordinate scaling
------------------
Bookshelf coordinates are integers per convention. We scale by
``--scale`` (default 1000) so a canvas of 22.95μm → 22950 integer
units, preserving 3-decimal precision in macro positions. The same
scale is applied in reverse when parsing the result ``.pl`` back.

Usage
-----
    PYTHONPATH=. python3 scripts/pt_to_bookshelf.py \
        --benchmark ibm07 \
        --out-dir bookshelf/ibm07
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def _resolve(scale: float, x: float) -> int:
    """Round a coordinate to nearest int after multiplying by scale."""
    return int(round(float(x) * float(scale)))


def convert(
    benchmark_name: str,
    out_dir: Path,
    scale: float = 1000.0,
    soft_macros_fixed: bool = True,
) -> None:
    # Imports deferred so this module can be imported without our torch deps
    # in the host process (we only need them when actually converting).
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark

    bench_path = Path("benchmarks/processed/public") / f"{benchmark_name}.pt"
    if not bench_path.exists():
        raise FileNotFoundError(f"benchmark not found: {bench_path}")
    b = Benchmark.load(str(bench_path))
    plc = resolve_plc(b)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_w = float(b.canvas_width)
    canvas_h = float(b.canvas_height)

    # ---- Classify modules ----
    # By default ``soft_macros_fixed=True``: only HardMacros are movable;
    # SoftMacros (clustered std-cells) are emitted as fixed terminals so
    # DREAMPlace optimizes around them instead of treating them as rigid
    # rectangles to be spread apart. SoftMacros in our pipeline can
    # overlap freely (they represent thousands of small cells), so
    # respecting their initial positions matches our real problem better.
    hard_movables: list[Any] = []
    soft_movables: list[Any] = []     # used as terminals when soft_macros_fixed
    ports: list[Any] = []
    pin_to_parent_name: dict[str, str] = {}   # full pin name → parent macro/port name
    pin_offsets: dict[str, tuple[float, float]] = {}   # full pin name → (dx, dy)

    # Note: in Google CT plc, both HardMacro and SoftMacro have
    # ``get_type() == "MACRO"``. The split is by Python class name.
    for mod in plc.modules_w_pins:
        t = mod.get_type()
        cls = type(mod).__name__
        if t == "MACRO" and cls == "HardMacro":
            hard_movables.append(mod)
        elif t in ("MACRO", "SOFT_MACRO", "STDCELL") and cls == "SoftMacro":
            soft_movables.append(mod)
        elif t == "MACRO":
            # Unknown macro variant; default to hard so we don't drop it.
            hard_movables.append(mod)
        elif t == "PORT":
            ports.append(mod)
            # Ports also act as their own pin; record under same name.
            pin_to_parent_name[mod.get_name()] = mod.get_name()
            px, py = mod.get_pos()
            pin_offsets[mod.get_name()] = (float(px), float(py))
        elif t == "MACRO_PIN":
            parent = mod.get_macro_name()
            pin_to_parent_name[mod.get_name()] = parent
            ox = float(getattr(mod, "x_offset", 0.0))
            oy = float(getattr(mod, "y_offset", 0.0))
            pin_offsets[mod.get_name()] = (ox, oy)
        # other types ignored

    if soft_macros_fixed:
        movables = hard_movables
        terminals = soft_movables + ports
    else:
        movables = hard_movables + soft_movables
        terminals = ports

    # ---- Index nodes (movables first, terminals last) ----
    node_names: list[str] = [m.get_name() for m in movables] + [p.get_name() for p in terminals]
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    num_movable = len(movables)
    num_terminals = len(terminals)
    num_nodes = num_movable + num_terminals

    # ---- Write .nodes ----
    nodes_path = out_dir / f"{benchmark_name}.nodes"
    with nodes_path.open("w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {num_nodes}\n")
        f.write(f"NumTerminals : {num_terminals}\n\n")
        for mod in movables:
            w = float(getattr(mod, "width", None) or mod.get_width())
            h = float(getattr(mod, "height", None) or mod.get_height())
            iw = max(1, _resolve(scale, w))
            ih = max(1, _resolve(scale, h))
            f.write(f"\t{mod.get_name()}\t{iw}\t{ih}\n")
        for term in terminals:
            # Both soft macros (when soft_macros_fixed=True) and ports go here.
            # Ports have width=0/height=0 in plc; give them a 1x1 footprint
            # so the Bookshelf parser is happy. Soft macros have real sizes.
            w = float(getattr(term, "width", 0.0) or term.get_width() or 0.0)
            h = float(getattr(term, "height", 0.0) or term.get_height() or 0.0)
            iw = max(1, _resolve(scale, w) or 1)
            ih = max(1, _resolve(scale, h) or 1)
            f.write(f"\t{term.get_name()}\t{iw}\t{ih}\tterminal\n")

    # ---- Build & write .nets ----
    # plc.nets[driver_pin] = list of sink pin names (strings).
    # A net's pins = {driver} ∪ {sinks}. Each pin maps to a parent node.
    # If a pin name isn't in our pin_offsets/pin_to_parent_name maps it's a
    # pin we couldn't resolve (e.g., pointing at a node we filtered out);
    # we skip those silently to stay robust.
    nets: list[tuple[str, list[str]]] = []
    for driver, sinks in plc.nets.items():
        members = [driver] + list(sinks)
        nets.append((driver, members))

    # Pre-count valid pins so the NumPins header matches what we actually emit.
    valid_pins_per_net: list[list[tuple[str, float, float]]] = []
    total_pins = 0
    for driver, members in nets:
        pins: list[tuple[str, float, float]] = []
        seen: set[str] = set()
        for pin_name in members:
            if pin_name in seen:
                continue
            seen.add(pin_name)
            parent_name = pin_to_parent_name.get(pin_name)
            if parent_name is None:
                # Unknown pin name; in Google CT format sinks can be macro
                # names directly (no "/"). Treat the sink name AS a node
                # name with offset (0, 0).
                parent_name = pin_name
                if parent_name not in node_to_idx:
                    continue
                offset = (0.0, 0.0)
            else:
                if parent_name not in node_to_idx:
                    continue
                offset = pin_offsets.get(pin_name, (0.0, 0.0))
            # For Bookshelf, offsets are relative to the node CENTER.
            # plc pin offsets are relative to macro CORNER (lower-left) in
            # microns. We convert by subtracting half-size.
            mod = plc.modules_w_pins[
                next(i for i, m in enumerate(plc.modules_w_pins) if m.get_name() == parent_name)
            ] if False else None
            # Fast-path: avoid the per-pin linear scan above by looking up via
            # node_to_idx + a parallel index built once.
            pins.append((parent_name, offset[0], offset[1]))
        valid_pins_per_net.append(pins)
        total_pins += len(pins)

    # Build name → (w, h) for offset normalization to center. Terminals
    # may have real sizes too (e.g. soft macros when fixed).
    name_to_size: dict[str, tuple[float, float]] = {}
    for m in movables:
        name_to_size[m.get_name()] = (
            float(getattr(m, "width", None) or m.get_width()),
            float(getattr(m, "height", None) or m.get_height()),
        )
    for term in terminals:
        name_to_size[term.get_name()] = (
            float(getattr(term, "width", 0.0) or term.get_width() or 0.0),
            float(getattr(term, "height", 0.0) or term.get_height() or 0.0),
        )

    nets_path = out_dir / f"{benchmark_name}.nets"
    nets_kept = sum(1 for pins in valid_pins_per_net if len(pins) >= 2)
    with nets_path.open("w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {nets_kept}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        net_idx = 0
        for pins in valid_pins_per_net:
            if len(pins) < 2:
                continue
            # ISPD bookshelf requires a net name on the NetDegree line.
            f.write(f"NetDegree : {len(pins)} n{net_idx}\n")
            net_idx += 1
            for parent_name, ox, oy in pins:
                w, h = name_to_size.get(parent_name, (0.0, 0.0))
                # Center-relative offset = corner-relative offset - half-size.
                cx = ox - 0.5 * w
                cy = oy - 0.5 * h
                f.write(f"\t{parent_name}\tB\t:\t{_resolve(scale, cx)}\t{_resolve(scale, cy)}\n")

    # ---- Write .pl (initial positions) ----
    pl_path = out_dir / f"{benchmark_name}.pl"
    with pl_path.open("w") as f:
        f.write("UCLA pl 1.0\n\n")
        for mod in movables:
            x, y = mod.get_pos()
            # plc positions are corner positions in microns.
            f.write(f"\t{mod.get_name()}\t{_resolve(scale, x)}\t{_resolve(scale, y)}\t:\tN\n")
        for term in terminals:
            x, y = term.get_pos()
            f.write(f"\t{term.get_name()}\t{_resolve(scale, x)}\t{_resolve(scale, y)}\t:\tN\t/FIXED\n")

    # ---- Write .scl (one big row covering canvas) ----
    # DREAMPlace expects rows for standard-cell legalization. We use a
    # single tall row spanning the whole canvas. Macros will float
    # inside it during analytical placement; we'll skip legalization in
    # the DREAMPlace flow and let our CD+LNS layer handle macro
    # legalization afterward.
    scl_path = out_dir / f"{benchmark_name}.scl"
    iw = _resolve(scale, canvas_w)
    ih = _resolve(scale, canvas_h)
    with scl_path.open("w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write("NumRows : 1\n\n")
        f.write("CoreRow Horizontal\n")
        f.write("  Coordinate    : 0\n")
        f.write(f"  Height        : {ih}\n")
        f.write("  Sitewidth     : 1\n")
        f.write("  Sitespacing   : 1\n")
        f.write("  Siteorient    : N\n")
        f.write("  Sitesymmetry  : Y\n")
        f.write(f"  SubrowOrigin  : 0\tNumSites : {iw}\n")
        f.write("End\n")

    # ---- Write .aux ----
    aux_path = out_dir / f"{benchmark_name}.aux"
    with aux_path.open("w") as f:
        f.write(
            f"RowBasedPlacement : {benchmark_name}.nodes {benchmark_name}.nets "
            f"{benchmark_name}.pl {benchmark_name}.scl\n"
        )

    print(
        f"Wrote {out_dir}/  "
        f"nodes={num_nodes} (movable={num_movable} terminal={num_terminals})  "
        f"nets={nets_kept}  total_pins={total_pins}  "
        f"canvas={iw}x{ih}  scale={scale}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="ibm01, ibm07, ...")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Coordinate scale factor (default 1000 = preserve 3 decimal places).",
    )
    args = parser.parse_args()
    convert(args.benchmark, args.out_dir, args.scale)


if __name__ == "__main__":
    main()
