"""Tests for clock-port plumbing in ORFS design generator (unblocks ariane/nvdla)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_with_orfs import get_clock_port_name
from orfs_integration.design_generator import ORFSDesign, create_orfs_design


@pytest.mark.unit
def test_get_clock_port_name_known_designs() -> None:
    """Clock port name varies by design — wrapper must look it up correctly."""
    assert get_clock_port_name("ariane133_ng45") == "clk_i"
    assert get_clock_port_name("ariane136_ng45") == "clk_i"
    assert get_clock_port_name("mempool_tile_ng45") == "clk_i"
    assert get_clock_port_name("nvdla_ng45") == "nvdla_core_clk"


@pytest.mark.unit
def test_get_clock_port_name_default_fallback() -> None:
    """Unknown benchmark falls back to 'clk' (most common default)."""
    assert get_clock_port_name("some_unknown_benchmark") == "clk"


@pytest.mark.unit
def test_orfs_design_writes_sdc_with_specified_clock_port(tmp_path: Path) -> None:
    """create_orfs_design must use design.clock_port in the generated SDC."""
    # Build minimal inputs
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    verilog = src_dir / "design.v"
    verilog.write_text("module foo(input clk_i); endmodule\n")
    macros_tcl = src_dir / "macros.tcl"
    macros_tcl.write_text("# empty\n")

    orfs_root = tmp_path / "orfs"
    (orfs_root / "flow" / "designs" / "nangate45").mkdir(parents=True)

    design = ORFSDesign(
        name="test_design",
        tech="nangate45",
        verilog_files=[verilog],
        macro_placement_tcl=macros_tcl,
        clock_period=4.0,
        core_utilization=0.65,
        top_module="foo",
        clock_port="clk_i",
    )

    design_dir = create_orfs_design(design, orfs_root, src_dir)
    sdc = (design_dir / "constraint.sdc").read_text()
    assert "[get_ports clk_i]" in sdc
    assert "create_clock -period 4.000" in sdc


@pytest.mark.unit
def test_orfs_design_clock_port_defaults_to_clk(tmp_path: Path) -> None:
    """Backward compat — if clock_port is omitted, SDC uses 'clk'."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    verilog = src_dir / "design.v"
    verilog.write_text("module foo(input clk); endmodule\n")
    macros_tcl = src_dir / "macros.tcl"
    macros_tcl.write_text("# empty\n")

    orfs_root = tmp_path / "orfs"
    (orfs_root / "flow" / "designs" / "nangate45").mkdir(parents=True)

    design = ORFSDesign(
        name="legacy_design",
        tech="nangate45",
        verilog_files=[verilog],
        macro_placement_tcl=macros_tcl,
        clock_period=4.0,
        core_utilization=0.65,
        top_module="foo",
    )

    design_dir = create_orfs_design(design, orfs_root, src_dir)
    sdc = (design_dir / "constraint.sdc").read_text()
    assert "[get_ports clk]" in sdc
