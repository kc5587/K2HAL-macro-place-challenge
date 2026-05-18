"""Tests for the generic NG45 fakeram blackbox writer (unblocks ariane136)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_with_orfs import (
    _detect_fakeram_types,
    _fakeram_blackbox,
    _patch_ng45_design_fakerams,
)


@pytest.mark.unit
def test_detect_fakeram_types_extracts_sizes_from_rtl(tmp_path: Path) -> None:
    rtl = tmp_path / "design.v"
    rtl.write_text(
        "module foo;\n"
        "  fakeram45_256x16 mem_a (.clk(clk));\n"
        "  fakeram45_64x64 mem_b (.clk(clk));\n"
        "  fakeram45_256x16 mem_c (.clk(clk));  // duplicate type\n"
        "endmodule\n"
    )
    types = _detect_fakeram_types([rtl])
    assert types == [(64, 64), (256, 16)]  # sorted, deduplicated


@pytest.mark.unit
def test_fakeram_blackbox_has_correct_port_widths() -> None:
    """fakeram45_256x16 → 16-bit data, 8-bit addr (log2(256) = 8)."""
    text = _fakeram_blackbox(256, 16)
    assert "module fakeram45_256x16" in text
    assert "(* blackbox *)" in text
    assert "output [15:0] rd_out" in text
    assert "input [7:0] addr_in" in text
    assert "input [15:0] wd_in" in text
    assert "input clk" in text


@pytest.mark.unit
def test_fakeram_blackbox_widths_for_64x64() -> None:
    """fakeram45_64x64 → 64-bit data, 6-bit addr."""
    text = _fakeram_blackbox(64, 64)
    assert "output [63:0] rd_out" in text
    assert "input [5:0] addr_in" in text


@pytest.mark.unit
def test_patch_ng45_design_fakerams_writes_macros_and_updates_config(tmp_path: Path) -> None:
    """End-to-end: scan RTL → write macros.v → add LEF/LIB refs to config.mk."""
    design_dir = tmp_path / "ariane136_ng45"
    design_dir.mkdir()
    rtl = design_dir / "ariane.v"
    rtl.write_text(
        "module ariane;\n  fakeram45_256x16 i_ram (.clk(clk));\nendmodule\n"
    )
    (design_dir / "config.mk").write_text(
        "export DESIGN_NICKNAME = ariane136_ng45\n"
        "export VERILOG_FILES = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/ariane.v\n"
    )

    applied = _patch_ng45_design_fakerams(design_dir, [rtl])

    assert applied is True
    macros = (design_dir / "macros.v").read_text()
    assert "module fakeram45_256x16" in macros
    config = (design_dir / "config.mk").read_text()
    assert "macros.v" in config
    assert "fakeram45_256x16.lef" in config
    assert "fakeram45_256x16.lib" in config


@pytest.mark.unit
def test_patch_ng45_design_fakerams_noop_when_no_fakerams(tmp_path: Path) -> None:
    design_dir = tmp_path / "design_no_fakeram"
    design_dir.mkdir()
    rtl = design_dir / "design.v"
    rtl.write_text("module foo;\n  wire w;\nendmodule\n")
    (design_dir / "config.mk").write_text("export DESIGN_NICKNAME = foo\n")

    applied = _patch_ng45_design_fakerams(design_dir, [rtl])

    assert applied is False
    assert not (design_dir / "macros.v").exists()
