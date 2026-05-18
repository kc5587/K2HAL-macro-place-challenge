from __future__ import annotations

import pytest

from scripts.generate_macro_placement_tcl import (
    _legalize_direct_macros_to_core,
    _plc_to_odb_name_brackets,
    _repair_direct_macro_overlaps,
)


@pytest.mark.unit
def test_repair_direct_macros_enforces_min_spacing_between_non_overlapping_pair() -> None:
    """Two non-overlapping macros that are <12 μm apart on both axes must be
    pushed apart when min_spacing=12. Required for PDN channel routing
    (nvdla failed PDN-0179 because macros were ~1 μm apart)."""
    # Two 30×30 macros, lower-lefts at (0, 0) and (32, 32). Gap is 2 μm both axes.
    direct_data = [
        ("a", 0.0, 0.0, 30.0, 30.0, "N", "a"),
        ("b", 32.0, 32.0, 30.0, 30.0, "N", "b"),
    ]
    repaired = _repair_direct_macro_overlaps(
        direct_data, padding=2.0, max_iters=200, min_spacing=12.0
    )
    _, xi, yi, wi, hi, _, _ = repaired[0]
    _, xj, yj, wj, hj, _, _ = repaired[1]
    gap_x = max(xi, xj) - min(xi + wi, xj + wj)
    gap_y = max(yi, yj) - min(yi + hi, yj + hj)
    # At least one axis must clear the spacing threshold.
    assert gap_x >= 12.0 - 1e-3 or gap_y >= 12.0 - 1e-3


@pytest.mark.unit
def test_repair_direct_macros_min_spacing_defaults_to_zero_no_regression() -> None:
    """Backward compat — without min_spacing, two near-but-non-overlapping
    macros must NOT be moved (preserves ariane133's working behavior)."""
    direct_data = [
        ("a", 0.0, 0.0, 30.0, 30.0, "N", "a"),
        ("b", 32.0, 32.0, 30.0, 30.0, "N", "b"),
    ]
    repaired = _repair_direct_macro_overlaps(direct_data, padding=2.0)
    assert repaired[0][1] == 0.0 and repaired[0][2] == 0.0
    assert repaired[1][1] == 32.0 and repaired[1][2] == 32.0


@pytest.mark.unit
def test_plc_to_odb_name_brackets_emits_yosys_backslash_escapes() -> None:
    r"""Yosys stores bracketed hierarchical names with literal backslash escapes
    in ODB (e.g. sram_block\[0\]). Tcl findInst needs the exact name including
    backslashes — verified by dumping ariane136's floorplan ODB on 2026-05-15."""
    plc = "i_cache_subsystem/i_nbdcache/sram_block[7].data_sram/macro_mem[6].i_ram"
    odb = _plc_to_odb_name_brackets(plc)
    assert odb == r"i_cache_subsystem.i_nbdcache.sram_block\[7\].data_sram.macro_mem\[6\].i_ram"


@pytest.mark.unit
def test_plc_to_odb_name_brackets_handles_icache_path() -> None:
    plc = "i_cache_subsystem/i_icache/sram_block[0].data_sram/macro_mem[0].i_ram"
    odb = _plc_to_odb_name_brackets(plc)
    assert odb == r"i_cache_subsystem.i_icache.sram_block\[0\].data_sram.macro_mem\[0\].i_ram"


@pytest.mark.unit
def test_legalize_direct_macros_preserves_core_bounds_after_overlap_repair() -> None:
    direct_data = [
        ("cache0", 92.365, 82.775, 77.900, 62.200, "N", "cache0"),
        ("cache1", 81.010, 206.395, 77.900, 62.200, "N", "cache1"),
        ("cache2", 13.950, 52.955, 77.900, 62.200, "N", "cache2"),
        ("cache3", 1.140, 164.083, 77.900, 62.200, "N", "cache3"),
    ]

    legalized = _legalize_direct_macros_to_core(
        direct_data,
        core_area=(10.07, 9.94, 1990.0, 1990.0),
    )

    for _, x_ll, y_ll, width, height, _, _ in legalized:
        assert x_ll >= 10.07
        assert y_ll >= 9.94
        assert x_ll + width <= 1990.0
        assert y_ll + height <= 1990.0

    for i, item_i in enumerate(legalized):
        _, xi, yi, wi, hi, _, _ = item_i
        for item_j in legalized[i + 1 :]:
            _, xj, yj, wj, hj, _, _ = item_j
            overlap_x = min(xi + wi, xj + wj) - max(xi, xj)
            overlap_y = min(yi + hi, yj + hj) - max(yi, yj)
            assert overlap_x <= 0.0 or overlap_y <= 0.0

