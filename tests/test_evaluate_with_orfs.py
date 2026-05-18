from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_with_orfs import _patch_mempool_ng45_fakerams


@pytest.mark.unit
def test_mempool_ng45_patch_adds_routability_grt_config(tmp_path: Path) -> None:
    design_dir = tmp_path / "mempool_tile_ng45"
    design_dir.mkdir()
    (design_dir / "config.mk").write_text(
        "\n".join(
            [
                "export DESIGN_NICKNAME = mempool_tile_ng45",
                "export VERILOG_FILES = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/mempool_tile_wrap.v",
                "export CORE_UTILIZATION = 0.650",
            ]
        )
    )

    _patch_mempool_ng45_fakerams(design_dir)

    config = (design_dir / "config.mk").read_text()
    fastroute = (design_dir / "fastroute_mempool.tcl").read_text()

    assert "export GLOBAL_ROUTE_ARGS = -congestion_iterations 50 -allow_congestion" in config
    assert "export FASTROUTE_TCL = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/fastroute_mempool.tcl" in config
    assert "export DIE_AREA = 0 0 2000 2600" in config
    assert "export CORE_AREA = 10.07 9.94 1990 2590" in config
    assert "CORE_UTILIZATION" not in config
    # Three-bucket adjustment: push routes off M2/M3, let M4/M5 absorb,
    # keep M6+ reserved for power/clock.
    assert "set_global_routing_layer_adjustment metal2-metal3 0.50" in fastroute
    assert "set_global_routing_layer_adjustment metal4-metal5 0.50" in fastroute
    assert "set_global_routing_layer_adjustment metal6-$::env(MAX_ROUTING_LAYER) 0.40" in fastroute
    # Looser std-cell density widens routing channels around macros.
    assert "export PLACE_DENSITY = 0.55" in config
