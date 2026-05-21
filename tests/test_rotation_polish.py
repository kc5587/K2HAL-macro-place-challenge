"""TDD for Lever C MVP — Rotation polish.

These tests use a tiny stub plc + benchmark; the actual evaluator integration
is exercised by the smoke script, not the unit suite (compute_proxy_cost
needs a real PlacementCost object).
"""
from __future__ import annotations

from typing import Any

import pytest
import torch


@pytest.mark.unit
def test_orientation_class_grouping() -> None:
    from macro_place.rotation_polish import _orientation_class, NS_ORIENTATIONS, EW_ORIENTATIONS

    assert _orientation_class("N") is NS_ORIENTATIONS
    assert _orientation_class("FN") is NS_ORIENTATIONS
    assert _orientation_class("S") is NS_ORIENTATIONS
    assert _orientation_class("FS") is NS_ORIENTATIONS
    assert _orientation_class("E") is EW_ORIENTATIONS
    assert _orientation_class("W") is EW_ORIENTATIONS
    # Unknown defaults to NS.
    assert _orientation_class("?") is NS_ORIENTATIONS


class _StubPlc:
    """Minimal plc stub tracking orientation calls."""

    def __init__(self, initial: dict[int, str]):
        self._ori: dict[int, str] = dict(initial)
        self.calls: list[tuple[int, str]] = []

    def get_macro_orientation(self, node_idx: int) -> str:
        return self._ori.get(node_idx, "N")

    def update_macro_orientation(self, node_idx: int, ori: str) -> None:
        self._ori[node_idx] = ori
        self.calls.append((node_idx, ori))


class _StubBench:
    def __init__(self, hard_macro_indices: list[int]):
        self.hard_macro_indices = hard_macro_indices
        self.macro_sizes = torch.tensor(
            [[10.0, 5.0] for _ in hard_macro_indices], dtype=torch.float32
        )
        self.macro_to_nets = [[0, 1, 2] for _ in hard_macro_indices]


def _make_cost_fn(min_ori: dict[int, str], base: float = 1.0):
    """Build a cost function that returns lowest proxy when each macro is at its min_ori."""
    def _cost_fn(positions, *, benchmark, plc):
        # Penalty: 0.05 per macro not at its preferred orientation.
        penalty = 0.0
        for idx in benchmark.hard_macro_indices:
            if plc.get_macro_orientation(idx) != min_ori.get(idx, plc.get_macro_orientation(idx)):
                penalty += 0.05
        return {
            "proxy_cost": base + penalty,
            "wirelength_cost": 0.1,
            "density_cost": 0.4,
            "congestion_cost": 0.5,
            "overlap_count": 0,
        }
    return _cost_fn


@pytest.mark.unit
def test_polish_finds_best_orientations_same_class() -> None:
    from macro_place.rotation_polish import polish_orientations

    # 2 hard macros initially at "N"; want to land on "FN" and "S" (same-NS-class).
    plc = _StubPlc(initial={10: "N", 11: "N"})
    bench = _StubBench(hard_macro_indices=[10, 11])
    cost_fn = _make_cost_fn(min_ori={10: "FN", 11: "S"})
    pos = torch.zeros((2, 2), dtype=torch.float32)

    out = polish_orientations(
        positions=pos, benchmark=bench, plc=plc, top_k=10, proxy_cost_fn=cost_fn
    )

    assert out["improved_count"] == 2
    assert out["chosen_orientations"][10] == "FN"
    assert out["chosen_orientations"][11] == "S"
    # Final proxy must be strictly less than initial.
    assert out["final_proxy"] < out["initial_proxy"]


@pytest.mark.unit
def test_polish_no_op_when_already_optimal() -> None:
    from macro_place.rotation_polish import polish_orientations

    plc = _StubPlc(initial={10: "FN"})
    bench = _StubBench(hard_macro_indices=[10])
    cost_fn = _make_cost_fn(min_ori={10: "FN"})
    out = polish_orientations(
        positions=torch.zeros((1, 2)), benchmark=bench, plc=plc,
        top_k=10, proxy_cost_fn=cost_fn,
    )
    assert out["improved_count"] == 0
    assert out["chosen_orientations"] == {}
    assert out["final_proxy"] == out["initial_proxy"]


@pytest.mark.unit
def test_polish_respects_top_k_cap() -> None:
    """Only top-K most-connected macros are searched."""
    from macro_place.rotation_polish import polish_orientations

    plc = _StubPlc(initial={i: "N" for i in range(10)})
    bench = _StubBench(hard_macro_indices=list(range(10)))
    cost_fn = _make_cost_fn(min_ori={i: "FN" for i in range(10)})
    out = polish_orientations(
        positions=torch.zeros((10, 2)), benchmark=bench, plc=plc,
        top_k=3, proxy_cost_fn=cost_fn,
    )
    # Only 3 macros searched → at most 3 can flip.
    assert out["improved_count"] <= 3


@pytest.mark.unit
def test_polish_does_not_cross_orientation_class() -> None:
    """A macro initially at 'E' must only be searched in EW_ORIENTATIONS."""
    from macro_place.rotation_polish import polish_orientations

    plc = _StubPlc(initial={10: "E"})
    bench = _StubBench(hard_macro_indices=[10])
    # Build a cost fn that prefers "N" (NS class) over "E" — but rotation
    # MUST NOT pick "N" since that would cross classes.
    cost_fn = _make_cost_fn(min_ori={10: "N"})
    out = polish_orientations(
        positions=torch.zeros((1, 2)), benchmark=bench, plc=plc,
        top_k=10, proxy_cost_fn=cost_fn,
    )
    # Either stays at "E" (current) or flips to FE/W/FW (EW class).
    final_ori = plc.get_macro_orientation(10)
    from macro_place.rotation_polish import EW_ORIENTATIONS
    assert final_ori in EW_ORIENTATIONS, f"Crossed class: {final_ori}"


@pytest.mark.unit
def test_polish_empty_bench_safe() -> None:
    from macro_place.rotation_polish import polish_orientations

    plc = _StubPlc(initial={})
    bench = _StubBench(hard_macro_indices=[])
    cost_fn = _make_cost_fn(min_ori={})
    out = polish_orientations(
        positions=torch.zeros((0, 2)), benchmark=bench, plc=plc,
        top_k=10, proxy_cost_fn=cost_fn,
    )
    assert out["improved_count"] == 0
    assert out["chosen_orientations"] == {}
