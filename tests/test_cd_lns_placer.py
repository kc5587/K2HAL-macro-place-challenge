"""End-to-end TDD for the new CD+LNS placer entry."""
from __future__ import annotations

from pathlib import Path

import torch
import pytest
import numpy as np

from macro_place.adapter import resolve_plc
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def _marker_positions(marker: float) -> np.ndarray:
    return np.full((2, 2), marker, dtype=np.float64)


@pytest.mark.unit
def test_placer_has_zero_arg_constructor() -> None:
    """Contest contract: the entry class must be constructible with no args."""
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer
    placer = CDLNSPlacer()
    assert placer is not None


@pytest.mark.unit
def test_restart_worker_loads_absolute_benchmark_path(monkeypatch, tmp_path) -> None:
    """Spawned workers must not depend on the parent's current directory."""
    from submissions.macro_placer import cd_lns_placer

    captured: dict[str, str] = {}

    def fake_load(path: str) -> Benchmark:
        captured["path"] = path
        raise RuntimeError("stop after path capture")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cd_lns_placer.Benchmark, "load", staticmethod(fake_load))

    bench_path = cd_lns_placer._benchmark_path_for("ibm01")
    with pytest.raises(RuntimeError, match="stop after path capture"):
        cd_lns_placer._restart_worker("ibm01", str(bench_path), 0, 1.0, {})

    loaded_path = Path(captured["path"])
    assert loaded_path.is_absolute()
    assert loaded_path.name == "ibm01.pt"


@pytest.mark.unit
def test_submission_default_config() -> None:
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    placer = CDLNSPlacer()

    # Broad-SA and Step F rotation-SA disabled for submission: partial 11-bench
    # full IBM run showed +0.49% mean regression vs v1.1. Re-enable only after a
    # true "no-worse" gate is in place. Lever C polish stays on (proven on ibm10).
    assert placer._config["sa_generator_enabled"] is False
    assert placer._config["sa_generator_num_candidates"] == 4
    assert placer._config["sa_rotation_extra_source_enabled"] is False
    assert placer._config["sa_rotation_extra_source_probability"] == pytest.approx(0.0)
    assert placer._config["rotation_polish_enabled"] is True
    assert placer._config["lns_rotation_probability"] == pytest.approx(0.0)
    assert placer._config["sa_rotation_probability"] == pytest.approx(0.0)
    assert placer._config["cd_orientation_search_enabled"] is False
    assert placer._config["targeted_sa_escape_enabled"] is False
    assert placer._config["targeted_sa_escape_num_candidates"] == 4
    assert placer._config["cd_congestion_tiebreak_enabled"] is False
    assert placer._config["cd_congestion_tiebreak_epsilon"] == pytest.approx(1e-3)
    assert placer._config["topk_polish_enabled"] is True
    assert placer._config["topk_polish_k"] == 8
    assert placer._config["topk_polish_time_budget_s"] == 480.0
    # Tier-1 lever: 3000s buys a 5-min safety cushion under the 60-min contest cap.
    assert placer._config["time_budget_s"] == 3000.0
    # 4 restarts match the 4 restart_modes (conservative/light/aggressive/aggressive).
    assert placer._config["num_restarts"] == 4
    # Hessian saddle escape (E12) — default ON after ibm06 v3 probe confirmed
    # 0.184% proxy improvement (1.16358 -> 1.16144) with zero overlap regression.
    assert placer._config["hessian_escape_enabled"] is True
    assert placer._config["hessian_escape_lanczos_iters"] == 16
    assert placer._config["hessian_escape_curvature_threshold"] == -1e-3


@pytest.mark.unit
def test_sa_generator_candidate_can_enter_final_selection(monkeypatch) -> None:
    from macro_place.sa_generator import AnnealCandidate
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    def fake_generate_sa_candidates(**kwargs: object) -> list[AnnealCandidate]:
        return [
            AnnealCandidate(
                positions=_marker_positions(3.0),
                proxy_cost=0.3,
                objective=0.3,
                overlap_count=0,
                evaluations=12,
                accepted_moves=5,
            )
        ]

    def fake_score(
        pos_t: torch.Tensor, benchmark: object, plc: object
    ) -> dict[str, float | int]:
        marker = float(pos_t[0, 0])
        return {"overlap_count": 0, "proxy_cost": marker}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(
        cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object()
    )
    monkeypatch.setattr(
        CDLNSPlacer,
        "_initial_positions",
        lambda self, benchmark, plc: _marker_positions(9.0),
    )
    monkeypatch.setattr(
        cd_lns_placer, "generate_sa_candidates", fake_generate_sa_candidates
    )
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)

    placer = CDLNSPlacer()
    placer._config["num_restarts"] = 0
    placer._config["sa_generator_enabled"] = True
    placer._config["topk_polish_enabled"] = False
    placer._config["hessian_escape_enabled"] = False
    placer._config["orfs_guard_repair_enabled"] = False
    placer._config["orfs_spacing_polish_enabled"] = False
    selected = placer.place(FakeBenchmark())

    assert float(selected[0, 0]) == 3.0
    assert placer._last_run_stats["sa_generator_candidates"] == 1
    assert placer._last_run_stats["sa_generator_best_proxy"] == pytest.approx(0.3)
    candidate_summary = placer._last_run_stats["candidate_summary"]
    assert candidate_summary["best_by_source"]["sa_generator"]["proxy_cost"] == pytest.approx(3.0)
    assert candidate_summary["best_by_source"]["initial_guard"]["proxy_cost"] == pytest.approx(9.0)


@pytest.mark.unit
def test_sa_rotation_candidate_source_competes_with_base_sa(monkeypatch) -> None:
    from macro_place.sa_generator import AnnealCandidate
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    class FakeOrientationState:
        macro_orientation = np.asarray([0, 0, 0], dtype=np.int8)

    seen_probs: list[float] = []

    def fake_generate_sa_candidates(**kwargs: object) -> list[AnnealCandidate]:
        probability = float(kwargs["rotation_probability"])
        seen_probs.append(probability)
        marker = 2.0 if probability > 0.0 else 4.0
        return [
            AnnealCandidate(
                positions=_marker_positions(marker),
                proxy_cost=marker,
                objective=marker,
                overlap_count=0,
                evaluations=12,
                accepted_moves=5,
            )
        ]

    def fake_score(
        pos_t: torch.Tensor, benchmark: object, plc: object
    ) -> dict[str, float | int]:
        marker = float(pos_t[0, 0])
        return {"overlap_count": 0, "proxy_cost": marker}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(
        cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object()
    )
    monkeypatch.setattr(
        cd_lns_placer, "build_orientation_state",
        lambda ctx, plc, benchmark: FakeOrientationState(),
    )
    monkeypatch.setattr(
        CDLNSPlacer,
        "_initial_positions",
        lambda self, benchmark, plc: _marker_positions(9.0),
    )
    monkeypatch.setattr(
        cd_lns_placer, "generate_sa_candidates", fake_generate_sa_candidates
    )
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)

    placer = CDLNSPlacer()
    placer._config["num_restarts"] = 0
    placer._config["sa_generator_enabled"] = True
    placer._config["sa_rotation_extra_source_enabled"] = True
    placer._config["sa_rotation_extra_source_probability"] = 0.1
    placer._config["topk_polish_enabled"] = False
    placer._config["hessian_escape_enabled"] = False
    placer._config["orfs_guard_repair_enabled"] = False
    placer._config["orfs_spacing_polish_enabled"] = False
    selected = placer.place(FakeBenchmark())

    assert seen_probs == [0.0, 0.1]
    assert float(selected[0, 0]) == 2.0
    assert placer._last_run_stats["sa_generator_candidates"] == 2
    candidate_summary = placer._last_run_stats["candidate_summary"]
    assert candidate_summary["best_by_source"]["sa_generator"]["proxy_cost"] == pytest.approx(4.0)
    assert candidate_summary["best_by_source"]["sa_generator_rotation"]["proxy_cost"] == pytest.approx(2.0)


@pytest.mark.unit
def test_targeted_sa_escape_runs_from_current_proxy_best(monkeypatch) -> None:
    from macro_place.sa_generator import AnnealCandidate
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    captured: dict[str, float] = {}

    def fake_targeted_sa(**kwargs: object) -> list[AnnealCandidate]:
        initial_positions = np.asarray(kwargs["initial_positions"], dtype=np.float64)
        captured["source_marker"] = float(initial_positions[0, 0])
        return [
            AnnealCandidate(
                positions=_marker_positions(0.5),
                proxy_cost=0.5,
                objective=0.5,
                overlap_count=0,
                evaluations=7,
                accepted_moves=3,
            )
        ]

    def fake_score(
        pos_t: torch.Tensor, benchmark: object, plc: object
    ) -> dict[str, float | int]:
        marker = float(pos_t[0, 0])
        return {"overlap_count": 0, "proxy_cost": marker}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object())
    monkeypatch.setattr(CDLNSPlacer, "_initial_positions", lambda self, benchmark, plc: _marker_positions(9.0))
    monkeypatch.setattr(CDLNSPlacer, "_run_one_restart", lambda *args, **kwargs: (_marker_positions(1.0), 1.0))
    monkeypatch.setattr(cd_lns_placer, "generate_targeted_sa_escape_candidates", fake_targeted_sa)
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)

    placer = CDLNSPlacer()
    placer._config["num_restarts"] = 1
    placer._config["sa_generator_enabled"] = False
    placer._config["targeted_sa_escape_enabled"] = True
    placer._config["topk_polish_enabled"] = False
    placer._config["hessian_escape_enabled"] = False
    placer._config["orfs_guard_repair_enabled"] = False
    placer._config["orfs_spacing_polish_enabled"] = False
    selected = placer.place(FakeBenchmark())

    assert captured["source_marker"] == pytest.approx(1.0)
    assert float(selected[0, 0]) == pytest.approx(0.5)
    assert placer._last_run_stats["targeted_sa_escape_candidates"] == 1
    assert placer._last_run_stats["candidate_summary"]["best_by_source"][
        "targeted_sa_escape"
    ]["proxy_cost"] == pytest.approx(0.5)


@pytest.mark.unit
def test_tier2_metrics_report_clearance_channels_and_displacement() -> None:
    from submissions.macro_placer.cd_lns_placer import _tier2_metrics

    class FakeBenchmark:
        num_hard_macros = 3
        canvas_width = 40.0
        canvas_height = 40.0
        macro_sizes = torch.full((3, 2), 10.0)

    positions = torch.tensor(
        [
            [5.0, 5.0],
            [18.0, 5.0],
            [5.0, 30.0],
        ],
        dtype=torch.float32,
    )
    initial_positions = positions + torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )

    metrics = _tier2_metrics(positions, FakeBenchmark(), initial_positions)

    assert metrics["macro_pair_count"] == 3
    assert metrics["min_clearance_um"] == pytest.approx(3.0)
    assert metrics["clearance_lt_12um_count"] == 1
    assert metrics["clearance_lt_10um_count"] == 1
    assert metrics["clearance_lt_5um_count"] == 1
    assert metrics["narrow_channel_lt_12um_count"] == 1
    assert metrics["min_boundary_margin_um"] == pytest.approx(0.0)
    assert metrics["displacement_mean_um"] == pytest.approx(7.0 / 3.0)
    assert metrics["displacement_max_um"] == pytest.approx(4.0)
    assert metrics["displacement_gt_12um_count"] == 0


@pytest.mark.unit
def test_orfs_tiebreak_keeps_proxy_primary_outside_tiny_band() -> None:
    from submissions.macro_placer.cd_lns_placer import (
        _FinalCandidate,
        _select_final_candidate,
    )

    class FakeBenchmark:
        num_hard_macros = 2
        canvas_width = 100.0
        canvas_height = 100.0
        macro_sizes = torch.full((2, 2), 10.0)

    risky = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[5.0, 5.0], [5.0, 17.0]], dtype=torch.float32
        ),
        key=(0, 1.0),
        cost={"overlap_count": 0, "proxy_cost": 1.0},
        stats={"candidate": "risky"},
    )
    safer = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[30.0, 30.0], [60.0, 60.0]], dtype=torch.float32
        ),
        key=(0, 1.01),
        cost={"overlap_count": 0, "proxy_cost": 1.01},
        stats={"candidate": "safer"},
    )

    selected, selection_stats = _select_final_candidate(
        [risky, safer],
        FakeBenchmark(),
        {
            "orfs_tiebreak_enabled": True,
            "orfs_proxy_tie_rel_tol": 0.001,
            "orfs_core_margin_um": 12.0,
            "orfs_clearance_threshold_um": 12.0,
        },
    )

    assert selected is risky
    assert selection_stats["selected_by_orfs_tiebreak"] is False
    assert selection_stats["tie_pool_size"] == 1


@pytest.mark.unit
def test_orfs_tiebreak_prefers_post_clamp_overlap_free_within_tiny_band() -> None:
    from submissions.macro_placer.cd_lns_placer import (
        _FinalCandidate,
        _select_final_candidate,
    )

    class FakeBenchmark:
        num_hard_macros = 2
        canvas_width = 100.0
        canvas_height = 100.0
        macro_sizes = torch.full((2, 2), 10.0)

    risky = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[5.0, 5.0], [5.0, 17.0]], dtype=torch.float32
        ),
        key=(0, 1.0),
        cost={"overlap_count": 0, "proxy_cost": 1.0},
        stats={"candidate": "risky"},
    )
    safer = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[30.0, 30.0], [60.0, 60.0]], dtype=torch.float32
        ),
        key=(0, 1.0005),
        cost={"overlap_count": 0, "proxy_cost": 1.0005},
        stats={"candidate": "safer"},
    )

    selected, selection_stats = _select_final_candidate(
        [risky, safer],
        FakeBenchmark(),
        {
            "orfs_tiebreak_enabled": True,
            "orfs_proxy_tie_rel_tol": 0.001,
            "orfs_core_margin_um": 12.0,
            "orfs_clearance_threshold_um": 12.0,
        },
    )

    assert selected is safer
    assert selection_stats["selected_by_orfs_tiebreak"] is True
    assert selection_stats["tie_pool_size"] == 2
    selected_metrics = selection_stats["selected_orfs_metrics"]
    proxy_best_metrics = selection_stats["proxy_best_orfs_metrics"]
    assert selected_metrics["post_clamp_overlap_count"] == 0
    assert proxy_best_metrics["post_clamp_overlap_count"] == 1


@pytest.mark.unit
def test_orfs_tiebreak_allows_slightly_wider_post_clamp_overlap_repair() -> None:
    from submissions.macro_placer.cd_lns_placer import (
        _FinalCandidate,
        _select_final_candidate,
    )

    class FakeBenchmark:
        num_hard_macros = 2
        canvas_width = 100.0
        canvas_height = 100.0
        macro_sizes = torch.full((2, 2), 10.0)

    risky = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[5.0, 5.0], [5.0, 17.0]], dtype=torch.float32
        ),
        key=(0, 1.0),
        cost={"overlap_count": 0, "proxy_cost": 1.0},
        stats={"candidate": "risky"},
    )
    repaired = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[30.0, 30.0], [60.0, 60.0]], dtype=torch.float32
        ),
        key=(0, 1.003),
        cost={"overlap_count": 0, "proxy_cost": 1.003},
        stats={"candidate": "repaired"},
    )

    selected, selection_stats = _select_final_candidate(
        [risky, repaired],
        FakeBenchmark(),
        {
            "orfs_tiebreak_enabled": True,
            "orfs_proxy_tie_rel_tol": 0.001,
            "orfs_overlap_repair_proxy_rel_tol": 0.005,
            "orfs_core_margin_um": 12.0,
            "orfs_clearance_threshold_um": 12.0,
        },
    )

    assert selected is repaired
    assert selection_stats["selected_by_orfs_tiebreak"] is True
    assert selection_stats["tie_pool_size"] == 2
    assert selection_stats["overlap_repair_pool_size"] == 1
    assert selection_stats["proxy_best_orfs_metrics"]["post_clamp_overlap_count"] == 1
    assert selection_stats["selected_orfs_metrics"]["post_clamp_overlap_count"] == 0


@pytest.mark.unit
def test_orfs_tiebreak_prefers_fewer_tiny_post_clamp_channels() -> None:
    from submissions.macro_placer.cd_lns_placer import (
        _FinalCandidate,
        _select_final_candidate,
    )

    class FakeBenchmark:
        num_hard_macros = 6
        canvas_width = 160.0
        canvas_height = 160.0
        macro_sizes = torch.full((6, 2), 10.0)

    risky = _FinalCandidate(
        raw_positions=np.zeros((6, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [
                [20.0, 20.0],
                [31.0, 20.0],
                [20.0, 50.0],
                [34.0, 50.0],
                [20.0, 80.0],
                [39.0, 80.0],
            ],
            dtype=torch.float32,
        ),
        key=(0, 1.0),
        cost={"overlap_count": 0, "proxy_cost": 1.0},
        stats={"candidate": "risky"},
    )
    safer = _FinalCandidate(
        raw_positions=np.zeros((6, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [
                [20.0, 20.0],
                [31.0, 20.0],
                [20.0, 50.0],
                [39.0, 50.0],
                [20.0, 80.0],
                [39.0, 80.0],
            ],
            dtype=torch.float32,
        ),
        key=(0, 1.0005),
        cost={"overlap_count": 0, "proxy_cost": 1.0005},
        stats={"candidate": "safer"},
    )

    selected, selection_stats = _select_final_candidate(
        [risky, safer],
        FakeBenchmark(),
        {
            "orfs_tiebreak_enabled": True,
            "orfs_proxy_tie_rel_tol": 0.001,
            "orfs_core_margin_um": 12.0,
            "orfs_clearance_threshold_um": 12.0,
        },
    )

    assert selected is safer
    assert selection_stats["selected_by_orfs_tiebreak"] is True
    selected_metrics = selection_stats["selected_orfs_metrics"]
    proxy_best_metrics = selection_stats["proxy_best_orfs_metrics"]
    assert selected_metrics["post_clamp_clearance_lt_5um_count"] == 1
    assert proxy_best_metrics["post_clamp_clearance_lt_5um_count"] == 2
    assert selected_metrics["post_clamp_clearance_lt_12um_count"] == (
        proxy_best_metrics["post_clamp_clearance_lt_12um_count"]
    )
    assert selected_metrics["post_clamp_min_clearance_um"] == pytest.approx(
        proxy_best_metrics["post_clamp_min_clearance_um"]
    )


@pytest.mark.unit
def test_orfs_guard_repair_candidate_clears_post_clamp_overlap(monkeypatch) -> None:
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import (
        _FinalCandidate,
        _orfs_guard_repair_candidate,
        _orfs_post_clamp_metrics,
    )

    class FakeBenchmark:
        num_hard_macros = 2
        canvas_width = 100.0
        canvas_height = 100.0
        macro_sizes = torch.full((2, 2), 10.0)

        def get_movable_mask(self) -> torch.Tensor:
            return torch.ones(2, dtype=torch.bool)

    candidate = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[5.0, 5.0], [5.0, 17.0]], dtype=torch.float32
        ),
        key=(0, 1.0),
        cost={"overlap_count": 0, "proxy_cost": 1.0},
        stats={},
    )

    monkeypatch.setattr(
        cd_lns_placer,
        "compute_proxy_cost",
        lambda pos_t, benchmark, plc: {
            "overlap_count": 0,
            "proxy_cost": float(pos_t[0, 0]),
        },
    )

    repaired = _orfs_guard_repair_candidate(
        candidate,
        FakeBenchmark(),
        object(),
        {
            "orfs_guard_repair_enabled": True,
            "orfs_guard_repair_iters": 16,
            "orfs_guard_repair_legalize_iters": 50,
            "orfs_core_margin_um": 12.0,
            "orfs_clearance_threshold_um": 12.0,
        },
    )

    assert repaired is not None
    metrics = _orfs_post_clamp_metrics(repaired.legalized_positions, FakeBenchmark())
    assert metrics["post_clamp_overlap_count"] == 0
    assert repaired.stats["candidate_kind"] == "orfs_guard_repair"


@pytest.mark.unit
def test_orfs_spacing_polish_candidate_repairs_core_clamp_sliver(
    monkeypatch,
) -> None:
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import (
        _FinalCandidate,
        _orfs_post_clamp_metrics,
        _orfs_spacing_polish_candidate,
    )

    class FakeBenchmark:
        num_hard_macros = 2
        canvas_width = 100.0
        canvas_height = 100.0
        macro_sizes = torch.full((2, 2), 10.0)

        def get_movable_mask(self) -> torch.Tensor:
            return torch.ones(2, dtype=torch.bool)

    candidate = _FinalCandidate(
        raw_positions=np.zeros((2, 2), dtype=np.float64),
        legalized_positions=torch.tensor(
            [[26.6, 20.0], [16.5, 20.0]], dtype=torch.float32
        ),
        key=(0, 1.0),
        cost={"overlap_count": 0, "proxy_cost": 1.0},
        stats={},
    )
    base_metrics = _orfs_post_clamp_metrics(
        candidate.legalized_positions,
        FakeBenchmark(),
        core_margin_um=12.0,
    )
    assert base_metrics["post_clamp_overlap_count"] == 1

    monkeypatch.setattr(
        cd_lns_placer,
        "compute_proxy_cost",
        lambda pos_t, benchmark, plc: {
            "overlap_count": 0,
            "proxy_cost": float(pos_t[0, 0]) / 100.0,
        },
    )

    polished = _orfs_spacing_polish_candidate(
        candidate,
        FakeBenchmark(),
        object(),
        {
            "orfs_spacing_polish_enabled": True,
            "orfs_spacing_polish_iters": 8,
            "orfs_spacing_polish_target_um": 2.0,
            "orfs_core_margin_um": 12.0,
            "orfs_clearance_threshold_um": 12.0,
        },
    )

    assert polished is not None
    metrics = _orfs_post_clamp_metrics(
        polished.legalized_positions,
        FakeBenchmark(),
        core_margin_um=12.0,
    )
    assert metrics["post_clamp_overlap_count"] == 0
    assert metrics["post_clamp_min_clearance_um"] >= 2.0 - 1e-6
    assert polished.stats["candidate_kind"] == "orfs_spacing_polish"
    assert polished.key == (0, polished.cost["proxy_cost"])
    assert polished.key == polished.stats["polished_proxy_key"]


@pytest.mark.unit
def test_place_records_tier2_metrics_without_changing_proxy_selection(monkeypatch) -> None:
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        num_hard_macros = 2
        num_macros = 2
        canvas_width = 40.0
        canvas_height = 40.0
        macro_sizes = torch.full((2, 2), 10.0)

    initial = np.array([[5.0, 5.0], [30.0, 5.0]], dtype=np.float64)
    restart = np.array([[5.0, 5.0], [18.0, 5.0]], dtype=np.float64)

    def fake_score(
        pos_t: torch.Tensor,
        benchmark: object,
        plc: object,
    ) -> dict[str, float | int]:
        first_gap_marker = float(pos_t[1, 0])
        proxy = 1.0 if first_gap_marker == 18.0 else 10.0
        return {"overlap_count": 0, "proxy_cost": proxy}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(
        cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object()
    )
    monkeypatch.setattr(
        CDLNSPlacer,
        "_initial_positions",
        lambda self, benchmark, plc: initial.copy(),
    )
    monkeypatch.setattr(
        CDLNSPlacer,
        "_run_one_restart",
        lambda *args, **kwargs: (restart.copy(), 1.0),
    )
    monkeypatch.setattr(
        cd_lns_placer,
        "repair_overlaps",
        lambda pos_t, benchmark, **kwargs: pos_t,
    )
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 10.0
    placer._config["num_restarts"] = 1
    placer._config["topk_polish_enabled"] = False
    placer._config["orfs_guard_repair_enabled"] = False
    selected = placer.place(FakeBenchmark())

    assert float(selected[1, 0]) == 18.0
    metrics = placer._last_run_stats["tier2_metrics"]
    assert metrics["min_clearance_um"] == pytest.approx(3.0)
    assert metrics["clearance_lt_12um_count"] == 1
    assert metrics["displacement_max_um"] == pytest.approx(12.0)
    assert metrics["post_clamp_overlap_count"] == 1
    assert metrics["core_clamp_moved_macro_count"] == 2
    assert placer._last_run_stats["orfs_final_selection"]["tie_pool_size"] >= 1


@pytest.mark.unit
def test_aggressive_restart_caps_cd_phase_and_reaches_lns(monkeypatch) -> None:
    """Bet 6 must reserve time for LNS; otherwise saddle escape never fires."""
    from macro_place.cd import CDResult
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    calls: dict[str, list[float] | list[int] | int] = {
        "cd_budgets": [],
        "cd_k": [],
        "lns_k": [],
        "lns": 0,
    }
    positions = np.zeros((2, 2), dtype=np.float64)

    class FakeBenchmark:
        canvas_width = 10.0
        canvas_height = 10.0

    def fake_warm_start(*args: object, **kwargs: object) -> np.ndarray:
        return positions.copy()

    def fake_fast_proxy(pos: np.ndarray, ctx: object) -> object:
        return type("Proxy", (), {"proxy_cost": 1.0, "overlap_count": 0})()

    def fake_cd_loop(**kwargs: object) -> CDResult:
        calls["cd_budgets"].append(float(kwargs["time_budget_s"]))  # type: ignore[index]
        calls["cd_k"].append(int(kwargs["k_per_axis"]))  # type: ignore[index]
        return CDResult(
            positions=positions.copy(),
            final_cost=1.0,
            sweeps_completed=1,
            total_evals=1,
            plateaued=True,
        )

    def fake_lns_destroy_rebuild(**kwargs: object) -> tuple[np.ndarray, bool, int]:
        calls["lns"] = int(calls["lns"]) + 1
        calls["lns_k"].append(int(kwargs["k_per_axis"]))  # type: ignore[index]
        return positions.copy(), False, 1

    monkeypatch.setattr(cd_lns_placer, "_warm_start_positions", fake_warm_start)
    monkeypatch.setattr(cd_lns_placer, "fast_proxy", fake_fast_proxy)
    monkeypatch.setattr(cd_lns_placer, "cd_loop", fake_cd_loop)
    monkeypatch.setattr(cd_lns_placer, "lns_destroy_rebuild", fake_lns_destroy_rebuild)

    placer = CDLNSPlacer()
    placer._config["restart_modes"] = ("aggressive",)
    placer._config["cd_phase_time_budget_s"] = 2.0
    placer._config["lns_min_time_budget_s"] = 1.0
    placer._config["aggressive_cd_k_per_axis"] = 3
    placer._config["lns_k_per_axis"] = 2
    placer._config["max_consecutive_lns_failures"] = 1
    placer._run_one_restart(
        benchmark=FakeBenchmark(),
        ctx=object(),
        plc=object(),
        seed=0,
        time_budget_s=10.0,
        restart_idx=0,
    )

    assert calls["cd_budgets"]
    assert max(calls["cd_budgets"]) <= 2.0  # type: ignore[arg-type]
    assert calls["cd_k"] == [3]  # type: ignore[comparison-overlap]
    assert calls["lns_k"] == [2]  # type: ignore[comparison-overlap]
    assert calls["lns"] > 0
    assert placer._last_restart_stats["lns_attempts"] > 0






@pytest.mark.unit
def test_topk_polish_keeps_original_candidate_when_polish_worsens(monkeypatch) -> None:
    from macro_place.cd import CDResult
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    cost_by_marker = {9.0: 10.0, 1.0: 1.0, 2.0: 2.0}

    def fake_cd_loop(**kwargs: object) -> CDResult:
        return CDResult(
            positions=_marker_positions(2.0),
            final_cost=2.0,
            sweeps_completed=1,
            total_evals=1,
            plateaued=True,
        )

    def fake_score(pos_t: torch.Tensor, benchmark: object, plc: object) -> dict[str, float | int]:
        marker = float(pos_t[0, 0])
        return {"overlap_count": 0, "proxy_cost": cost_by_marker[marker]}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object())
    monkeypatch.setattr(CDLNSPlacer, "_initial_positions", lambda self, benchmark, plc: _marker_positions(9.0))
    monkeypatch.setattr(CDLNSPlacer, "_run_one_restart", lambda *args, **kwargs: (_marker_positions(1.0), 1.0))
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)
    monkeypatch.setattr(cd_lns_placer, "cd_loop", fake_cd_loop)

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 10.0
    placer._config["num_restarts"] = 1
    placer._config["topk_polish_enabled"] = True
    placer._config["topk_polish_k"] = 1
    placer._config["topk_polish_time_budget_s"] = 1.0
    selected = placer.place(FakeBenchmark())

    assert float(selected[0, 0]) == 1.0
    assert placer._last_run_stats["topk_polish_attempts"] == 1
    assert placer._last_run_stats["topk_polish_accepts"] == 0
    assert placer._last_run_stats["topk_polish_events"][0]["accepted"] is False


@pytest.mark.unit
def test_topk_polish_adds_and_selects_better_polished_candidate(monkeypatch) -> None:
    from macro_place.cd import CDResult
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    cost_by_marker = {9.0: 10.0, 1.0: 2.0, 2.0: 1.0}

    def fake_cd_loop(**kwargs: object) -> CDResult:
        return CDResult(
            positions=_marker_positions(2.0),
            final_cost=1.0,
            sweeps_completed=1,
            total_evals=1,
            plateaued=True,
        )

    def fake_score(pos_t: torch.Tensor, benchmark: object, plc: object) -> dict[str, float | int]:
        marker = float(pos_t[0, 0])
        return {"overlap_count": 0, "proxy_cost": cost_by_marker[marker]}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object())
    monkeypatch.setattr(CDLNSPlacer, "_initial_positions", lambda self, benchmark, plc: _marker_positions(9.0))
    monkeypatch.setattr(CDLNSPlacer, "_run_one_restart", lambda *args, **kwargs: (_marker_positions(1.0), 2.0))
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)
    monkeypatch.setattr(cd_lns_placer, "cd_loop", fake_cd_loop)

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 10.0
    placer._config["num_restarts"] = 1
    placer._config["topk_polish_enabled"] = True
    placer._config["topk_polish_k"] = 1
    placer._config["topk_polish_time_budget_s"] = 1.0
    selected = placer.place(FakeBenchmark())

    assert float(selected[0, 0]) == 2.0
    assert placer._last_run_stats["topk_polish_attempts"] == 1
    assert placer._last_run_stats["topk_polish_accepts"] == 1
    assert placer._last_run_stats["topk_polish_events"][0]["accepted"] is True


@pytest.mark.unit
def test_topk_overlap_positive_polish_cannot_beat_zero_overlap_original(monkeypatch) -> None:
    from macro_place.cd import CDResult
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    def fake_cd_loop(**kwargs: object) -> CDResult:
        return CDResult(
            positions=_marker_positions(2.0),
            final_cost=0.1,
            sweeps_completed=1,
            total_evals=1,
            plateaued=True,
        )

    def fake_score(pos_t: torch.Tensor, benchmark: object, plc: object) -> dict[str, float | int]:
        marker = float(pos_t[0, 0])
        if marker == 2.0:
            return {"overlap_count": 1, "proxy_cost": 0.1}
        return {"overlap_count": 0, "proxy_cost": 5.0 if marker == 1.0 else 10.0}

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object())
    monkeypatch.setattr(CDLNSPlacer, "_initial_positions", lambda self, benchmark, plc: _marker_positions(9.0))
    monkeypatch.setattr(CDLNSPlacer, "_run_one_restart", lambda *args, **kwargs: (_marker_positions(1.0), 5.0))
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(cd_lns_placer, "compute_proxy_cost", fake_score)
    monkeypatch.setattr(cd_lns_placer, "cd_loop", fake_cd_loop)

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 10.0
    placer._config["num_restarts"] = 1
    placer._config["topk_polish_enabled"] = True
    placer._config["topk_polish_k"] = 1
    placer._config["topk_polish_time_budget_s"] = 1.0
    selected = placer.place(FakeBenchmark())

    assert float(selected[0, 0]) == 1.0
    assert placer._last_run_stats["topk_polish_attempts"] == 1
    assert placer._last_run_stats["topk_polish_accepts"] == 0
    assert placer._last_run_stats["topk_polish_events"][0]["polished_overlap_count"] == 1


@pytest.mark.unit
def test_topk_polish_does_not_reduce_restart_budget(monkeypatch) -> None:
    from macro_place.cd import CDResult
    from submissions.macro_placer import cd_lns_placer
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    class FakeBenchmark:
        name = "fake"
        canvas_width = 10.0
        canvas_height = 10.0

    restart_budgets: list[float] = []

    def fake_cd_loop(**kwargs: object) -> CDResult:
        return CDResult(
            positions=_marker_positions(2.0),
            final_cost=1.0,
            sweeps_completed=1,
            total_evals=1,
            plateaued=True,
        )

    def fake_run_one_restart(
        self: CDLNSPlacer,
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, float]:
        restart_budgets.append(float(kwargs["time_budget_s"]))
        self._last_restart_stats = {"restart_idx": 0, "mode": "conservative"}
        return _marker_positions(1.0), 1.0

    monkeypatch.setattr(cd_lns_placer, "resolve_plc", lambda benchmark: object())
    monkeypatch.setattr(cd_lns_placer, "build_fast_proxy_context", lambda plc, benchmark: object())
    monkeypatch.setattr(CDLNSPlacer, "_initial_positions", lambda self, benchmark, plc: _marker_positions(9.0))
    monkeypatch.setattr(CDLNSPlacer, "_run_one_restart", fake_run_one_restart)
    monkeypatch.setattr(cd_lns_placer, "repair_overlaps", lambda pos_t, benchmark: pos_t)
    monkeypatch.setattr(
        cd_lns_placer,
        "compute_proxy_cost",
        lambda pos_t, benchmark, plc: {"overlap_count": 0, "proxy_cost": float(pos_t[0, 0])},
    )
    monkeypatch.setattr(cd_lns_placer, "cd_loop", fake_cd_loop)

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 10.0
    placer._config["num_restarts"] = 1
    placer._config["topk_polish_enabled"] = True
    placer._config["topk_polish_time_budget_s"] = 4.0
    placer.place(FakeBenchmark())

    assert restart_budgets == [10.0]
    assert placer._last_run_stats["per_restart_s"] == 10.0


@pytest.mark.integration
def test_placer_returns_legal_placement_on_ibm01_under_tiny_budget() -> None:
    """Place ibm01 with a 30-second budget; result must be valid: zero
    overlaps and finite proxy cost."""
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None

    placer = CDLNSPlacer()
    placer._config["time_budget_s"] = 30.0
    placer._config["num_restarts"] = 1
    placer._config["k_per_axis"] = 4  # smaller for speed
    positions = placer.place(b)

    pos_t = positions.detach().cpu().to(torch.float32) if isinstance(positions, torch.Tensor) else torch.as_tensor(positions, dtype=torch.float32)
    cost = compute_proxy_cost(pos_t, b, plc)
    assert int(cost["overlap_count"]) == 0
    assert float(cost["proxy_cost"]) > 0.0
    assert float(cost["proxy_cost"]) < 5.0  # sanity bound


@pytest.mark.integration
def test_placer_with_multi_restart_returns_at_least_as_good_as_single() -> None:
    """4 restarts (each with a smaller budget slice) must produce a cost
    at least as good as 1 restart with the full budget — within noise."""
    import time as _time
    from submissions.macro_placer.cd_lns_placer import CDLNSPlacer

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None

    def _run(num_restarts: int) -> float:
        placer = CDLNSPlacer()
        placer._config["time_budget_s"] = 240.0
        placer._config["num_restarts"] = num_restarts
        placer._config["k_per_axis"] = 4
        positions = placer.place(b)
        pos_t = positions.detach().cpu().to(torch.float32) if isinstance(positions, torch.Tensor) else torch.as_tensor(positions, dtype=torch.float32)
        return float(compute_proxy_cost(pos_t, b, plc)["proxy_cost"])

    cost_single = _run(1)
    cost_multi = _run(4)
    # Multi-restart should not be materially worse than single
    assert cost_multi <= cost_single * 1.1, (
        f"4-restart got {cost_multi}, single got {cost_single}"
    )
