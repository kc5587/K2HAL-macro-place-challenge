"""CD+LNS placer entry (Bet 7 clean-restart submission).

A zero-arg-constructor placer that runs a single CD-LNS-multi-restart
loop on the surrogate proxy and returns the best-of-restart placement
rescored with the official TILOS evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from submissions.macro_placer._audit import assert_self_contained
assert_self_contained()

from macro_place.adapter import resolve_plc
from macro_place.benchmark import Benchmark
from macro_place.cd import cd_loop
from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy
from macro_place.legality import repair_overlaps
from macro_place.lns_v2 import lns_destroy_rebuild
from macro_place.objective import compute_proxy_cost
from macro_place.sa_generator import (
    generate_sa_candidates,
    generate_targeted_sa_escape_candidates,
)
from macro_place.hessian_escape import block_diag_top_saddle_macros, hessian_escape
from macro_place.congestion_destroy import worst_congestion_bin_destroy_seeds
from macro_place.hybrid_target import hybrid_target_hard_macros
from macro_place.restart_modes import mode_params
from macro_place.adaptive_config import (
    extract_bench_metrics,
    adaptive_overrides_from_metrics,
)
from macro_place.orientation import build_orientation_state
from macro_place.rotation_polish import polish_orientations_fast
from macro_place.spatial_lns import spatial_window_destroy_seeds


_DEFAULT_CONFIG: dict[str, Any] = {
    # Contest cap is 60 min/bench. Tier-1 lever (2026-05-17): bumped from
    # 1800s → 3300s. Walked back to 3000s on 2026-05-17 evening: ibm17 at
    # 1800s budget overran by 24% (wall=2235s); at 3300s budget projected
    # wall on contest hardware sits at ~57 min — within cap but tight. 3000s
    # buys a 5-min safety cushion at no measured quality cost (time-budget
    # sweep at 7/10/14/20 on ibm10 was bit-identical: conservative wins
    # from plc init regardless of budget).
    "time_budget_s": 3000.0,
    # Tier-1 lever: 4 restarts match the 4 restart_modes below. Was 1.
    # Submission contract: judges call CDLNSPlacer().place(b) and we
    # need diversified search, not single-shot.
    "num_restarts": 4,
    "k_per_axis": 8,                 # 64 candidates per node
    "aggressive_cd_k_per_axis": 4,
    "lns_k_per_axis": 4,
    "max_sweeps": 30,
    "radius_init_ratio": 0.25,
    "radius_min_ratio": 1.0 / 64.0,
    "lns_num_destroy": 10,
    "lns_max_iters": 20,
    "max_consecutive_lns_failures": 3,
    "cd_phase_time_budget_s": 60.0,
    "lns_min_time_budget_s": 5.0,
    "warm_start_sigma": 0.05,
    "restart_modes": ("conservative", "light", "aggressive", "aggressive"),
    # Experimental basin generator. Disabled until a smoke/full-budget probe
    # proves it beats the D-enabled CD/LNS baseline under equal work.
    "sa_generator_enabled": False,
    "sa_generator_steps": 1_000,
    "sa_generator_num_candidates": 4,
    "sa_generator_seed": 20_000,
    "sa_generator_initial_temperature_ratio": 0.03,
    "sa_generator_final_temperature_ratio": 0.001,
    "sa_generator_global_move_probability": 0.70,
    "sa_generator_overlap_penalty": 0.02,
    "sa_generator_diversity_distance_ratio": 0.03,
    "sa_generator_exact_rescore_pool_size": 64,
    "sa_generator_pre_legalize_iters": 0,
    "targeted_sa_escape_enabled": False,
    "targeted_sa_escape_steps": 1_000,
    "targeted_sa_escape_num_candidates": 4,
    "targeted_sa_escape_seed": 30_000,
    "targeted_sa_escape_target_count": 16,
    "targeted_sa_escape_top_n_bins": 8,
    "targeted_sa_escape_macros_per_bin": 4,
    "targeted_sa_escape_exact_rescore_pool_size": 32,
    "targeted_sa_escape_polish_time_budget_s": 0.0,
    # Lever 2 (multi-source targeted SA). Default 1 = single source (current
    # behavior — SA seeds from proxy-best only). >1 runs SA from top-N
    # candidates by key; final selection unchanged.
    "targeted_sa_source_top_k": 1,
    # Lever 3 (hybrid target macro scoring). "congestion" (default) preserves
    # current selection (worst_congestion_bin_destroy_seeds). "hybrid" uses
    # hybrid_target_hard_macros combining congestion + area + degree + crowding.
    "targeted_sa_target_strategy": "congestion",
    # Lever 4 (adaptive SA temperature). "static" (default) preserves the
    # current schedule (initial_temperature_ratio=0.01). "adaptive" samples
    # ``adaptive_num_trials`` random moves before SA, computes the median
    # positive proxy delta, and sets T0 to make P(accept) = adaptive_target_accept.
    "targeted_sa_temperature_mode": "static",
    "targeted_sa_adaptive_num_trials": 64,
    "targeted_sa_adaptive_target_accept": 0.5,
    # Lever 6 (runtime-aware gating). Both default no-op:
    # - min_remaining_budget_s == 0.0: never skip targeted-SA on time.
    # - max_proxy_regression_to_polish == math.inf: always polish candidates.
    "targeted_sa_min_remaining_budget_s": 0.0,
    "targeted_sa_max_proxy_regression_to_polish": math.inf,
    "cd_congestion_tiebreak_enabled": False,
    "cd_congestion_tiebreak_epsilon": 1e-3,
    "topk_polish_enabled": True,
    # Tier-1 lever (polish-every-restart): was 2; bumped to 8 so the
    # polish stage runs on ALL candidates up to that many. With
    # num_restarts=4 in defaults this polishes every restart's output,
    # not just the top 2. Higher num_restarts auto-stays bounded at 8.
    "topk_polish_k": 8,
    # Tier-1 lever: bumped 120 → 480 to preserve ~60s of polish per
    # candidate at the new k=8 (was 120/2=60s/candidate; without this
    # bump it would have dropped to 120/8=15s/candidate). ibm02 smoke at
    # 3300s with k=8 + budget=120 actually regressed +0.87% vs 1800s/k=2
    # because of this squeeze; restoring 60s/candidate fixes it.
    "topk_polish_time_budget_s": 480.0,
    "topk_polish_sweeps": 4,
    "topk_polish_k_per_axis": 4,
    "topk_polish_radius_init_ratio": 1.0 / 64.0,
    "topk_polish_radius_min_ratio": 1.0 / 128.0,
    "orfs_tiebreak_enabled": True,
    "orfs_proxy_tie_rel_tol": 0.001,
    "orfs_overlap_repair_proxy_rel_tol": 0.005,
    "orfs_core_margin_um": 20.0,
    "orfs_clearance_threshold_um": 20.0,
    "orfs_guard_repair_enabled": True,
    "orfs_guard_repair_iters": 16,
    "orfs_guard_repair_legalize_iters": 500,
    "orfs_spacing_polish_enabled": True,
    "orfs_spacing_polish_iters": 50,
    "orfs_spacing_polish_target_um": 2.0,
    # Width PDN needs for routing straps between adjacent macros. When two
    # macros overlap on one axis (forming a channel along the other), this is
    # the minimum gap that channel must have so ORFS PDN can repair it.
    # NG45 metal4 PDN at 56 μm pitch (halo 2 μm + strap 0.48 μm) empirically
    # needs ≥20 μm channels — observed 13.75-μm channels still triggering
    # PDN-0179 on ariane133 2026-05-18. Set to 0 to disable the
    # narrow-channel polish and keep only the legacy bbox-overlap behavior.
    "orfs_spacing_polish_narrow_um": 20.0,
    # Hessian saddle escape (E12) — block-diag + Lanczos random-subspace
    # Rayleigh-Ritz. Designed to find coupled-macro saddles invisible to
    # single-macro CD. Plan + Bet-6 failure analysis in
    # docs/superpowers/plans/2026-05-16-hessian-saddle-escape-plan.md.
    # Lever 5 (hessian on top-K). Default 1 = current behavior (hessian runs
    # only on proxy-best candidate). When >1, sort candidates by key and run
    # hessian on each top-K; all strict-improvements join the candidate pool.
    "hessian_escape_top_k": 1,
    "hessian_escape_enabled": True,             # ON — ibm06 probe confirmed 0.18% win
    "hessian_escape_h_block": 0.5,              # FD step for block-diag, canvas units
    "hessian_escape_h_lanczos": 0.1,            # FD step for Lanczos quadratic form
    "hessian_escape_lanczos_iters": 16,         # K — random subspace dim
    "hessian_escape_curvature_threshold": -1e-3,
    "hessian_escape_line_search_alphas": (0.25, 0.5, 1.0, 2.0, 4.0),
    "hessian_escape_tolerance": 1e-4,
    "hessian_escape_seed": 0,
    # Hessian-guided LNS destroy (Priority 2). When enabled, the per-macro
    # block-diagonal Hessian min eigenvalue ranks macros by how "saddle-like"
    # their local 2x2 geometry is; the top-K seed every LNS-destroy call in
    # the restart. The Hessian compute is ~8N fast_proxy evals (~8–10s on
    # 1k macros), so we cache it once per restart and refresh every
    # ``hessian_lns_destroy_refresh_every`` LNS iters (0 = never refresh).
    "hessian_lns_destroy_enabled": True,  # smoke probe 2026-05-17
    "hessian_lns_destroy_top_k": 10,
    "hessian_lns_destroy_h": 0.5,
    "hessian_lns_destroy_refresh_every": 0,
    # Spatial-window LNS destroy (Lever #1, 2026-05-18). Selects hard macros
    # inside the densest grid cell(s) so LNS rebuilds a coherent cluster,
    # targeting congestion bottlenecks the per-macro Hessian view misses.
    # When both Hessian-guided and spatial-window destroy are enabled, the
    # spatial seeds take the first half of num_destroy (locality) and Hessian
    # seeds fill the rest (saddle escape).
    "spatial_window_destroy_enabled": True,
    "spatial_window_grid_size": 16,
    "spatial_window_share": 0.5,  # fraction of num_destroy that comes from spatial seeds
    # Lever L (worst-congestion-bin destroy). EDA showed congestion is 65.5%
    # of proxy on avg across 17 IBM benches; existing destroy sources do not
    # target it. Picks macros NEAREST to top congestion bins.
    # congestion_destroy_share is its FRACTION of the destroy budget — must
    # be reserved upfront, otherwise spatial+hessian saturate num_destroy
    # and L's seeds get sliced off (bug fix 2026-05-20).
    # Disabled by default: micro-A/B on ibm12/14/17 (LNS-loop level, 20 iters,
    # seeds refreshed per iter) showed L loses to internal-random by 1.2-3x.
    # L's seeds are intelligent but lack the exploration diversity that the
    # placer's existing random-fill provides. Kept as opt-in flag for future
    # variants (e.g. L + jitter, alternating exploit/explore).
    "congestion_destroy_enabled": False,
    "congestion_destroy_share": 0.3,
    "congestion_destroy_top_n_bins": 8,
    "congestion_destroy_macros_per_bin": 4,
    # Lever K' — adaptive config from measured bench properties.
    # No hardcoded bench names (would violate "must be general algorithm").
    # Same formula every bench; different bench shapes get different overrides.
    # Disabled by default: 60-90s smoke A/Bs on ibm12 showed null because LNS
    # never fires at those budgets (CD eats wall time on big benches). K'
    # tunes LNS knobs which need 3000s+ budgets to take effect.
    "adaptive_config_enabled": False,
    # Lever M — big-macro-first two-stage placement. Phase 1 sweeps only the
    # top `two_stage_big_pct` macros by area with small macros frozen; phase 2
    # (the rest of the restart) sweeps all macros from that warm-started
    # scaffold. cd_loop-level micro-A/B (2026-05-20, ibm12/14/17) showed
    # 0.98-1.23% proxy improvement, but placer-level A/B on ibm12 (60s/1r,
    # polish-off) was proxy-neutral: M correctly reduces CONGESTION by 1.05%
    # but density/wirelength compensate, total proxy unchanged. cd_loop-level
    # signal was a 1-sweep artifact. Disabled by default; kept as opt-in for
    # potential revisit at full 4-restart 3000s polish-on scale.
    "two_stage_enabled": False,
    "two_stage_big_pct": 0.3,
    "two_stage_phase1_share": 0.25,
    # Option 2 (Lever K' — adaptive config full rules). Default False preserves
    # the prior congestion-only inline rule bit-exactly. Set True to enable the
    # Rules A–D rule layer in macro_place.adaptive_config.
    "adaptive_config_full_rules": False,
    # Lever C — Rotation polish (final stage after candidate selection).
    # When enabled, iterates each hard macro in its same orientation class
    # (NS or EW) and picks the orientation that minimizes fast_proxy. Plc
    # orientations are synced after so compute_proxy_cost sees the choices.
    # Default False; flip to True after smoke + full-budget A/B confirm.
    "rotation_polish_enabled": True,
    "rotation_polish_top_k": 0,  # 0 = all hard macros
    # Lever C Step E — joint orientation search inside cd_grid_search.
    # When True, each CD sweep iterates the same-class orientations per macro
    # in addition to the k_per_axis² position grid. ~4× per-sweep cost.
    # Default False; opt-in via config.
    "cd_orientation_search_enabled": False,
    # Lever C Step F — LNS/SA rotation proposals.
    # Both default 0.0 (no-op). When > 0, lns_destroy_rebuild and
    # generate_sa_candidates also propose same-class rotations with the given
    # per-decision probability. Requires an orientation_state to be threaded
    # (built when cd_orientation_search_enabled=True or when either of these
    # is > 0). Default 0.0 is bit-exact identical to the prior path.
    "lns_rotation_probability": 0.0,
    "sa_rotation_probability": 0.0,
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def should_skip_targeted_sa_budget(
    *,
    elapsed_s: float,
    time_budget_s: float,
    min_remaining_s: float,
) -> bool:
    """Lever 6 (gate 1): skip targeted-SA generation if too little budget left.

    Default no-op when ``min_remaining_s == 0.0``: returns False for any
    elapsed/budget pair.
    """
    if float(min_remaining_s) <= 0.0:
        return False
    remaining = float(time_budget_s) - float(elapsed_s)
    return remaining < float(min_remaining_s)


def _targeted_sa_source_seed(base_seed: int, source_idx: int) -> int:
    """Lever 2 helper: distinct seed per source so multi-source pools diverge.

    Source 0 maps to ``base_seed`` exactly, preserving single-source behavior.
    Subsequent sources offset by a large stride.
    """
    return int(base_seed) + int(source_idx) * 10_000


def _top_k_candidates_by_key(candidates, k: int):
    """Lever 5 helper: return up to K candidates with the smallest .key.

    Sorts a copy ascending by .key (tuple lex compare matches
    ``min(candidates, key=...)`` behavior at k=1). Returns ``[]`` for k<=0.
    """
    if int(k) <= 0:
        return []
    sorted_pool = sorted(candidates, key=lambda c: c.key)
    return sorted_pool[: int(k)]


def should_skip_targeted_sa_polish(
    *,
    min_escape_proxy: float,
    source_proxy: float,
    max_regression: float,
) -> bool:
    """Lever 6 (gate 2): skip top-k polish on SA candidates if they regress badly.

    ``max_regression`` is in proxy units (absolute, not percent).
    Default no-op when ``max_regression == math.inf``: returns False always.
    Also returns False if the escape pool actually beat the source.
    """
    regression = float(min_escape_proxy) - float(source_proxy)
    if regression <= 0.0:
        return False
    return regression > float(max_regression)


@dataclass(frozen=True)
class _FinalCandidate:
    raw_positions: np.ndarray
    legalized_positions: torch.Tensor
    key: tuple[int, float]
    cost: dict[str, Any]
    stats: dict[str, Any]
    # Lever C Step G — orientation array carried from worker → main → final selection.
    # ``None`` means "no rotation lever applied; plc default orientations".
    orientations: np.ndarray | None = None


def _benchmark_path_for(bench_name: str) -> Path | None:
    """Resolve a path to a serialized Benchmark for worker re-load.

    Falls back to the public processed cache; returns ``None`` if no
    matching file exists, signaling the caller to run restarts serially
    in-process (avoids depending on a hardcoded path for hidden NG45
    designs or alternate benchmark dirs).
    """
    candidate = (
        _REPO_ROOT / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"
    ).resolve()
    return candidate if candidate.exists() else None


def _warm_start_positions(
    benchmark: Benchmark,
    plc: Any,
    seed: int,
    sigma: float = 0.05,
) -> np.ndarray:
    """Return a warm-start position array seeded from initial.plc.

    For seed == 0 returns the bare plc positions (deterministic baseline).
    For seed > 0 applies Gaussian perturbation N(0, sigma * canvas_max)
    per coordinate, clipped to canvas bounds.
    """
    placer = CDLNSPlacer()
    pos = placer._initial_positions(benchmark, plc)
    if seed == 0:
        return pos
    rng = np.random.default_rng(seed)
    canvas_max = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
    perturb = rng.normal(loc=0.0, scale=sigma * canvas_max, size=pos.shape)
    pos = pos + perturb
    pos[:, 0] = np.clip(pos[:, 0], 0.0, float(benchmark.canvas_width))
    pos[:, 1] = np.clip(pos[:, 1], 0.0, float(benchmark.canvas_height))
    return pos


def _to_numpy_positions(positions: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(positions, torch.Tensor):
        return positions.detach().cpu().numpy().astype(np.float64, copy=True)
    return np.asarray(positions, dtype=np.float64).copy()


def _hard_macro_sizes(benchmark: Benchmark) -> np.ndarray:
    num_hard = int(benchmark.num_hard_macros)
    sizes = benchmark.macro_sizes[:num_hard]
    if isinstance(sizes, torch.Tensor):
        return sizes.detach().cpu().numpy().astype(np.float64, copy=True)
    return np.asarray(sizes, dtype=np.float64).copy()


def _percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _clearance_metrics(
    pos: np.ndarray,
    half: np.ndarray,
    clearance_threshold_um: float,
) -> dict[str, Any]:
    clearances: list[float] = []
    narrow_channels = 0
    worst_pairs: list[tuple[float, int, int]] = []

    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            gap_x = abs(float(pos[i, 0] - pos[j, 0])) - float(
                half[i, 0] + half[j, 0]
            )
            gap_y = abs(float(pos[i, 1] - pos[j, 1])) - float(
                half[i, 1] + half[j, 1]
            )
            clearance = float(np.hypot(max(gap_x, 0.0), max(gap_y, 0.0)))
            clearances.append(clearance)
            if (
                (0.0 < gap_x < clearance_threshold_um and gap_y <= 0.0)
                or (0.0 < gap_y < clearance_threshold_um and gap_x <= 0.0)
            ):
                narrow_channels += 1
            worst_pairs.append((clearance, i, j))

    clearance_arr = np.asarray(clearances, dtype=np.float64)
    return {
        "macro_pair_count": int(clearance_arr.size),
        "min_clearance_um": _percentile_or_none(clearance_arr, 0.0),
        "p1_clearance_um": _percentile_or_none(clearance_arr, 1.0),
        "p5_clearance_um": _percentile_or_none(clearance_arr, 5.0),
        "p10_clearance_um": _percentile_or_none(clearance_arr, 10.0),
        "median_clearance_um": _percentile_or_none(clearance_arr, 50.0),
        "clearance_lt_12um_count": int(np.count_nonzero(clearance_arr < 12.0)),
        "clearance_lt_10um_count": int(np.count_nonzero(clearance_arr < 10.0)),
        "clearance_lt_5um_count": int(np.count_nonzero(clearance_arr < 5.0)),
        "narrow_channel_lt_12um_count": int(narrow_channels),
        "worst_clearance_pairs": [
            {"i": int(i), "j": int(j), "clearance_um": float(clearance)}
            for clearance, i, j in sorted(worst_pairs)[:5]
        ],
    }


def _overlap_metrics(pos: np.ndarray, half: np.ndarray) -> dict[str, Any]:
    overlap_count = 0
    worst_pairs: list[tuple[float, int, int]] = []

    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            gap_x = abs(float(pos[i, 0] - pos[j, 0])) - float(
                half[i, 0] + half[j, 0]
            )
            gap_y = abs(float(pos[i, 1] - pos[j, 1])) - float(
                half[i, 1] + half[j, 1]
            )
            if gap_x < -1e-6 and gap_y < -1e-6:
                overlap_count += 1
                worst_pairs.append((min(gap_x, gap_y), i, j))

    return {
        "post_clamp_overlap_count": int(overlap_count),
        "post_clamp_worst_overlap_pairs": [
            {"i": int(i), "j": int(j), "signed_gap_um": float(gap)}
            for gap, i, j in sorted(worst_pairs)[:5]
        ],
    }


def _boundary_margin_metrics(
    pos: np.ndarray,
    half: np.ndarray,
    benchmark: Benchmark,
) -> dict[str, float | None]:
    if pos.shape[0] == 0:
        return {"min_boundary_margin_um": None}
    margins = np.concatenate(
        [
            pos[:, 0] - half[:, 0],
            pos[:, 1] - half[:, 1],
            float(benchmark.canvas_width) - (pos[:, 0] + half[:, 0]),
            float(benchmark.canvas_height) - (pos[:, 1] + half[:, 1]),
        ]
    )
    return {"min_boundary_margin_um": _percentile_or_none(margins, 0.0)}


def _displacement_metrics(
    pos: np.ndarray,
    initial_positions: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    initial = _to_numpy_positions(initial_positions)[: pos.shape[0]]
    displacement = np.linalg.norm(pos - initial, axis=1)
    return {
        "displacement_mean_um": (
            float(displacement.mean()) if displacement.size else None
        ),
        "displacement_p95_um": _percentile_or_none(displacement, 95.0),
        "displacement_max_um": _percentile_or_none(displacement, 100.0),
        "displacement_gt_12um_count": int(np.count_nonzero(displacement > 12.0)),
    }


def _orfs_clamped_positions(
    pos: np.ndarray,
    half: np.ndarray,
    benchmark: Benchmark,
    core_margin_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _orfs_core_bounds(half, benchmark, core_margin_um)
    canvas = np.asarray(
        [float(benchmark.canvas_width), float(benchmark.canvas_height)],
        dtype=np.float64,
    )
    center = canvas * 0.5
    clamped = pos.copy()

    for axis in range(2):
        valid = lower[:, axis] <= upper[:, axis]
        clamped[:, axis] = np.where(
            valid,
            np.clip(clamped[:, axis], lower[:, axis], upper[:, axis]),
            center[axis],
        )

    displacement = np.linalg.norm(clamped - pos, axis=1)
    return clamped, displacement


def _orfs_core_bounds(
    half: np.ndarray,
    benchmark: Benchmark,
    core_margin_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.asarray(
        [float(benchmark.canvas_width), float(benchmark.canvas_height)],
        dtype=np.float64,
    )
    lower = half + float(core_margin_um)
    upper = canvas - half - float(core_margin_um)
    return lower, upper


def _orfs_move_axis(
    pos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    idx: int,
    axis: int,
    delta: float,
) -> float:
    before = float(pos[idx, axis])
    pos[idx, axis] = float(np.clip(before + delta, lower[idx, axis], upper[idx, axis]))
    return abs(float(pos[idx, axis]) - before)


def _orfs_post_clamp_metrics(
    positions: torch.Tensor | np.ndarray,
    benchmark: Benchmark,
    *,
    clearance_threshold_um: float = 12.0,
    core_margin_um: float = 12.0,
) -> dict[str, Any]:
    if not all(
        hasattr(benchmark, attr)
        for attr in (
            "num_hard_macros",
            "macro_sizes",
            "canvas_width",
            "canvas_height",
        )
    ):
        return {
            "orfs_post_clamp_available": False,
            "orfs_post_clamp_reason": "missing_benchmark_geometry",
        }

    num_hard = int(benchmark.num_hard_macros)
    pos = _to_numpy_positions(positions)[:num_hard]
    sizes = _hard_macro_sizes(benchmark)
    half = sizes * 0.5
    clamped, displacement = _orfs_clamped_positions(
        pos, half, benchmark, core_margin_um
    )
    clearance = _clearance_metrics(clamped, half, clearance_threshold_um)
    overlap = _overlap_metrics(clamped, half)

    return {
        "orfs_post_clamp_available": True,
        "orfs_core_margin_um": float(core_margin_um),
        "post_clamp_min_clearance_um": clearance["min_clearance_um"],
        "post_clamp_clearance_lt_5um_count": clearance[
            "clearance_lt_5um_count"
        ],
        "post_clamp_clearance_lt_12um_count": clearance[
            "clearance_lt_12um_count"
        ],
        "post_clamp_narrow_channel_lt_12um_count": clearance[
            "narrow_channel_lt_12um_count"
        ],
        "core_clamp_moved_macro_count": int(np.count_nonzero(displacement > 1e-6)),
        "core_clamp_max_displacement_um": _percentile_or_none(displacement, 100.0),
        **overlap,
    }


def _tier2_metrics(
    positions: torch.Tensor | np.ndarray,
    benchmark: Benchmark,
    initial_positions: torch.Tensor | np.ndarray | None = None,
    clearance_threshold_um: float = 12.0,
) -> dict[str, Any]:
    """Report ORFS-sensitive geometry without changing candidate selection."""
    if not hasattr(benchmark, "num_hard_macros") or not hasattr(
        benchmark, "macro_sizes"
    ):
        return {
            "available": False,
            "reason": "missing_benchmark_geometry",
        }
    num_hard = int(benchmark.num_hard_macros)
    pos = _to_numpy_positions(positions)[:num_hard]
    sizes = _hard_macro_sizes(benchmark)
    half = sizes * 0.5
    metrics = _clearance_metrics(pos, half, clearance_threshold_um)
    metrics.update(_boundary_margin_metrics(pos, half, benchmark))
    metrics.update(
        {
            "available": True,
            "clearance_threshold_um": float(clearance_threshold_um),
        }
    )
    metrics.update(
        _orfs_post_clamp_metrics(
            positions,
            benchmark,
            clearance_threshold_um=clearance_threshold_um,
            core_margin_um=clearance_threshold_um,
        )
    )
    if initial_positions is not None:
        metrics.update(_displacement_metrics(pos, initial_positions))
    return metrics


class CDLNSPlacer:
    """Contest entry: zero-arg constructor + place(benchmark) -> tensor."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = dict(_DEFAULT_CONFIG)
        self._last_restart_stats: dict[str, Any] = {}
        self._last_run_stats: dict[str, Any] = {}
        self._last_final_candidates: list[_FinalCandidate] = []
        self._last_fast_proxy_context: Any | None = None

    def _initial_positions(self, benchmark: Benchmark, plc: Any) -> np.ndarray:
        """Read the plc's current placement as a float64 [num_macros, 2] array."""
        pos = np.zeros((benchmark.num_macros, 2), dtype=np.float64)
        for i, idx in enumerate(benchmark.hard_macro_indices):
            x, y = plc.modules_w_pins[idx].get_pos()
            pos[i, 0], pos[i, 1] = x, y
        for i, idx in enumerate(benchmark.soft_macro_indices):
            x, y = plc.modules_w_pins[idx].get_pos()
            pos[benchmark.num_hard_macros + i, 0] = x
            pos[benchmark.num_hard_macros + i, 1] = y
        return pos

    def _run_one_restart(
        self,
        benchmark: Benchmark,
        ctx: Any,
        plc: Any,
        seed: int,
        time_budget_s: float,
        restart_idx: int = 0,
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        """One restart: CD until plateau, then LNS until repeated failure or time-out."""
        cfg = self._config
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        modes = cfg.get("restart_modes", ("aggressive",)) or ("aggressive",)
        mode = modes[restart_idx % len(modes)]
        params = mode_params(mode, cfg)
        radius_init_ratio = float(params["radius_init_ratio"])
        max_sweeps = int(params["max_sweeps"])
        do_lns = bool(params["do_lns"])
        warm_sigma = float(params["warm_sigma"])
        cd_k_per_axis = (
            int(cfg["aggressive_cd_k_per_axis"]) if do_lns else int(cfg["k_per_axis"])
        )
        lns_k_per_axis = int(cfg["lns_k_per_axis"])

        pos = _warm_start_positions(benchmark, plc, seed=seed, sigma=warm_sigma)

        # Lever C Steps E/F — build orientation state if any of CD orientation
        # search, LNS rotation, or SA rotation is enabled. State lives across
        # the restart's cd_loop/lns/sa iterations so each macro's current
        # orientation persists. Cross-class transitions are disallowed by
        # apply_rotation_to_cache (would corrupt density/overlap caches).
        ori_search_enabled = bool(cfg.get("cd_orientation_search_enabled", False))
        lns_rot_prob = float(cfg.get("lns_rotation_probability", 0.0))
        sa_rot_prob = float(cfg.get("sa_rotation_probability", 0.0))
        ori_state_needed = (
            ori_search_enabled or lns_rot_prob > 0.0 or sa_rot_prob > 0.0
        )
        ori_state = None
        if ori_state_needed:
            try:
                ori_state = build_orientation_state(ctx, plc, benchmark)
            except Exception:
                ori_state = None
                ori_search_enabled = False
                lns_rot_prob = 0.0
                sa_rot_prob = 0.0

        start = time.perf_counter()
        consecutive_lns_failures = 0
        best_pos = pos.copy()
        best_cost = float(fast_proxy(pos, ctx).proxy_cost)
        stats: dict[str, Any] = {
            "restart_idx": int(restart_idx),
            "seed": int(seed),
            "mode": str(mode),
            "initial_cost": float(best_cost),
            "final_cost": float(best_cost),
            "cd_phase_budget_s": float(cfg["cd_phase_time_budget_s"]) if do_lns else 0.0,
            "lns_min_time_budget_s": float(cfg["lns_min_time_budget_s"]) if do_lns else 0.0,
            "cd_k_per_axis": int(cd_k_per_axis),
            "lns_k_per_axis": int(lns_k_per_axis),
            "cd_improvements": 0,
            "lns_attempts": 0,
            "lns_accepts": 0,
            "lns_failures": 0,
            "plateau_count": 0,
            "hessian_lns_destroy_enabled": bool(
                cfg.get("hessian_lns_destroy_enabled", False)
            ),
            "hessian_lns_destroy_computes": 0,
            "hessian_lns_destroy_top_k": int(
                cfg.get("hessian_lns_destroy_top_k", 0)
            ),
        }

        hessian_destroy_seeds: np.ndarray | None = None
        lns_iters_since_refresh = 0

        # Lever M — big-macro-first warm-up. Phase 1 sweeps ONLY the top
        # `two_stage_big_pct` macros by area, with small ones frozen at
        # initial positions. Big macros dominate congestion (proportional to
        # area covered by routing bins), so settling them first gives the
        # full sweep a cleaner congestion field to work with.
        # Micro-A/B (2026-05-20, ibm12/14/17): two-stage beat single-stage
        # by 0.98-1.23% on each bench at cd_loop level.
        if bool(cfg.get("two_stage_enabled", True)):
            num_hard = int(getattr(benchmark, "num_hard_macros", 0))
            if num_hard >= 4:
                hard_areas = (
                    ctx.macro_w[:num_hard] * ctx.macro_h[:num_hard]
                ).astype(np.float64)
                big_pct = float(cfg.get("two_stage_big_pct", 0.3))
                big_threshold = float(np.percentile(hard_areas, (1.0 - big_pct) * 100.0))
                # freeze_mask covers ALL positions: small hard macros + every soft
                freeze_mask = np.ones(best_pos.shape[0], dtype=bool)
                freeze_mask[:num_hard] = hard_areas < big_threshold
                n_big = int((~freeze_mask[:num_hard]).sum())
                if n_big >= 2:
                    phase1_share = float(cfg.get("two_stage_phase1_share", 0.25))
                    phase1_budget = max(1.0, time_budget_s * phase1_share)
                    p1_result = cd_loop(
                        initial_positions=best_pos,
                        ctx=ctx,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        max_sweeps=max_sweeps,
                        k_per_axis=cd_k_per_axis,
                        radius_init_ratio=radius_init_ratio,
                        radius_min_ratio=float(cfg["radius_min_ratio"]),
                        time_budget_s=phase1_budget,
                        seed=seed,
                        freeze_mask=freeze_mask,
                        tiebreak_enabled=bool(
                            cfg.get("cd_congestion_tiebreak_enabled", False)
                        ),
                        tie_epsilon_rel=float(
                            cfg.get("cd_congestion_tiebreak_epsilon", 1e-3)
                        ),
                    )
                    if p1_result.final_cost + 1e-9 < best_cost:
                        best_pos = p1_result.positions.copy()
                        best_cost = p1_result.final_cost
                        stats["two_stage_phase1_improvement"] = True
                    stats["two_stage_phase1_proxy"] = float(p1_result.final_cost)
                    stats["two_stage_n_big"] = n_big

        while time.perf_counter() - start < time_budget_s:
            # CD phase
            cd_remaining = time_budget_s - (time.perf_counter() - start)
            if cd_remaining <= 0:
                break
            reserve_s = float(cfg["lns_min_time_budget_s"]) if do_lns else 0.0
            cd_budget = _phase_budget(
                remaining_s=cd_remaining,
                phase_cap_s=(
                    float(cfg["cd_phase_time_budget_s"]) if do_lns else cd_remaining
                ),
                reserve_s=reserve_s,
                reserve_enabled=do_lns,
            )
            if cd_budget <= 0.0:
                break
            cd_result = cd_loop(
                initial_positions=best_pos,
                ctx=ctx,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                max_sweeps=max_sweeps,
                k_per_axis=cd_k_per_axis,
                radius_init_ratio=radius_init_ratio,
                radius_min_ratio=float(cfg["radius_min_ratio"]),
                time_budget_s=cd_budget,
                seed=seed,
                tiebreak_enabled=bool(
                    cfg.get("cd_congestion_tiebreak_enabled", False)
                ),
                tie_epsilon_rel=float(
                    cfg.get("cd_congestion_tiebreak_epsilon", 1e-3)
                ),
                orientation_state=ori_state,
                search_orientations=ori_search_enabled,
            )
            if cd_result.final_cost + 1e-9 < best_cost:
                best_pos = cd_result.positions.copy()
                best_cost = cd_result.final_cost
                stats["cd_improvements"] += 1

            if not do_lns:
                break

            # LNS phase
            lns_remaining = time_budget_s - (time.perf_counter() - start)
            if lns_remaining <= 0:
                break
            stats["lns_attempts"] += 1

            # Hessian-guided destroy seeding (Priority 2). Lazily compute
            # the ranking on first LNS attempt; refresh every N iters if
            # configured. Cache in ``hessian_destroy_seeds``.
            if bool(cfg.get("hessian_lns_destroy_enabled", False)):
                num_hard_macros = int(getattr(benchmark, "num_hard_macros", 0))
                refresh_every = int(cfg.get("hessian_lns_destroy_refresh_every", 0))
                refresh_due = hessian_destroy_seeds is None or (
                    refresh_every > 0 and lns_iters_since_refresh >= refresh_every
                )
                if num_hard_macros >= 2 and refresh_due:
                    hard_pos = best_pos[:num_hard_macros].astype(
                        np.float64, copy=True
                    )
                    soft_pos = best_pos[num_hard_macros:].astype(
                        np.float64, copy=True
                    )

                    def _hard_eval_fn(hp: np.ndarray) -> float:
                        full = np.empty_like(best_pos)
                        full[:num_hard_macros] = hp
                        if full.shape[0] > num_hard_macros:
                            full[num_hard_macros:] = soft_pos
                        return float(fast_proxy(full, ctx).proxy_cost)

                    top_k = max(
                        int(cfg.get("hessian_lns_destroy_top_k", 10)),
                        int(cfg["lns_num_destroy"]),
                    )
                    hessian_destroy_seeds = block_diag_top_saddle_macros(
                        positions=hard_pos,
                        eval_fn=_hard_eval_fn,
                        num_select=top_k,
                        h=float(cfg.get("hessian_lns_destroy_h", 0.5)),
                    )
                    stats["hessian_lns_destroy_computes"] += 1
                    lns_iters_since_refresh = 0
            lns_iters_since_refresh += 1

            # Spatial-window destroy seeding (Lever #1, 2026-05-18). Selects
            # hard macros inside the densest grid cell(s); recomputed every
            # LNS iteration since best_pos shifts. Cheap (~O(num_hard *
            # grid_size^2) cells touched), runs in microseconds for 1k macros.
            # Seed priority: spatial seeds first, with Hessian extras filling
            # the remaining destroy budget.
            # Round-robin interleaving regressed ibm01 smoke 0.8362 → 0.8661
            # because it pushed hessian seeds ahead of spatial ones.
            num_destroy_target = int(cfg["lns_num_destroy"])
            destroy_seeds = hessian_destroy_seeds
            if bool(cfg.get("spatial_window_destroy_enabled", False)):
                num_hard_macros = int(getattr(benchmark, "num_hard_macros", 0))
                if num_hard_macros >= 2:
                    spatial_share = float(cfg.get("spatial_window_share", 0.5))
                    spatial_count = max(
                        1, int(round(num_destroy_target * spatial_share))
                    )
                    spatial_seeds = spatial_window_destroy_seeds(
                        positions=best_pos,
                        macro_w=ctx.macro_w,
                        macro_h=ctx.macro_h,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        num_select=spatial_count,
                        num_hard_macros=num_hard_macros,
                        grid_size=int(cfg.get("spatial_window_grid_size", 16)),
                    )
                    stats.setdefault("spatial_window_seeds_total", 0)
                    stats["spatial_window_seeds_total"] += int(spatial_seeds.size)
                    if spatial_seeds.size > 0:
                        if hessian_destroy_seeds is not None:
                            spatial_set = set(spatial_seeds.tolist())
                            extras = np.array(
                                [
                                    int(s)
                                    for s in hessian_destroy_seeds.tolist()
                                    if int(s) not in spatial_set
                                ],
                                dtype=np.int64,
                            )
                            destroy_seeds = np.concatenate(
                                [spatial_seeds, extras]
                            )[:num_destroy_target]
                        else:
                            destroy_seeds = spatial_seeds
            # Lever L — worst-congestion-bin destroy seeds. Reserves
            # ``congestion_destroy_share`` of the destroy budget UPFRONT
            # (before being merged with spatial+hessian) so its seeds
            # aren't sliced off by capacity. Prepends to destroy_seeds
            # so L's macros are LNS-rebuilt first.
            if bool(cfg.get("congestion_destroy_enabled", True)):
                cong_share = float(cfg.get("congestion_destroy_share", 0.3))
                cong_count = max(1, int(round(num_destroy_target * cong_share)))
                cong_seeds = worst_congestion_bin_destroy_seeds(
                    positions=best_pos,
                    ctx=ctx,
                    num_seeds=cong_count,
                    top_n_bins=int(cfg.get("congestion_destroy_top_n_bins", 8)),
                    macros_per_bin=int(
                        cfg.get("congestion_destroy_macros_per_bin", 4)
                    ),
                )
                stats.setdefault("congestion_destroy_seeds_total", 0)
                stats["congestion_destroy_seeds_total"] += int(cong_seeds.size)
                if cong_seeds.size > 0:
                    if destroy_seeds is not None and destroy_seeds.size > 0:
                        existing_after_cong = set(cong_seeds.tolist())
                        existing_tail = np.array(
                            [
                                int(s)
                                for s in destroy_seeds.tolist()
                                if int(s) not in existing_after_cong
                            ],
                            dtype=np.int64,
                        )
                        destroy_seeds = np.concatenate(
                            [cong_seeds, existing_tail]
                        )[:num_destroy_target]
                    else:
                        destroy_seeds = cong_seeds[:num_destroy_target]
            if destroy_seeds is not None and destroy_seeds.size == 0:
                destroy_seeds = None

            new_pos, accepted, _ = lns_destroy_rebuild(
                positions=best_pos,
                ctx=ctx,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                num_destroy=num_destroy_target,
                max_lns_iters=int(cfg["lns_max_iters"]),
                k_per_axis=lns_k_per_axis,
                seed=seed + 100,
                destroy_seed_indices=destroy_seeds,
                orientation_state=ori_state if lns_rot_prob > 0.0 else None,
                rotation_probability=lns_rot_prob,
            )
            if accepted:
                best_pos = new_pos
                best_cost = float(fast_proxy(best_pos, ctx).proxy_cost)
                consecutive_lns_failures = 0
                stats["lns_accepts"] += 1
            else:
                consecutive_lns_failures += 1
                stats["lns_failures"] += 1
                if consecutive_lns_failures >= int(cfg["max_consecutive_lns_failures"]):
                    stats["plateau_count"] += 1
                    break

        stats["final_cost"] = float(best_cost)
        stats["runtime_s"] = float(time.perf_counter() - start)
        self._last_restart_stats = stats
        # Step G: return final orientations alongside positions so the main
        # process can sync plc before scoring. None when CD orientation search
        # was disabled (default).
        final_orientations: np.ndarray | None = None
        if ori_state is not None and (
            ori_search_enabled or lns_rot_prob > 0.0 or sa_rot_prob > 0.0
        ):
            final_orientations = np.asarray(
                ori_state.macro_orientation, dtype=np.int8
            ).copy()
        return best_pos, best_cost, final_orientations

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = resolve_plc(benchmark)
        if plc is None:
            raise RuntimeError(f"resolve_plc returned None for {benchmark.name}")
        # Expose the internal plc so callers can read the final orientations
        # / state after place() returns (e.g. for in-memory rescoring).
        self._last_plc = plc

        ctx = build_fast_proxy_context(plc, benchmark)
        self._last_fast_proxy_context = ctx

        # Lever K' — adaptive config from measured bench properties.
        # General algorithm (no hardcoded bench names — that would violate
        # the "must be general algorithm" rule). Same formula for every
        # bench; different bench shapes get different config overrides.
        # Restored at the end so the next call starts from defaults.
        _saved_config: dict[str, Any] | None = None
        if bool(self._config.get("adaptive_config_enabled", True)):
            adaptive_overrides = self._compute_adaptive_overrides(plc, ctx, benchmark)
            if adaptive_overrides:
                _saved_config = dict(self._config)
                self._config.update(adaptive_overrides)

        try:
            return self._place_inner(benchmark, plc, ctx)
        finally:
            if _saved_config is not None:
                self._config = _saved_config

    def _compute_adaptive_overrides(
        self, plc: Any, ctx: Any, benchmark: Benchmark
    ) -> dict[str, Any]:
        """Measure bench shape, return config overrides (Lever K').

        Default: legacy Rule A only (cong_share > 0.6 path). Set
        ``adaptive_config_full_rules=True`` to enable the Rules A–D layer
        from ``macro_place.adaptive_config``.
        """
        try:
            initial = self._initial_positions(benchmark, plc)
        except Exception:
            return {}
        init = fast_proxy(initial, ctx)
        if init.proxy_cost <= 0.0:
            return {}
        if bool(self._config.get("adaptive_config_full_rules", False)):
            num_macros = int(getattr(benchmark, "num_macros", 0))
            metrics = extract_bench_metrics(
                initial_proxy_cost=float(init.proxy_cost),
                initial_wirelength=float(init.wirelength),
                initial_density=float(init.density),
                initial_congestion=float(init.congestion),
                num_macros=num_macros,
            )
            return adaptive_overrides_from_metrics(metrics, self._config)
        # Legacy Rule A only (bit-exact prior behavior).
        cong_share = 0.5 * init.congestion / init.proxy_cost
        overrides: dict[str, Any] = {}
        if cong_share > 0.6:
            base_cd = float(self._config.get("cd_phase_time_budget_s", 60.0))
            base_fails = int(self._config.get("max_consecutive_lns_failures", 3))
            base_destroy = int(self._config.get("lns_num_destroy", 10))
            overrides["cd_phase_time_budget_s"] = base_cd * 0.5
            overrides["max_consecutive_lns_failures"] = base_fails * 2
            overrides["lns_num_destroy"] = max(base_destroy, int(base_destroy * 1.5))
        return overrides

    def _place_inner(
        self, benchmark: Benchmark, plc: Any, ctx: Any
    ) -> torch.Tensor:
        place_start = time.perf_counter()
        time_budget_s = float(self._config["time_budget_s"])
        topk_budget_s = _topk_polish_budget(self._config)
        num_restarts = int(self._config["num_restarts"])
        # Each restart self-caps at per_restart_s. Restarts run in PARALLEL
        # (4 workers via ProcessPoolExecutor), so total wall ≈ per_restart_s
        # + topk_budget + setup overhead, NOT N * per_restart_s. Kept as
        # time_budget_s / num_restarts for backwards compatibility with
        # smoke-validated behavior; under-utilizes wall time relative to
        # the contest cap but extra budget did not improve scores in our
        # A/B tests (ibm10 at 1800s vs 3300s was bit-identical).
        per_restart_s = time_budget_s / max(num_restarts, 1)

        initial_pos = self._initial_positions(benchmark, plc)
        initial_stats: dict[str, Any] = {"restart_idx": -1, "mode": "initial_guard"}
        results: list[tuple[np.ndarray, float, dict[str, Any], np.ndarray | None]] = [
            (initial_pos, 0.0, initial_stats, None)
        ]
        self._last_run_stats = {
            "benchmark": benchmark.name,
            "num_restarts": num_restarts,
            "per_restart_s": float(per_restart_s),
            "restarts": [],
            "sa_generator_enabled": bool(self._config["sa_generator_enabled"]),
            "sa_generator_candidates": 0,
            "targeted_sa_escape_enabled": bool(
                self._config["targeted_sa_escape_enabled"]
            ),
            "targeted_sa_escape_candidates": 0,
            "cd_congestion_tiebreak_enabled": bool(
                self._config["cd_congestion_tiebreak_enabled"]
            ),
            "cd_congestion_tiebreak_epsilon": float(
                self._config["cd_congestion_tiebreak_epsilon"]
            ),
            "topk_polish_enabled": bool(self._config["topk_polish_enabled"]),
            "topk_polish_attempts": 0,
            "topk_polish_accepts": 0,
            "topk_polish_events": [],
            # Lever 6 (runtime-aware gating). Populated lazily when targeted-SA runs.
            "targeted_sa_gated_reason": None,
            "targeted_sa_gated_elapsed_s": None,
            "targeted_sa_gated_remaining_s": None,
            "targeted_sa_polish_gated_reason": None,
            "targeted_sa_polish_gated_regression": None,
        }

        if bool(self._config["sa_generator_enabled"]):
            # Step F: if sa_rotation_probability > 0, build an orientation
            # state so the SA inner loop can also propose same-class rotations.
            # The internal _rescore_archive re-evaluates with the final ctx
            # pin offsets, so the final orientations are attached to every
            # candidate to keep the worker's plc state consistent.
            sa_gen_rot_prob = float(
                self._config.get("sa_rotation_probability", 0.0)
            )
            sa_ori_state = None
            if sa_gen_rot_prob > 0.0:
                try:
                    sa_ori_state = build_orientation_state(ctx, plc, benchmark)
                except Exception:
                    sa_ori_state = None
                    sa_gen_rot_prob = 0.0
            sa_candidates = generate_sa_candidates(
                initial_positions=initial_pos,
                ctx=ctx,
                canvas_w=float(benchmark.canvas_width),
                canvas_h=float(benchmark.canvas_height),
                seed=int(self._config["sa_generator_seed"]),
                steps=int(self._config["sa_generator_steps"]),
                num_candidates=int(self._config["sa_generator_num_candidates"]),
                initial_temperature_ratio=float(
                    self._config["sa_generator_initial_temperature_ratio"]
                ),
                final_temperature_ratio=float(
                    self._config["sa_generator_final_temperature_ratio"]
                ),
                global_move_probability=float(
                    self._config["sa_generator_global_move_probability"]
                ),
                overlap_penalty=float(self._config["sa_generator_overlap_penalty"]),
                diversity_distance_ratio=float(
                    self._config["sa_generator_diversity_distance_ratio"]
                ),
                exact_rescore_pool_size=int(
                    self._config["sa_generator_exact_rescore_pool_size"]
                ),
                pre_legalize_iters=int(
                    self._config["sa_generator_pre_legalize_iters"]
                ),
                orientation_state=sa_ori_state,
                rotation_probability=sa_gen_rot_prob,
            )
            sa_candidate_orientations: np.ndarray | None = None
            if sa_ori_state is not None and sa_gen_rot_prob > 0.0:
                sa_candidate_orientations = np.asarray(
                    sa_ori_state.macro_orientation, dtype=np.int8
                ).copy()
            for candidate_idx, candidate in enumerate(sa_candidates):
                results.append(
                    (
                        candidate.positions,
                        float(candidate.proxy_cost),
                        {
                            "restart_idx": -2,
                            "mode": "sa_generator",
                            "candidate_kind": "sa_generator",
                            "candidate_idx": int(candidate_idx),
                            "sa_objective": float(candidate.objective),
                            "sa_proxy_cost": float(candidate.proxy_cost),
                            "sa_overlap_count": int(candidate.overlap_count),
                            "sa_evaluations": int(candidate.evaluations),
                            "sa_accepted_moves": int(candidate.accepted_moves),
                        },
                        sa_candidate_orientations,
                    )
                )
            self._last_run_stats["sa_generator_candidates"] = int(len(sa_candidates))
            if sa_candidates:
                self._last_run_stats["sa_generator_best_proxy"] = float(
                    min(candidate.proxy_cost for candidate in sa_candidates)
                )

        if num_restarts >= 1:
            if num_restarts == 1:
                pos, _, orientations = self._run_one_restart(
                    benchmark=benchmark, ctx=ctx, plc=plc, seed=0,
                    time_budget_s=per_restart_s, restart_idx=0,
                )
                restart_stats = dict(self._last_restart_stats)
                results.append((pos, 0.0, restart_stats, orientations))
                self._last_run_stats["restarts"].append(restart_stats)
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                import multiprocessing as _mp
                bench_path = _benchmark_path_for(benchmark.name)
                # Fallback for benchmarks not in the public processed cache
                # (hidden NG45 designs, alternate benchmark dirs): run
                # restarts serially in-process so we don't depend on a
                # hardcoded disk path. Slower wall-clock but preserves
                # correctness for any Benchmark the harness hands us.
                if bench_path is None:
                    self._last_run_stats.setdefault("restart_errors", []).append(
                        f"no cached benchmark file for {benchmark.name}; running serially"
                    )
                    for seed in range(num_restarts):
                        pos, _, orientations_serial = self._run_one_restart(
                            benchmark=benchmark, ctx=ctx, plc=plc, seed=seed,
                            time_budget_s=per_restart_s, restart_idx=seed,
                        )
                        restart_stats = dict(self._last_restart_stats)
                        results.append((pos, 0.0, restart_stats, orientations_serial))
                        self._last_run_stats["restarts"].append(restart_stats)
                    bench_path = None  # skip the ProcessPool block below
                if bench_path is not None:
                    # Force fork start method when available so worker
                    # children inherit sys.modules (including this module
                    # under whatever name the judge's harness loaded it as).
                    # Under spawn, the child would re-import by module name;
                    # if the judge loaded us via spec_from_file_location, the
                    # name isn't on the import path in a fresh interpreter.
                    try:
                        _ctx = _mp.get_context("fork")
                    except (ValueError, RuntimeError):
                        _ctx = None  # Windows or environments without fork
                    with ProcessPoolExecutor(
                        max_workers=num_restarts, mp_context=_ctx
                    ) as ex:
                        futures = [
                            ex.submit(
                                _restart_worker,
                                benchmark.name,
                                str(bench_path),
                                seed,
                                per_restart_s,
                                dict(self._config),
                                seed,
                            )
                            for seed in range(num_restarts)
                        ]
                        self._last_run_stats.setdefault("restart_errors", [])
                        for fut in as_completed(futures):
                            try:
                                item = fut.result()
                            except Exception as exc:
                                # Per-restart isolation: a single worker
                                # crash (OOM, plc init error, bad bench path)
                                # must NOT kill the placement. We always
                                # have the initial_guard candidate plus any
                                # surviving sibling workers.
                                self._last_run_stats["restart_errors"].append(
                                    f"{type(exc).__name__}: {str(exc)[:200]}"
                                )
                                continue
                            orientations: np.ndarray | None = None
                            if len(item) == 2:
                                pos, surrogate = item
                                restart_stats = {}
                            elif len(item) == 3:
                                pos, surrogate, restart_stats = item
                            else:  # 4-tuple (Step G)
                                pos, surrogate, restart_stats, orientations = item
                            results.append((pos, surrogate, restart_stats, orientations))
                            self._last_run_stats["restarts"].append(restart_stats)

        # Legalize every candidate and pick by (overlap_count, proxy_cost) so
        # that ANY zero-overlap candidate beats every overlap-positive one.
        # Some dense benchmarks (ibm13) have plateau layouts where the
        # pair-pushing legalizer leaves a few residual overlaps; without this
        # gate, a low-cost overlapping candidate could DQ the submission.
        candidates = [
            _score_legalized_candidate(
                raw_positions=pos_arr,
                benchmark=benchmark,
                plc=plc,
                stats=candidate_stats,
                orientations=cand_orientations,
            )
            for pos_arr, _surrogate, candidate_stats, cand_orientations in results
        ]
        self._last_final_candidates = list(candidates)
        candidates.extend(
            _topk_final_polish(
                candidates=candidates,
                benchmark=benchmark,
                plc=plc,
                ctx=ctx,
                cfg=self._config,
                time_budget_s=topk_budget_s,
                run_stats=self._last_run_stats,
            )
        )
        proxy_best_candidate = min(candidates, key=lambda candidate: candidate.key)

        # Lever 6 gate 1: skip targeted-SA if elapsed wall leaves too little
        # budget. Default no-op (min_remaining_s == 0.0).
        _sa_enabled = bool(self._config["targeted_sa_escape_enabled"])
        _sa_skip_budget = False
        if _sa_enabled:
            _sa_elapsed_s = time.perf_counter() - place_start
            self._last_run_stats["targeted_sa_gated_elapsed_s"] = float(_sa_elapsed_s)
            self._last_run_stats["targeted_sa_gated_remaining_s"] = float(
                time_budget_s - _sa_elapsed_s
            )
            _sa_skip_budget = should_skip_targeted_sa_budget(
                elapsed_s=_sa_elapsed_s,
                time_budget_s=time_budget_s,
                min_remaining_s=float(
                    self._config["targeted_sa_min_remaining_budget_s"]
                ),
            )
            if _sa_skip_budget:
                self._last_run_stats["targeted_sa_gated_reason"] = "budget"

        if _sa_enabled and not _sa_skip_budget:
            # Lever 2: SA from top-K source candidates. k=1 (default) = current
            # single-source behavior (proxy_best). Seed offset per source keeps
            # SA pools divergent.
            sa_source_top_k = int(self._config["targeted_sa_source_top_k"])
            sa_sources = _top_k_candidates_by_key(candidates, sa_source_top_k)
            scored_escape_candidates: list[_FinalCandidate] = []
            sa_target_strategy = str(
                self._config.get("targeted_sa_target_strategy", "congestion")
            )
            for source_idx, source_candidate in enumerate(sa_sources):
                escape_seed_positions = (
                    source_candidate.legalized_positions.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                source_seed = _targeted_sa_source_seed(
                    int(self._config["targeted_sa_escape_seed"]), source_idx
                )
                # Lever 3: pick targets via strategy. "hybrid" overrides the
                # default congestion-bin selector with a multi-factor score.
                _target_override = None
                if sa_target_strategy == "hybrid":
                    _target_override = hybrid_target_hard_macros(
                        escape_seed_positions,
                        ctx,
                        num_seeds=int(
                            self._config["targeted_sa_escape_target_count"]
                        ),
                    )
                escape_candidates = generate_targeted_sa_escape_candidates(
                    initial_positions=escape_seed_positions,
                    ctx=ctx,
                    canvas_w=float(benchmark.canvas_width),
                    canvas_h=float(benchmark.canvas_height),
                    seed=source_seed,
                    steps=int(self._config["targeted_sa_escape_steps"]),
                    num_candidates=int(
                        self._config["targeted_sa_escape_num_candidates"]
                    ),
                    target_count=int(self._config["targeted_sa_escape_target_count"]),
                    top_n_bins=int(self._config["targeted_sa_escape_top_n_bins"]),
                    macros_per_bin=int(
                        self._config["targeted_sa_escape_macros_per_bin"]
                    ),
                    exact_rescore_pool_size=int(
                        self._config["targeted_sa_escape_exact_rescore_pool_size"]
                    ),
                    target_indices_override=_target_override,
                    adaptive_temperature=(
                        str(self._config.get("targeted_sa_temperature_mode", "static"))
                        == "adaptive"
                    ),
                    adaptive_num_trials=int(
                        self._config.get("targeted_sa_adaptive_num_trials", 64)
                    ),
                    adaptive_target_accept=float(
                        self._config.get("targeted_sa_adaptive_target_accept", 0.5)
                    ),
                )
                for candidate_idx, candidate in enumerate(escape_candidates):
                    scored = _score_legalized_candidate(
                        raw_positions=candidate.positions,
                        benchmark=benchmark,
                        plc=plc,
                        stats={
                            "restart_idx": -3,
                            "mode": "targeted_sa_escape",
                            "candidate_kind": "targeted_sa_escape",
                            "candidate_idx": int(candidate_idx),
                            "source_idx": int(source_idx),
                            "escape_source_key": tuple(source_candidate.key),
                            "sa_objective": float(candidate.objective),
                            "sa_proxy_cost": float(candidate.proxy_cost),
                            "sa_overlap_count": int(candidate.overlap_count),
                            "sa_evaluations": int(candidate.evaluations),
                            "sa_accepted_moves": int(candidate.accepted_moves),
                        },
                    )
                    candidates.append(scored)
                    scored_escape_candidates.append(scored)
            self._last_run_stats["targeted_sa_escape_candidates"] = int(
                len(scored_escape_candidates)
            )
            self._last_run_stats["targeted_sa_source_top_k"] = sa_source_top_k
            self._last_run_stats["targeted_sa_source_count"] = len(sa_sources)
            if scored_escape_candidates:
                self._last_run_stats["targeted_sa_escape_best_proxy"] = float(
                    min(candidate.key[1] for candidate in scored_escape_candidates)
                )
                # Keep the single-source "source_key" for back-compat with the
                # probe / auto-memory; it points at the first (proxy-best) source.
                self._last_run_stats["targeted_sa_escape_source_key"] = tuple(
                    sa_sources[0].key if sa_sources else proxy_best_candidate.key
                )
            escape_polish_budget_s = float(
                self._config["targeted_sa_escape_polish_time_budget_s"]
            )
            # Lever 6 gate 2: skip polish if SA candidates regressed badly vs
            # source. Default no-op (max_regression == math.inf).
            _sa_polish_skip = False
            if scored_escape_candidates and escape_polish_budget_s > 0.0:
                _sa_source_proxy = float(proxy_best_candidate.key[1])
                _sa_min_escape_proxy = float(
                    min(c.key[1] for c in scored_escape_candidates)
                )
                _sa_polish_skip = should_skip_targeted_sa_polish(
                    min_escape_proxy=_sa_min_escape_proxy,
                    source_proxy=_sa_source_proxy,
                    max_regression=float(
                        self._config["targeted_sa_max_proxy_regression_to_polish"]
                    ),
                )
                if _sa_polish_skip:
                    self._last_run_stats["targeted_sa_polish_gated_reason"] = "quality"
                    self._last_run_stats["targeted_sa_polish_gated_regression"] = (
                        _sa_min_escape_proxy - _sa_source_proxy
                    )
            if (
                scored_escape_candidates
                and escape_polish_budget_s > 0.0
                and not _sa_polish_skip
            ):
                candidates.extend(
                    _topk_final_polish(
                        candidates=scored_escape_candidates,
                        benchmark=benchmark,
                        plc=plc,
                        ctx=ctx,
                        cfg=self._config,
                        time_budget_s=escape_polish_budget_s,
                        run_stats=self._last_run_stats,
                    )
                )
            proxy_best_candidate = min(candidates, key=lambda candidate: candidate.key)

        # Hessian saddle escape (E12): try to escape coupled-macro saddles
        # invisible to single-macro CD. Accepts only on strict improvement.
        # Lever 5: when hessian_escape_top_k > 1, run on the top-K candidates
        # by key. Default (k=1) preserves the proven single-shot behavior.
        hessian_top_k = int(self._config.get("hessian_escape_top_k", 1))
        hessian_sources = _top_k_candidates_by_key(candidates, hessian_top_k)
        hessian_diagnostics_all: list[dict[str, Any]] = []
        hessian_accepted_count = 0
        first_accepted: _FinalCandidate | None = None
        first_accepted_source = None
        for source in hessian_sources:
            diag: dict[str, Any] = {}
            new_c = _hessian_escape_polish_candidate(
                source,
                benchmark,
                plc,
                ctx,
                self._config,
                diagnostics=diag,
            )
            hessian_diagnostics_all.append(diag)
            if new_c is not None:
                candidates.append(new_c)
                hessian_accepted_count += 1
                if first_accepted is None:
                    first_accepted = new_c
                    first_accepted_source = source
        # Stats — preserve the single-slot keys for backwards compatibility
        # (probe scripts + auto-memory still read these). Diagnostics dict
        # is the first one; full list is in *_all.
        self._last_run_stats["hessian_diagnostics"] = (
            hessian_diagnostics_all[0] if hessian_diagnostics_all else {}
        )
        self._last_run_stats["hessian_diagnostics_all"] = hessian_diagnostics_all
        self._last_run_stats["hessian_escape_top_k"] = hessian_top_k
        self._last_run_stats["hessian_escape_accepted_count"] = hessian_accepted_count
        self._last_run_stats["hessian_escape_accepted"] = hessian_accepted_count > 0
        if first_accepted is not None and first_accepted_source is not None:
            self._last_run_stats["hessian_escape_source_key"] = list(
                first_accepted_source.key
            )
            self._last_run_stats["hessian_escape_new_key"] = list(first_accepted.key)
            self._last_run_stats["hessian_escape_source"] = (
                first_accepted.stats.get("hessian_escape_source")
            )
        # Re-pick proxy-best from the (possibly extended) candidate pool. For
        # k=1 this matches the prior `proxy_best_candidate = hessian_candidate`
        # assignment because hessian only returns on strict improvement.
        proxy_best_candidate = min(candidates, key=lambda c: c.key)

        spacing_candidate = _orfs_spacing_polish_candidate(
            proxy_best_candidate,
            benchmark,
            plc,
            self._config,
        )
        if spacing_candidate is not None:
            candidates.append(spacing_candidate)

        guard_candidate = _orfs_guard_repair_candidate(
            proxy_best_candidate,
            benchmark,
            plc,
            self._config,
        )
        if guard_candidate is not None:
            candidates.append(guard_candidate)

        best_candidate, selection_stats = _select_final_candidate(
            candidates,
            benchmark,
            self._config,
        )
        self._last_run_stats["candidate_summary"] = _summarize_final_candidates(
            candidates
        )
        self._last_run_stats["selected_key"] = best_candidate.key
        self._last_run_stats["selected_restart"] = dict(best_candidate.stats)
        self._last_run_stats["orfs_final_selection"] = selection_stats

        # Lever C Step G — sync plc orientations to the SELECTED candidate so
        # any downstream scoring (polish, eval) sees its orientations. Without
        # this, the last candidate _score_legalized_candidate touched is what
        # plc remembers.
        try:
            from macro_place.orientation import orientation_name
            hard_idx = list(getattr(benchmark, "hard_macro_indices", []))
            if best_candidate.orientations is not None:
                arr = np.asarray(best_candidate.orientations, dtype=np.int8)
                for i in range(min(len(hard_idx), arr.shape[0])):
                    ori_idx = int(arr[i])
                    if 0 <= ori_idx < 8:
                        plc.update_macro_orientation(
                            int(hard_idx[i]), orientation_name(ori_idx)
                        )
            else:
                for plc_idx in hard_idx:
                    plc.update_macro_orientation(int(plc_idx), "N")
        except Exception:
            pass

        # Lever C — Rotation polish (gated). Operates on the selected
        # candidate's positions; mutates plc orientations in place so the
        # next compute_proxy_cost reflects the polished orientations.
        if bool(self._config.get("rotation_polish_enabled", False)):
            try:
                ori_state = build_orientation_state(ctx, plc, benchmark)
                pos_np_for_polish = (
                    best_candidate.legalized_positions
                    .detach().cpu().numpy().astype(np.float64, copy=True)
                )
                top_k = int(self._config.get("rotation_polish_top_k", 0)) or len(
                    getattr(benchmark, "hard_macro_indices", [])
                )
                polish = polish_orientations_fast(
                    positions_np=pos_np_for_polish,
                    ctx=ctx,
                    state=ori_state,
                    benchmark=benchmark,
                    plc=plc,
                    top_k=top_k,
                )
                self._last_run_stats["rotation_polish_enabled"] = True
                self._last_run_stats["rotation_polish_improved_count"] = int(
                    polish["improved_count"]
                )
                self._last_run_stats["rotation_polish_initial_proxy"] = float(
                    polish["initial_proxy"]
                )
                self._last_run_stats["rotation_polish_final_proxy"] = float(
                    polish["final_proxy"]
                )
                self._last_run_stats["rotation_polish_official_proxy"] = float(
                    polish["final_official_proxy"]
                )
            except Exception as exc:
                self._last_run_stats["rotation_polish_error"] = (
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )

        self._last_run_stats["tier2_metrics"] = _tier2_metrics(
            best_candidate.legalized_positions,
            benchmark,
            initial_pos,
        )
        return best_candidate.legalized_positions


def _score_legalized_candidate(
    *,
    raw_positions: np.ndarray,
    benchmark: Benchmark,
    plc: Any,
    stats: dict[str, Any],
    orientations: np.ndarray | None = None,
) -> _FinalCandidate:
    raw_copy = np.asarray(raw_positions, dtype=np.float64).copy()
    legalized = repair_overlaps(
        torch.as_tensor(raw_copy, dtype=torch.float32), benchmark
    )
    # Lever C Step G — sync plc orientations before scoring. If orientations
    # provided, apply them; otherwise reset to all-N (the default state) so
    # scoring is consistent regardless of prior plc mutations.
    try:
        from macro_place.orientation import orientation_name
        hard_idx = list(getattr(benchmark, "hard_macro_indices", []))
        if orientations is not None:
            arr = np.asarray(orientations, dtype=np.int8)
            for i in range(min(len(hard_idx), arr.shape[0])):
                ori_idx = int(arr[i])
                if 0 <= ori_idx < 8:
                    plc.update_macro_orientation(
                        int(hard_idx[i]), orientation_name(ori_idx)
                    )
        else:
            # Reset to N to avoid leaking prior candidate's orientations.
            for plc_idx in hard_idx:
                plc.update_macro_orientation(int(plc_idx), "N")
    except Exception:
        pass
    cost = dict(compute_proxy_cost(legalized, benchmark, plc))
    key = (int(cost["overlap_count"]), float(cost["proxy_cost"]))
    ori_copy = (
        np.asarray(orientations, dtype=np.int8).copy()
        if orientations is not None
        else None
    )
    return _FinalCandidate(
        raw_positions=raw_copy,
        legalized_positions=legalized,
        key=key,
        cost=cost,
        stats=dict(stats),
        orientations=ori_copy,
    )


def _summarize_final_candidates(
    candidates: list[_FinalCandidate],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_by_source: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(candidates):
        source = _candidate_source(candidate.stats)
        row = {
            "idx": int(idx),
            "source": source,
            "mode": str(candidate.stats.get("mode", "unknown")),
            "candidate_kind": str(candidate.stats.get("candidate_kind", source)),
            "restart_idx": int(candidate.stats.get("restart_idx", -99)),
            "overlap_count": int(candidate.key[0]),
            "proxy_cost": float(candidate.key[1]),
            "raw_proxy_cost": float(
                candidate.stats.get("sa_proxy_cost", candidate.key[1])
            ),
        }
        rows.append(row)
        current = best_by_source.get(source)
        if current is None or (row["overlap_count"], row["proxy_cost"]) < (
            int(current["overlap_count"]),
            float(current["proxy_cost"]),
        ):
            best_by_source[source] = dict(row)

    rows.sort(key=lambda row: (int(row["overlap_count"]), float(row["proxy_cost"])))
    return {
        "candidate_count": int(len(candidates)),
        "best_by_source": best_by_source,
        "top_candidates": rows[: min(12, len(rows))],
    }


def _candidate_source(stats: dict[str, Any]) -> str:
    kind = str(stats.get("candidate_kind", ""))
    if kind:
        return kind
    mode = str(stats.get("mode", ""))
    if mode:
        return mode
    restart_idx = stats.get("restart_idx")
    return f"restart_{restart_idx}" if restart_idx is not None else "unknown"


def _proxy_tie_limit(best_proxy: float, rel_tol: float) -> float:
    return float(best_proxy) + max(abs(float(best_proxy)) * float(rel_tol), 1e-12)


def _orfs_candidate_sort_key(
    metrics: dict[str, Any],
    candidate: _FinalCandidate,
) -> tuple[Any, ...]:
    if not metrics.get("orfs_post_clamp_available", False):
        return (1, int(candidate.key[0]), float(candidate.key[1]))
    min_clearance = metrics.get("post_clamp_min_clearance_um")
    return (
        0,
        int(metrics.get("post_clamp_overlap_count", 0)),
        int(metrics.get("post_clamp_clearance_lt_5um_count", 0)),
        int(metrics.get("post_clamp_narrow_channel_lt_12um_count", 0)),
        int(metrics.get("post_clamp_clearance_lt_12um_count", 0)),
        -float(min_clearance if min_clearance is not None else 0.0),
        int(metrics.get("core_clamp_moved_macro_count", 0)),
        float(candidate.key[1]),
    )


def _select_final_candidate(
    candidates: list[_FinalCandidate],
    benchmark: Benchmark,
    cfg: dict[str, Any],
) -> tuple[_FinalCandidate, dict[str, Any]]:
    proxy_best = min(candidates, key=lambda candidate: candidate.key)
    if not bool(cfg.get("orfs_tiebreak_enabled", False)):
        return proxy_best, {
            "enabled": False,
            "candidate_count": int(len(candidates)),
            "tie_pool_size": 1,
            "selected_by_orfs_tiebreak": False,
        }

    rel_tol = float(cfg.get("orfs_proxy_tie_rel_tol", 0.0))
    proxy_limit = _proxy_tie_limit(float(proxy_best.key[1]), rel_tol)
    tie_pool = [
        candidate
        for candidate in candidates
        if int(candidate.key[0]) == int(proxy_best.key[0])
        and float(candidate.key[1]) <= proxy_limit
    ]
    if not tie_pool:
        tie_pool = [proxy_best]

    def metrics_for(candidate: _FinalCandidate) -> dict[str, Any]:
        return _orfs_post_clamp_metrics(
            candidate.legalized_positions,
            benchmark,
            clearance_threshold_um=float(
                cfg.get("orfs_clearance_threshold_um", 12.0)
            ),
            core_margin_um=float(cfg.get("orfs_core_margin_um", 12.0)),
        )

    metrics_by_idx = {id(candidate): metrics_for(candidate) for candidate in tie_pool}
    proxy_best_metrics = metrics_by_idx.get(id(proxy_best))
    if proxy_best_metrics is None:
        proxy_best_metrics = metrics_for(proxy_best)

    repair_pool_size = 0
    proxy_best_post_clamp_overlaps = int(
        proxy_best_metrics.get("post_clamp_overlap_count", 0)
    )
    if proxy_best_post_clamp_overlaps > 0:
        repair_rel_tol = float(
            cfg.get("orfs_overlap_repair_proxy_rel_tol", rel_tol)
        )
        repair_limit = _proxy_tie_limit(float(proxy_best.key[1]), repair_rel_tol)
        tie_ids = {id(candidate) for candidate in tie_pool}
        for candidate in candidates:
            if id(candidate) in tie_ids:
                continue
            if int(candidate.key[0]) != int(proxy_best.key[0]):
                continue
            if float(candidate.key[1]) > repair_limit:
                continue
            metrics = metrics_for(candidate)
            if (
                int(metrics.get("post_clamp_overlap_count", 0))
                < proxy_best_post_clamp_overlaps
            ):
                tie_pool.append(candidate)
                tie_ids.add(id(candidate))
                metrics_by_idx[id(candidate)] = metrics
                repair_pool_size += 1

    selected = min(
        tie_pool,
        key=lambda candidate: _orfs_candidate_sort_key(
            metrics_by_idx[id(candidate)], candidate
        ),
    )
    selected_metrics = metrics_by_idx[id(selected)]

    return selected, {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "tie_pool_size": int(len(tie_pool)),
        "overlap_repair_pool_size": int(repair_pool_size),
        "proxy_tie_rel_tol": float(rel_tol),
        "overlap_repair_proxy_rel_tol": float(
            cfg.get("orfs_overlap_repair_proxy_rel_tol", rel_tol)
        ),
        "proxy_best_key": tuple(proxy_best.key),
        "selected_key": tuple(selected.key),
        "selected_by_orfs_tiebreak": bool(selected is not proxy_best),
        "selected_orfs_metrics": selected_metrics,
        "proxy_best_orfs_metrics": proxy_best_metrics,
    }


def _orfs_guard_repair_positions(
    positions: torch.Tensor,
    benchmark: Benchmark,
    cfg: dict[str, Any],
) -> torch.Tensor:
    repaired = positions.detach().cpu().to(torch.float32)
    core_margin_um = float(cfg.get("orfs_core_margin_um", 12.0))
    legalize_iters = int(cfg.get("orfs_guard_repair_legalize_iters", 500))
    guard_iters = max(1, int(cfg.get("orfs_guard_repair_iters", 1)))
    half = _hard_macro_sizes(benchmark) * 0.5
    num_hard = int(benchmark.num_hard_macros)

    for _ in range(guard_iters):
        pos = _to_numpy_positions(repaired)
        clamped, _ = _orfs_clamped_positions(
            pos[:num_hard],
            half,
            benchmark,
            core_margin_um,
        )
        pos[:num_hard] = clamped
        repaired = repair_overlaps(
            torch.as_tensor(pos, dtype=torch.float32),
            benchmark,
            max_iters=legalize_iters,
        )
        metrics = _orfs_post_clamp_metrics(
            repaired,
            benchmark,
            clearance_threshold_um=float(
                cfg.get("orfs_clearance_threshold_um", 12.0)
            ),
            core_margin_um=core_margin_um,
        )
        if int(metrics.get("post_clamp_overlap_count", 0)) == 0:
            break

    return repaired


def _orfs_guard_repair_candidate(
    candidate: _FinalCandidate,
    benchmark: Benchmark,
    plc: Any,
    cfg: dict[str, Any],
) -> _FinalCandidate | None:
    if not bool(cfg.get("orfs_guard_repair_enabled", False)):
        return None

    base_metrics = _orfs_post_clamp_metrics(
        candidate.legalized_positions,
        benchmark,
        clearance_threshold_um=float(cfg.get("orfs_clearance_threshold_um", 12.0)),
        core_margin_um=float(cfg.get("orfs_core_margin_um", 12.0)),
    )
    if not base_metrics.get("orfs_post_clamp_available", False):
        return None
    if int(base_metrics.get("post_clamp_overlap_count", 0)) == 0:
        return None

    repaired = _orfs_guard_repair_positions(
        candidate.legalized_positions,
        benchmark,
        cfg,
    )
    cost = dict(compute_proxy_cost(repaired, benchmark, plc))
    return _FinalCandidate(
        raw_positions=_to_numpy_positions(repaired),
        legalized_positions=repaired,
        key=(int(cost["overlap_count"]), float(cost["proxy_cost"])),
        cost=cost,
        stats={
            **candidate.stats,
            "candidate_kind": "orfs_guard_repair",
            "source_key": tuple(candidate.key),
            "orfs_guard_base_metrics": base_metrics,
        },
    )


def _orfs_push_pair_axis(
    pos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    movable: np.ndarray,
    i: int,
    j: int,
    axis: int,
    needed: float,
) -> bool:
    if needed <= 1e-6 or (not movable[i] and not movable[j]):
        return False
    sign = 1.0 if float(pos[j, axis]) >= float(pos[i, axis]) else -1.0
    remaining = float(needed)
    moved = 0.0
    if movable[i]:
        step = min(remaining, needed * 0.5)
        moved_i = _orfs_move_axis(pos, lower, upper, i, axis, -sign * step)
        remaining = max(0.0, remaining - moved_i)
        moved += moved_i
    if movable[j] and remaining > 1e-6:
        moved_j = _orfs_move_axis(pos, lower, upper, j, axis, sign * remaining)
        remaining = max(0.0, remaining - moved_j)
        moved += moved_j
    if movable[i] and remaining > 1e-6:
        moved += _orfs_move_axis(pos, lower, upper, i, axis, -sign * remaining)
    return moved > 1e-6


def _orfs_spacing_polish_positions(
    positions: torch.Tensor,
    benchmark: Benchmark,
    cfg: dict[str, Any],
) -> torch.Tensor:
    target_um = max(0.0, float(cfg.get("orfs_spacing_polish_target_um", 0.0)))
    if target_um <= 0.0:
        return positions.detach().cpu().to(torch.float32)

    repaired = _to_numpy_positions(positions)
    num_hard = int(benchmark.num_hard_macros)
    half = _hard_macro_sizes(benchmark) * 0.5
    hard, _ = _orfs_clamped_positions(
        repaired[:num_hard],
        half,
        benchmark,
        float(cfg.get("orfs_core_margin_um", 12.0)),
    )
    lower, upper = _orfs_core_bounds(
        half,
        benchmark,
        float(cfg.get("orfs_core_margin_um", 12.0)),
    )
    if hasattr(benchmark, "get_movable_mask"):
        movable = (
            benchmark.get_movable_mask()[:num_hard]
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
        )
    else:
        movable = np.ones(num_hard, dtype=np.bool_)

    narrow_um = max(0.0, float(cfg.get("orfs_spacing_polish_narrow_um", 0.0)))

    for _ in range(max(1, int(cfg.get("orfs_spacing_polish_iters", 1)))):
        changed = False
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                gap_x = abs(float(hard[i, 0] - hard[j, 0])) - float(
                    half[i, 0] + half[j, 0]
                )
                gap_y = abs(float(hard[i, 1] - hard[j, 1])) - float(
                    half[i, 1] + half[j, 1]
                )
                # Case A — bbox overlap on both axes: push apart by target_um.
                if gap_x < 0.0 and gap_y < 0.0:
                    need_x = target_um - gap_x
                    need_y = target_um - gap_y
                    if need_x <= 1e-6 and need_y <= 1e-6:
                        continue
                    axis = 0 if need_x <= need_y else 1
                    needed = need_x if axis == 0 else need_y
                    changed |= _orfs_push_pair_axis(
                        hard, lower, upper, movable, i, j, axis, needed
                    )
                    continue
                # Case B — y-overlap (side-by-side macros), narrow vertical
                # channel on the x axis. Widen x to narrow_um.
                if narrow_um > 0.0 and gap_y < 0.0 and 0.0 <= gap_x < narrow_um:
                    needed = narrow_um - gap_x
                    if needed > 1e-6:
                        changed |= _orfs_push_pair_axis(
                            hard, lower, upper, movable, i, j, 0, needed
                        )
                    continue
                # Case C — x-overlap (stacked macros), narrow horizontal
                # channel on the y axis. Widen y to narrow_um.
                if narrow_um > 0.0 and gap_x < 0.0 and 0.0 <= gap_y < narrow_um:
                    needed = narrow_um - gap_y
                    if needed > 1e-6:
                        changed |= _orfs_push_pair_axis(
                            hard, lower, upper, movable, i, j, 1, needed
                        )
                    continue
        if not changed:
            break

    repaired[:num_hard] = hard
    return torch.as_tensor(repaired, dtype=torch.float32)


def _hessian_escape_polish_candidate(
    candidate: _FinalCandidate,
    benchmark: Benchmark,
    plc: Any,
    ctx: Any,
    cfg: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> _FinalCandidate | None:
    """Hessian saddle escape (E12) — block-diag + Lanczos random-subspace
    Rayleigh-Ritz. Targets coupled-macro saddles invisible to single-macro
    CD. Accepts only if ``(overlap_count, proxy_cost)`` strictly improves.

    Implementation notes vs Bet 6 failure modes (saddle.py):
      - F1 (random dirs miss good direction): eigenvector IS the
        analytically optimal escape direction.
      - F2 (first-order picking misses curvature): we measure curvature
        directly via finite differences.
      - F3 (cd_loop polish wipes escape): no polish here — accept the
        raw line-searched eigenvector step.
    """
    if not bool(cfg.get("hessian_escape_enabled", False)):
        return None
    if not hasattr(benchmark, "num_hard_macros"):
        return None
    if not hasattr(benchmark, "macro_sizes"):
        return None
    # Defensive: tests pass a mock ctx; skip cleanly if it lacks the fields
    # fast_proxy needs.
    if not hasattr(ctx, "pin_macro_idx"):
        return None
    num_hard = int(benchmark.num_hard_macros)
    if num_hard < 1:
        return None

    full_positions = _to_numpy_positions(candidate.legalized_positions)
    half = _hard_macro_sizes(benchmark) * 0.5
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    # eval_fn scores hard-macro positions through fast_proxy. Soft macros
    # remain fixed at their current positions.
    soft_positions = full_positions[num_hard:].astype(np.float64, copy=True)

    def _eval_fn(hard_pos: np.ndarray) -> float:
        full = np.empty_like(full_positions)
        full[:num_hard] = hard_pos
        if full.shape[0] > num_hard:
            full[num_hard:] = soft_positions
        return float(fast_proxy(full, ctx).proxy_cost)

    hard_in = full_positions[:num_hard].astype(np.float64, copy=True)
    new_hard, accepted, escape_stats = hessian_escape(
        positions=hard_in,
        eval_fn=_eval_fn,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        half_sizes=half,
        h_block=float(cfg.get("hessian_escape_h_block", 0.5)),
        h_lanczos=float(cfg.get("hessian_escape_h_lanczos", 0.1)),
        lanczos_max_iters=int(cfg.get("hessian_escape_lanczos_iters", 16)),
        curvature_threshold=float(cfg.get("hessian_escape_curvature_threshold", -1e-3)),
        line_search_alphas=tuple(
            cfg.get("hessian_escape_line_search_alphas", (1.0, 2.0, 4.0, 8.0))
        ),
        tolerance=float(cfg.get("hessian_escape_tolerance", 1e-4)),
        rng_seed=int(cfg.get("hessian_escape_seed", 0)),
    )

    # Always capture the eigenvalue diagnostics — even on reject — so we
    # can tell whether the placement was at a true minimum (eigval ≥ 0)
    # vs a saddle that we couldn't escape (eigval < 0, line search failed).
    if diagnostics is not None:
        diagnostics["hessian_block_diag_eigenvalue"] = escape_stats.get(
            "block_diag_eigenvalue"
        )
        diagnostics["hessian_lanczos_eigenvalue"] = escape_stats.get(
            "lanczos_eigenvalue"
        )
        diagnostics["hessian_source"] = escape_stats.get("source")
        diagnostics["hessian_reason"] = escape_stats.get("reason")
        diagnostics["hessian_f0"] = escape_stats.get("f0")
        diagnostics["hessian_f_best"] = escape_stats.get("f_best")
        diagnostics["hessian_best_alpha"] = escape_stats.get("best_alpha")

    if not accepted:
        return None

    # Legalize the hessian-stepped positions through the standard
    # _score_legalized_candidate path so an overlap induced by the step
    # gets repaired before final scoring (otherwise compute_proxy_cost
    # rejects every saddle escape that nudges a macro into a neighbor).
    polished_full = full_positions.copy()
    polished_full[:num_hard] = new_hard
    raw_positions = _to_numpy_positions(
        torch.as_tensor(polished_full, dtype=torch.float32)
    )
    polished_candidate = _score_legalized_candidate(
        raw_positions=raw_positions,
        benchmark=benchmark,
        plc=plc,
        stats={
            **candidate.stats,
            "candidate_kind": "hessian_escape",
            "source_key": tuple(candidate.key),
            "hessian_escape_source": escape_stats.get("source"),
            "hessian_escape_eigenvalue": escape_stats.get("eigenvalue"),
            "hessian_escape_best_alpha": escape_stats.get("best_alpha"),
            "hessian_escape_reason": escape_stats.get("reason"),
            "hessian_escape_fast_f0": escape_stats.get("f0"),
            "hessian_escape_fast_f_best": escape_stats.get("f_best"),
        },
    )
    if polished_candidate.key >= candidate.key:
        return None
    return polished_candidate

def _orfs_metrics_rank(metrics: dict[str, Any]) -> tuple[int, int, int, int, float]:
    min_clearance = metrics.get("post_clamp_min_clearance_um")
    return (
        int(metrics.get("post_clamp_overlap_count", 0)),
        int(metrics.get("post_clamp_clearance_lt_5um_count", 0)),
        int(metrics.get("post_clamp_narrow_channel_lt_12um_count", 0)),
        int(metrics.get("post_clamp_clearance_lt_12um_count", 0)),
        -float(min_clearance if min_clearance is not None else 0.0),
    )


def _orfs_spacing_polish_candidate(
    candidate: _FinalCandidate,
    benchmark: Benchmark,
    plc: Any,
    cfg: dict[str, Any],
) -> _FinalCandidate | None:
    if not bool(cfg.get("orfs_spacing_polish_enabled", False)):
        return None

    base_metrics = _orfs_post_clamp_metrics(
        candidate.legalized_positions,
        benchmark,
        clearance_threshold_um=float(cfg.get("orfs_clearance_threshold_um", 12.0)),
        core_margin_um=float(cfg.get("orfs_core_margin_um", 12.0)),
    )
    if not base_metrics.get("orfs_post_clamp_available", False):
        return None
    # Trigger polish when either close-overlap pairs OR narrow channels exist,
    # so the PDN-safe widening fires even on placements without near-overlaps.
    if (
        int(base_metrics.get("post_clamp_clearance_lt_5um_count", 0)) == 0
        and int(base_metrics.get("post_clamp_narrow_channel_lt_12um_count", 0)) == 0
    ):
        return None

    polished = _orfs_spacing_polish_positions(
        candidate.legalized_positions,
        benchmark,
        cfg,
    )
    # Re-legalize: widening narrow channels (cases B/C in the polish) can
    # create bbox overlaps elsewhere. Without this re-pass, polished_metrics
    # has overlap_count > 0 and the rank check below silently rejects the
    # candidate — leaving narrow channels in the final placement and crashing
    # ORFS PDN. See PDN-0179 incident on ariane133 2026-05-18.
    polished = repair_overlaps(polished, benchmark)
    polished_metrics = _orfs_post_clamp_metrics(
        polished,
        benchmark,
        clearance_threshold_um=float(cfg.get("orfs_clearance_threshold_um", 12.0)),
        core_margin_um=float(cfg.get("orfs_core_margin_um", 12.0)),
    )
    if _orfs_metrics_rank(polished_metrics) >= _orfs_metrics_rank(base_metrics):
        return None

    cost = dict(compute_proxy_cost(polished, benchmark, plc))
    polished_key_real = (int(cost["overlap_count"]), float(cost["proxy_cost"]))
    # Rank the polished placement by its actual proxy. Reusing the source
    # candidate's key can make a high-cost polish win final selection under a
    # stale low proxy.
    return _FinalCandidate(
        raw_positions=_to_numpy_positions(polished),
        legalized_positions=polished,
        key=polished_key_real,
        cost=cost,
        stats={
            **candidate.stats,
            "candidate_kind": "orfs_spacing_polish",
            "source_key": tuple(candidate.key),
            "polished_proxy_key": polished_key_real,
            "orfs_spacing_base_metrics": base_metrics,
            "orfs_spacing_polished_metrics": polished_metrics,
        },
    )


def _topk_polish_budget(
    cfg: dict[str, Any],
) -> float:
    if not bool(cfg.get("topk_polish_enabled", False)):
        return 0.0
    return max(0.0, float(cfg.get("topk_polish_time_budget_s", 0.0)))


def _topk_final_polish(
    *,
    candidates: list[_FinalCandidate],
    benchmark: Benchmark,
    plc: Any,
    ctx: Any,
    cfg: dict[str, Any],
    time_budget_s: float,
    run_stats: dict[str, Any],
) -> list[_FinalCandidate]:
    if not bool(cfg.get("topk_polish_enabled", False)):
        return []
    k = max(0, int(cfg.get("topk_polish_k", 0)))
    if k <= 0 or time_budget_s <= 0.0 or not candidates:
        return []

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    selected_bases = sorted(candidates, key=lambda candidate: candidate.key)[:k]
    polished_candidates: list[_FinalCandidate] = []
    remaining_s = float(time_budget_s)

    for polish_idx, base in enumerate(selected_bases):
        if remaining_s <= 0.0:
            break
        base_stats = base.stats
        source_restart_idx = int(base_stats.get("restart_idx", -1))
        source_mode = str(base_stats.get("mode", "unknown"))
        attempts_left = max(1, len(selected_bases) - polish_idx)
        polish_budget_s = max(0.0, remaining_s / attempts_left)
        if polish_budget_s <= 0.0:
            break

        run_stats["topk_polish_attempts"] += 1
        start = time.perf_counter()
        work_positions = (
            base.legalized_positions.detach().cpu().numpy().astype(
                np.float64,
                copy=True,
            )
        )
        max_sweeps = int(cfg["topk_polish_sweeps"])
        radius_init_ratio = float(cfg["topk_polish_radius_init_ratio"])
        radius_min_ratio = float(cfg["topk_polish_radius_min_ratio"])
        base_seed = 10_000 + polish_idx
        total_evals = 0
        sweeps_completed = 0
        last_surrogate_cost = float(base.key[1])
        intermediate_count = 0
        best_polished: _FinalCandidate | None = None

        for sweep_idx in range(max_sweeps):
            elapsed_s = time.perf_counter() - start
            sweep_budget_s = max(0.0, polish_budget_s - elapsed_s)
            if sweep_budget_s <= 0.0:
                break
            decay_steps = sweep_idx // 4
            sweep_radius_ratio = max(
                radius_min_ratio,
                radius_init_ratio * (0.5 ** decay_steps),
            )
            cd_result = cd_loop(
                initial_positions=work_positions,
                ctx=ctx,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                max_sweeps=1,
                k_per_axis=int(cfg["topk_polish_k_per_axis"]),
                radius_init_ratio=sweep_radius_ratio,
                radius_min_ratio=radius_min_ratio,
                time_budget_s=sweep_budget_s,
                seed=base_seed + sweep_idx,
                tiebreak_enabled=bool(
                    cfg.get("cd_congestion_tiebreak_enabled", False)
                ),
                tie_epsilon_rel=float(
                    cfg.get("cd_congestion_tiebreak_epsilon", 1e-3)
                ),
            )
            total_evals += int(cd_result.total_evals)
            sweeps_completed += int(cd_result.sweeps_completed)
            last_surrogate_cost = float(cd_result.final_cost)
            if cd_result.sweeps_completed <= 0:
                break
            work_positions = cd_result.positions.copy()

            polished = _score_legalized_candidate(
                raw_positions=work_positions,
                benchmark=benchmark,
                plc=plc,
                stats={
                    **base_stats,
                    "candidate_kind": "topk_polish",
                    "source_restart_idx": source_restart_idx,
                    "source_mode": source_mode,
                    "topk_polish_sweep": int(sweep_idx + 1),
                },
            )
            intermediate_count += 1
            if best_polished is None or polished.key < best_polished.key:
                best_polished = polished
            polished_candidates.append(
                _FinalCandidate(
                    raw_positions=polished.raw_positions,
                    legalized_positions=polished.legalized_positions,
                    key=polished.key,
                    cost=polished.cost,
                    stats={
                        **polished.stats,
                        "topk_polish_accepted": bool(polished.key < base.key),
                    },
                )
            )
            if cd_result.plateaued:
                break

        runtime_s = float(time.perf_counter() - start)
        remaining_s = max(0.0, remaining_s - runtime_s)

        accepted = best_polished is not None and best_polished.key < base.key
        if accepted:
            run_stats["topk_polish_accepts"] += 1
        event = {
            "source_restart_idx": source_restart_idx,
            "source_mode": source_mode,
            "base_proxy_cost": float(base.key[1]),
            "polished_proxy_cost": (
                float(best_polished.key[1]) if best_polished is not None else float(base.key[1])
            ),
            "base_overlap_count": int(base.key[0]),
            "polished_overlap_count": (
                int(best_polished.key[0]) if best_polished is not None else int(base.key[0])
            ),
            "accepted": bool(accepted),
            "runtime_s": runtime_s,
            "sweeps_completed": int(sweeps_completed),
            "total_evals": int(total_evals),
            "surrogate_final_cost": float(last_surrogate_cost),
            "seed": int(base_seed),
            "intermediate_candidates": int(intermediate_count),
        }
        run_stats["topk_polish_events"].append(event)

    return polished_candidates


def _phase_budget(
    *,
    remaining_s: float,
    phase_cap_s: float,
    reserve_s: float,
    reserve_enabled: bool,
) -> float:
    if not reserve_enabled:
        return max(0.0, remaining_s)
    usable_s = max(0.0, remaining_s - max(0.0, reserve_s))
    return min(max(0.0, phase_cap_s), usable_s)


def _restart_worker(
    bench_name: str,
    bench_path: str,
    seed: int,
    time_budget_s: float,
    config: dict[str, Any],
    restart_idx: int = 0,
) -> tuple[np.ndarray, float, dict[str, Any], np.ndarray | None]:
    """Module-level worker for ProcessPool — picklable, builds its own
    benchmark + plc + ctx so nothing is shared across processes.

    Step G: returns a 4-tuple including a per-macro orientation index array
    (or None when CD orientation search wasn't used).
    """
    benchmark = Benchmark.load(bench_path)
    plc = resolve_plc(benchmark)
    if plc is None:
        raise RuntimeError(f"resolve_plc None in worker for {bench_name}")
    ctx = build_fast_proxy_context(plc, benchmark)

    # Re-create a placer to reuse _run_one_restart
    placer = CDLNSPlacer()
    placer._config = dict(config)
    pos, surrogate_cost, orientations = placer._run_one_restart(
        benchmark=benchmark, ctx=ctx, plc=plc, seed=seed,
        time_budget_s=time_budget_s, restart_idx=restart_idx,
    )
    return pos, surrogate_cost, dict(placer._last_restart_stats), orientations
