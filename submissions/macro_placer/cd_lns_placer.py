"""CD+LNS placer entry (Bet 7 clean-restart submission).

A zero-arg-constructor placer that runs a single CD-LNS-multi-restart
loop on the surrogate proxy and returns the best-of-restart placement
rescored with the official TILOS evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from macro_place.hessian_escape import block_diag_top_saddle_macros, hessian_escape


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
    "orfs_core_margin_um": 12.0,
    "orfs_clearance_threshold_um": 12.0,
    "orfs_guard_repair_enabled": True,
    "orfs_guard_repair_iters": 16,
    "orfs_guard_repair_legalize_iters": 500,
    "orfs_spacing_polish_enabled": True,
    "orfs_spacing_polish_iters": 8,
    "orfs_spacing_polish_target_um": 2.0,
    # Hessian saddle escape (E12) — block-diag + Lanczos random-subspace
    # Rayleigh-Ritz. Designed to find coupled-macro saddles invisible to
    # single-macro CD. Plan + Bet-6 failure analysis in
    # docs/superpowers/plans/2026-05-16-hessian-saddle-escape-plan.md.
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
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _FinalCandidate:
    raw_positions: np.ndarray
    legalized_positions: torch.Tensor
    key: tuple[int, float]
    cost: dict[str, Any]
    stats: dict[str, Any]


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
    ) -> tuple[np.ndarray, float]:
        """One restart: CD until plateau, then LNS until repeated failure or time-out."""
        cfg = self._config
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        modes = cfg.get("restart_modes", ("aggressive",)) or ("aggressive",)
        mode = modes[restart_idx % len(modes)]
        if mode == "conservative":
            radius_init_ratio = 1.0 / 32.0
            max_sweeps = 5
            do_lns = False
            warm_sigma = 0.0
        elif mode == "light":
            radius_init_ratio = 1.0 / 16.0
            max_sweeps = 10
            do_lns = False
            warm_sigma = 0.02
        else:
            radius_init_ratio = float(cfg["radius_init_ratio"])
            max_sweeps = int(cfg["max_sweeps"])
            do_lns = True
            warm_sigma = float(cfg["warm_start_sigma"])
        cd_k_per_axis = (
            int(cfg["aggressive_cd_k_per_axis"]) if do_lns else int(cfg["k_per_axis"])
        )
        lns_k_per_axis = int(cfg["lns_k_per_axis"])

        pos = _warm_start_positions(benchmark, plc, seed=seed, sigma=warm_sigma)

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

            new_pos, accepted, _ = lns_destroy_rebuild(
                positions=best_pos,
                ctx=ctx,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                num_destroy=int(cfg["lns_num_destroy"]),
                max_lns_iters=int(cfg["lns_max_iters"]),
                k_per_axis=lns_k_per_axis,
                seed=seed + 100,
                destroy_seed_indices=hessian_destroy_seeds,
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
        return best_pos, best_cost

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = resolve_plc(benchmark)
        if plc is None:
            raise RuntimeError(f"resolve_plc returned None for {benchmark.name}")

        ctx = build_fast_proxy_context(plc, benchmark)
        self._last_fast_proxy_context = ctx

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
        results: list[tuple[np.ndarray, float, dict[str, Any]]] = [
            (initial_pos, 0.0, initial_stats)
        ]
        self._last_run_stats = {
            "benchmark": benchmark.name,
            "num_restarts": num_restarts,
            "per_restart_s": float(per_restart_s),
            "restarts": [],
            "topk_polish_enabled": bool(self._config["topk_polish_enabled"]),
            "topk_polish_attempts": 0,
            "topk_polish_accepts": 0,
            "topk_polish_events": [],
        }

        if num_restarts >= 1:
            if num_restarts == 1:
                pos, _ = self._run_one_restart(
                    benchmark=benchmark, ctx=ctx, plc=plc, seed=0,
                    time_budget_s=per_restart_s, restart_idx=0,
                )
                restart_stats = dict(self._last_restart_stats)
                results.append((pos, 0.0, restart_stats))
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
                        pos, _ = self._run_one_restart(
                            benchmark=benchmark, ctx=ctx, plc=plc, seed=seed,
                            time_budget_s=per_restart_s, restart_idx=seed,
                        )
                        restart_stats = dict(self._last_restart_stats)
                        results.append((pos, 0.0, restart_stats))
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
                            if len(item) == 2:
                                pos, surrogate = item
                                restart_stats = {}
                            else:
                                pos, surrogate, restart_stats = item
                            results.append((pos, surrogate, restart_stats))
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
            )
            for pos_arr, _surrogate, candidate_stats in results
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

        # Hessian saddle escape (E12): try to escape coupled-macro saddles
        # invisible to single-macro CD. Accepts only on strict improvement.
        # Pass a diagnostics dict so we capture eigenvalues even on reject.
        hessian_diag: dict[str, Any] = {}
        hessian_candidate = _hessian_escape_polish_candidate(
            proxy_best_candidate,
            benchmark,
            plc,
            ctx,
            self._config,
            diagnostics=hessian_diag,
        )
        self._last_run_stats["hessian_diagnostics"] = hessian_diag
        if hessian_candidate is not None:
            candidates.append(hessian_candidate)
            self._last_run_stats["hessian_escape_accepted"] = True
            self._last_run_stats["hessian_escape_source_key"] = list(
                proxy_best_candidate.key
            )
            self._last_run_stats["hessian_escape_new_key"] = list(
                hessian_candidate.key
            )
            self._last_run_stats["hessian_escape_source"] = hessian_candidate.stats.get(
                "hessian_escape_source"
            )
            proxy_best_candidate = hessian_candidate
        else:
            self._last_run_stats["hessian_escape_accepted"] = False

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
        self._last_run_stats["selected_key"] = best_candidate.key
        self._last_run_stats["selected_restart"] = dict(best_candidate.stats)
        self._last_run_stats["orfs_final_selection"] = selection_stats
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
) -> _FinalCandidate:
    raw_copy = np.asarray(raw_positions, dtype=np.float64).copy()
    legalized = repair_overlaps(
        torch.as_tensor(raw_copy, dtype=torch.float32), benchmark
    )
    cost = dict(compute_proxy_cost(legalized, benchmark, plc))
    key = (int(cost["overlap_count"]), float(cost["proxy_cost"]))
    return _FinalCandidate(
        raw_positions=raw_copy,
        legalized_positions=legalized,
        key=key,
        cost=cost,
        stats=dict(stats),
    )


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
                if gap_x >= 0.0 or gap_y >= 0.0:
                    continue
                need_x = target_um - gap_x
                need_y = target_um - gap_y
                if need_x <= 1e-6 and need_y <= 1e-6:
                    continue
                axis = 0 if need_x <= need_y else 1
                needed = need_x if axis == 0 else need_y
                changed |= _orfs_push_pair_axis(
                    hard, lower, upper, movable, i, j, axis, needed
                )
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
    if int(base_metrics.get("post_clamp_clearance_lt_5um_count", 0)) == 0:
        return None

    polished = _orfs_spacing_polish_positions(
        candidate.legalized_positions,
        benchmark,
        cfg,
    )
    polished_metrics = _orfs_post_clamp_metrics(
        polished,
        benchmark,
        clearance_threshold_um=float(cfg.get("orfs_clearance_threshold_um", 12.0)),
        core_margin_um=float(cfg.get("orfs_core_margin_um", 12.0)),
    )
    if _orfs_metrics_rank(polished_metrics) >= _orfs_metrics_rank(base_metrics):
        return None

    cost = dict(compute_proxy_cost(polished, benchmark, plc))
    return _FinalCandidate(
        raw_positions=_to_numpy_positions(polished),
        legalized_positions=polished,
        key=(int(cost["overlap_count"]), float(cost["proxy_cost"])),
        cost=cost,
        stats={
            **candidate.stats,
            "candidate_kind": "orfs_spacing_polish",
            "source_key": tuple(candidate.key),
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
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Module-level worker for ProcessPool — picklable, builds its own
    benchmark + plc + ctx so nothing is shared across processes."""
    benchmark = Benchmark.load(bench_path)
    plc = resolve_plc(benchmark)
    if plc is None:
        raise RuntimeError(f"resolve_plc None in worker for {bench_name}")
    ctx = build_fast_proxy_context(plc, benchmark)

    # Re-create a placer to reuse _run_one_restart
    placer = CDLNSPlacer()
    placer._config = dict(config)
    pos, surrogate_cost = placer._run_one_restart(
        benchmark=benchmark, ctx=ctx, plc=plc, seed=seed,
        time_budget_s=time_budget_s, restart_idx=restart_idx,
    )
    return pos, surrogate_cost, dict(placer._last_restart_stats)
