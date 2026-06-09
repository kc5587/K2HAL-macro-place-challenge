"""Simulated-annealing seed generator for basin-diverse macro placements."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from macro_place.fast_proxy import FastProxyContext, fast_proxy
from macro_place.fast_proxy_incremental import (
    apply_move,
    build_cache,
    cache_result,
    revert_move,
)
from macro_place.congestion_destroy import worst_congestion_bin_destroy_seeds
from macro_place.orientation import OrientationState, orientation_class_indices
from macro_place.orientation_cache import apply_rotation_to_cache


@dataclass(frozen=True)
class AnnealCandidate:
    positions: np.ndarray
    proxy_cost: float
    objective: float
    overlap_count: int
    evaluations: int
    accepted_moves: int


def generate_sa_candidates(
    *,
    initial_positions: np.ndarray,
    ctx: FastProxyContext,
    canvas_w: float,
    canvas_h: float,
    seed: int = 0,
    steps: int = 1_000,
    num_candidates: int = 4,
    initial_temperature_ratio: float = 0.03,
    final_temperature_ratio: float = 0.001,
    global_move_probability: float = 0.70,
    overlap_penalty: float = 0.02,
    diversity_distance_ratio: float = 0.03,
    max_overlap_count: int | None = None,
    exact_rescore_pool_size: int = 64,
    pre_legalize_iters: int = 0,
    target_indices: np.ndarray | None = None,
    adaptive_temperature: bool = False,
    adaptive_num_trials: int = 64,
    adaptive_target_accept: float = 0.5,
    orientation_state: OrientationState | None = None,
    rotation_probability: float = 0.0,
) -> list[AnnealCandidate]:
    """Generate basin-diverse seeds via nonlocal single-macro annealing.

    The generator deliberately uses large relocations most of the time. Its
    output is intended as extra seed material for CD/LNS polish, not as a final
    placement.
    """
    requested = int(num_candidates)
    if requested <= 0:
        return []

    work = np.asarray(initial_positions, dtype=np.float64).copy()
    if work.ndim != 2 or work.shape[1] != 2:
        raise ValueError("initial_positions must have shape [num_macros, 2]")
    if work.shape[0] == 0:
        return []

    rng = np.random.default_rng(seed)
    movable = _movable_indices(ctx, work.shape[0], target_indices)
    if movable.size == 0:
        return []

    _clamp_all_in_place(work, ctx, float(canvas_w), float(canvas_h))
    cache = build_cache(work, ctx)
    current = cache_result(cache)
    current_objective = _objective(current.proxy_cost, current.overlap_count, overlap_penalty)
    overlap_limit = (
        int(current.overlap_count)
        if max_overlap_count is None
        else max(0, int(max_overlap_count))
    )
    scale = max(abs(current_objective), 1e-9)

    # Lever 4: adaptive temperature calibration. Sample trial moves with a
    # separate RNG (so the main SA RNG state is unaffected), collect proxy
    # deltas, and override ``initial_temperature_ratio`` from the median
    # positive delta. Skipped when adaptive_temperature is False (default).
    if (
        bool(adaptive_temperature)
        and int(adaptive_num_trials) > 0
        and int(steps) > 0
    ):
        calib_rng = np.random.default_rng(int(seed) + 1_000_007)
        canvas_max_calib = max(float(canvas_w), float(canvas_h))
        trial_deltas: list[float] = []
        for _ in range(int(adaptive_num_trials)):
            m_idx = int(calib_rng.choice(movable))
            old = cache.positions[m_idx].copy()
            new_xy = _propose_xy(
                old_xy=old,
                macro_idx=m_idx,
                ctx=ctx,
                rng=calib_rng,
                canvas_w=float(canvas_w),
                canvas_h=float(canvas_h),
                canvas_max=canvas_max_calib,
                global_move_probability=float(global_move_probability),
                progress=0.0,
            )
            trial = apply_move(
                cache, ctx, m_idx, new_xy, exact_hpwl=False, update_congestion=False
            )
            delta = _objective(
                float(trial.proxy_cost),
                int(trial.overlap_count),
                float(overlap_penalty),
            ) - current_objective
            trial_deltas.append(delta)
            revert_move(cache, ctx, m_idx, old)
        adaptive_ratio = estimate_initial_temperature_ratio(
            trial_deltas,
            target_accept=float(adaptive_target_accept),
            scale=scale,
        )
        if adaptive_ratio > 0.0:
            initial_temperature_ratio = float(adaptive_ratio)

    archive: list[AnnealCandidate] = [
        _candidate(
            cache.positions,
            float(current.proxy_cost),
            current_objective,
            int(current.overlap_count),
            evaluations=1,
            accepted_moves=0,
        )
    ]

    total_steps = max(0, int(steps))
    accepted_moves = 0
    evaluations = 1
    canvas_max = max(float(canvas_w), float(canvas_h))

    rotation_enabled = (
        orientation_state is not None and float(rotation_probability) > 0.0
    )
    rot_prob = float(rotation_probability) if rotation_enabled else 0.0

    for step in range(total_steps):
        macro_idx = int(rng.choice(movable))
        temperature = _temperature(
            step=step,
            total_steps=total_steps,
            scale=scale,
            initial_ratio=float(initial_temperature_ratio),
            final_ratio=float(final_temperature_ratio),
        )

        propose_rotation = False
        rot_alts: list[int] = []
        if rotation_enabled and orientation_state is not None:
            if rng.random() < rot_prob:
                cur_ori = int(orientation_state.macro_orientation[macro_idx])
                class_oris = orientation_class_indices(cur_ori)
                rot_alts = [int(o) for o in class_oris if int(o) != cur_ori]
                if rot_alts:
                    propose_rotation = True

        if propose_rotation and orientation_state is not None:
            ori_state = orientation_state
            new_ori = int(rng.choice(np.asarray(rot_alts, dtype=np.int64)))
            prev = apply_rotation_to_cache(
                cache, ctx, ori_state, macro_idx, new_ori
            )
            trial = cache_result(cache)
            evaluations += 1
            trial_objective = _objective(
                float(trial.proxy_cost),
                int(trial.overlap_count),
                float(overlap_penalty),
            )
            if int(trial.overlap_count) > overlap_limit:
                accepted = False
            else:
                accepted = _accept(
                    delta=trial_objective - current_objective,
                    temperature=temperature,
                    rng=rng,
                )
            if accepted:
                current_objective = trial_objective
                accepted_moves += 1
                archive.append(
                    _candidate(
                        cache.positions,
                        float(trial.proxy_cost),
                        trial_objective,
                        int(trial.overlap_count),
                        evaluations=evaluations,
                        accepted_moves=accepted_moves,
                    )
                )
            else:
                apply_rotation_to_cache(
                    cache, ctx, ori_state, macro_idx, prev
                )
            continue

        old_xy = cache.positions[macro_idx].copy()
        new_xy = _propose_xy(
            old_xy=old_xy,
            macro_idx=macro_idx,
            ctx=ctx,
            rng=rng,
            canvas_w=float(canvas_w),
            canvas_h=float(canvas_h),
            canvas_max=canvas_max,
            global_move_probability=float(global_move_probability),
            progress=(step + 1) / max(total_steps, 1),
        )
        trial = apply_move(
            cache,
            ctx,
            macro_idx,
            new_xy,
            exact_hpwl=False,
            update_congestion=False,
        )
        evaluations += 1
        trial_objective = _objective(
            float(trial.proxy_cost),
            int(trial.overlap_count),
            float(overlap_penalty),
        )
        if int(trial.overlap_count) > overlap_limit:
            accepted = False
        else:
            accepted = _accept(
                delta=trial_objective - current_objective,
                temperature=temperature,
                rng=rng,
            )
        if accepted:
            current_objective = trial_objective
            accepted_moves += 1
            archive.append(
                _candidate(
                    cache.positions,
                    float(trial.proxy_cost),
                    trial_objective,
                    int(trial.overlap_count),
                    evaluations=evaluations,
                    accepted_moves=accepted_moves,
                )
            )
        else:
            revert_move(cache, ctx, macro_idx, old_xy)

    archive.append(
        _scored_candidate(
            cache.positions,
            ctx,
            float(overlap_penalty),
            evaluations,
            accepted_moves,
        )
    )
    rescored_archive = _rescore_archive(
        archive,
        ctx,
        canvas_w=float(canvas_w),
        canvas_h=float(canvas_h),
        overlap_penalty=float(overlap_penalty),
        pool_size=max(requested, int(exact_rescore_pool_size)),
        pre_legalize_iters=int(pre_legalize_iters),
    )
    selected = _select_diverse(
        rescored_archive,
        max_candidates=requested,
        min_distance=float(diversity_distance_ratio) * canvas_max,
    )
    return selected


def generate_targeted_sa_escape_candidates(
    *,
    initial_positions: np.ndarray,
    ctx: FastProxyContext,
    canvas_w: float,
    canvas_h: float,
    seed: int = 0,
    steps: int = 1_000,
    num_candidates: int = 4,
    target_count: int = 16,
    top_n_bins: int = 8,
    macros_per_bin: int = 4,
    exact_rescore_pool_size: int = 32,
    target_indices_override: np.ndarray | None = None,
    adaptive_temperature: bool = False,
    adaptive_num_trials: int = 64,
    adaptive_target_accept: float = 0.5,
) -> list[AnnealCandidate]:
    """Generate SA escape candidates by moving congestion-hot hard macros only.

    Lever 3 hook: if ``target_indices_override`` is provided (e.g. by the
    hybrid scorer), use those indices instead of running the default
    worst-congestion-bin selector.

    Lever 4 hook: when ``adaptive_temperature=True`` the SA initial temperature
    is calibrated from trial-move deltas (see ``estimate_initial_temperature_ratio``).
    """
    if target_indices_override is not None:
        targets = np.asarray(target_indices_override, dtype=np.int64).ravel()
    else:
        targets = worst_congestion_bin_destroy_seeds(
            np.asarray(initial_positions, dtype=np.float64),
            ctx,
            num_seeds=max(1, int(target_count)),
            top_n_bins=max(1, int(top_n_bins)),
            macros_per_bin=max(1, int(macros_per_bin)),
            seed=int(seed),
        )
    if targets.size == 0:
        hard = np.flatnonzero(np.asarray(ctx.macro_is_hard, dtype=bool))
        areas = np.asarray(ctx.macro_w, dtype=np.float64) * np.asarray(
            ctx.macro_h, dtype=np.float64
        )
        hard = hard[hard < areas.shape[0]]
        if hard.size > 0:
            order = hard[np.argsort(-areas[hard])]
            targets = order[: max(1, int(target_count))].astype(np.int64)

    return generate_sa_candidates(
        initial_positions=initial_positions,
        ctx=ctx,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        seed=seed,
        steps=steps,
        num_candidates=num_candidates,
        initial_temperature_ratio=0.01,
        final_temperature_ratio=0.0005,
        global_move_probability=0.35,
        overlap_penalty=0.05,
        diversity_distance_ratio=0.01,
        max_overlap_count=0,
        exact_rescore_pool_size=exact_rescore_pool_size,
        pre_legalize_iters=0,
        target_indices=targets,
        adaptive_temperature=bool(adaptive_temperature),
        adaptive_num_trials=int(adaptive_num_trials),
        adaptive_target_accept=float(adaptive_target_accept),
    )


def _movable_indices(
    ctx: FastProxyContext,
    num_positions: int,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    hard = np.flatnonzero(np.asarray(ctx.macro_is_hard, dtype=bool))
    hard = hard[hard < num_positions]
    if target_indices is not None:
        targets = np.asarray(target_indices, dtype=np.int64).ravel()
        targets = targets[(targets >= 0) & (targets < num_positions)]
        if targets.size > 0:
            allowed = set(int(idx) for idx in hard.tolist())
            filtered = [int(idx) for idx in targets.tolist() if int(idx) in allowed]
            if filtered:
                return np.asarray(filtered, dtype=np.int64)
    if hard.size > 0:
        return hard.astype(np.int64, copy=False)
    return np.arange(num_positions, dtype=np.int64)


def _objective(proxy_cost: float, overlap_count: int, overlap_penalty: float) -> float:
    return float(proxy_cost) + max(0.0, float(overlap_penalty)) * int(overlap_count)


def _candidate(
    positions: np.ndarray,
    proxy_cost: float,
    objective: float,
    overlap_count: int,
    evaluations: int,
    accepted_moves: int,
) -> AnnealCandidate:
    return AnnealCandidate(
        positions=np.asarray(positions, dtype=np.float64).copy(),
        proxy_cost=float(proxy_cost),
        objective=float(objective),
        overlap_count=int(overlap_count),
        evaluations=int(evaluations),
        accepted_moves=int(accepted_moves),
    )


def _scored_candidate(
    positions: np.ndarray,
    ctx: FastProxyContext,
    overlap_penalty: float,
    evaluations: int,
    accepted_moves: int,
) -> AnnealCandidate:
    result = fast_proxy(positions, ctx)
    return _candidate(
        positions=positions,
        proxy_cost=float(result.proxy_cost),
        objective=_objective(
            float(result.proxy_cost), int(result.overlap_count), overlap_penalty
        ),
        overlap_count=int(result.overlap_count),
        evaluations=evaluations,
        accepted_moves=accepted_moves,
    )


def _rescore_archive(
    archive: list[AnnealCandidate],
    ctx: FastProxyContext,
    *,
    canvas_w: float,
    canvas_h: float,
    overlap_penalty: float,
    pool_size: int,
    pre_legalize_iters: int,
) -> list[AnnealCandidate]:
    pool = sorted(archive, key=lambda item: item.objective)[: max(1, int(pool_size))]
    rescored: list[AnnealCandidate] = []
    for candidate in pool:
        positions = _pre_legalize(
            candidate.positions,
            ctx,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            max_iters=max(0, int(pre_legalize_iters)),
        )
        rescored.append(
            _scored_candidate(
                positions,
                ctx,
                overlap_penalty,
                candidate.evaluations,
                candidate.accepted_moves,
            )
        )
    return rescored


def _pre_legalize(
    positions: np.ndarray,
    ctx: FastProxyContext,
    *,
    canvas_w: float,
    canvas_h: float,
    max_iters: int,
) -> np.ndarray:
    work = np.asarray(positions, dtype=np.float64).copy()
    if max_iters <= 0:
        return work
    hard_indices = np.flatnonzero(np.asarray(ctx.macro_is_hard, dtype=bool))
    hard_indices = hard_indices[hard_indices < work.shape[0]]
    if hard_indices.size <= 1:
        return work

    for _ in range(max_iters):
        moved = False
        for offset, left_raw in enumerate(hard_indices):
            left = int(left_raw)
            for right_raw in hard_indices[offset + 1:]:
                right = int(right_raw)
                moved |= _push_pair_apart(work, ctx, left, int(right))
        _clamp_all_in_place(work, ctx, canvas_w, canvas_h)
        if not moved:
            break
    return work


def _push_pair_apart(
    positions: np.ndarray,
    ctx: FastProxyContext,
    left: int,
    right: int,
) -> bool:
    dx = float(positions[left, 0] - positions[right, 0])
    dy = float(positions[left, 1] - positions[right, 1])
    sep_x = 0.5 * (float(ctx.macro_w[left]) + float(ctx.macro_w[right]))
    sep_y = 0.5 * (float(ctx.macro_h[left]) + float(ctx.macro_h[right]))
    overlap_x = sep_x - abs(dx)
    overlap_y = sep_y - abs(dy)
    if overlap_x <= 0.0 or overlap_y <= 0.0:
        return False

    if overlap_x < overlap_y:
        sign = 1.0 if dx >= 0.0 else -1.0
        push = 0.5 * overlap_x + 0.01
        positions[left, 0] += sign * push
        positions[right, 0] -= sign * push
    else:
        sign = 1.0 if dy >= 0.0 else -1.0
        push = 0.5 * overlap_y + 0.01
        positions[left, 1] += sign * push
        positions[right, 1] -= sign * push
    return True


def estimate_initial_temperature_ratio(
    deltas,
    *,
    target_accept: float,
    scale: float,
) -> float:
    """Lever 4 helper: calibrate an SA initial-temperature ratio from observed deltas.

    Given a sample of trial proxy deltas (only positive ones inform temperature
    since negatives are always accepted), compute the median positive delta and
    return the temperature ratio that makes P(accept that delta) = ``target_accept``:

        T0 = median_positive_delta / -ln(target_accept)
        ratio = T0 / scale

    Returns ``0.0`` when no positive deltas are available, signalling the caller
    to use the static default ratio.
    """
    positives = [float(d) for d in deltas if float(d) > 0.0]
    if not positives:
        return 0.0
    target = max(min(float(target_accept), 0.999), 1e-6)
    median = float(sorted(positives)[len(positives) // 2])
    denom = -math.log(target)
    t0 = median / denom
    scale_v = max(float(scale), 1e-12)
    return t0 / scale_v


def _temperature(
    *,
    step: int,
    total_steps: int,
    scale: float,
    initial_ratio: float,
    final_ratio: float,
) -> float:
    if total_steps <= 1:
        return max(scale * final_ratio, 1e-12)
    progress = min(1.0, max(0.0, step / (total_steps - 1)))
    initial = max(scale * initial_ratio, 1e-12)
    final = max(scale * final_ratio, 1e-12)
    return initial * ((final / initial) ** progress)


def _accept(delta: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta <= 0.0:
        return True
    if temperature <= 0.0:
        return False
    return bool(rng.random() < math.exp(-float(delta) / float(temperature)))


def _propose_xy(
    *,
    old_xy: np.ndarray,
    macro_idx: int,
    ctx: FastProxyContext,
    rng: np.random.Generator,
    canvas_w: float,
    canvas_h: float,
    canvas_max: float,
    global_move_probability: float,
    progress: float,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = _legal_center_bounds(
        macro_idx, ctx, canvas_w, canvas_h
    )
    if rng.random() < global_move_probability:
        return np.asarray(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
            dtype=np.float64,
        )

    radius = canvas_max * max(0.02, 0.25 * (1.0 - progress))
    proposal = np.asarray(old_xy, dtype=np.float64).copy()
    proposal += rng.normal(loc=0.0, scale=radius, size=2)
    proposal[0] = np.clip(proposal[0], x_min, x_max)
    proposal[1] = np.clip(proposal[1], y_min, y_max)
    return proposal


def _legal_center_bounds(
    macro_idx: int,
    ctx: FastProxyContext,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float, float, float]:
    half_w = max(0.0, float(ctx.macro_w[macro_idx]) * 0.5)
    half_h = max(0.0, float(ctx.macro_h[macro_idx]) * 0.5)
    x_min = min(max(0.0, half_w), float(canvas_w))
    x_max = max(x_min, float(canvas_w) - half_w)
    y_min = min(max(0.0, half_h), float(canvas_h))
    y_max = max(y_min, float(canvas_h) - half_h)
    return x_min, x_max, y_min, y_max


def _clamp_all_in_place(
    positions: np.ndarray,
    ctx: FastProxyContext,
    canvas_w: float,
    canvas_h: float,
) -> None:
    for macro_idx in range(positions.shape[0]):
        x_min, x_max, y_min, y_max = _legal_center_bounds(
            macro_idx, ctx, canvas_w, canvas_h
        )
        positions[macro_idx, 0] = np.clip(positions[macro_idx, 0], x_min, x_max)
        positions[macro_idx, 1] = np.clip(positions[macro_idx, 1], y_min, y_max)


def _select_diverse(
    archive: list[AnnealCandidate],
    *,
    max_candidates: int,
    min_distance: float,
) -> list[AnnealCandidate]:
    selected: list[AnnealCandidate] = []
    ordered = sorted(archive, key=lambda item: item.objective)
    for candidate in ordered:
        if len(selected) >= max_candidates:
            break
        if all(
            _rms_distance(candidate.positions, kept.positions) >= min_distance
            for kept in selected
        ):
            selected.append(candidate)
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in ordered:
        if len(selected) >= max_candidates:
            break
        if id(candidate) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(id(candidate))
    return selected


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))
