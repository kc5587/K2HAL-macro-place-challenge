"""Unit tests for macro_place/hessian_escape.py — Hessian-based saddle escape (E12).

Validates correctness on analytical toy quadratics BEFORE the algorithm
is run against IBM benchmarks. Bet 6 (saddle.py) skipped this step and we
never knew whether its disappointing results meant "algorithm fails" or
"benchmarks have no saddles." E12 avoids that ambiguity.

Test layers:
  T1 — 2D saddle f = -x²+y²: block_diag finds eigenvalue -2 along (1,0).
  T2 — 2D paraboloid f = x²+y²: rejects (no negative eigenvalues).
  T3 — Coupled 4D saddle f = 4·x₁·y₂: block_diag finds nothing (per-macro
       Hessians are zero), Lanczos must find eigenvalue -4 along (x₁, y₂).
  T4 — Canvas clip: huge step stays in bounds.
  T5 — Deterministic: same seed → bit-identical output.
"""
from __future__ import annotations

import numpy as np
import pytest

from macro_place.hessian_escape import (
    block_diag_min_eigenvalue,
    hessian_escape,
    lanczos_min_eigenvalue,
)


# ---------------------------------------------------------------------------
# T1: classic 2D saddle. One macro at (0, 0). f = -x² + y².
# Hessian = diag(-2, +2). Block-diag (only block) has eigenvalues (-2, +2).
# The negative eigenvalue eigenvector is (1, 0).
# ---------------------------------------------------------------------------


def _eval_2d_saddle(positions: np.ndarray) -> float:
    x, y = float(positions[0, 0]), float(positions[0, 1])
    return -x * x + y * y


@pytest.mark.unit
def test_block_diag_finds_negative_eigenvalue_on_2d_saddle() -> None:
    positions = np.array([[0.0, 0.0]], dtype=np.float64)
    eigval, macro_idx, eigvec = block_diag_min_eigenvalue(
        positions=positions, eval_fn=_eval_2d_saddle, h=0.1
    )
    assert macro_idx == 0
    # Eigenvalue should be ≈ -2 (within FD discretization error).
    assert eigval < -1.0, f"expected negative eigenvalue, got {eigval}"
    assert eigval == pytest.approx(-2.0, abs=0.1)
    # Eigenvector should be aligned with the x-axis (sign indeterminate).
    assert abs(eigvec[0]) > 0.95, f"eigenvec not along x: {eigvec}"
    assert abs(eigvec[1]) < 0.1, f"eigenvec has y-component: {eigvec}"


@pytest.mark.unit
def test_hessian_escape_steps_along_negative_eigenvalue_direction_2d_saddle() -> None:
    """End-to-end on T1: should accept and return positions with f(x') < f(x)."""
    positions = np.array([[0.0, 0.0]], dtype=np.float64)
    new_positions, accepted, stats = hessian_escape(
        positions=positions,
        eval_fn=_eval_2d_saddle,
        canvas_w=100.0,
        canvas_h=100.0,
        h_block=0.1,
        h_lanczos=0.1,
        curvature_threshold=-1e-3,
        line_search_alphas=(0.5, 1.0, 2.0, 4.0),
        rng_seed=0,
    )
    assert accepted, f"saddle escape rejected; stats={stats}"
    # f at new position should be strictly less than f at saddle (=0).
    new_f = _eval_2d_saddle(new_positions)
    assert new_f < -1e-3, f"new cost {new_f} did not improve over 0"
    # Must have stepped along x (the negative-curvature direction).
    assert abs(new_positions[0, 0]) > 0.1
    assert stats["source"] in ("block_diag", "lanczos")


# ---------------------------------------------------------------------------
# T2: 2D paraboloid (true local minimum). f = x² + y². All eigenvalues > 0.
# hessian_escape must return accepted=False and unchanged positions.
# ---------------------------------------------------------------------------


def _eval_2d_paraboloid(positions: np.ndarray) -> float:
    x, y = float(positions[0, 0]), float(positions[0, 1])
    return x * x + y * y


@pytest.mark.unit
def test_block_diag_returns_positive_eigenvalue_on_paraboloid() -> None:
    positions = np.array([[0.0, 0.0]], dtype=np.float64)
    eigval, _macro_idx, _eigvec = block_diag_min_eigenvalue(
        positions=positions, eval_fn=_eval_2d_paraboloid, h=0.1
    )
    assert eigval > 0.0, f"paraboloid eigenvalue should be positive, got {eigval}"


@pytest.mark.unit
def test_hessian_escape_rejects_paraboloid_local_minimum() -> None:
    positions = np.array([[0.0, 0.0]], dtype=np.float64)
    new_positions, accepted, stats = hessian_escape(
        positions=positions,
        eval_fn=_eval_2d_paraboloid,
        canvas_w=100.0,
        canvas_h=100.0,
        h_block=0.1,
        h_lanczos=0.1,
        curvature_threshold=-1e-3,
        line_search_alphas=(0.5, 1.0, 2.0, 4.0),
        rng_seed=0,
    )
    assert accepted is False, f"paraboloid wrongly accepted: stats={stats}"
    assert np.allclose(new_positions, positions)
    assert "reason" in stats
    assert "min_eigenval_nonneg" in stats["reason"] or "no_improvement" in stats["reason"]


# ---------------------------------------------------------------------------
# T3: coupled 4D saddle. Two macros. f(x₁,y₁,x₂,y₂) = 4·x₁·y₂.
# Hessian has a single off-diagonal entry of 4 (at (x₁, y₂) and (y₂, x₁));
# eigenvalues are (4, -4, 0, 0). Per-macro 2x2 block Hessians are zero,
# so block_diag finds NOTHING — proving Variant A is insufficient. Lanczos
# (Variant B) must find eigenvalue ≈ -4 with eigenvector spanning (x₁, y₂).
# ---------------------------------------------------------------------------


def _eval_coupled_4d_saddle(positions: np.ndarray) -> float:
    """Coupled saddle centered at (50, 50, 50, 50) so escape moves don't
    hit canvas clipping. f - f0 has Hessian eigenvalues (4, -4, 0, 0)
    with the negative-eigenvalue eigenvector spanning (x₁, y₂)."""
    x1 = float(positions[0, 0]) - 50.0
    y2 = float(positions[1, 1]) - 50.0
    return 4.0 * x1 * y2


@pytest.mark.unit
def test_block_diag_misses_coupled_saddle_t3() -> None:
    """T3 case: per-macro block Hessians are zero. Block-diag cannot help.
    This documents the limitation that motivates Variant B (Lanczos).
    """
    positions = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float64)
    eigval, _macro_idx, _eigvec = block_diag_min_eigenvalue(
        positions=positions, eval_fn=_eval_coupled_4d_saddle, h=0.1
    )
    # All per-macro blocks are zero; the smallest eigenvalue must be ≈ 0.
    assert abs(eigval) < 0.1, (
        f"block-diag spuriously found eigenvalue {eigval} on coupled-only saddle"
    )


@pytest.mark.unit
def test_lanczos_finds_coupled_saddle_t3() -> None:
    """Variant B must catch the eigenvalue -4 invisible to block-diag."""
    positions = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float64)
    eigval, eigvec = lanczos_min_eigenvalue(
        positions=positions,
        eval_fn=_eval_coupled_4d_saddle,
        h=0.05,
        max_iters=20,
        rng_seed=0,
    )
    # Eigenvalue ≈ -4 (or smaller magnitude is fine — we just need negative).
    assert eigval < -1.0, f"lanczos did not find coupled negative eigenvalue: {eigval}"
    # Eigenvector should span (x₁, y₂): entries at positions[0,0] and
    # positions[1,1] should be significant; entries at positions[0,1],
    # positions[1,0] should be near zero. eigvec shape is (2, 2).
    assert eigvec.shape == positions.shape
    # The non-zero components should be at (0, 0) and (1, 1).
    nonzero = abs(eigvec[0, 0]) + abs(eigvec[1, 1])
    near_zero = abs(eigvec[0, 1]) + abs(eigvec[1, 0])
    assert nonzero > 0.5, f"x₁/y₂ components too small: {eigvec}"
    assert near_zero < 0.3, f"y₁/x₂ components too large (uncoupled): {eigvec}"


@pytest.mark.unit
def test_hessian_escape_uses_lanczos_when_block_diag_finds_nothing_t3() -> None:
    """End-to-end on T3: Variant C falls back to Lanczos and accepts."""
    positions = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float64)
    new_positions, accepted, stats = hessian_escape(
        positions=positions,
        eval_fn=_eval_coupled_4d_saddle,
        canvas_w=100.0,
        canvas_h=100.0,
        h_block=0.1,
        h_lanczos=0.05,
        curvature_threshold=-1e-3,
        line_search_alphas=(0.5, 1.0, 2.0, 4.0),
        rng_seed=0,
    )
    assert accepted, f"coupled saddle rejected; stats={stats}"
    new_f = _eval_coupled_4d_saddle(new_positions)
    assert new_f < -1e-3, f"new cost {new_f} did not improve"
    assert stats["source"] == "lanczos", (
        f"expected Lanczos to be the escape source on T3; got {stats['source']}"
    )


# ---------------------------------------------------------------------------
# T4: canvas clip. Saddle at canvas edge; large step must stay in bounds.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_is_clipped_to_canvas() -> None:
    # One "macro" with half-size 1, canvas 10x10. Place at (1.0, 5.0) — right
    # at the left edge. Saddle f = -(x-1)² + (y-5)² has negative curvature
    # along x; saddle escape will push x leftward, which would exit canvas.
    positions = np.array([[1.0, 5.0]], dtype=np.float64)

    def _eval(p: np.ndarray) -> float:
        dx = float(p[0, 0] - 1.0)
        dy = float(p[0, 1] - 5.0)
        return -dx * dx + dy * dy

    half_sizes = np.array([[1.0, 1.0]], dtype=np.float64)
    new_positions, _accepted, _stats = hessian_escape(
        positions=positions,
        eval_fn=_eval,
        canvas_w=10.0,
        canvas_h=10.0,
        half_sizes=half_sizes,
        h_block=0.1,
        h_lanczos=0.1,
        curvature_threshold=-1e-3,
        line_search_alphas=(8.0, 16.0),  # large step
        rng_seed=0,
    )
    # Macro center must stay in [half_w, canvas_w - half_w] = [1, 9].
    assert new_positions[0, 0] >= 1.0 - 1e-9
    assert new_positions[0, 0] <= 9.0 + 1e-9
    assert new_positions[0, 1] >= 1.0 - 1e-9
    assert new_positions[0, 1] <= 9.0 + 1e-9


# ---------------------------------------------------------------------------
# T5: deterministic with seed. Same input → bit-identical output.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hessian_escape_is_deterministic_for_same_seed() -> None:
    positions = np.array([[0.0, 0.0]], dtype=np.float64)
    kw = dict(
        eval_fn=_eval_2d_saddle,
        canvas_w=100.0,
        canvas_h=100.0,
        h_block=0.1,
        h_lanczos=0.1,
        curvature_threshold=-1e-3,
        line_search_alphas=(1.0, 2.0, 4.0),
        rng_seed=42,
    )
    out_a, acc_a, _ = hessian_escape(positions=positions.copy(), **kw)
    out_b, acc_b, _ = hessian_escape(positions=positions.copy(), **kw)
    assert acc_a == acc_b
    assert np.array_equal(out_a, out_b)
