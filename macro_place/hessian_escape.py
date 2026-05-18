"""Hessian-based saddle escape (E12).

Designed to find escapes that Bet 6 (saddle.py, random-direction sampling)
missed. The plan in
``docs/superpowers/plans/2026-05-16-hessian-saddle-escape-plan.md``
documents Bet 6's failure modes and how this module addresses each.

Two variants composed (Variant C):

  1. ``block_diag_min_eigenvalue``: per-macro 2x2 Hessian via 5-point finite
     differences; solve each block's 2x2 eigenproblem analytically; return
     the smallest eigenvalue across macros plus its eigenvector. Catches
     single-macro saddles (negative curvature in a single macro's local
     (x, y) plane).

  2. ``lanczos_min_eigenvalue``: random K-dim subspace + Rayleigh-Ritz.
     Sample K random unit directions; build K x K matrix C with
     C[i,j] = v_i^T H v_j via the polarization identity
       v_i^T H v_j = (1/4)*[(v_i+v_j)^T H (v_i+v_j) - (v_i-v_j)^T H (v_i-v_j)]
     and each quadratic form via the 3-point Hessian formula
       u^T H u ≈ (f(x + h*u) + f(x - h*u) - 2*f(x)) / h².
     Diagonalize C; lift the smallest-eigenvalue eigenvector back to the
     full (N, 2) space. Catches coupled saddles that block-diag misses.

  3. ``hessian_escape``: try block-diag first (cheap); if no negative
     eigenvalue, fall back to Lanczos; if still no negative eigenvalue,
     return ``accepted=False`` (we're at a true local minimum). When a
     direction is found, do a small line search along it (no ``cd_loop``
     polish — that was Bet 6 failure mode F3 that pulled escapes back
     into the same basin).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _clip_to_canvas(
    positions: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    half_sizes: np.ndarray | None = None,
) -> np.ndarray:
    out = np.array(positions, dtype=np.float64, copy=True)
    if half_sizes is None:
        lo_x = lo_y = 0.0
        hi_x = float(canvas_w)
        hi_y = float(canvas_h)
        np.clip(out[:, 0], lo_x, hi_x, out=out[:, 0])
        np.clip(out[:, 1], lo_y, hi_y, out=out[:, 1])
    else:
        np.clip(out[:, 0], half_sizes[:, 0], canvas_w - half_sizes[:, 0], out=out[:, 0])
        np.clip(out[:, 1], half_sizes[:, 1], canvas_h - half_sizes[:, 1], out=out[:, 1])
    return out


def block_diag_min_eigenvalue(
    positions: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    h: float = 0.05,
) -> tuple[float, int, np.ndarray]:
    """For each macro, compute its 2x2 block of the Hessian
        H_i = [[∂²f/∂x_i², ∂²f/∂x_i∂y_i],
               [∂²f/∂y_i∂x_i, ∂²f/∂y_i²]]
    via 5-point finite differences. Solve each 2x2 eigenproblem.
    Return the smallest eigenvalue found across all macros, plus the
    macro index and its corresponding 2D eigenvector (normalized).

    NOTE: This is the *block-diagonal* minimum eigenvalue of the Hessian,
    NOT the minimum eigenvalue of the full 2N x 2N Hessian. Coupled
    saddles whose negative-eigenvalue eigenvectors span multiple macros
    are invisible to this method; use ``lanczos_min_eigenvalue`` for those.

    Returns:
        (smallest_eigenvalue, macro_idx, eigvec_2d)
        eigvec_2d is shape (2,), unit-normalized.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2); got shape {positions.shape}")
    n = positions.shape[0]
    f0 = float(eval_fn(positions))

    best_eigval = float("inf")
    best_macro = 0
    best_eigvec = np.array([1.0, 0.0])

    for i in range(n):
        # Perturb macro i by ±h along x and y.
        p_xp = positions.copy()
        p_xp[i, 0] += h
        p_xm = positions.copy()
        p_xm[i, 0] -= h
        p_yp = positions.copy()
        p_yp[i, 1] += h
        p_ym = positions.copy()
        p_ym[i, 1] -= h
        # Cross terms.
        p_pp = positions.copy()
        p_pp[i, 0] += h
        p_pp[i, 1] += h
        p_pm = positions.copy()
        p_pm[i, 0] += h
        p_pm[i, 1] -= h
        p_mp = positions.copy()
        p_mp[i, 0] -= h
        p_mp[i, 1] += h
        p_mm = positions.copy()
        p_mm[i, 0] -= h
        p_mm[i, 1] -= h

        fx_p = float(eval_fn(p_xp))
        fx_m = float(eval_fn(p_xm))
        fy_p = float(eval_fn(p_yp))
        fy_m = float(eval_fn(p_ym))
        f_pp = float(eval_fn(p_pp))
        f_pm = float(eval_fn(p_pm))
        f_mp = float(eval_fn(p_mp))
        f_mm = float(eval_fn(p_mm))

        h_xx = (fx_p + fx_m - 2.0 * f0) / (h * h)
        h_yy = (fy_p + fy_m - 2.0 * f0) / (h * h)
        h_xy = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)

        block = np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(block)
        # eigvals are sorted ascending; smallest is at index 0.
        if eigvals[0] < best_eigval:
            best_eigval = float(eigvals[0])
            best_macro = i
            best_eigvec = eigvecs[:, 0].astype(np.float64, copy=True)

    return best_eigval, best_macro, best_eigvec


def block_diag_per_macro_min_eigenvalues(
    positions: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    h: float = 0.05,
) -> np.ndarray:
    """Per-macro minimum eigenvalue of the 2x2 Hessian block.

    Returns a length-N float array; entry ``i`` is the smaller eigenvalue
    of macro ``i``'s 2x2 block computed by 5-point finite differences.
    Macros with more-negative entries are at sharper local saddles and
    are the targets ``block_diag_top_saddle_macros`` selects.

    Cost: ~8N evaluations of ``eval_fn`` plus one base evaluation.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2); got shape {positions.shape}")
    n = positions.shape[0]
    f0 = float(eval_fn(positions))
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        p_xp = positions.copy()
        p_xp[i, 0] += h
        p_xm = positions.copy()
        p_xm[i, 0] -= h
        p_yp = positions.copy()
        p_yp[i, 1] += h
        p_ym = positions.copy()
        p_ym[i, 1] -= h
        p_pp = positions.copy()
        p_pp[i, 0] += h
        p_pp[i, 1] += h
        p_pm = positions.copy()
        p_pm[i, 0] += h
        p_pm[i, 1] -= h
        p_mp = positions.copy()
        p_mp[i, 0] -= h
        p_mp[i, 1] += h
        p_mm = positions.copy()
        p_mm[i, 0] -= h
        p_mm[i, 1] -= h

        h_xx = (float(eval_fn(p_xp)) + float(eval_fn(p_xm)) - 2.0 * f0) / (h * h)
        h_yy = (float(eval_fn(p_yp)) + float(eval_fn(p_ym)) - 2.0 * f0) / (h * h)
        h_xy = (
            float(eval_fn(p_pp))
            - float(eval_fn(p_pm))
            - float(eval_fn(p_mp))
            + float(eval_fn(p_mm))
        ) / (4.0 * h * h)
        block = np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(block)
        out[i] = float(eigvals[0])

    return out


def block_diag_top_saddle_macros(
    positions: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    num_select: int,
    h: float = 0.05,
) -> np.ndarray:
    """Return indices of the ``num_select`` macros with the most-negative
    per-macro 2x2-block minimum eigenvalue (i.e. the macros most stuck
    at local saddles). Used to seed LNS destroy.

    If ``num_select`` exceeds the macro count, all macros are returned
    sorted ascending by min eigenvalue.
    """
    eigvals = block_diag_per_macro_min_eigenvalues(positions, eval_fn, h=h)
    n = eigvals.shape[0]
    k = min(int(num_select), n)
    # ``np.argsort`` ascending — most-negative first.
    order = np.argsort(eigvals, kind="stable")
    return order[:k].astype(np.int64, copy=True)


def _quadratic_form(
    positions: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    direction: np.ndarray,
    h: float,
    f0: float,
) -> float:
    """Compute v^T H v via 3-point finite difference:
        v^T H v ≈ (f(x + h*v) + f(x - h*v) - 2*f(x)) / h².
    """
    p_plus = positions + h * direction
    p_minus = positions - h * direction
    f_plus = float(eval_fn(p_plus))
    f_minus = float(eval_fn(p_minus))
    return (f_plus + f_minus - 2.0 * f0) / (h * h)


def lanczos_min_eigenvalue(
    positions: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    h: float = 0.05,
    max_iters: int = 16,
    rng_seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Approximate the smallest eigenvalue of the full 2N x 2N Hessian
    via random-subspace Rayleigh-Ritz.

    Builds a K-dim random subspace (K = max_iters) of (N, 2)-shaped unit
    vectors. Forms the K x K matrix C with C[i,j] = v_i^T H v_j, computed
    via the polarization identity. Diagonalizes C. Returns the smallest
    eigenvalue with its corresponding full-space eigenvector.

    This is NOT classical Lanczos (which needs H*v products that we can't
    compute without a gradient oracle). It's the projection of H onto a
    random K-dim subspace — sufficient to catch coupled saddles when K
    is large relative to the ambient dimension or when the negative
    eigenvalue is "loud" (e.g., the toy T3 4D saddle).

    Returns:
        (smallest_eigenvalue, eigvec_full)
        eigvec_full has shape positions.shape, unit-normalized.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2); got shape {positions.shape}")
    n = positions.shape[0]
    K = int(max_iters)
    rng = np.random.default_rng(rng_seed)

    f0 = float(eval_fn(positions))

    # Sample K random Gaussian vectors and orthonormalize via QR.
    # Orthonormality is required for Rayleigh-Ritz to give eigenvalues in
    # [λ_min(H), λ_max(H)] — diagonalizing C in a non-orthonormal basis
    # solves the wrong eigenproblem (true form is generalized: Cu = λMu
    # where M is the Gram matrix; only M=I makes plain eigh correct).
    full_dim = n * 2
    K_eff = min(K, full_dim)
    raw = rng.standard_normal((full_dim, K_eff))
    # QR yields orthonormal columns in raw_q (R is discarded — we only
    # need an orthonormal basis for the random subspace).
    raw_q, _ = np.linalg.qr(raw)
    basis: list[np.ndarray] = []
    for k in range(raw_q.shape[1]):
        v_flat = raw_q[:, k]
        v = v_flat.reshape(n, 2)
        # raw_q columns are already unit; reshape preserves norm.
        basis.append(v.astype(np.float64, copy=True))
    if not basis:
        return float("inf"), np.zeros_like(positions)

    Kb = len(basis)
    # Compute diagonal v_i^T H v_i via 3-point formula.
    diag = np.zeros(Kb, dtype=np.float64)
    for i in range(Kb):
        diag[i] = _quadratic_form(positions, eval_fn, basis[i], h, f0)

    # Polarization for off-diagonals:
    #   v_i^T H v_j = (1/4) * [(v_i+v_j)^T H (v_i+v_j) - (v_i-v_j)^T H (v_i-v_j)]
    C = np.zeros((Kb, Kb), dtype=np.float64)
    for i in range(Kb):
        C[i, i] = diag[i]
        for j in range(i + 1, Kb):
            sum_dir = basis[i] + basis[j]
            diff_dir = basis[i] - basis[j]
            # Normalize each to keep h-step consistent
            sum_norm = float(np.linalg.norm(sum_dir))
            diff_norm = float(np.linalg.norm(diff_dir))
            if sum_norm < 1e-12 or diff_norm < 1e-12:
                C[i, j] = C[j, i] = 0.0
                continue
            q_sum = _quadratic_form(
                positions, eval_fn, sum_dir / sum_norm, h, f0
            ) * (sum_norm ** 2)
            q_diff = _quadratic_form(
                positions, eval_fn, diff_dir / diff_norm, h, f0
            ) * (diff_norm ** 2)
            cij = 0.25 * (q_sum - q_diff)
            C[i, j] = cij
            C[j, i] = cij

    # Diagonalize the (symmetric) Rayleigh-Ritz matrix.
    eigvals, eigvecs = np.linalg.eigh(C)
    # Smallest at index 0.
    lam_min = float(eigvals[0])
    coeffs = eigvecs[:, 0]
    # Lift back to full space.
    eigvec_full = np.zeros_like(positions)
    for k in range(Kb):
        eigvec_full += coeffs[k] * basis[k]
    norm = float(np.linalg.norm(eigvec_full))
    if norm > 1e-12:
        eigvec_full = eigvec_full / norm
    return lam_min, eigvec_full


def hessian_escape(
    positions: np.ndarray,
    eval_fn: Callable[[np.ndarray], float],
    canvas_w: float,
    canvas_h: float,
    half_sizes: np.ndarray | None = None,
    h_block: float = 0.05,
    h_lanczos: float = 0.05,
    lanczos_max_iters: int = 16,
    curvature_threshold: float = -1e-3,
    line_search_alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    tolerance: float = 1e-6,
    rng_seed: int = 0,
) -> tuple[np.ndarray, bool, dict]:
    """Variant C: block-diag first; if it doesn't find a saddle, fall back
    to Lanczos; if still no saddle, return ``accepted=False``. After the
    direction is found, do a small line search (no ``cd_loop`` polish —
    Bet 6 F3 mitigation).

    Args:
        positions: (N, 2) float array of macro center positions.
        eval_fn: callable mapping (N, 2) -> float (the cost we're escaping).
        canvas_w, canvas_h: canvas extent. New positions are clipped.
        half_sizes: optional (N, 2) half-sizes for per-macro clipping.
            If None, positions are clipped to [0, canvas_*].
        h_block: FD step for block-diag.
        h_lanczos: FD step for Lanczos quadratic forms.
        lanczos_max_iters: K — random subspace dimension for Lanczos.
        curvature_threshold: eigenvalue must be < this to count as a saddle
            (defaults to -1e-3 so FD noise doesn't trigger false saddles).
        line_search_alphas: step magnitudes to try along the eigenvector.
        tolerance: required strict improvement to accept.
        rng_seed: RNG seed for Lanczos basis.

    Returns:
        (new_positions, accepted, stats)
        stats includes keys: ``source`` ("block_diag"/"lanczos"/"none"),
        ``eigenvalue``, ``reason`` (if rejected), and per-stage counters.
    """
    stats: dict = {
        "source": "none",
        "block_diag_eigenvalue": None,
        "lanczos_eigenvalue": None,
        "best_alpha": None,
        "f0": None,
        "f_best": None,
        "reason": "",
    }

    f0 = float(eval_fn(positions))
    stats["f0"] = f0

    # Stage 1: block-diag (cheap).
    bd_eigval, bd_macro, bd_eigvec_2d = block_diag_min_eigenvalue(
        positions=positions, eval_fn=eval_fn, h=h_block
    )
    stats["block_diag_eigenvalue"] = bd_eigval

    direction: np.ndarray | None = None
    if bd_eigval < curvature_threshold:
        # Lift the per-macro 2D eigenvector into the full (N, 2) space.
        direction = np.zeros_like(positions)
        direction[bd_macro] = bd_eigvec_2d
        # Normalize to unit norm in (N, 2) (already unit because eigvec_2d is).
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction = direction / norm
        stats["source"] = "block_diag"
        stats["eigenvalue"] = bd_eigval
    else:
        # Stage 2: fall back to Lanczos.
        lz_eigval, lz_eigvec = lanczos_min_eigenvalue(
            positions=positions,
            eval_fn=eval_fn,
            h=h_lanczos,
            max_iters=lanczos_max_iters,
            rng_seed=rng_seed,
        )
        stats["lanczos_eigenvalue"] = lz_eigval
        if lz_eigval < curvature_threshold:
            direction = lz_eigvec
            stats["source"] = "lanczos"
            stats["eigenvalue"] = lz_eigval
        else:
            stats["reason"] = "min_eigenval_nonneg"
            return positions.copy(), False, stats

    if direction is None:
        stats["reason"] = "no_direction_found"
        return positions.copy(), False, stats

    # Stage 3: line search along direction (positive and negative — the
    # eigenvector's sign is arbitrary; both directions descend on a saddle).
    best_f = f0
    best_pos = positions.copy()
    best_alpha: float | None = None
    for alpha in line_search_alphas:
        for sign in (1.0, -1.0):
            trial = positions + (sign * alpha) * direction
            trial = _clip_to_canvas(trial, canvas_w, canvas_h, half_sizes)
            f_trial = float(eval_fn(trial))
            if f_trial < best_f:
                best_f = f_trial
                best_pos = trial
                best_alpha = float(sign * alpha)
    stats["best_alpha"] = best_alpha
    stats["f_best"] = best_f

    if best_f < f0 - tolerance:
        stats["reason"] = "accepted"
        return best_pos, True, stats
    stats["reason"] = "no_improvement"
    return positions.copy(), False, stats
