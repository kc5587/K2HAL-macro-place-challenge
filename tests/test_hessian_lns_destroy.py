"""Unit tests for Hessian-guided LNS destroy (Priority 2).

Two pieces:
  - ``block_diag_top_saddle_macros`` (new in macro_place/hessian_escape.py):
    ranks macros by per-macro min eigenvalue of the 2x2 Hessian block;
    returns the top ``num_select`` macro indices.
  - ``lns_destroy_rebuild`` (modified in macro_place/lns_v2.py): accepts
    optional ``destroy_seed_indices`` to override random destroy selection.

Test layers:
  H1 — block_diag identifies the saddle-macro first.
  H2 — block_diag clamps num_select to N.
  L1 — lns_destroy_rebuild backward-compat: None → identical to no kwarg.
  L2 — lns_destroy_rebuild seeded destroy: only the seed indices move.
"""
from __future__ import annotations

import numpy as np
import pytest

from macro_place.hessian_escape import block_diag_top_saddle_macros


@pytest.mark.unit
def test_h1_block_diag_ranks_saddle_macro_first() -> None:
    """Three macros. f = (macro0 paraboloid) + (macro1 paraboloid) +
    (macro2 saddle). block_diag_top_saddle_macros should return macro2
    first (its 2x2 block has a negative eigenvalue).
    """
    def _eval(p: np.ndarray) -> float:
        # Each macro's contribution is its own quadratic about (0,0).
        f0 = p[0, 0] ** 2 + p[0, 1] ** 2  # convex
        f1 = p[1, 0] ** 2 + p[1, 1] ** 2  # convex
        f2 = -p[2, 0] ** 2 + p[2, 1] ** 2  # saddle (neg eigval along x)
        return float(f0 + f1 + f2)

    positions = np.zeros((3, 2), dtype=np.float64)
    indices = block_diag_top_saddle_macros(
        positions=positions,
        eval_fn=_eval,
        num_select=1,
        h=0.1,
    )

    assert indices.shape == (1,)
    assert int(indices[0]) == 2


@pytest.mark.unit
def test_h2_block_diag_clamps_num_select_to_n() -> None:
    """Asking for more macros than exist returns N indices, not an error."""
    def _eval(p: np.ndarray) -> float:
        return float(np.sum(p * p))

    positions = np.zeros((4, 2), dtype=np.float64)
    indices = block_diag_top_saddle_macros(
        positions=positions,
        eval_fn=_eval,
        num_select=20,
        h=0.1,
    )

    assert indices.shape == (4,)
    # All four macros must appear exactly once (it's a permutation).
    assert sorted(indices.tolist()) == [0, 1, 2, 3]


@pytest.mark.integration
def test_l1_lns_destroy_rebuild_backward_compat_with_none() -> None:
    """Calling lns_destroy_rebuild with destroy_seed_indices=None must
    produce identical output to omitting the kwarg entirely (same seed).
    """
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context
    from macro_place.lns_v2 import lns_destroy_rebuild

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos = np.zeros((b.num_macros, 2), dtype=np.float64)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[i, 0], pos[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[b.num_hard_macros + i, 0], pos[b.num_hard_macros + i, 1] = x, y

    kw = dict(
        positions=pos,
        ctx=ctx,
        canvas_w=b.canvas_width,
        canvas_h=b.canvas_height,
        num_destroy=4,
        max_lns_iters=2,
        k_per_axis=4,
        seed=0,
    )
    out_a, acc_a, _ = lns_destroy_rebuild(**kw)
    out_b, acc_b, _ = lns_destroy_rebuild(destroy_seed_indices=None, **kw)

    assert acc_a == acc_b
    assert np.array_equal(out_a, out_b)


@pytest.mark.unit
def test_l2_lns_destroy_rebuild_only_moves_seeded_indices() -> None:
    """With destroy_seed_indices=[1, 3], only macros 1 and 3 may move;
    macros 0 and 2 must be bit-equal to their input positions.

    Uses a fake FastProxyContext-shaped object and a fake fast_proxy/
    cd_grid_search via monkeypatch so the test stays in pure NumPy.
    """
    from unittest.mock import patch

    from macro_place.lns_v2 import lns_destroy_rebuild

    class _FakeCtx:
        pin_macro_idx = np.array([], dtype=np.int32)

    positions = np.array(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
        dtype=np.float64,
    )

    class _FakeProxyResult:
        proxy_cost = 1.0

    # Each cd_grid_search call returns a strictly lower cost than the
    # previous, so every seeded macro's rebuild is accepted.
    call_count = {"n": 0}

    def _fake_cd_grid_search(*, node_idx: int, positions, ctx, radius, k_per_axis):
        new_pos = positions[node_idx].copy()
        new_pos[0] += 5.0
        new_pos[1] += 5.0
        call_count["n"] += 1
        cost = 1.0 - 0.1 * call_count["n"]
        return new_pos, cost

    def _fake_fast_proxy(positions, ctx):
        return _FakeProxyResult()

    with patch("macro_place.lns_v2.fast_proxy", _fake_fast_proxy), patch(
        "macro_place.lns_v2.cd_grid_search", _fake_cd_grid_search
    ):
        new_pos, accepted, _ = lns_destroy_rebuild(
            positions=positions,
            ctx=_FakeCtx(),
            canvas_w=100.0,
            canvas_h=100.0,
            num_destroy=2,
            max_lns_iters=1,
            k_per_axis=4,
            seed=0,
            destroy_seed_indices=np.array([1, 3], dtype=np.int64),
        )

    assert accepted is True
    # Untouched macros bit-equal.
    np.testing.assert_array_equal(new_pos[0], positions[0])
    np.testing.assert_array_equal(new_pos[2], positions[2])
    # Seeded macros moved (by (+5,+5) per fake cd_grid_search).
    np.testing.assert_allclose(new_pos[1], positions[1] + 5.0)
    np.testing.assert_allclose(new_pos[3], positions[3] + 5.0)
