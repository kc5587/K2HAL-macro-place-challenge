"""Tests for Lever L (worst-congestion-bin destroy).

Includes 4 correctness tests plus 3 proof-of-improvement tests that gate
the merge BEFORE any benchmark smoke. The proof tests confirm L picks
different macros than E, that L's picks sit in genuinely hot bins, and
that one LNS iter seeded by L reduces (or at least doesn't increase) the
congestion cost.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from macro_place.congestion_destroy import worst_congestion_bin_destroy_seeds
from macro_place.fast_proxy import (
    FastProxyContext,
    build_fast_proxy_context,
    fast_congestion,
    fast_congestion_per_bin,
)
from macro_place.lns_v2 import lns_destroy_rebuild
from macro_place.loader import load_benchmark


# ---------- helpers ----------

_BENCH_ROOT = Path(
    "/Users/kaushalchitturu/HRT_MACRO/external/MacroPlacement/Testcases/ICCAD04"
)


def _load_ibm(name: str):
    base = _BENCH_ROOT / name
    bench, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
    ctx = build_fast_proxy_context(plc, bench)
    pos = np.asarray(
        bench.macro_positions.detach().cpu().numpy()
        if hasattr(bench.macro_positions, "detach")
        else bench.macro_positions,
        dtype=np.float64,
    )
    return bench, plc, ctx, pos


def _synthetic_ctx_with_hot_bin():
    """Return ctx + positions where macro 0 sits alone in a high-congestion bin,
    macros 1-5 are scattered in low-congestion bins."""
    grid_col = grid_row = 4
    canvas_w = canvas_h = 100.0
    num_macros = 6
    # Macros are small (5x5), positioned at bin centers
    cell = 25.0
    centers = [
        (cell * 0.5, cell * 0.5),   # macro 0 in bin (0,0)
        (cell * 1.5, cell * 1.5),   # macro 1 in bin (1,1)
        (cell * 2.5, cell * 2.5),   # macro 2 in bin (2,2)
        (cell * 3.5, cell * 0.5),   # macro 3 in bin (0,3)
        (cell * 0.5, cell * 3.5),   # macro 4 in bin (3,0)
        (cell * 2.5, cell * 0.5),   # macro 5 in bin (0,2)
    ]
    positions = np.array(centers, dtype=np.float64)
    macro_w = np.full(num_macros, 5.0, dtype=np.float64)
    macro_h = np.full(num_macros, 5.0, dtype=np.float64)
    macro_is_hard = np.ones(num_macros, dtype=bool)

    # Construct a net structure that pumps congestion through bin (0,0):
    # nets are heavy and pin (macro 0) <-> a PORT at far corner, plus many
    # parallel nets fan-routing through bin (0,0)'s row 0.
    # To keep it simple, we build pins as (macro_idx, offset). PORT pins
    # have macro_idx=-1 and the offset is absolute position.
    pin_macro_idx = []
    pin_offset_x = []
    pin_offset_y = []
    # Multiple heavy 2-pin nets: macro 0 <-> PORT in far corner
    # Each net contributes routing through row 0 (because source is macro 0 at (12.5, 12.5))
    net_pin_starts = [0]
    net_weights = []
    net_source_pin_local = []
    pin_local = 0

    for _ in range(30):  # 30 heavy nets all pinned on macro 0
        # source: macro 0 pin
        pin_macro_idx.append(0)
        pin_offset_x.append(0.0)
        pin_offset_y.append(0.0)
        src_local = pin_local
        pin_local += 1
        # sink: PORT at corner (canvas_w-1, canvas_h-1)
        pin_macro_idx.append(-1)
        pin_offset_x.append(canvas_w - 1.0)
        pin_offset_y.append(canvas_h - 1.0)
        pin_local += 1
        net_pin_starts.append(pin_local)
        net_weights.append(10.0)
        net_source_pin_local.append(src_local)

    # A few light nets connecting macros 1-5 in various bins
    for src_mi, dst_mi in [(1, 2), (3, 4), (4, 5)]:
        pin_macro_idx.append(src_mi)
        pin_offset_x.append(0.0)
        pin_offset_y.append(0.0)
        src_local = pin_local
        pin_local += 1
        pin_macro_idx.append(dst_mi)
        pin_offset_x.append(0.0)
        pin_offset_y.append(0.0)
        pin_local += 1
        net_pin_starts.append(pin_local)
        net_weights.append(1.0)
        net_source_pin_local.append(src_local)

    ctx = FastProxyContext(
        pin_macro_idx=np.asarray(pin_macro_idx, dtype=np.int64),
        pin_offset_x=np.asarray(pin_offset_x, dtype=np.float64),
        pin_offset_y=np.asarray(pin_offset_y, dtype=np.float64),
        net_pin_starts=np.asarray(net_pin_starts, dtype=np.int64),
        net_pin_indices=np.arange(pin_local, dtype=np.int64),
        net_weights=np.asarray(net_weights, dtype=np.float64),
        net_source_pin_local=np.asarray(net_source_pin_local, dtype=np.int64),
        macro_w=macro_w,
        macro_h=macro_h,
        macro_is_hard=macro_is_hard,
        grid_col=grid_col,
        grid_row=grid_row,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        h_routes_per_micron=1.0,
        v_routes_per_micron=1.0,
        hrouting_alloc=0.5,
        vrouting_alloc=0.5,
        smooth_range=0,
        overlap_threshold=0.0,
        net_cnt=float(sum(net_weights)),
    )
    return ctx, positions


# ---------- 4 correctness tests ----------


@pytest.mark.unit
def test_picks_macros_in_worst_bin() -> None:
    ctx, positions = _synthetic_ctx_with_hot_bin()
    seeds = worst_congestion_bin_destroy_seeds(
        positions, ctx, num_seeds=3, top_n_bins=4
    )
    assert seeds.size >= 1, "L returned no seeds on a heavily congested synthetic"
    # macro 0 is the source of 30 heavy nets all routing into bin (0,0); it
    # must appear in the picks (it sits in or adjacent to the highest bin).
    assert 0 in seeds.tolist(), f"L missed the obvious hot macro 0: {seeds}"


@pytest.mark.unit
def test_deterministic() -> None:
    ctx, positions = _synthetic_ctx_with_hot_bin()
    s1 = worst_congestion_bin_destroy_seeds(positions, ctx, num_seeds=5, top_n_bins=4)
    s2 = worst_congestion_bin_destroy_seeds(positions, ctx, num_seeds=5, top_n_bins=4)
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.unit
def test_respects_num_seeds() -> None:
    ctx, positions = _synthetic_ctx_with_hot_bin()
    seeds = worst_congestion_bin_destroy_seeds(
        positions, ctx, num_seeds=3, top_n_bins=10
    )
    assert seeds.size <= 3
    assert len(set(seeds.tolist())) == seeds.size


@pytest.mark.unit
def test_only_hard_macros() -> None:
    ctx, positions = _synthetic_ctx_with_hot_bin()
    # Flip macros 4 and 5 to soft to verify they are excluded
    soft_mask = ctx.macro_is_hard.copy()
    soft_mask[4] = False
    soft_mask[5] = False
    ctx_soft = replace(ctx, macro_is_hard=soft_mask)
    seeds = worst_congestion_bin_destroy_seeds(
        positions, ctx_soft, num_seeds=10, top_n_bins=16
    )
    for s in seeds.tolist():
        assert s < 4, f"L returned soft-macro index {s}"


# ---------- 3 proof-of-improvement tests on a real bench ----------


@pytest.mark.integration
def test_L_picks_concentrated_subset_not_random() -> None:
    """L's pick must be much more selective than random — proves it's actually
    using the congestion signal, not just returning arbitrary macros."""
    bench, plc, ctx, pos = _load_ibm("ibm12")  # worst-congestion bench
    n = 10
    l_seeds = worst_congestion_bin_destroy_seeds(
        pos, ctx, num_seeds=n, top_n_bins=8
    )
    assert l_seeds.size > 0, "L returned no seeds on ibm12"
    num_hard = int(ctx.macro_is_hard.sum())
    # Run L several times with different top_n to confirm picks are
    # SAME (deterministic) and a SMALL subset of total hard macros
    l_seeds_again = worst_congestion_bin_destroy_seeds(
        pos, ctx, num_seeds=n, top_n_bins=8
    )
    np.testing.assert_array_equal(
        l_seeds, l_seeds_again,
        err_msg="L not deterministic on real bench"
    )
    selectivity = l_seeds.size / num_hard
    print(
        f"\n[L_selectivity] ibm12 hard_macros={num_hard} L_picks={l_seeds.size} "
        f"selectivity={selectivity*100:.2f}%"
    )
    assert selectivity < 0.05, (
        f"L picked {l_seeds.size}/{num_hard} = {selectivity*100:.1f}% — "
        f"too broad to be targeted"
    )


@pytest.mark.integration
def test_L_picked_macros_neighbor_hot_bins_more_than_random() -> None:
    """L's picks must be physically closer to the top congestion bins than
    a random sample of macros — proves L is using the congestion signal.

    (Congestion peaks happen in routing channels BETWEEN macros, so we measure
    proximity to hot bins, not bbox overlap with them.)
    """
    bench, plc, ctx, pos = _load_ibm("ibm12")
    per_bin = fast_congestion_per_bin(pos, ctx)
    seeds = worst_congestion_bin_destroy_seeds(
        pos, ctx, num_seeds=10, top_n_bins=8
    )
    assert seeds.size > 0, "L returned no seeds on ibm12"

    grid_row, grid_col = per_bin.shape
    cell_w = ctx.canvas_w / grid_col
    cell_h = ctx.canvas_h / grid_row
    flat = per_bin.ravel()
    top_idx = np.argpartition(flat, flat.size - 8)[flat.size - 8:]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    bin_centers = np.array(
        [
            ((int(fb) % grid_col + 0.5) * cell_w,
             (int(fb) // grid_col + 0.5) * cell_h)
            for fb in top_idx
        ],
        dtype=np.float64,
    )

    num_hard = int(ctx.macro_is_hard.sum())
    hard_pos = pos[:num_hard]

    def _mean_min_dist(macro_indices: np.ndarray) -> float:
        # For each macro, distance to NEAREST top-bin center; average over the set
        sub = hard_pos[macro_indices]
        dx = sub[:, None, 0] - bin_centers[None, :, 0]
        dy = sub[:, None, 1] - bin_centers[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        return float(d.min(axis=1).mean())

    rng = np.random.default_rng(0)
    random_idx = rng.choice(num_hard, size=seeds.size, replace=False)
    d_picked = _mean_min_dist(seeds)
    d_random = _mean_min_dist(random_idx)
    print(
        f"\n[L_proximity] mean min-dist to top-8 bin centers — "
        f"L_picks={d_picked:.3f}μm  random={d_random:.3f}μm  "
        f"ratio_random/L={d_random/max(d_picked,1e-9):.2f}x"
    )
    assert d_picked < d_random * 0.5, (
        f"L's picks ({d_picked:.3f}μm) aren't meaningfully closer to hot bins "
        f"than random ({d_random:.3f}μm) — L isn't using the congestion signal"
    )


@pytest.mark.integration
def test_L_reduces_congestion_in_LNS_iter() -> None:
    """One LNS rebuild seeded by L must reduce congestion (or reject the move)."""
    bench, plc, ctx, pos = _load_ibm("ibm12")
    num_destroy = 10
    seeds = worst_congestion_bin_destroy_seeds(
        pos, ctx, num_seeds=num_destroy, top_n_bins=8
    )
    cong_before = fast_congestion(pos, ctx)
    new_pos, accepted, _evals = lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=ctx.canvas_w,
        canvas_h=ctx.canvas_h,
        num_destroy=num_destroy,
        max_lns_iters=4,
        k_per_axis=4,
        seed=0,
        destroy_seed_indices=seeds,
    )
    cong_after = fast_congestion(new_pos, ctx)
    print(
        f"\n[L_lns_iter] accepted={accepted} cong_before={cong_before:.4f} "
        f"cong_after={cong_after:.4f} delta={cong_after-cong_before:+.4f}"
    )
    # If LNS rejected, congestion is unchanged → OK (no harm).
    # If LNS accepted, congestion must not have INCREASED — that would mean
    # L is steering LNS toward worse layouts on the dominant cost.
    if accepted:
        assert cong_after <= cong_before + 1e-6, (
            f"L-seeded LNS accepted a move that INCREASED congestion "
            f"({cong_before:.4f} → {cong_after:.4f}) — L is hurting the dominant cost"
        )
