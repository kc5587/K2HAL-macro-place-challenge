from __future__ import annotations
import numpy as np
import pytest

from macro_place.fast_proxy import FastProxyContext, FastProxyResult


@pytest.mark.unit
def test_fast_proxy_context_is_frozen() -> None:
    ctx = FastProxyContext(
        pin_macro_idx=np.zeros(0, dtype=np.int32),
        pin_offset_x=np.zeros(0, dtype=np.float32),
        pin_offset_y=np.zeros(0, dtype=np.float32),
        net_pin_starts=np.zeros(1, dtype=np.int32),
        net_pin_indices=np.zeros(0, dtype=np.int32),
        net_weights=np.zeros(0, dtype=np.float32),
        net_source_pin_local=np.zeros(0, dtype=np.int32),
        macro_w=np.zeros(0, dtype=np.float32),
        macro_h=np.zeros(0, dtype=np.float32),
        macro_is_hard=np.zeros(0, dtype=bool),
        grid_col=10,
        grid_row=10,
        canvas_w=100.0,
        canvas_h=100.0,
        h_routes_per_micron=1.0,
        v_routes_per_micron=1.0,
        hrouting_alloc=0.0,
        vrouting_alloc=0.0,
        smooth_range=2,
        overlap_threshold=0.004,
        net_cnt=1.0,
    )
    with pytest.raises(Exception):
        ctx.grid_col = 99  # type: ignore[misc]


@pytest.mark.unit
def test_fast_proxy_result_is_frozen() -> None:
    r = FastProxyResult(
        proxy_cost=1.0, wirelength=1.0, density=0.0,
        congestion=0.0, overlap_count=0,
    )
    with pytest.raises(Exception):
        r.proxy_cost = 2.0  # type: ignore[misc]


@pytest.mark.integration
def test_build_fast_proxy_context_matches_plc_shape() -> None:
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.fast_proxy import build_fast_proxy_context

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None, "resolve_plc returned None for ibm01"
    ctx = build_fast_proxy_context(plc, b)

    assert ctx.macro_w.shape == (b.num_macros,)
    assert ctx.macro_h.shape == (b.num_macros,)
    assert ctx.macro_is_hard.sum() == b.num_hard_macros
    assert ctx.net_pin_starts.shape[0] >= 2
    assert ctx.net_pin_indices.shape[0] == ctx.net_pin_starts[-1]
    assert ctx.pin_macro_idx.shape == ctx.pin_offset_x.shape
    assert ctx.grid_col == plc.grid_col
    assert ctx.grid_row == plc.grid_row
    assert ctx.canvas_w == pytest.approx(plc.width)
    assert ctx.canvas_h == pytest.approx(plc.height)


@pytest.mark.integration
def test_fast_hpwl_matches_plc_get_cost() -> None:
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.objective import compute_proxy_cost
    from macro_place.fast_proxy import build_fast_proxy_context, fast_hpwl

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos_t = torch.zeros(b.num_macros, 2, dtype=torch.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[i, 0], pos_t[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[b.num_hard_macros + i, 0], pos_t[b.num_hard_macros + i, 1] = x, y
    pos_np = pos_t.numpy()

    official = compute_proxy_cost(pos_t, b, plc)["wirelength_cost"]
    fast = fast_hpwl(pos_np, ctx)
    rel = abs(fast - official) / max(abs(official), 1e-9)
    assert rel < 5e-3, f"hpwl mismatch: fast={fast} vs official={official} rel={rel}"


@pytest.mark.integration
def test_fast_density_matches_plc() -> None:
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.objective import compute_proxy_cost
    from macro_place.fast_proxy import build_fast_proxy_context, fast_density

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos_t = torch.zeros(b.num_macros, 2, dtype=torch.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[i, 0], pos_t[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[b.num_hard_macros + i, 0], pos_t[b.num_hard_macros + i, 1] = x, y
    pos_np = pos_t.numpy()

    official = compute_proxy_cost(pos_t, b, plc)["density_cost"]
    fast = fast_density(pos_np, ctx)
    rel = abs(fast - official) / max(abs(official), 1e-9)
    assert rel < 5e-3, f"density mismatch: fast={fast} vs official={official} rel={rel}"


@pytest.mark.integration
def test_fast_overlap_count_exact() -> None:
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.objective import compute_overlap_metrics
    from macro_place.fast_proxy import build_fast_proxy_context, fast_overlap_count

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos_t = torch.zeros(b.num_macros, 2, dtype=torch.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[i, 0], pos_t[i, 1] = x, y
    pos_np = pos_t.numpy()

    official = int(compute_overlap_metrics(pos_t, b)["overlap_count"])
    fast = fast_overlap_count(pos_np, ctx)
    assert fast == official


@pytest.mark.integration
def test_fast_congestion_matches_plc() -> None:
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.objective import compute_proxy_cost
    from macro_place.fast_proxy import build_fast_proxy_context, fast_congestion

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos_t = torch.zeros(b.num_macros, 2, dtype=torch.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[i, 0], pos_t[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[b.num_hard_macros + i, 0], pos_t[b.num_hard_macros + i, 1] = x, y
    pos_np = pos_t.numpy()

    official = compute_proxy_cost(pos_t, b, plc)["congestion_cost"]
    fast = fast_congestion(pos_np, ctx)
    rel = abs(fast - official) / max(abs(official), 1e-9)
    assert rel < 1e-2, f"congestion mismatch: fast={fast} vs official={official} rel={rel}"


@pytest.mark.integration
@pytest.mark.parametrize("bench", ["ibm01", "ibm03"])
def test_fast_proxy_total_matches_official(bench: str) -> None:
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.objective import compute_proxy_cost
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy

    b = Benchmark.load(f"benchmarks/processed/public/{bench}.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos_t = torch.zeros(b.num_macros, 2, dtype=torch.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[i, 0], pos_t[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[b.num_hard_macros + i, 0], pos_t[b.num_hard_macros + i, 1] = x, y
    pos_np = pos_t.numpy()

    official = compute_proxy_cost(pos_t, b, plc)
    fast = fast_proxy(pos_np, ctx)

    rel = abs(fast.proxy_cost - official["proxy_cost"]) / max(abs(official["proxy_cost"]), 1e-9)
    assert rel < 1e-2, (
        f"{bench}: total proxy mismatch fast={fast.proxy_cost} "
        f"official={official['proxy_cost']} rel={rel}"
    )
    assert fast.overlap_count == int(official["overlap_count"])


@pytest.mark.integration
def test_fast_proxy_throughput_ibm01() -> None:
    import time
    import numpy as np
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy

    b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos = np.zeros((b.num_macros, 2), dtype=np.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[i, 0], pos[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos[b.num_hard_macros + i, 0], pos[b.num_hard_macros + i, 1] = x, y

    # Warmup (Numba JIT compile)
    for _ in range(3):
        fast_proxy(pos, ctx)

    N = 100
    t0 = time.perf_counter()
    for k in range(N):
        pos[k % b.num_hard_macros, 0] += 0.05
        fast_proxy(pos, ctx)
    dt = time.perf_counter() - t0
    rate = N / dt
    print(f"\nibm01 fast_proxy: {rate:.1f} calls/s, {dt*1000/N:.2f} ms/call")
    assert rate >= 100, f"throughput target missed: {rate:.1f}/s < 100/s"


@pytest.mark.integration
def test_fast_proxy_total_matches_official_ibm10() -> None:
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.adapter import resolve_plc
    from macro_place.objective import compute_proxy_cost
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy

    b = Benchmark.load("benchmarks/processed/public/ibm10.pt")
    plc = resolve_plc(b)
    assert plc is not None
    ctx = build_fast_proxy_context(plc, b)

    pos_t = torch.zeros(b.num_macros, 2, dtype=torch.float32)
    for i, idx in enumerate(b.hard_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[i, 0], pos_t[i, 1] = x, y
    for i, idx in enumerate(b.soft_macro_indices):
        x, y = plc.modules_w_pins[idx].get_pos()
        pos_t[b.num_hard_macros + i, 0], pos_t[b.num_hard_macros + i, 1] = x, y
    pos_np = pos_t.numpy()

    official = compute_proxy_cost(pos_t, b, plc)
    fast = fast_proxy(pos_np, ctx)
    rel = abs(fast.proxy_cost - official["proxy_cost"]) / max(abs(official["proxy_cost"]), 1e-9)
    assert rel < 1e-2, (
        f"ibm10: total proxy mismatch fast={fast.proxy_cost} "
        f"official={official['proxy_cost']} rel={rel}"
    )
    assert fast.overlap_count == int(official["overlap_count"])
