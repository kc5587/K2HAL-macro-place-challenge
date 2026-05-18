"""Profile fast_proxy throughput on ibm01 and ibm03 (investigation only).

Do NOT commit. Used by Path B profiler for T7 budget feasibility analysis.
"""

from __future__ import annotations

import time

import numpy as np

from macro_place.adapter import resolve_plc
from macro_place.benchmark import Benchmark
from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy


def profile_bench(bench_name: str, n_eval: int = 2000) -> dict:
    bench = Benchmark.load(f"benchmarks/processed/public/{bench_name}.pt")
    plc = resolve_plc(bench)
    ctx = build_fast_proxy_context(plc, bench)

    rng = np.random.default_rng(0)
    pos = np.zeros((bench.num_macros, 2), dtype=np.float64)
    pos[:, 0] = rng.uniform(0.0, bench.canvas_width, size=bench.num_macros)
    pos[:, 1] = rng.uniform(0.0, bench.canvas_height, size=bench.num_macros)

    # Warmup (JIT compile + cache hot paths)
    for _ in range(50):
        _ = fast_proxy(pos, ctx)

    # Timed run with mutation to defeat any naive caching
    t0 = time.perf_counter()
    for i in range(n_eval):
        idx = i % bench.num_macros
        pos[idx, 0] = (pos[idx, 0] + 1.0) % bench.canvas_width
        _ = fast_proxy(pos, ctx)
    dt = time.perf_counter() - t0

    throughput = n_eval / dt
    ms_per = dt / n_eval * 1e3

    # Also profile components individually for bottleneck attribution
    from macro_place.fast_proxy import (
        fast_congestion,
        fast_density,
        fast_hpwl,
        fast_overlap_count,
    )

    # Warmup component kernels
    for _ in range(20):
        fast_hpwl(pos, ctx)
        fast_density(pos, ctx)
        fast_congestion(pos, ctx)
        fast_overlap_count(pos, ctx)

    n_comp = max(200, n_eval // 5)

    t = time.perf_counter()
    for i in range(n_comp):
        idx = i % bench.num_macros
        pos[idx, 0] = (pos[idx, 0] + 1.0) % bench.canvas_width
        fast_hpwl(pos, ctx)
    dt_hpwl = time.perf_counter() - t

    t = time.perf_counter()
    for i in range(n_comp):
        idx = i % bench.num_macros
        pos[idx, 0] = (pos[idx, 0] + 1.0) % bench.canvas_width
        fast_density(pos, ctx)
    dt_den = time.perf_counter() - t

    t = time.perf_counter()
    for i in range(n_comp):
        idx = i % bench.num_macros
        pos[idx, 0] = (pos[idx, 0] + 1.0) % bench.canvas_width
        fast_congestion(pos, ctx)
    dt_cong = time.perf_counter() - t

    t = time.perf_counter()
    for i in range(n_comp):
        idx = i % bench.num_macros
        pos[idx, 0] = (pos[idx, 0] + 1.0) % bench.canvas_width
        fast_overlap_count(pos, ctx)
    dt_ov = time.perf_counter() - t

    return {
        "bench": bench_name,
        "num_macros": int(bench.num_macros),
        "num_hard": int(bench.num_hard_macros),
        "throughput": throughput,
        "ms_per_eval": ms_per,
        "n_eval": n_eval,
        "comp_ms": {
            "hpwl": dt_hpwl / n_comp * 1e3,
            "density": dt_den / n_comp * 1e3,
            "congestion": dt_cong / n_comp * 1e3,
            "overlap": dt_ov / n_comp * 1e3,
        },
    }


def main():
    results = []
    for bench in ("ibm01", "ibm03"):
        # Use smaller N for ibm03 to keep wall time reasonable
        n = 2000 if bench == "ibm01" else 1000
        r = profile_bench(bench, n_eval=n)
        results.append(r)
        print(
            f"{r['bench']}: num_macros={r['num_macros']} (hard={r['num_hard']}), "
            f"throughput={r['throughput']:.0f} evals/s, "
            f"mean_per_eval={r['ms_per_eval']:.3f} ms (N={r['n_eval']})"
        )
        c = r["comp_ms"]
        print(
            f"  components (ms/call): hpwl={c['hpwl']:.3f} density={c['density']:.3f} "
            f"congestion={c['congestion']:.3f} overlap={c['overlap']:.3f}"
        )

    # T7 budget analysis: k_per_axis=8 -> 65 evals per node-sweep
    K = 8
    PER_NODE_EVALS = K * K + 1  # 65
    DESTROY_NODES = 10  # LNS destroy size
    RESTART_BUDGET_S = 1800.0
    print()
    print("T7 budget feasibility (k_per_axis=8, restart=1800s):")
    for r in results:
        per_sweep_evals = r["num_macros"] * PER_NODE_EVALS
        per_sweep_s = per_sweep_evals / r["throughput"]
        sweeps_per_restart = RESTART_BUDGET_S / per_sweep_s
        # LNS sub-sweep: only DESTROY_NODES nodes touched
        per_lns_evals = DESTROY_NODES * PER_NODE_EVALS
        per_lns_s = per_lns_evals / r["throughput"]
        lns_per_restart = RESTART_BUDGET_S / per_lns_s
        print(
            f"  {r['bench']}: full-sweep={per_sweep_evals} evals -> "
            f"{per_sweep_s:.2f}s ({sweeps_per_restart:.1f} sweeps/restart); "
            f"LNS-sweep={per_lns_evals} evals -> {per_lns_s:.3f}s "
            f"({lns_per_restart:.0f} LNS-iters/restart)"
        )


if __name__ == "__main__":
    main()
