"""TDD for the post-CD LNS destroy-rebuild step (Bet 7 restart)."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_lns_destroy_rebuild_does_not_regress_cost() -> None:
    """A single LNS destroy-rebuild iteration must produce a placement
    whose cost is <= the cost going in (it accepts only on improvement
    or rejects to original)."""
    from macro_place.adapter import resolve_plc
    from macro_place.benchmark import Benchmark
    from macro_place.fast_proxy import build_fast_proxy_context, fast_proxy
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

    cost_before = float(fast_proxy(pos, ctx).proxy_cost)

    new_pos, accepted, _ = lns_destroy_rebuild(
        positions=pos,
        ctx=ctx,
        canvas_w=b.canvas_width,
        canvas_h=b.canvas_height,
        num_destroy=8,
        max_lns_iters=10,
        k_per_axis=8,
        seed=0,
    )

    cost_after = float(fast_proxy(new_pos, ctx).proxy_cost)
    # Either accepted with strictly lower cost, or rejected with original cost.
    if accepted:
        assert cost_after < cost_before + 1e-9
    else:
        assert abs(cost_after - cost_before) < 1e-9
