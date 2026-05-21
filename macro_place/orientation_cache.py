"""Lever C Step E — Cache-aware orientation switching.

``apply_rotation_to_cache`` swaps a macro's orientation and updates the
incremental fast_proxy cache for affected nets only. Same-class rotation
(NS↔NS or EW↔EW) leaves macro w/h unchanged, so density and congestion
contributions stay valid — only per-net HPWL changes when pin offsets shift.

Cross-class rotation (NS↔EW) would swap w/h and require density/overlap
recomputation; that case is explicitly out of scope here and asserts.
"""
from __future__ import annotations

from typing import Any

from macro_place.orientation import (
    NS_CLASS_INDICES,
    EW_CLASS_INDICES,
    apply_orientation,
)
from macro_place.fast_proxy_incremental import _update_hpwl, _normalize_hpwl


def _same_class(a: int, b: int) -> bool:
    a, b = int(a), int(b)
    if a in NS_CLASS_INDICES and b in NS_CLASS_INDICES:
        return True
    if a in EW_CLASS_INDICES and b in EW_CLASS_INDICES:
        return True
    return False


def apply_rotation_to_cache(
    cache: Any,
    ctx: Any,
    ori_state: Any,
    macro_idx: int,
    new_ori_idx: int,
) -> int:
    """Rotate ``macro_idx`` to ``new_ori_idx`` and update cache HPWL for affected nets.

    Returns the previous orientation index so the caller can revert symmetrically
    by calling this function again with the returned value.

    Restricted to same-class rotation (NS↔NS or EW↔EW). Cross-class raises
    ``ValueError`` since cross-class swaps macro w/h and would require
    density/overlap cache rebuild that this helper does not perform.
    """
    macro_idx_i = int(macro_idx)
    new_ori = int(new_ori_idx)
    prev_ori = int(ori_state.macro_orientation[macro_idx_i])
    if new_ori == prev_ori:
        return prev_ori
    if not _same_class(prev_ori, new_ori):
        raise ValueError(
            f"cross-class rotation not supported: {prev_ori} → {new_ori}"
        )
    # Mutate ctx pin offsets and macro_orientation.
    apply_orientation(ctx, ori_state, macro_idx_i, new_ori)
    # Recompute per-net HPWL for affected nets only.
    nets = cache.macro_to_nets[macro_idx_i]
    if nets.size > 0:
        _update_hpwl(cache, ctx, nets)
        # _update_hpwl updates total_hpwl_raw incrementally but does NOT
        # refresh the normalized total_hpwl that cache_result reads from.
        cache.total_hpwl = _normalize_hpwl(cache.total_hpwl_raw, ctx)
    return prev_ori
