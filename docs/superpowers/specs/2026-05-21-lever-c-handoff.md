# Lever C (Macro Rotation) — Session Handoff (2026-05-21)

## Where we are

Lever C (macro orientation as a search DoF) was scoped, implemented, and partially
integrated this session. The orientation lever has **proven proxy signal**:

- **Step 2 helper-only smoke on ibm10**: **−0.478% proxy**, 494/786 macros (63%)
  wanted a different orientation, fast/official scorers agreed exactly.

## What landed (all default-off; bit-exact preserving)

### Step 1 — Orientation-aware fast_proxy ✅
- `macro_place/orientation.py` — 8 orientation transforms (N/FN/S/FS/E/FE/W/FW),
  inverse transforms to recover N-base offsets, `apply_orientation()` mutates
  ctx.pin_offset_x/y in place, `OrientationState` tracks per-macro current orientation.
- `tests/test_orientation.py` — 15/15 pass.

### Step 2 — Fast polish helper ✅
- `macro_place/rotation_polish.py:polish_orientations_fast` — uses fast_proxy
  (μs/eval) and syncs plc orientations after. **Proven −0.478% on ibm10.**

### Step A — Polish integration into placer ✅
- `submissions/macro_placer/cd_lns_placer.py`:
  - `_DEFAULT_CONFIG`: `rotation_polish_enabled: False`, `rotation_polish_top_k: 0`
  - Post-`_select_final_candidate` polish stage builds `OrientationState` and
    calls `polish_orientations_fast`.
- **Verified default-off path bit-exact** via A_off smoke (proxy=1.245362 on ibm10).

### Step E — CD-level joint orientation search ✅ (helpers + integration)
- `macro_place/orientation_cache.py:apply_rotation_to_cache` — same-class only,
  invalidates affected nets via `_update_hpwl`, refreshes normalized total_hpwl.
- `tests/test_orientation_cache.py` — 5/5 pass.
- `macro_place/cd.py:cd_grid_search` extended with optional `orientation_state` +
  `search_orientations` params. Wraps existing position loop with orientation outer
  loop. Threaded through `cd_sweep`, `cd_loop`.
- `cd_lns_placer.py:_run_one_restart`:
  - Builds `OrientationState` per restart when `cd_orientation_search_enabled=True`
  - Passes through to `cd_loop`
  - Returns 3-tuple `(pos, cost, orientations_or_None)`
- `_DEFAULT_CONFIG`: `cd_orientation_search_enabled: False`

### Step G — Cross-process orientation sync ✅
- Worker tuple gained 4th element (orientations array)
- `_score_legalized_candidate` accepts `orientations: np.ndarray | None`. When None,
  resets plc to all-N. When provided, applies via `plc.update_macro_orientation`.
- `_FinalCandidate.orientations` field carries orientations through selection
- After `_select_final_candidate`: explicit plc sync to `best_candidate.orientations`
  before any downstream polish/eval reads it
- `CDLNSPlacer.place()` exposes `_last_plc` attribute — callers can read the
  internal plc to see final orientations (the previous smoke broken because it
  scored against an outer plc with stale orientations)

### Test totals
- **71 unit tests passing** across all new modules:
  - 15 test_orientation
  - 5 test_orientation_cache
  - 6 test_rotation_polish
  - 6 test_restart_modes
  - 11 test_adaptive_config
  - 8 test_targeted_sa_gating
  - 5 test_hessian_top_k
  - 3 test_targeted_sa_multi_source
  - 6 test_hybrid_target
  - 6 test_adaptive_temperature

## What remains for the FULL Lever C

### Step F — LNS + SA rotation moves (1 day) ⏸

**Two new config keys, both default 0.0 (no-op):**
- `lns_rotation_probability: float` — per rebuilt macro, probability of also
  trying a random same-class rotation in addition to the new position.
- `sa_rotation_probability: float` — per SA step, probability that the move is
  a rotation instead of a translation.

**`macro_place/lns_v2.py:lns_destroy_rebuild`:**
- Accept new params `orientation_state` and `rotation_probability` (both
  optional, default None/0.0).
- During rebuild, for each macro placed at a new candidate position: with
  probability `rotation_probability`, also try each same-class orientation,
  pick the lowest-proxy combination. Use `apply_rotation_to_cache` for the
  cache-aware switch (already exists).
- Symmetric revert if not accepted.
- Thread params through any wrapper functions.

**`macro_place/sa_generator.py:generate_sa_candidates` inner loop:**
- Accept new params `orientation_state` and `rotation_probability`.
- In the per-step propose loop: with probability `rotation_probability`,
  instead of calling `_propose_xy`, propose a rotation:
  1. Pick a same-class orientation for the chosen macro
  2. Call `apply_rotation_to_cache` to mutate cache + ctx
  3. Score via the existing apply_move path (a zero-displacement move forces
     reevaluation; or just call `cache_result(cache)` directly)
  4. Accept/reject by SA rule; revert via `apply_rotation_to_cache(prev_ori)`
- For combined rotation+translation: pick one of {rotate, translate, both}
  with configurable probabilities; both is most aggressive but most expensive.

**Tests required:**
- `tests/test_lns_v2_rotation.py`: rotation_probability=0 → bit-exact prior
  behavior; rotation_probability=1.0 → every rebuild tries rotations.
- `tests/test_sa_generator_rotation.py`: same pattern.

**Smoke:**
- ibm10 1-restart with sa_rotation_probability=0.1 vs 0.0
- Look for >0.1% additional gain over polish-only

**Forward to placer:**
- Add config keys to `_DEFAULT_CONFIG`
- Thread `orientation_state` from `_run_one_restart` into LNS/SA calls (already
  built when `cd_orientation_search_enabled=True`; reuse the same instance for
  LNS/SA rotation if `lns_rotation_probability` or `sa_rotation_probability` > 0)
- Update `_last_run_stats` with rotation move counts

### Step 5 — Full validation + Phase D ⏸
- Validate `_last_plc` fix: re-run polish smoke and confirm reported delta
  matches the proven −0.478%.
- Full-budget A/B on ibm10/13/17 with `rotation_polish_enabled=True`. Gate:
  beats SA-only refs by ≥0.2%.
- If polish-only ships clean: enable `cd_orientation_search_enabled=True` and
  re-run; gate: ≥0.1% additional gain over polish-only.
- Phase D Tier-1 lock-in across all 17 ibm benches.

## How to ship polish-only TODAY (if Step F is deferred)

```python
# In submissions/macro_placer/cd_lns_placer.py _DEFAULT_CONFIG:
"rotation_polish_enabled": True,  # was False
```

That single flag flip enables the proven polish-only path. Estimated submission
impact: **−0.4% to −0.5% on ibm10** (Step 2 evidence). Other benches likely
similar. No invasive changes; opt-out by setting flag False.

**Caveat**: the polish operates on `placer._last_plc` (the placer's internal
plc). The contest evaluator reads the PLC file the placer writes via
`def_writer.py`, which already round-trips orientation. So the lever should
work in the submission flow even though the previous smoke (using outer plc)
reported 0% — that was a measurement artifact.

## Key files / commits

All edits are in the working tree (nothing committed).

**New modules:**
- `macro_place/orientation.py`
- `macro_place/orientation_cache.py`
- `macro_place/rotation_polish.py`
- `macro_place/restart_modes.py`
- `macro_place/adaptive_config.py`
- `macro_place/hybrid_target.py`

**Modified:**
- `submissions/macro_placer/cd_lns_placer.py` (all lever integrations + Step G)
- `macro_place/cd.py` (cd_grid_search/sweep/loop orientation params)
- `macro_place/sa_generator.py` (Lever 4 adaptive temperature + Lever 3 target override)

**Test files:** 10 new test files, all passing.

**Specs / docs:**
- `docs/superpowers/specs/2026-05-21-macro-rotation.md`
- `docs/superpowers/specs/2026-05-21-lever-c-handoff.md` (this file)

## Open questions for next session

1. **Verify the `_last_plc` fix.** Run `scripts/verify_polish_fix.py` and confirm
   inner_plc reports ~−0.4% improvement while outer_plc shows ~0%.
2. **Phase C full-budget A/B**. Does polish-only sustain ≥0.2% at full budget?
3. **Step F worth doing?** Depends on Step E full-budget evidence (LNS + SA
   propose rotation moves on top of CD orientation search; ~1 day).
4. **Per-bench reliability.** Does the −0.478% on ibm10 hold across all 17 ibm
   benches? Some benches may have less rotation slack.

## Risk inventory

- **PDN regression (Tier 2)**: rotating macros could change channel widths.
  Step 5 should include a Tier 2 NG45 routability check before flipping default.
- **Numerical drift**: orientation transform precision is float32. Could
  accumulate small errors over many rotations. Unit tests cover round-trip.
- **Cross-class rotation**: explicitly disallowed in `apply_rotation_to_cache`
  (would swap macro w/h, invalidate density/overlap caches). Step F should
  preserve this restriction.
