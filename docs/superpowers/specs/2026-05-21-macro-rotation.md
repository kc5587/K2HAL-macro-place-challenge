# Spec — Macro rotation as discrete DoF (Lever C)

Date: 2026-05-21
Status: **Scoped; implementation deferred** (multi-day, touches legalizer + proxy + CD + LNS).
Risk: HIGH — invasive change across the placer pipeline.

## Motivation

The current CD/LNS placer treats macro position `(x, y) ∈ R^2` as the only
degree of freedom. Each macro has a fixed orientation (R0). In real designs
each macro can be placed at 4 rotations (R0, R90, R180, R270) — and for pad-
asymmetric macros, 8 with mirroring. Our pipeline ignores this DoF entirely.

Why this matters:

- Rotating a macro can shorten the bounding box of all nets touching it (WL
  reduction) by moving pins to the natural side of the canvas.
- Rotated macros can fit into channels that the R0 footprint blocks, freeing
  routing congestion.
- The competition's evaluator scores rotated placements identically to
  R0 placements — there is no rule restricting orientation. Our family of
  algorithms is leaving free DoF on the table.

Closely related: the fresh-angles spec lists this as **Lever C** with an
estimated −1 to −5% proxy band (median for rotation-aware analytical placers
in published literature).

## Why this is a multi-day implementation

Macro rotation is **not** a localized lever — it touches every component
that reads macro dimensions or pin offsets:

1. **`macro_place/fast_proxy.py`** — the `FastProxyContext` packs `macro_w`,
   `macro_h`, `pin_offset_x`, `pin_offset_y`. These are baked at context
   build time. Rotation requires either (a) precomputed CSR per orientation
   (4× memory) or (b) on-the-fly transform inside the proxy kernel
   (slower per-evaluation). The fast-proxy is in Numba-jit'd kernels —
   changing the kernel signature is non-trivial.

2. **`macro_place/legality.py`** — `repair_overlaps` works on bbox geometry.
   When a macro rotates, its bbox `(w, h) → (h, w)`. Existing pair-pushing
   logic doesn't account for orientation changes during legalization.

3. **`macro_place/cd.py`** — `cd_grid_search` searches positions for ONE
   macro. We'd need to extend it to search `(x, y, rotation)` — either as a
   nested loop (4× cost per macro per sweep, i.e. 256 evaluations per macro
   instead of 64) or by interleaving rotation flips between position sweeps.

4. **`macro_place/lns_v2.py`** — destroy/rebuild picks new positions for K
   macros. Rebuild needs to also pick orientations. Doable but multiplies
   the rebuild search by 4.

5. **`macro_place/sa_generator.py`** — `_propose_xy` proposes a new (x, y).
   To respect rotation, propose `(x, y, r)` where `r ∈ {0,1,2,3}` with some
   probability of rotation per move.

6. **`submissions/macro_placer/cd_lns_placer.py`** — `_score_legalized_candidate`
   and ORFS spacing polish work on positions only; rotation needs to be
   threaded through. Eval reports must include orientation; output PLC must
   round-trip.

7. **`macro_place/objective.py:compute_proxy_cost`** — the official
   evaluator path needs the orientation; if PLC format doesn't carry it,
   we need to pre-rotate macros before scoring.

## Risk inventory

- **Eval-format mismatch.** The contest evaluator may not accept rotated
  placements in the PLC format we currently emit. Verify rotation support
  in the evaluator before any code is written.
- **PDN regression.** Rotating a macro changes which power straps cross
  which pin — could break PDN routability silently on NG45 designs.
- **Performance.** 4× state space in CD/LNS approximately quadruples
  per-sweep cost. Need to verify wall-time per restart stays within the
  60-minute contest cap.
- **Pin offset cache invalidation.** Pin offsets become orientation-
  dependent. Every cache that stores absolute pin positions needs
  invalidation hooks on rotation changes.

## Implementation plan (if we commit)

**Phase 1 — design + risk discharge (1 day):**
- Verify the evaluator accepts rotated placements (smoke a hand-rotated PLC).
- Decide between precomputed-per-orientation CSR vs on-the-fly transform.
- Measure baseline per-sweep cost; estimate rotation overhead.

**Phase 2 — fast_proxy + legality (1-2 days):**
- Extend `FastProxyContext` with `macro_orientation: np.ndarray[int8]`.
- Update CSR pin layout to be orientation-aware (or compute pin abs pos
  with rotation transform).
- Update `repair_overlaps` to use orientation-aware bboxes.
- Unit tests: proxy of rotated placement equals proxy of equivalent
  un-rotated placement modulo evaluator semantics.

**Phase 3 — CD/LNS/SA (1-2 days):**
- Add rotation as a parameter to `cd_grid_search` (search 4 rotations × k²
  positions). Gated behind `cd_rotation_enabled = False` default.
- Add rotation flips to `lns_destroy_rebuild` (probabilistic flip during
  rebuild).
- Add rotation moves to `_propose_xy` (small probability per SA step).

**Phase 4 — placer integration + lockin (1 day):**
- Thread orientation through `_score_legalized_candidate` and ORFS spacing.
- Verify output PLC carries orientation field.
- Phase A regression (all existing tests pass at orientation=R0).
- Smoke on ibm10/13/17 with rotation enabled.

**Total: 4-6 days.**

## Falsifiable smoke plan

Once Phase 4 lands:

1. **Validate evaluator round-trip.** Hand-rotate one macro in a known-good
   placement, write PLC, re-read, score. Must produce a finite proxy and
   match an independent hand calculation.
2. **Small smoke (1 restart, 5 min):** ibm10 with rotation enabled vs
   disabled. Look for any movement on proxy.
3. **Full-budget A/B:** if smoke shows ≥0.5% improvement potential, run
   the standard 3000s × 4 restarts on ibm10/13/17. Win condition: ≥1 bench
   improves by ≥0.3% vs SA-only reference with no overlap/PDN regression.

## What lands tonight (not this spec, but related)

- ✅ Option 1: `macro_place/restart_modes.py` + tests — new `exploratory`
  mode. Pending integration into `cd_lns_placer.py` AFTER Phase C finishes.
- ✅ Option 2: `macro_place/adaptive_config.py` + tests — Rules A-D for
  runtime-measured adaptive config. Pending integration AFTER Phase C
  finishes.
- ⏸️ Option 3 (this spec): scoped only. No code.

## Recommendation

**Defer.** Macro rotation is the highest-EV remaining lever in the placer
family (the fresh-angles spec gives it −1 to −5% proxy on similar pipelines),
but it's also the most invasive. Land Options 1 and 2 first; their compounded
effect will tell us whether basin-diversity / adaptive-config has any
remaining headroom. If Phase C-equivalent runs show Options 1+2 ≥0.5%
improvement, rotation is worth the 4-6 day investment. If they're flat,
the convergence wall is likely structural and rotation may not move it
either.
