# K2HAL — Macro Placement Challenge 2026 Submission

**Team:** K2HAL · Kaushal Chitturu ([kc5587](https://github.com/kc5587))
**Method:** CD-LNS with Hessian Saddle Escape
**Tag:** [`v1.0-submission`](https://github.com/kc5587/K2HAL-macro-place-challenge/releases/tag/v1.0-submission) (commit `69a7673`)
**License:** Apache 2.0 (see [LICENSE.md](LICENSE.md))

---

## Method

The placer is a parallel multi-restart search built around a calibrated fast
proxy and a strict legality gate. Each call to `CDLNSPlacer.place(benchmark)`
runs the following pipeline:

1. **Warm-start.** All restarts begin from the contest's `initial.plc`. The
   legalized initial layout is also retained as an "initial guard" candidate
   so the placer never returns worse than the contest's hand-crafted seed.
2. **Four parallel restarts.** A `ProcessPoolExecutor` launches four workers
   with the mode mix `(conservative, light, aggressive, aggressive)` — varying
   the warm-start σ, CD radius, sweep cap, and whether LNS runs at all — so
   the search covers different basins instead of stacking redundant runs.
3. **CD ↔ LNS inner loop.** Each restart alternates coordinate descent
   (multi-scale, shrinking radius) with large-neighborhood-search
   destroy-and-rebuild. LNS is **Hessian-guided**: a block-diagonal
   approximation of the proxy Hessian ranks macros by their local
   negative-curvature ("saddle-like") score, and the top-ranked macros seed
   the destroy set, biasing search toward genuinely stuck regions instead of
   random subsets.
4. **Legalize every candidate.** A tiered overlap-repair stack (local
   pair-pushing → minimum-disturbance reshuffle → shelf-pack fallback)
   produces a fully legal placement for each restart output. Candidates are
   sorted by `(overlap_count, proxy_cost)` using the official
   `compute_proxy_cost`, so any zero-overlap placement beats every
   overlap-positive one regardless of score.
5. **Top-K final polish.** Within a bounded tail budget (~480 s), the top
   `K = 8` legal candidates are refined by a small-radius CD polish that
   operates on the already-legal placements and is re-scored with the
   official proxy.
6. **Hessian saddle escape.** The proxy-best candidate then undergoes a
   deterministic saddle escape: a block-diagonal eigendecomposition (refined
   by a Lanczos Rayleigh–Ritz pass) finds the most-negative-curvature
   direction of the proxy cost, and a small line search along ±that direction
   accepts only on strict improvement.
7. **ORFS protection.** A spacing polish enforces ≥ 12 μm clearance and a
   guard-repair pass keeps macros inside the core, giving Tier-2 OpenROAD
   placements a clean starting point without compromising Tier-1 proxy.
8. **Final selection.** Candidates are tied within 0.1 % proxy and broken by
   ORFS-aware metrics, then the legalized positions of the winning candidate
   are returned to the harness.

The result is a fully open-source, GPU-free placer that returns zero-overlap
legal placements on every IBM benchmark and substantially improves on the
RePlAce baseline.

## Performance

### Tier 1 — IBM (17 benchmarks)

| Metric | Value |
|---|---|
| **Average proxy cost** | **1.1262** |
| RePlAce baseline | 1.4578 |
| SA baseline | 2.1251 |
| Improvement vs RePlAce | **~22.7%** |
| Average runtime / bench | 16.8 min (Apple M3 reference) |
| Max runtime / bench | 37.3 min (ibm17) |
| Contest runtime cap | 60 min / bench |
| Overlaps | **0 / 17** |

Source: `output/17bench_final/` (`time_budget_s=1800`, `num_restarts=4`).
A confirmation run at the placer's default `time_budget_s=3000` is in progress
and will refresh these numbers before final form submission.

### Tier 2 — NG45 OpenROAD flow (ariane133)

| Metric | Value |
|---|---|
| WNS | 0.267703 ns |
| TNS | 0 |
| Hold WNS | 0.0115249 ns |
| Hold TNS | 0 |
| Wirelength | 4,680,914 |
| Area | 4,306,330 μm² |
| Power | 0.198774 W |
| Fmax | 267.93 MHz |
| Proxy cost (NG45) | 0.7628 |

## How to reproduce

```bash
git clone https://github.com/kc5587/K2HAL-macro-place-challenge.git
cd K2HAL-macro-place-challenge
git submodule update --init external/MacroPlacement
uv sync

# Run a single benchmark (~12 min on a modern CPU)
uv run evaluate submissions/macro_placer/cd_lns_placer.py -b ibm01

# Run the full 17-bench IBM suite
uv run evaluate submissions/macro_placer/cd_lns_placer.py --all

# Visualize a placement
uv run evaluate submissions/macro_placer/cd_lns_placer.py -b ibm01 --vis
```

The placer requires no GPU. Dependencies (Python ≥ 3.8): `torch`, `numpy`,
`numba`, plus the standard contest evaluator from the `external/MacroPlacement`
submodule.

## File map

| Path | Purpose |
|---|---|
| `submissions/macro_placer/cd_lns_placer.py` | Submission entrypoint — `class CDLNSPlacer`, `.place(benchmark) → Tensor[N,2]` |
| `submissions/macro_placer/_audit.py` | Self-containment guard run at import time |
| `macro_place/cd.py` | Coordinate descent inner loop |
| `macro_place/lns_v2.py` | Large-neighborhood search rebuild |
| `macro_place/hessian_escape.py` | Block-diagonal + Lanczos Rayleigh–Ritz saddle escape |
| `macro_place/legality.py` | Tiered overlap-repair stack including shelf-pack fallback |
| `macro_place/fast_proxy.py` | Calibrated fast proxy used during search (1800× speedup vs `compute_proxy_cost`, <1 ppm error) |
| `tests/` | Unit and integration tests |

## Compliance

| Rule | Status |
|---|---|
| Apache 2.0 / GPL open source | ✓ Apache 2.0 |
| Self-contained submission | ✓ `_audit.py` fails import if required modules are external |
| TILOS evaluator unmodified | ✓ |
| No benchmark-specific hardcoding | ✓ Generic algorithm, identical config across all 17 IBM benches |
| No external proprietary tools | ✓ Pure Python / NumPy / PyTorch / Numba |
| Zero overlaps | ✓ Strict gate enforced |
| ≤ 1 h runtime per benchmark | ✓ Max 37.3 min observed |

## Contact

- Email: kaushalchitturu@gmail.com
- LinkedIn: https://www.linkedin.com/in/kaushal-chitturu/
- GitHub: [@kc5587](https://github.com/kc5587)
