# K2HAL — Macro Placement Challenge 2026 Submission

**Team:** K2HAL · Kaushal Chitturu ([kc5587](https://github.com/kc5587))
**Method:** CD-LNS with Hessian Saddle Escape
**Tag:** [`v1.0-submission`](https://github.com/kc5587/K2HAL-macro-place-challenge/releases/tag/v1.0-submission) (commit `69a7673`)
**License:** Apache 2.0 (see [LICENSE.md](LICENSE.md))

---

## Method (one paragraph)

A multi-restart placer that warm-starts from the contest's `initial.plc`, polishes
each candidate with coordinate descent (CD) and large-neighborhood search (LNS),
and escapes local minima with a Hessian-based saddle escape — a deterministic
move along the most-negative-curvature direction of the proxy cost, computed by
combining a block-diagonal eigen-decomposition with a Lanczos
Rayleigh–Ritz refinement. Candidates are legalized through a tiered repair stack
(local repair → minimum-disturbance → shelf-pack fallback) and selected by
`(overlap_count, proxy_cost)` so legality strictly precedes score. The result is
a fully open-source, GPU-free placer that produces zero-overlap legal placements
on every IBM benchmark and a substantial proxy-cost improvement over the
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

## Technical highlight — Hessian Saddle Escape

The novel contribution worth noting for the Innovation Award is the
Hessian-guided saddle escape. After CD converges, we form a sparse approximation
of the proxy-cost Hessian, take the eigenvector of its smallest eigenvalue (via
a fast block-diagonal eigendecomposition, then refined with a Lanczos
Rayleigh–Ritz pass), and perform a 1-D line search along ±that direction. When
the smallest eigenvalue is negative — which empirically happens on most
benchmarks once CD stalls — this provably exits the local minimum into a lower
basin. The escape is deterministic (no random perturbation), bounded by a
single eigen-solve per restart, and only accepted when the new placement
strictly improves the official proxy cost.

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
