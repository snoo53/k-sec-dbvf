# k-SEC Nature-tier Roadmap

Target venue: **npj Computational Materials** (IF ~13) first, **Nature
Communications** (IF ~17) stretch. Without wet-lab validation, Nature
main is out of reach; this roadmap maximizes what's achievable
computationally.

Timeline: **7 months solo at 20 hr/week** (~560 hours of research + ~200
GPU-days of consumer-hardware compute).

Commitment: **full pivot** — SaaS work deferred during this period.

---

## Summary of work packages

| WP | Focus | Duration | Deliverable |
|---|---|---|---|
| WP1 | Data breadth (~2k samples across 3 ion chemistries) | 1–2 mo | unified `data/σ_unified.parquet` |
| WP2 | Matbench generalization (demonstrate method beyond σ) | 2–3 mo | top-3 on 2+ Matbench tasks |
| WP3 | Multi-fidelity learning (DFT → MACE → exp σ) | 2 mo | MAE ≤0.85 on σ |
| WP4 | Interpretability + scientific insight | 1–2 mo | Filter-attribution figures + correlations with known physics |
| WP5 | Computational "validation" (screening + temporal holdout + DFT cross-check) | 1 mo | Top-50 MP Li candidate list + MACE-barrier correlation |
| WP6 | Rigor + reproducibility | concurrent | 10-seed ensembles, conformal intervals, dockerized release |

---

## Critical path & milestones

**Month 1 — WP1 complete**
- [ ] Hargreaves + OBELiX unified with temperature as input and Arrhenius output
- [ ] Na-ion literature mined (target ≥300 entries)
- [ ] LLM-extracted post-2023 σ triples (target ≥200 entries)
- [ ] Cross-dataset deduplication by reduced formula
- [ ] **~1800 total σ-labelled samples across Li+Na+K chemistries**

**Month 2 — WP3 pipeline alive**
- [ ] 50k non-Li MP crystal fetch (broadens encoder beyond Li)
- [ ] CHGNet-based Li migration barrier computation on 100 representative CIFs
- [ ] Three-fidelity training harness: MP (DFT) → MACE (mid) → experimental (high)
- [ ] Per-seed MAE target ≤0.95 on unified σ set

**Month 3 — WP2 Matbench wins**
- [ ] MP-E-form: top-3 leaderboard position
- [ ] MP-Gap: top-5 leaderboard position
- [ ] At least one additional Matbench task evaluated

**Month 4 — WP3 + WP6**
- [ ] MAE on unified σ ≤0.85 (target)
- [ ] 10-seed ensemble with std ≤0.025
- [ ] Conformal prediction intervals (target coverage 95% ± 2%)

**Month 5 — WP4 + WP5**
- [ ] Cubic-harmonic filter attribution analysis; visualize learned symmetry channels
- [ ] Correlate k-SEC embeddings with known physics descriptors (bottleneck radii, BV pathway)
- [ ] Virtual screening on 150k MP Li crystals → top-50 candidates
- [ ] Temporal holdout: 30 post-2023 SSE candidates evaluated
- [ ] DFT cross-check on top-10 predictions (MACE/CHGNet barriers)

**Month 6 — writing + WP6 polish**
- [ ] Full manuscript with SI
- [ ] Docker image + HuggingFace model weights
- [ ] Dataset release (unified σ parquet + MP fetch code)
- [ ] Pre-registered evaluation protocol

**Month 7 — submission**
- [ ] Submit to **npj Computational Materials** (primary target)
- [ ] Prepare Nature Communications version as fallback
- [ ] Respond to reviewers; target 1 revision cycle

---

## Execution risks (ranked by probability × impact)

| Risk | Mitigation |
|---|---|
| WP3 multi-fidelity doesn't reduce MAE | fall back to the current stacked-ensemble result, downgrade target to JMST |
| Na-ion literature too sparse | broaden scope to include polymer + glassy electrolytes (different physics, but more data) |
| Matbench tasks too computationally heavy | precompute features once; cache aggressively |
| MP pretraining scaling doesn't transfer to σ | try feature-level fusion: concatenate CGCNN embeddings into k-SEC stream |
| Time overrun | cut WP4 first (interpretability is nice-to-have, not critical) |

---

## What "done" looks like

A 20–30 page manuscript titled something like:

> **k-SEC: a reciprocal-space equivariant architecture for crystal
> property prediction, with multi-fidelity transfer learning for ionic
> conductivity**

With:
- Clear architectural novelty (cubic-harmonic filters + cross-shell gated attention)
- Multi-fidelity framework (DFT + MACE + experimental)
- Matbench competitiveness on 2+ tasks
- Ionic conductivity prediction beating LightGBM+Magpie *standalone* on unified σ set
- Virtual screening output + computational validation
- Full reproducibility stack

Honest acceptance probability at npj CM: **~55–65%** if all six work
packages complete cleanly. At Nature Communications: **~20–30%** as
primary, **~40%** as revised fallback from npj CM rejection.
