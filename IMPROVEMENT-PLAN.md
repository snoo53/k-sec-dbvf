# k-SEC improvement plan

Plan for pushing standalone MAE below the LightGBM+Magpie ceiling (1.099)
and/or improving the scientific story for JMST. Ordered by expected
MAE delta ÷ engineering cost. Each item lists: expected result, cost,
risks.

## Current standing

| Model | MAE | R² | AUC | Notes |
|---|---|---|---|---|
| LightGBM + Magpie (ceiling) | 1.099 | 0.606 | 0.918 | — |
| **k-SEC Hybrid ensemble (best)** | **1.195** | **0.594** | **0.886** | 5 seeds |
| k-SEC Hybrid + Magpie pretrain | 1.205 | 0.581 | 0.882 | null result |
| k-SEC v2 per-seed | 1.374 | 0.458 | 0.854 | — |

Gap to ceiling: 0.096 (8.7 %). Data scale: **n = 285** (OBELiX CIF-bearing
labelled entries at 298 K).

---

## Group A — architecture (keep the same data)

### A1. Lattice-matrix conditioning — MAE, OOD  (HIGH priority)

**The issue.** k-SEC currently uses fractional coordinates + atomic Z but
**discards the 3×3 lattice matrix entirely**. k-points are built on a unit
cube of integer indices. This means a cubic cell of 4 Å edge and a cubic
cell of 12 Å edge produce the same k-space features given identical
fractional coords — a very large information loss.

**Fix.** Feed the lattice matrix through a small MLP, use its output as
(a) an additive bias to the k-space filter, (b) a scaling factor on |k|,
and (c) a concat feature at readout. Alternative: construct real-space
k-points by multiplying integer indices by the reciprocal lattice.

**Expected.** 5–10 % MAE lift. This is the single biggest missed signal.

**Cost.** Half a day of code + retrain. Low risk.

### A2. Heteroscedastic output head — calibration, R², maybe MAE  (HIGH)

Replace MSE with NLL = ½ (log σ² + (y − μ)²/σ²). Already stubbed in
[scripts/08_train_hybrid.py](scripts/08_train_hybrid.py) via
`HybridHead`; just needs `--hetero` flag retrain. Addresses the
catastrophic MC-dropout coverage (6.7 % @ 1σ).

**Expected.** 2–4 % MAE (noise-downweighting), 0.05–0.10 R² improvement,
well-calibrated uncertainty.

**Cost.** 2 GPU-hours. Very low risk.

### A3. Stronger k-space resolution  (MED)

Current `n_max = 2` → ~30 k-points. Bump to `n_max = 3` → ~80 k-points.
Cross-shell attention is O(K²), so ~7× more memory; requires batch size
halving. Probably 2–4 % MAE.

**Cost.** 1 GPU-day. Moderate risk (GPU OOM on slow folds).

### A4. Graph-conv parallel path + fuse at readout  (MED)

Add a CGCNN-style real-space message-passing branch in parallel to the
k-SEC branch, concat at readout. Different inductive biases may provide
complementary gradients. Common strategy in successful models
(ReGNet does this).

**Expected.** 3–6 % MAE. But the architecture becomes more complex —
may be harder to publish as "one new component."

**Cost.** 2–3 days.

### A5. Deeper 10-seed ensemble  (LOW)

Diminishing returns. 5 → 10 seeds typically 2–3 % extra.

**Cost.** 4 GPU-hours.

---

## Group B — data (expand or curate the input)

The single most impactful lever. At **n = 285** we are structurally in
the tabular-dominance regime. Options, ordered by feasibility:

### B1. Materials Project pretraining — MAE, OOD  (HIGHEST single lever)

**What.** Use the MP API to download ~150k crystals with formation
energy, bandgap, stability. Pretrain k-SEC on these multi-task targets
(CIF → E_formation + bandgap, MSE). Fine-tune on OBELiX σ.

**Why.** A 150k-sample pretraining is ~500× our data. The k-space
features learned during pretraining encode generic crystal structure
information (symmetry, chemistry, density) that transfers across
downstream tasks. This is the "more data" lever.

**Expected.** 8–15 % MAE. If standalone MAE drops to ~1.00–1.10, the
JMST story changes from "publishable with caveats" to "clean beat on
all baselines."

**Cost.** 1–2 days: MP downloader + curation (filter to crystals with
both targets, remove too-large cells, handle spin) + pretraining run
(4–8 GPU-hours for 150k samples, 40 epochs).

**Risks.** MP requires a free API key (user provides). Some MP
structures are too large for n_max=2 k-grid without truncation.

### B2. Temperature-conditioned Hargreaves joint training  (MED-HIGH)

**What.** Hargreaves dataset has 820 entries across 5–873 °C. Currently
we use only 465 near-RT entries as pretraining. Instead:
1. Include ALL 820 entries in hybrid training as a separate composition-
   only task head, with temperature as an input feature.
2. Parameterize the model to predict σ(T) via an Arrhenius form:
   `log σ = log σ_0 − E_a / (k_B T)`, with the network predicting log σ_0
   and E_a.
3. Auxiliary loss on Arrhenius residuals + direct log σ loss.

**Why.** 820 extra composition-only samples, and the Arrhenius structure
gives the model a physics-informed inductive bias.

**Expected.** 5–10 % MAE on OBELiX (by better-regularizing the shared
Magpie head through a much larger auxiliary dataset).

**Cost.** 2 days: multi-task head + Arrhenius param + retrain.

**Risks.** The 355 non-RT Hargreaves entries have temperature that needs
careful handling (C vs. K).

### B3. Bond-valence-sum (BV) pathway features — MAE, AUC  (HIGH)

**What.** BV-pathway analysis identifies plausible Li migration channels
from CIF geometry alone (no DFT). Per-crystal scalars:
- percolation threshold (minimum bottleneck radius for connected path)
- channel dimensionality (1D/2D/3D)
- site-multiplicity stats

Pymatgen has `bvanalyzer`; a CIF takes ~1–10 s to analyze.

**Why.** These features are explicitly geometric and directly
causally tied to ionic conductivity. Not compressible from Magpie.

**Expected.** 3–7 % MAE on top of hybrid. May help OOD (geometric
features are transferable across chemical families).

**Cost.** 1 day: BV analyzer + per-sample featurization + concat to
readout.

**Risks.** Some CIFs may fail BV analysis (undefined oxidation states).

### B4. Na-ion and K-ion conductor transfer learning  (MED)

**What.** Na-ion solid electrolyte literature (e.g., Na-β-alumina,
Na₃PS₄, NaSICON) overlaps significantly with Li chemistry. Pretrain k-SEC
on a combined Na+Li dataset, fine-tune on Li-only OBELiX.

**Why.** Triples the training set; chemistry is closely related.

**Expected.** 3–6 % MAE.

**Cost.** 3–4 days: dataset curation is the bottleneck (no OBELiX-style
curation for Na-ion yet).

### B5. DFT-computed Li migration barriers as auxiliary target  (HIGH effort)

**What.** For a subset of OBELiX CIFs, run NEB calculations (via VASP or
MACE) to get Li activation energies. Use as a multi-task regression
target.

**Why.** E_a directly determines σ at a given T; this is the most
physically informative auxiliary target.

**Expected.** If 50–100 NEB samples available: 5–10 % MAE on OBELiX.

**Cost.** 1+ weeks of DFT compute (heavy). Unless we use MACE or CHGNet
as a cheap NEB surrogate.

### B6. CALiSol-23 electrolyte dataset  (LOW relevance — liquid)

Note: CALiSol-23 is liquid electrolyte data, not SSE. Different problem;
skip.

---

## Group C — training / optimization

### C1. Cosine warm-restart + SWA  (LOW)

Standard tricks. 1–3 % MAE. Low priority given bigger levers exist.

### C2. Two-LR setup (separate LR for k-SEC vs. Magpie branches)  (MED)

Magpie branch already has pretrained init; it should fine-tune at a
lower LR than the k-SEC branch which trains from scratch. Simple
implementation via AdamW param groups.

**Expected.** 2–4 % MAE improvement when combined with B2/B3.

### C3. Heteroscedastic + SWA + larger ensemble combined  (MED)

Stack all three training tricks. Compounding ~5–8 %.

---

## Group D — evaluation / scientific story

### D1. Complete OOD-by-family (13 families)

Currently have 3/13 done. Run the remaining 10. This is pure narrative
strength for JMST — doesn't move any single MAE, but addresses a
reviewer concern directly.

**Cost.** 5–8 GPU-hours. Parallelizable.

### D2. Interface-stability companion prediction

Reframe k-SEC as a two-head model:
- Head 1: log σ (current task)
- Head 2: interfacial decomposition energy with common cathodes (LFP, LCO)

The Head-2 labels don't exist at scale yet — would need a custom data
effort. Not immediate.

---

## Recommended sequence

Given that **R² and AUC are already near-parity with LightGBM** and only
MAE lags by 9 %, the focused path is:

### Phase 1 — JMST-ready (within 1 week)

1. **A1 (lattice conditioning)** — biggest pure-architecture miss.
2. **A2 (heteroscedastic loss)** — fixes the MC-dropout calibration problem
   and likely 2–4 % MAE.
3. **B3 (BV pathway features)** — physics-informed, complementary to
   Magpie.
4. Stack with LightGBM corroboratively (already half-done in
   scripts/10_stacking.py).
5. **D1 (finish OOD)** — narrative completion.

Target: MAE **1.00–1.05** standalone, stacked 0.95–1.00, R² > 0.62,
AUC > 0.90. Defensible clean beat.

### Phase 2 — ambitious follow-up (next 1–2 months)

1. **B1 (MP pretraining)** — the real ceiling-break, requires API key.
2. **B2 (Hargreaves multi-task + Arrhenius)**
3. **A4 (real-space parallel path)**

Target: MAE **< 0.90**, publishable at higher tier (npj Computational
Materials, ACS Energy Letters).

### Phase 3 — research direction (6+ months)

1. **B4 (cross-ion transfer learning)**
2. **B5 (NEB surrogate multi-task)**
3. **D2 (interface stability head)**

This is thesis-scale work — the research direction discussed in the
previous strategy conversation.
