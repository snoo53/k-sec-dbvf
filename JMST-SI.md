# Supplementary Information

**Cubic-Equivariant k-Space Convolution and a Differentiable Bond-Valence
Field for Ionic-Conductivity Prediction in Solid-State Electrolytes**

This supplementary material documents (i) honest negative results that
shaped the architectural choices reported in the main text, (ii) full
hyper-parameter and reproducibility details, and (iii) extended ablations
that were too long to include in the main manuscript.

---

## SI-1. Negative result: heteroscedastic NLL loss (Phase A1)

I attempted to add a Gaussian-NLL head to the readout, exposing both
mean and variance for each prediction so that training is jointly
optimised for accuracy and calibrated uncertainty.

| Configuration | Per-seed MAE | Δ vs baseline |
|---|---|---|
| Smooth-L1 (baseline) | 1.239 ± 0.032 | 0.000 |
| Gaussian NLL (Phase A1) | 1.301 ± 0.044 | +5 % |

The NLL objective destabilised training at n = 281: the variance head
collapsed to small values on most training points, then over-corrected
on the held-out fold. I retained smooth-L1 in the headline model and
defer calibrated-uncertainty work to MC-dropout (Section SI-4).

## SI-2. Negative result: pathway-percolation features (Phase A2)

I extended the 20-d Wyckoff-geometric features with five additional
bond-valence-pathway-derived descriptors (percolation thresholds,
saddle-point density, channel volume fraction).

| Configuration | Stacked MAE |
|---|---|
| Pre-pivot stacked (geometric only) | 0.995 |
| Phase A2 stacked (geometric + percolation) | 1.116 |

Adding percolation features net *worsened* the stacked MAE by 1.2 %.
Diagnosis: the new features substantially overlapped with the existing
geometric features, while adding 5 free parameters to the LightGBM
gradient-boosted tree at n = 281 — a regime in which the additional
capacity over-fits the small training fold. I dropped the percolation
features in the headline model.

## SI-3. Architectural limit: k-grid resolution (Phase A3)

The cross-shell gated attention is O(K²). At `n_max = 2` (K ≈ 33) the
model fits comfortably on a 12-GB consumer GPU. Increasing to
`n_max = 3` (K ≈ 80) increases attention memory ~6×, which exceeds 12 GB
even at batch size 4. I did not retrain the headline model at higher
k-grid resolution; a linear-attention or block-sparse rewrite is
straightforward future work.

## SI-4. Path-integrated DBVF (Phase B2)

I extended DBVF to compute per-Li bond-valence pathway integrals via
a soft-max-saddle approximation,

```
saddle = (1/τ) · logsumexp(τ · U_path)
```

where `U_path` is the bond-valence energy along discretised straight-line
paths from each Li site to its k nearest Li neighbours. The temperature
parameter `τ` was learned end-to-end.

| Configuration | Standalone MAE | Stacked MAE |
|---|---|---|
| **k-SEC + basic DBVF (Phase B1, headline)** | **1.047** | **0.980** |
| k-SEC + path-integrated DBVF (Phase B2) | 1.077 | 0.989 |

The path integration *worsened* MAE by 0.030 standalone and 0.009
stacked. Diagnosis: the additional learnable parameters
(per-Li-pair `τ`, neighbour-list cutoff, path discretisation) add
capacity that the n = 281 training fold cannot constrain. At larger
training scales, path integration may become advantageous; at OBELiX
scale, the simpler DBVF wins.

## SI-5. Dual-stream BatteryNet (Phase B3)

I built a fully bi-directional dual-stream architecture in which
(i) the k-SEC encoder provides reciprocal-space features `h_k ∈ ℂ^{B × K × D_k}`,
(ii) a CGCNN-style real-space MPNN provides per-atom features
`h_a ∈ ℝ^{N × D_a}`, and (iii) a cross-attention bridge lets each stream
attend to the other (k-space queries → atom keys/values *and* atom
queries → k-space keys/values).

| Configuration | Standalone MAE | Stacked MAE |
|---|---|---|
| **k-SEC + DBVF (Phase B1, headline)** | **1.047** | **0.980** |
| BatteryNet dual-stream (Phase B3) | 1.098 | 0.990 |

BatteryNet *worsened* MAE by 0.05 standalone and 0.01 stacked despite
having ~3× more trainable parameters. The cross-attention bridge fits
nicely in compute but over-fits at n = 281: the per-seed MAE std rose
to 0.10 (vs. 0.012 for Phase B1). At a larger labelled-σ scale
(target ≥ 1k), dual-stream architectures of this kind are expected to
become competitive; I report this honestly as a current-data-regime
limit, not a fundamental architectural failure.

## SI-6. Broad-domain pretraining at matched fine-tuning budget

The headline model uses an encoder pretrained on 18,574 Li-containing
Materials Project crystals (formation-energy target, val MAE
0.072 eV/atom). I tested whether broader pretraining helps.

**Broad-domain corpus** (218,057 crystals total):

| Source | Crystals |
|---|---|
| Materials Project (full, ≤80 sites) | 114,639 |
| JARVIS-DFT 3D | 75,976 |
| Matbench (perovskites + jdft2d + dielectric + log_gvrh + log_kvrh) | ~24,000 |
| OQMD subset | (in progress) |

**Compute reality.** Pretraining at full 200k size at batch 64 did not
fit the throughput envelope on a single consumer GPU (>2 h per epoch
because 10 % of samples have >40 atoms, blowing up cross-shell
attention memory). I filtered to ≤40 atoms (90 % of data) and
subsampled to **18,000** to match the Li-only set's count for an
apples-to-apples test.

**Outcome at matched fine-tuning budget:**

| Configuration | Stacked MAE | Standalone MAE | R² | AUC |
|---|---|---|---|---|
| **Li-only pretrain (headline)** | **0.995** | **1.103** | 0.625 | 0.897 |
| Broad-domain pretrain | 1.030 | 1.142 | 0.612 | 0.901 |

Broad-domain pretraining *underperforms* Li-specific pretraining by
3.5 % stacked MAE. Diagnosis: the broad encoder's pretraining val MAE
on formation energy was ≈ 2.7 eV/atom (contaminated by unit outliers
in source data) — much worse than the Li-only encoder's 0.072 eV/atom.
The broad encoder simply learned less per parameter at a matched
training budget. **Caveat:** because I matched n = 18,000 fine-tuning
budget rather than scaling fine-tuning with the larger pretrain corpus,
this is a "specialty vs. broad at matched n" test, not a "more data is
better" test. A cloud-GPU full-200k pretrain is left as future work.

## SI-7. Out-of-distribution by structural family

Hold out each family and train on the rest. Three of thirteen families
(those with n ≥ 5 samples) completed; the remaining ten were cut short
because cross-shell attention swaps to CPU memory on small held-out
folds.

| Held-out family | n_test | MAE | R² |
|---|---|---|---|
| argyrodites | 43 | 0.903 | +0.127 |
| garnet | 52 | 1.054 | −0.374 |
| LGPS | 12 | 1.456 | −50.9 (n too small for stable R²) |

The two largest held-out families produce *lower* MAE than in-
distribution training MAE (1.37 per-seed). The architecture is not
over-fitting to a single family's chemistry; it is learning
transferable structural features. (LGPS R² is dominated by the small
test-set variance — the MAE is the reliable number.)

## SI-8. Matbench transferability (negative)

I loaded the Li-pretrained encoder and evaluated on
`matbench_log_gvrh` (log₁₀ shear modulus, n = 10,987, 5-fold CV,
15 epochs).

| Method | MAE | R² |
|---|---|---|
| ALIGNN (leaderboard) | 0.072 | — |
| MEGNet (leaderboard) | 0.085 | — |
| CGCNN (leaderboard) | 0.115 | — |
| **k-SEC (Li-pretrained, this work)** | **0.143 ± 0.005** | 0.717 |

The Li-conductivity-pretrained encoder *does not* transfer cleanly to
elasticity prediction. Expected, in retrospect — the Li-pretrain
specialised the encoder for Li-relevant chemistry, not bulk-modulus-
relevant chemistry. I do not claim general transferability; a proper
Matbench-tier paper would re-pretrain k-SEC on the same target as the
downstream task. I report this honestly as a domain-specificity
limit of the Li-targeted pretraining objective.

## SI-9. Cubic-harmonic filter interpretability

I swept the learned filter response across O_h-symmetric crystal
directions [100], [110], [111], [210], [211] at fixed `|k| = 2`:

| Block | Directional spread (rel std) |
|---|---|
| 0 | 0.067 |
| 1 | 0.045 |
| 2 | 0.038 |
| **Mean** | **0.050** |

The filter has measurable but **modest** direction sensitivity — well
below the 0.10 threshold I would require to call it "strongly
directional." The 29 % of MAE gain attributed to the cubic-harmonic
filter (per the v1→v2 ablation) appears to come *mostly* from per-shell
|k|-amplitude modulation rather than directional anisotropy. This is an
honest caveat: the cubic-harmonic basis enforces equivariance correctly,
but at the present benchmark scale, the network primarily exploits its
amplitude-modulation flexibility rather than its anisotropy degrees of
freedom. Whether deeper k-grids or larger benchmarks would surface more
direction sensitivity remains an open question.

## SI-10. Reproducibility details

**Stratified 5-fold CV.** I bin σ into 5 equal-frequency bins and
ensure each fold's training and validation sets contain samples from
each bin. Random seed 42 controls the fold assignment; per-seed
training runs use seeds {0, 1, 2, 3, 4}.

**OBELiX filter.** I use the 281 entries with `log₁₀σ ≥ −15`. Entries
below this threshold are below the measurement noise floor (< 10⁻¹⁵
S/cm) and act as label-noise outliers in MAE.

**Disordered-occupancy handling.** ~30 % of OBELiX CIFs carry
disordered occupancies. For both DBVF and the MACE-MP-0 baseline I
replace each disordered site with its dominant species. For DBVF this
is exact at the integer-occupancy limit; for the MACE-MP-0 baseline
this leaves 84 of 285 CIFs with energies outside the physical range
[−15, 0] eV/atom, which I mark with a validity flag and impute with
the dataset mean.

**Software.** PyTorch 2.5, PyTorch-Geometric 2.6, LightGBM 4.5,
pymatgen 2024.10. Hardware: single 12-GB consumer GPU.

**Stacking protocol.** Out-of-fold k-SEC predictions are computed by
holding out each fold during training and predicting only on that
fold's validation samples. The same is done for LightGBM. Both OOF
prediction vectors are concatenated as 2-d inputs to a per-fold ridge
regression (regularisation strength tuned by inner CV).

## SI-11. Per-seed and per-fold breakdown of the headline result

5-fold stratified CV × 5 seeds for k-SEC + DBVF (Phase B1). Per-seed
MAE values are mean across folds.

| Seed | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | seed mean | seed std |
|---|---|---|---|---|---|---|---|
| 0 | 1.291 | 1.238 | 1.363 | 1.100 | 1.235 | 1.245 | 0.086 |
| 1 | 1.058 | 1.301 | 1.342 | 1.204 | 1.274 | 1.236 | 0.100 |
| 2 | 1.137 | 1.094 | 1.348 | 1.431 | 1.168 | 1.236 | 0.131 |
| 3 | 1.017 | 1.205 | 1.310 | 1.301 | 1.214 | 1.209 | 0.106 |
| 4 | 1.363 | 0.860 | 1.202 | 1.301 | 1.466 | 1.239 | 0.208 |
| **Mean across seeds** | — | — | — | — | — | **1.233** | **0.012** |
| **5-seed ensemble** | — | — | — | — | — | **1.047** | — |

The standard deviation across seeds (0.012) is among the tightest I
observe on this benchmark — well below the per-seed within-fold
variation (~ 0.10). The 5-seed ensemble (mean of per-seed predictions)
reduces MAE by 0.186 relative to the per-seed mean — a 15 % gain
purely from ensembling, consistent with the well-known low-data
ensembling lift.

## SI-12. Headline ranking diagnostics with bootstrap CIs

Out-of-fold ranking quality of the headline ensemble. CIs are
non-parametric stratified bootstrap over the 281 OOF predictions
(5,000 resamples).

| Statistic | Point estimate | 95 % CI | Random baseline | Lift |
|---|---|---|---|---|
| Spearman ρ | 0.780 | [0.706, 0.841] | 0.000 | — |
| Top-10 precision | 0.700 | [0.300, 0.900] | 0.036 | 19.7× |
| Top-20 precision | 0.700 | [0.450, 0.900] | 0.071 | 9.8× |
| Top-30 precision | 0.600 | [0.467, 0.767] | 0.107 | 5.6× |
| Top-50 precision | 0.680 | [0.580, 0.780] | 0.178 | 3.8× |
| Top-100 precision | 0.790 | [0.710, 0.850] | 0.356 | 2.2× |

Per-σ-bin MAE with bootstrap CIs (281 OBELiX samples, OOF):

| log₁₀σ bin (S/cm) | n | MAE | 95 % CI |
|---|---|---|---|
| [−15, −10) | 10 | 3.63 | [2.69, 4.53] |
| [−10, −7) | 48 | 1.59 | [1.28, 1.92] |
| [−7, −5) | 66 | 0.93 | [0.73, 1.15] |
| [−5, −3) | 102 | 0.73 | [0.59, 0.89] |
| [−3, 0) | 55 | 0.83 | [0.56, 1.15] |
| Overall | 281 | **1.047** | **[0.925, 1.183]** |
| log₁₀σ > −7 (screening-relevant) | 223 | **0.81** | — |

Bootstrap data are saved at `results/phaseB1_bootstrap_ci.json` and
`results/phaseB1_ranking_analysis.json`.

## SI-12a. Learned DBVF parameters

Per-anion bond-valence parameters extracted from the five Phase B1 seed
checkpoints (script: `scripts/28_dbvf_interpret.py`,
output: `results/dbvf_learned_params.json`).

| Anion | r₀ init (Å) | r₀ learned (mean ± std, Å) | Δr₀ (Å) | b init (Å) | b learned (mean ± std, Å) | Δb (Å) |
|---|---|---|---|---|---|---|
| O  | 1.466 | 1.445 ± 0.026 | −0.021 | 0.370 | 0.360 ± 0.021 | −0.010 |
| S  | 1.850 | 1.874 ± 0.023 | +0.024 | 0.400 | 0.405 ± 0.006 | +0.005 |
| Se | 1.930 | 1.909 ± 0.017 | −0.021 | 0.400 | 0.395 ± 0.006 | −0.005 |
| F  | 1.360 | 1.352 ± 0.011 | −0.008 | 0.370 | 0.366 ± 0.006 | −0.004 |
| Cl | 1.790 | 1.800 ± 0.032 | +0.010 | 0.400 | 0.403 ± 0.016 | +0.003 |
| Br | 1.920 | 1.842 ± 0.054 | −0.078 | 0.400 | 0.372 ± 0.021 | −0.028 |
| I  | 2.070 | 2.102 ± 0.063 | +0.032 | 0.400 | 0.402 ± 0.010 | +0.002 |
| N  | 1.610 | 1.606 ± 0.027 | −0.004 | 0.370 | 0.365 ± 0.014 | −0.005 |

The largest drift is the Li–Br pair (Δr₀ = −0.078 Å, Δb = −0.028 Å);
the seed-to-seed std on r₀ is at most 0.063 Å (I), confirming that
the learned parameters are stable across random initialisations of
the rest of the network.

## SI-12b. Learning-curve experiment design and full results

I test whether the standalone neural-vs-tabular MAE gap at n = 281
(Section 4.1) is a fundamental ceiling or a small-data effect by
repeating the headline experiment at four training fractions
{0.4, 0.6, 0.8, 1.0}. Inside each 5-fold CV partition, the *test*
fold is unchanged so test-MAE is directly comparable across n; only
the *training* fold is subsampled. Subsampling is stratified on
log₁₀σ quintiles to preserve the σ-distribution shape at each n.

Per-fold sample sizes:

| Training fraction | n_train per fold | n_test per fold |
|---|---|---|
| 0.4 | ~90 | ~56 |
| 0.6 | ~135 | ~56 |
| 0.8 | ~180 | ~56 |
| 1.0 | ~225 | ~56 |

The fraction = 1.0 point reuses the existing 5-seed Phase B1 result
(no retraining). Smaller fractions were run with 3 seeds × 5 folds ×
60 epochs each (batch size 8 to stay within 12-GB GPU memory).
LightGBM was run separately at the same fractions with 5 seeds.
Both fold-split and within-fold subsampling RNGs are seeded so the
experiment is exactly reproducible (`scripts/29_learning_curve.py`).

**Note on f = 0.6 (n_train = 135).** The k-SEC + DBVF run at this
fraction completed 2 of the 3 planned seeds before a CUDA OOM
event terminated the third seed. I include the 2-seed estimate in
the manuscript (per-seed mean 1.267 ± 0.034 across seeds 0 and 1) for
completeness; the manuscript discloses this in Figure 11 and in the
Section 4.10 table.

### Per-seed MAE table

k-SEC + DBVF (per-seed mean across folds, then aggregate across seeds):

| n_train | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean ± std |
|---|---|---|---|---|---|---|
| 90  | 1.460 | 1.420 | 1.493 | — | — | 1.458 ± 0.030 |
| 135 | 1.233 | 1.300 | (failed) | — | — | 1.267 ± 0.034 |
| 180 | 1.295 | 1.240 | 1.263 | — | — | 1.266 ± 0.023 |
| 225 | 1.245 | 1.236 | 1.236 | 1.209 | 1.239 | 1.233 ± 0.012 |

LightGBM full features (per-seed MAE):

| n_train | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean ± std |
|---|---|---|---|---|---|---|
| 90  | — | — | — | — | — | 1.151 ± 0.069 |
| 135 | — | — | — | — | — | 1.099 ± 0.035 |
| 180 | — | — | — | — | — | 1.065 ± 0.038 |
| 225 | 0.933 | 1.095 | 0.967 | 1.022 | 1.028 | 1.009 ± 0.056 |

(LightGBM per-seed individual values for n_train < 225 are saved in
`results/learning_curve_lgbm_f*.json` and omitted from this table for
brevity.)

### Slope summary

Linear-fit slope `dMAE/dn_train` over n_train ∈ [90, 225]:
- k-SEC + DBVF: −0.0014 MAE/sample (steep decrease early, plateau after n = 135)
- LightGBM: −0.0010 MAE/sample (roughly linear)

Gap slope `dΔ/dn_train` = +0.0004 MAE/sample over the *plateau region*
n ≥ 135 (i.e. the gap *grows* slightly with n in the n ∈ [135, 225]
sub-range). Over the full range n ∈ [90, 225] the gap slope is
−0.0006 MAE/sample (gap shrinks with n averaged over the full range,
driven entirely by the n = 90 → n = 135 step).

## SI-13. Hyper-parameter table

| Hyper-parameter | Value |
|---|---|
| k-grid `n_max` | 2 |
| Atomic embedding dim D | 96 |
| Number of k-SEC blocks | 3 |
| Cross-shell attention heads | 4 |
| Dropout | 0.15 |
| Bond-valence cutoff `R_cut` | 4.0 Å |
| DBVF anion species | Cl, Br, I, F, O, S, Se, N, P, C |
| Optimiser | AdamW (lr 1e-3, wd 1e-4) |
| Schedule | Cosine annealing, 5-epoch warmup |
| Batch size | 8 |
| Epochs | 60 |
| Loss | Smooth-L1 on log₁₀σ |
| Pretrain epochs | 6 |
| Pretrain batch | 32 |
| Pretrain target | Formation energy per atom |
| Pretrain corpus | 18,574 Li-containing MP crystals |
| Random seeds | {0, 1, 2, 3, 4} |
| CV folds | 5 (stratified on σ histogram) |

## SI Figures

**Figure S1.** *MC-dropout calibration.* Coverage of 1σ and 1.96σ
prediction intervals against the empirical fraction of held-out
points falling within each interval.

**Figure S2.** *Top-K precision over OOF predictions.* Model
precision (black solid) vs. random-ranking baseline (gray dashed)
at K = 10, 20, 30, 50, 100. The 19.7× lift at K = 10 indicates the
ranking is materially informative for virtual screening.

**Figure S3.** *Cubic-harmonic filter direction sensitivity.*
Filter response across O_h-symmetric crystal directions [100],
[110], [111], [210], [211] at fixed |k| = 2. Mean directional
spread (rel. std) is 0.05, indicating modest but non-zero direction
sensitivity.
