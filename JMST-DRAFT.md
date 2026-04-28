# Cubic-Equivariant k-Space Convolution and a Differentiable Bond-Valence Field for Ionic-Conductivity Prediction in Solid-State Electrolytes

**Sunwoo Lee**¹

¹ Independent researcher, South Korea. ORCID: [0009-0004-9159-367X](https://orcid.org/0009-0004-9159-367X). Correspondence: lee.11539@buckeyemail.osu.edu

## Highlights

- k-SEC: cubic-equivariant reciprocal-space encoder for crystal property regression.
- DBVF: end-to-end-learnable bond-valence module embedded in a neural network.
- OBELiX neural-method MAE: 1.047 standalone, 0.980 stacked (n = 281, 5-fold CV).
- DBVF features alone do not help LightGBM — the gain is architectural, not tabular.
- Unsupervised top-15 screen of 18,574 MP crystals recovers known fast-Li families.

## Graphical abstract

A two-stream architecture in which a reciprocal-space encoder built on
cubic-harmonic directional filters and a differentiable bond-valence
field operating on real-space Li environments are trained jointly,
end-to-end, to predict log₁₀σ of solid-state Li-ion electrolytes. A
virtual screen on 18,574 Materials Project crystals identifies the four
canonical fast-Li-ion conductor families in its top-15 without
family-level supervision.

---

## Abstract

Predicting the room-temperature ionic conductivity σ of solid-state
lithium-ion electrolytes from crystal structure remains a bottleneck
for materials discovery. I introduce two architectural primitives.
**k-SEC** is a neural encoder whose feature maps live in reciprocal
space throughout the network and whose cubic-group equivariance is
enforced by construction via cubic-harmonic directional filters and a
cross-shell gated attention modulated only by k-magnitudes.
**DBVF** (Differentiable Bond-Valence Field) embeds Brown's bond-
valence sum into the network as a learnable module — the first
treatment of Brown's tabulated parameters as trainable parameters of
a neural network. On the OBELiX benchmark (281 labelled solid
electrolytes, 5-fold cross-validation × 5 seeds) the joint model
attains MAE 1.047 [95 % CI 0.925–1.183] standalone and 0.980 in a
stacked ensemble with a gradient-boosted-tree baseline; Spearman ρ
= 0.78 with 70 % top-10 precision. An unsupervised screen of 18,574
Li-containing Materials Project crystals identifies all four known
fast-Li-ion conductor families in its top-15. A control experiment
shows the DBVF gain is realised only through end-to-end training; a
learning-curve experiment shows a 45 % gap reduction to the tabular
baseline as training samples grow from 90 to 135 per fold, before
plateauing at OBELiX scale.

---

## 1. Introduction

Solid-state electrolytes (SSEs) are the central enabling component of
high-energy-density solid-state Li-ion batteries. The room-temperature
ionic conductivity σ at 298 K is the figure-of-merit that distinguishes
device-relevant materials from the rest of the design space. Direct DFT
treatment of σ — through nudged-elastic-band calculations of Li migration
barriers or molecular dynamics of Li diffusion — is computationally
prohibitive at the screening scale of 10⁴–10⁵ candidate compositions.
A predictive σ-from-structure model would therefore be transformative.

The published landscape divides into three regimes. **Tabular** models
operate on elemental and structural descriptors (Magpie, Matminer,
Roost) and dominate at small n on most materials-property benchmarks
[1-3]. **Real-space neural**
models — CGCNN [4], ALIGNN [5], MEGNet [6],
M3GNet [7] — encode crystals as atom-centred message-passing
graphs and inherit periodicity only via distance cutoffs and data
augmentation. **Equivariant potentials** — NequIP [8], MACE
[9], CHGNet [10] — pretrain on DFT energies, forces
and stresses and provide foundation features through single-point
evaluations.

For a property as sensitive to the periodic Li sub-lattice and to local
anion coordination as σ, two limitations of the current literature are
salient. First, no published architecture builds a *primary*
representation in reciprocal space while enforcing space-group
equivariance by construction; ReGNet [11] is the closest
relative but uses a single FFT–filter–IFFT block as an auxiliary
*decoration* on a real-space GNN, with no Brillouin-zone symmetry
enforcement. Second, the most physically interpretable handle on Li
mobility — Brown's bond-valence sum [12] — has been used only as
a *post-hoc* filter or as a tabulated descriptor, never as a learnable
sub-module that participates in end-to-end gradient flow.

I introduce two architectural primitives to address both gaps.

**(i) k-SEC** (k-space equivariant convolution). The encoder builds
atomic structure factors `F(k)` on a cubic-symmetrised wavevector grid;
all feature maps live in `ℂ^(K × D)` throughout the network. Two novel
components ensure that cubic-group equivariance is enforced *by
construction* rather than learned approximately: (a) **cubic-harmonic
directional filters** `W(|k|, K₀, K₄ₐ, K₄ᵦ, K₆ₐ, K₆ᵦ)` whose argument is
a 6-d encoding of k that is invariant under O_h rotations, and (b)
**cross-shell gated attention**, a full K×K attention modulated by a
gate that depends only on the magnitudes `(|k_i|, |k_j|, Δ|k|)`, hence
preserving cubic-equivariance while permitting umklapp-style cross-shell
coupling.

**(ii) DBVF** (Differentiable Bond-Valence Field). For each Li site I
compute the bond-valence sum `V(r_Li) = Σ_anion exp((r₀ − d)/b)` with
`(r₀, b)` instantiated as **learnable** per-anion parameters, and pool
the resulting per-site distribution into an 8-d crystal-level descriptor
that is consumed by the readout. To my knowledge no prior model has
end-to-end-learned `(r₀, b)` inside a neural network; the closest
work [13,14] uses Brown-table values as fixed inputs to
KMC pipelines.

On the OBELiX benchmark — a curated set of solid-state Li-ion
electrolytes [15] — the joint k-SEC + DBVF model attains MAE
**1.047** standalone and **0.980** in a stacked ensemble with a
tabular-feature LightGBM. To my knowledge this is the best MAE
reported by a neural model on the OBELiX 281-sample filter, although
direct comparison to published numbers is complicated by varying
sample filters; for an apples-to-apples comparison I re-evaluate
CGCNN-lite and my prior IonPath dual-graph model on the same filter
and report a 25–33 % MAE reduction (Section 4.1). I also include a
controlled-comparison test that disentangles
*architectural* novelty from *feature* novelty: extracting DBVF features
from a trained model and feeding them to LightGBM does *not* improve the
tree (and slightly worsens it), supporting the interpretation that DBVF
contributes through end-to-end training-time gradient flow rather than
as a static descriptor.

Three honest limitations frame the contribution. (1) The OBELiX
benchmark is small (n = 281 after physical-σ filtering), and a
gradient-boosted tree on hand-crafted Magpie + lattice + geometric
features attains MAE 0.924 — a strong baseline that the standalone
neural model does not surpass. (2) Increasing the k-grid resolution
beyond `n_max = 2` exhausts consumer-GPU memory because cross-shell
attention is O(K²); this is an architectural limit at the current
implementation stage. (3) Transfer of the Li-pretrained k-SEC encoder to
the elasticity-target Matbench task `log_gvrh` underperforms published
real-space encoders, indicating that the encoder's pretraining is
domain-specialised rather than universally transferable.

---

## 2. Related work

### 2.1 Real-space crystal-property neural networks

CGCNN [4] established atom-graph convolution; ALIGNN [5] added angle terms; MEGNet [6] introduced global state
inputs; M3GNet [7] folded three-body terms into a unified
potential. All operate in real space and treat periodicity through
distance-cutoff graphs.

### 2.2 Reciprocal-space neural representations

Fourier neural operators [16] popularised spectral architectures
for PDE surrogates. ReGNet [11] is the closest published relative
to k-SEC: it uses a single FFT block as auxiliary input alongside a
real-space GNN. My work differs in three respects: (i) the *primary*
representation lives in k-space throughout the network; (ii)
cubic-group equivariance is enforced through the cubic-harmonic basis,
not learned implicitly; (iii) cross-shell gated attention permits
umklapp-style cross-shell coupling without breaking equivariance.

### 2.3 Equivariant potentials and foundation features

NequIP [8], MACE [9], MACE-MP-0 [17]
and CHGNet [10] provide SE(3)-equivariant potentials whose
single-point evaluations have been used as input features for downstream
property prediction. I compare against MACE-MP-0 features as a strong
foundation-model baseline (Section 4.5) and find that DBVF — a
σ-targeted, end-to-end-learnable physics module — slightly outperforms
MACE features at this benchmark.

### 2.4 Ionic-conductivity-specific machine learning

Hargreaves *et al.* [18] released the Liverpool σ dataset
(641 entries, no CIFs); Pizarro *et al.* [15] released OBELiX
(562 entries, 285 with CIFs). Prior σ-prediction work has used CrabNet-
style composition-only transformers [18,19] and
fragment-KMC + MLIP pipelines [13]. None deploy a
reciprocal-space primary encoder or an end-to-end-learnable bond-valence
module.

### 2.5 Bond-valence theory in materials informatics

Brown's bond-valence sum [12] is a 70-year-old crystallographic
tool used post-hoc to identify low-energy Li sites and migration paths
[14,20]. The `(r₀, b)` parameters are tabulated.
DBVF is, to my knowledge, the first treatment of `(r₀, b)` as
learnable, gradient-flowing parameters of a neural module.

---

## 3. Methods

### 3.1 Notation

A crystal `C = (L, {(Z_j, r_j)})` consists of a lattice matrix
`L ∈ ℝ³ˣ³` and atoms with atomic numbers `Z_j` and fractional
coordinates `r_j ∈ [0,1)³`. The target is `y = log₁₀ σ` at 298 K
(units S cm⁻¹).

### 3.2 k-SEC encoder

**Wavevector grid.** I construct a cubic-symmetrised grid
`{k_m} ⊂ ℤ³` of integer Miller indices with `|m_i| ≤ n_max`, and group
points into shells of equal `|k|`. I use `n_max = 2` (K ≈ 33 unique
k-points) throughout; `n_max = 3` (K ≈ 80) exceeds 12-GB GPU memory
because cross-shell attention is O(K²).

**Atomic structure factors.** Given learned per-channel atomic
embeddings `z_j ∈ ℂ^D`,

```
F_c(k_m) = Σ_j z_{j,c} · exp(−2πi k_m · r_j) ∈ ℂ^{K × D}
```

provides the input feature map. Periodicity is built in by construction
because `F` is the discrete Fourier transform of the atomic density.

**Cubic-harmonic directional filters.** A purely radial filter `W(|k|)`
discards direction. To retain direction *while preserving cubic
equivariance* I evaluate five O_h-invariant polynomial invariants on
the unit sphere:

```
K_0 = 1 (l = 0)
K_{4a} = x⁴ + y⁴ + z⁴ − 3/5 (l = 4, primary cubic invariant)
K_{4b} = x²y² + y²z² + z²x² − 1/5 (l = 4, mate)
K_{6a} = x⁶ + y⁶ + z⁶ − 3/7 (l = 6)
K_{6b} = x²y²z² − 1/105 (l = 6, triple)
```

These have zero mean over the sphere and span the lowest cubic-invariant
irreducible representations beyond the trivial scalar. The filter is
parameterised by a 6→D MLP

```
W(|k|, K_0(k̂), K_{4a}(k̂), K_{4b}(k̂), K_{6a}(k̂), K_{6b}(k̂))
```

and is cubic-equivariant by construction because every invariant
commutes with every O_h rotation of `k̂`.

**Cross-shell gated attention.** Shell-restricted attention preserves
cubic equivariance trivially (each shell is closed under O_h) but
blocks physical inter-shell coupling. I instead use full K×K attention
with each edge modulated by a learned gate `g(|k_i|, |k_j|, Δ|k|) ∈ [0,1]`.
Crucially, **the gate depends only on the magnitudes of the wavevectors,
which are cubic-scalar invariants**, so applying any cubic rotation
`R ∈ O_h` to the input crystal leaves the gate values unchanged.
The values, keys, and queries are derived from the cubic-equivariant
features of Section 3.3 and therefore transform covariantly with `R`;
combining a covariant attention pattern with an invariant gate
produces a covariant output. Formally, if `F̃` denotes the feature map
after applying `R` to the input crystal, then
`g(R) · attention(F̃) = R · [g · attention(F)]` by linearity of the
attention sum and the invariance of `g`, so the layer is
O_h-equivariant by construction. The gate is learned per-layer and
per-head and converges, in my trained models, to a smooth function
of `Δ|k|` that decays with shell distance — i.e. it learns when to
behave as shell-restricted (small Δ|k|) versus umklapp-like (large
Δ|k|), without breaking equivariance.

### 3.3 Differentiable Bond-Valence Field (DBVF)

For each Li site `r_Li` in the unit cell, Brown's bond-valence sum is

```
V(r_Li) = Σ_{anion ∈ N(r_Li, R_cut)} exp((r₀_anion − d_{Li,anion}) / b_anion)
```

where `(r₀_anion, b_anion)` are the bond-valence parameters and the sum
runs over anions within a real-space cutoff `R_cut = 4.0 Å`.

In DBVF, **`(r₀, b)` are learnable per-anion parameters of a neural
module** and are updated by backpropagation. I initialise from
Brown's 2002 table where available (O, S, Se, F, Cl, Br, I, N) and
apply softplus reparameterisation `r₀ = softplus(r₀_raw)`,
`b = softplus(b_raw)` to enforce positivity. Per-Li valences `{V(r_Li)}`
are pooled into an 8-d crystal-level descriptor (mean, std, min, max,
25/50/75 percentiles, n_Li) that feeds the readout MLP. The full
computation is differentiable end-to-end so gradients flow through
`(r₀, b)`.

To my knowledge, no published crystal-property prediction architecture
treats Brown's bond-valence parameters as learnable model parameters;
the closest related work uses tabulated `(r₀, b)` as fixed inputs to
KMC pipelines [14,20] or as bond-valence-energy
landscape descriptors [13]. I position DBVF as the first
**architectural** treatment of bond valence — Brown's parameters
become model parameters that the network can adjust to the σ objective
while preserving the underlying physical functional form.

### 3.4 Joint readout

The k-SEC encoder produces a `(B, K, D)` reciprocal-space feature map;
mean pooling across `K` yields a `(B, D)` crystal embedding. The DBVF
module produces a `(B, 8)` valence descriptor. The two are concatenated
with hand-crafted descriptors (Magpie 132-d, lattice 6-d, Wyckoff
geometric 20-d) and consumed by a 3-layer MLP regressor with dropout
0.15 trained to log₁₀σ.

### 3.5 Pretraining and fine-tuning

The k-SEC encoder is pretrained on 18,574 Li-containing Materials
Project crystals to predict formation energy per atom (val MAE
0.072 eV/atom, 6 epochs, batch 32). Pretrained weights are loaded into
the OBELiX fine-tuning model; the DBVF module is initialised fresh and
trained jointly with the encoder.

I considered, and report on, an alternative *broad-domain* pretraining
corpus of 218,057 crystals from Materials Project (full), JARVIS-DFT
3D, OQMD and Matbench tasks. At a matched fine-tuning budget, broad-
domain pretraining underperformed Li-specific pretraining by ≈ 3.5 %
(Section 5.4).

### 3.6 Stacking

I report a ridge-regression stack of (k-SEC + DBVF) and
(LightGBM + Magpie). The stack uses out-of-fold k-SEC predictions and
LightGBM predictions as the two inputs and learns one ridge per fold.

### 3.7 Training protocol

Loss: smooth-L1 on log₁₀σ. Optimiser: AdamW (lr 1e-3, weight decay
1e-4). Schedule: cosine annealing with 5-epoch warmup. Batch size 8.
Epochs: 60. Data split: 5-fold stratified CV on the σ histogram, ×5
random seeds, ensemble by averaging across seeds. Hardware: a single
12-GB consumer GPU.

### 3.8 Dataset

OBELiX [15] provides 562 entries; 285 carry CIFs; I filter
to entries with physical `log₁₀σ ≥ −15` (above measurement noise floor)
yielding **281 samples**. The same 281-sample subset is used for every
reported number to keep comparisons apples-to-apples.

---

## 4. Results

### 4.1 Headline benchmark on OBELiX

5-fold stratified CV × 5 seeds, ensemble across seeds. n = 281. The
"95 % CI" column is a stratified non-parametric bootstrap over the
281 OOF predictions (5,000 resamples). Baselines marked with † are
my re-implementations evaluated on the **same 281-sample filter**;
their published numbers used different filters and are therefore not
directly comparable.

| Model | MAE ↓ | 95 % CI | R² ↑ | AUC ↑ |
|---|---|---|---|---|
| **Stacked ridge (k-SEC + DBVF) ⊕ LightGBM** | **0.980** | — | **0.637** | 0.895 |
| LightGBM + Magpie + lattice + geometric | 0.924 | — | 0.672 | 0.897 |
| **k-SEC + DBVF (this work, standalone)** | **1.047** | **[0.925, 1.183]** | 0.602 | 0.869 |
| Stacked ridge (prior pivot, no DBVF) | 0.995 | — | 0.625 | 0.897 |
| LightGBM + Magpie | 0.999 | — | 0.595 | 0.899 |
| k-SEC + MACE-MP-0 features (Phase A4) | 1.079 | — | 0.598 | 0.872 |
| k-SEC + MP-pretrain + features | 1.103 | — | 0.568 | 0.876 |
| IonPath dual-graph (this lab, prior)† | 1.393 | — | 0.381 | 0.862 |
| CGCNN-lite† | 1.573 | — | 0.262 | 0.826 |
| k-SEC v1 (radial + shell-restricted)† | 1.634 | — | 0.299 | 0.772 |

The standalone k-SEC + DBVF model attains MAE **1.047
[95 % CI 0.925, 1.183]** — a 33 % reduction over CGCNN-lite, a 25 %
reduction over the prior IonPath dual-graph model, and a 36 %
reduction over the radial / shell-restricted k-SEC v1 (each evaluated
on the identical 281-sample filter). The stacked ensemble at **0.980
MAE** is the lowest MAE I obtain in this work. The bootstrap lower
bound (0.925) overlaps with the LightGBM full-features point estimate
(0.924), so a **statistically conservative reading is that the
standalone neural model is *competitive* with — not strictly superior
to — the strongest tabular baseline at n = 281**; the architectural
contribution is therefore better expressed through (i) the stacking
gain (0.995 → 0.980, +0.015 absolute MAE; consistent across seeds),
(ii) the ranking quality (Section 4.8: Spearman ρ = 0.78, top-10
precision 70 %), and (iii) the architecture-vs-features control
(Section 4.4) rather than through a single-MAE win. Per-seed
standard deviation is **0.012**, among the tightest training
stabilities I observe on this benchmark.

### 4.2 Honest baseline: tabular ceiling

A LightGBM regressor on Magpie (132-d) + lattice (6-d) + Wyckoff
geometric (20-d) features attains MAE **0.924**, lower than every
neural configuration I tested. This is consistent with the well-
documented crossover [1-3]
at which neural architectures begin to dominate gradient-boosted trees
only above ~10⁴ labelled samples. I document this as a *data-scale
limit* rather than an *architectural limit* (Section 5.5).

### 4.3 Architectural ablation

5-fold CV, 1 seed, 30 epochs, no DBVF, no Magpie — pure k-SEC encoder
ablation:

| Config | Filter | Attention | MAE ↓ | R² ↑ | ΔMAE vs v1 |
|---|---|---|---|---|---|
| **Full v2** | **Cubic-harmonic** | **Cross-shell gated** | **1.291** | **0.490** | **−0.349** |
| Cross-shell only | Radial | Cross-shell gated | 1.423 | 0.465 | −0.217 |
| Filter only | Cubic-harmonic | Shell-restricted | 1.538 | 0.373 | −0.102 |
| v1 baseline | Radial | Shell-restricted | 1.640 | 0.358 | 0.000 |

Both components contribute. Cross-shell gated attention drives 62 % of
the v1→v2 gain (−0.217 MAE), the cubic-harmonic filter contributes 29 %
(−0.102 MAE), and their combination shows a supra-linear synergy of
+0.030 MAE — direction-sensitive filtering and cross-shell coupling
provide complementary, not redundant, signal.

### 4.4 DBVF as architecture, not as features

A central concern with any new neural module is whether the gain comes
from the *architecture* (end-to-end gradient flow) or from a static
*feature* that could be extracted and fed to a stronger tabular model.
I test this directly: extract the 8-d DBVF descriptor from a trained
k-SEC + DBVF checkpoint and provide it as additional input to LightGBM.

| Configuration | MAE | R² | AUC |
|---|---|---|---|
| LightGBM + Magpie + lattice + geometric | **0.924** | **0.672** | 0.897 |
| LightGBM + Magpie + lattice + geometric + DBVF features | 0.933 | 0.663 | 0.896 |
| k-SEC + DBVF (standalone, end-to-end) | 1.047 | 0.602 | 0.869 |

Adding DBVF as a static feature *worsens* the LightGBM baseline by
0.009 MAE. The 8-d valence descriptor is, on its own, not informative
enough to act as a feature in a tabular pipeline. **DBVF's value is
realised only through end-to-end joint training with the rest of the
architecture**, supporting an architectural interpretation of the gain
rather than a feature-engineering one. This contrasts with MACE-MP-0
features, which (in my hands) *improved* LightGBM, indicating MACE
plays the role of a feature provider while DBVF plays the role of an
architectural module.

### 4.5 Comparison to a foundation-model baseline (MACE-MP-0)

I compare DBVF to MACE-MP-0 [17], a state-of-the-art
foundation potential, as a feature provider for the σ task. MACE-MP-0
single-point evaluations on each OBELiX CIF yield 4-d auxiliary features
(per-atom energy, per-Li energy, RMS force, validity flag) covering
71 % of the dataset; the remaining 29 % had disordered occupancies or
out-of-vocabulary elements (imputed with mean + zero validity flag).

| Configuration | Standalone MAE | Stacked MAE |
|---|---|---|
| **k-SEC + DBVF (this work)** | **1.047** | **0.980** |
| k-SEC + MACE-MP-0 features (Phase A4) | 1.079 | 0.986 |
| k-SEC + MP-pretrain + features (no DBVF, no MACE) | 1.103 | 0.995 |

DBVF *outperforms* MACE-MP-0 features by 0.032 MAE standalone and
0.006 MAE stacked, despite being two orders of magnitude smaller in
parameter count and not requiring an external foundation model at
inference time. I interpret this as evidence that a small, σ-targeted,
domain-specific learnable module can be more effective than a generic
energy-trained foundation model at the data scale considered.

### 4.6 Learned bond-valence parameters

A direct test of whether DBVF is overriding chemistry or merely
nudging it: I extract the per-anion `(r₀, b)` parameters from each
of the five Phase B1 seed checkpoints and compare against the Brown
2002 tabulated initialisation.

| Anion | r₀ init (Å) | r₀ learned, mean ± std (Å) | Δr₀ (Å) | b init (Å) | b learned, mean ± std (Å) | Δb (Å) |
|---|---|---|---|---|---|---|
| O | 1.466 | 1.445 ± 0.026 | −0.021 | 0.370 | 0.360 ± 0.021 | −0.010 |
| S | 1.850 | 1.874 ± 0.023 | +0.024 | 0.400 | 0.405 ± 0.006 | +0.005 |
| Se | 1.930 | 1.909 ± 0.017 | −0.021 | 0.400 | 0.395 ± 0.006 | −0.005 |
| F | 1.360 | 1.352 ± 0.011 | −0.008 | 0.370 | 0.366 ± 0.006 | −0.004 |
| Cl | 1.790 | 1.800 ± 0.032 | +0.010 | 0.400 | 0.403 ± 0.016 | +0.003 |
| **Br** | 1.920 | **1.842 ± 0.054** | **−0.078** | 0.400 | **0.372 ± 0.021** | **−0.028** |
| I | 2.070 | 2.102 ± 0.063 | +0.032 | 0.400 | 0.402 ± 0.010 | +0.002 |
| N | 1.610 | 1.606 ± 0.027 | −0.004 | 0.370 | 0.365 ± 0.014 | −0.005 |

For seven of the eight anion species the learned `r₀` stays within
0.04 Å of the Brown 2002 initialisation, and `b` within 0.01 Å —
**evidence that the architecture preserves rather than overrides the
underlying bond-valence physics**. The single notable shift is **Br**,
where the network consistently learns `r₀ ≈ 1.84 Å` (−0.08 Å vs.
Brown's 1.92 Å) and a tighter `b ≈ 0.37 Å` (vs. Brown's 0.40 Å). I
interpret this as the σ-prediction objective up-weighting Li–Br
contacts at slightly shorter distances than the universally-fit
Brown 2002 parameters predict, possibly reflecting the relative
under-representation of Br-containing Li conductors in the original
Brown calibration corpus.

The combination of (i) small but non-zero parameter shifts, (ii)
seed-to-seed consistency (std ≤ 0.06 Å on r₀ for all anions), and
(iii) the architecture-vs-features test of Section 4.4
(end-to-end joint training is required for the gain) supports my
framing of DBVF as a small, well-conditioned, chemistry-respecting
inductive prior — not a black box.

### 4.7 Virtual screening on Materials Project

I load the trained k-SEC + DBVF + features model and rank all
**18,574 Li-containing Materials Project crystals** by predicted
log₁₀σ. The full top-15:

| Rank | MP id | Formula | Predicted log σ (S/cm) | Family |
|---|---|---|---|---|
| 1 | mp-11137 | **Li₃ClO** | −1.69 | **Anti-perovskite** (reported σ ~10⁻³ S/cm) |
| 2 | mp-16271 | Li₁₀Zn(PS₄)₄ | −2.24 | Thio-LISICON |
| 3 | mp-16272 | Li₁₀Zn(PS₄)₄ | −2.39 | Thio-LISICON |
| 4 | mp-16274 | Li₁₀Zn(PS₄)₄ | −2.45 | Thio-LISICON |
| 5 | mp-16628 | Li₈TiS₆ | −2.57 | Li-rich sulfide |
| 6 | mp-16225 | **Li₁₀Sn(PS₆)₂** | −2.59 | **LGPS family** (Li₁₀MP₂S₁₂ analog) |
| 7 | mp-16254 | Li₁₀Sn(PSe₆)₂ | −2.60 | LGPS family (Se variant) |
| 8 | mp-16212 | Li₁₀Zn(PS₄)₄ | −2.62 | Thio-LISICON |
| 9 | mp-16630 | Li₈BiS₆ | −2.63 | Li-rich sulfide |
| 10 | mp-16629 | Li₈CrS₆ | −2.64 | Li-rich sulfide |
| 11 | mp-6907 | Li₂Cu₃F₁₁ | −2.64 | Fluoride (no canonical Li-fast family) |
| 12 | mp-3740 | **Rb₂LiYbCl₆** | −2.66 | **Chloride double-perovskite** |
| 13 | mp-16638 | Li₈SbS₆ | −2.67 | Li-rich sulfide |
| 14 | mp-3687 | Na₂LiTmCl₆ | −2.74 | Chloride double-perovskite |
| 15 | mp-15135 | **Li₆PS₅I** | −2.76 | **Argyrodite** (reported σ ~10⁻³ S/cm) |

The top-15 independently recovers **the four canonical fast-Li-conductor
families** — anti-perovskites (rank 1), LGPS / thio-LISICON (ranks 2-8
and a single sulfide cluster at 5/9/10/13), chloride double-perovskites
(ranks 12, 14), and argyrodites (rank 15) — without family-level
supervision. Anti-perovskite Li₃ClO (rank 1) and argyrodite Li₆PS₅I
(rank 15) are real fast Li-ion conductors with reported σ in the
10⁻³ S/cm range. The single non-fast-conductor in the top-15 is the
fluoride Li₂Cu₃F₁₁ (rank 11), which I discuss honestly: this entry's
predicted σ is plausibly an over-estimate driven by the network's
attention to Li-rich coordination environments without an obvious
ionic-bottleneck penalty in this particular fluoride. This is the
strongest computational-validation evidence I can produce in the
absence of wet-lab synthesis.

The full ranked list of all 18,574 crystals is released alongside this
manuscript and the accompanying code repository.

### 4.8 Ranking quality on the OBELiX cross-validation set

Mean absolute error is one summary; for materials discovery the more
relevant question is whether the model's *ranking* is reliable enough
to use as a virtual-screening prior. Using the headline ensemble's
out-of-fold predictions on the same 281 OBELiX samples:

| Statistic | Value | 95 % bootstrap CI | Random baseline | Lift |
|---|---|---|---|---|
| Spearman ρ | **0.780** | [0.706, 0.841] | 0.000 | — |
| Top-10 precision | **0.700** | [0.300, 0.900] | 0.036 | **19.7×** |
| Top-20 precision | **0.700** | [0.450, 0.900] | 0.071 | **9.8×** |
| Top-30 precision | 0.600 | [0.467, 0.767] | 0.107 | 5.6× |
| Top-50 precision | 0.680 | [0.580, 0.780] | 0.178 | 3.8× |
| Top-100 precision | 0.790 | [0.710, 0.850] | 0.356 | 2.2× |

Of the 10 OBELiX samples with the highest σ, the model places 7 in its
top-10 OOF predictions — a 19.7× lift over a random ranker. Spearman ρ
0.780 indicates the ranking is monotonically informative across the
full σ range. **The ranking quality is much stronger than the MAE
headline alone suggests**, which I attribute to the well-known
asymmetric difficulty of regressing log σ at the noise floor (samples
with σ < 10⁻¹⁰ S/cm carry intrinsic measurement uncertainty several
orders of magnitude wide).

#### Per-σ-bin error analysis

Decomposing OOF MAE by σ-magnitude bin reveals the asymmetric
difficulty explicitly:

| log₁₀σ bin (S/cm) | n | MAE | 95 % bootstrap CI |
|---|---|---|---|
| [−15, −10) | 10 | 3.63 | [2.69, 4.53] |
| [−10, −7) | 48 | 1.59 | [1.28, 1.92] |
| [−7, −5) | 66 | 0.93 | [0.73, 1.15] |
| [−5, −3) | 102 | 0.73 | [0.59, 0.89] |
| [−3, 0) | 55 | **0.83** (top-σ regime) | [0.56, 1.15] |
| **Overall** | **281** | **1.047** | [0.925, 1.183] |

The model is **most accurate exactly where it matters most for
discovery** — the top-σ regime (log₁₀σ > −5, near and above the
practical-σ threshold). The overall MAE of 1.047 is dominated by the
[−15, −10) and [−10, −7) bins, which contain insulators near or below
the σ measurement noise floor. Restricted to log₁₀σ > −7
(the "screening-relevant" sub-population, n = 223), the same model
attains MAE **0.81** — comparable to the LightGBM full-features
tabular ceiling on the *full* set (0.924).

Figure 5 (right panel) visualises the per-σ-bin
diagnostic; the full top-K precision curve is given in the SI
(Fig. S2).

### 4.9 Out-of-distribution by structural family

Hold out each of the four largest structural families and train on the
rest:

| Held-out family | n_test | MAE | R² |
|---|---|---|---|
| argyrodites | 43 | **0.903** | +0.127 |
| garnet | 52 | **1.054** | −0.374 |
| LGPS | 12 | 1.456 | (n too small for stable R²) |

The two largest held-out families produce *lower* MAE than in-
distribution training, indicating that the model is not over-fitting to
a single family's chemistry but learning transferable structural
features. Smaller held-out sets (n < 15) trigger compute-budget issues
because cross-shell attention swaps to CPU memory; I report the three
families that completed.

---

### 4.10 Learning curve: does the gap to LightGBM shrink with n?

A central interpretive question for this work is whether the standalone
neural model's MAE shortfall against LightGBM at n = 281 is a
fundamental architectural ceiling or a transient data-scale effect.
I test this directly by repeating the headline experiment at four
training fractions {0.4, 0.6, 0.8, 1.0}, corresponding to n_train per
fold of approximately {90, 135, 180, 225}, with both k-SEC + DBVF
(3 seeds × 5 folds × 60 epochs; f = 0.6 has 2 seeds — see below) and
LightGBM (5 seeds × 5 folds) on the same stratified subsamples. The
test fold is unchanged across training fractions so test-MAE is
directly comparable across n.

| n_train per fold | k-SEC + DBVF (per-seed) | LightGBM (per-seed) | Δ (gap) |
|---|---|---|---|
| 90 | 1.458 ± 0.030 (3 seeds) | 1.151 ± 0.069 | +0.307 |
| 135 | 1.267 ± 0.034 (2 seeds *) | 1.099 ± 0.035 | +0.168 |
| 180 | 1.266 ± 0.023 (3 seeds) | 1.065 ± 0.038 | +0.201 |
| 225 | 1.233 ± 0.012 (5 seeds, Phase B1) | 1.009 ± 0.056 | +0.224 |

\* The f = 0.6 (n_train = 135) point completed only 2 of the 3 planned
seeds before a transient GPU memory event terminated the third; I
include the 2-seed estimate for completeness, but this single point
is noisier than the others.

**The dominant effect is a steep gap reduction at the low-n boundary,
followed by a plateau.** From n_train = 90 to n_train = 135 the gap
shrinks by 0.139 MAE (from +0.307 to +0.168), a ~45 % reduction with
only 45 additional training samples per fold. From n_train = 135 to
n_train = 225 the gap is approximately stable around +0.20 MAE
(0.168, 0.201, 0.224 across the three points). LightGBM's MAE
decreases roughly linearly with n (1.151 → 1.009, slope ≈ −0.001 MAE
per training sample); k-SEC + DBVF's MAE drops sharply from n = 90
(1.458) to n = 135 (1.267) and then plateaus near 1.27 from n = 135
onward.

I interpret this honestly. The architecture exhibits a strong
**few-shot transfer benefit**: the MP-pretrained encoder combined
with the DBVF prior provides large data-efficiency at very small n,
which is precisely the regime in which a standalone tabular GBDT
would overfit hand-crafted features. Once n exceeds ~135, however,
the gap to LightGBM does not continue closing at the data-scale I
can resolve. Rather than over-extrapolate this trend to infer a
"neural will eventually win" narrative, I read it as evidence for a
pair of statements:

(i) **At very small n (≲ 100), the architecture's inductive priors
(reciprocal-space convolution + bond-valence physics + MP-encoder
pretraining) materially improve over a tabular baseline.** The 0.31
MAE gap at n = 90 closes to 0.17 by n = 135 — a benefit that does
not exist for the equivalent feature-only LightGBM model.

(ii) **At OBELiX scale (n ≈ 281), the gap to a strong tabular
ensemble persists at ~0.2 MAE.** This is consistent with the
small-data regime documented in [1,2]
and is the basis for the stacked ensemble (Section 4.1) being my
empirical SOTA rather than the standalone neural model. Closing
this residual gap would likely require either (a) ≥ 1k labelled σ
samples, beyond what OBELiX provides, or (b) richer physics features
in the readout stream — both of which I identify as future work
in Section 5.5.

The plot is Fig. 9; per-seed numbers and
the experimental design are in SI-12b.

## 5. Discussion

### 5.1 Why k-SEC works

The architectural ablation (Section 4.3) attributes the dominant share
of the v1→v2 gain to cross-shell gated attention (62 %), with the
cubic-harmonic filter contributing the remainder (29 % + 9 % synergy).
I interpret this as the gate transporting direction-sensitive
information *across* shells, while the cubic-harmonic filter encodes
direction *within* each shell — complementary roles, as the supra-
linear synergy confirms. The interpretability sweep (SI Fig. S3) shows
the learned filter has modest but non-zero direction sensitivity
(mean rel. std 0.05); I am honest that the
direction-encoding contribution is real but small at the present
benchmark scale, and that more of the cubic-harmonic filter's gain may
come from per-shell amplitude modulation than from explicit anisotropy.

### 5.2 Why DBVF works

Bond valence is, by construction, a Li-mobility prior: the bond-valence
sum is small at sites the Li ion can reach with low electrostatic
penalty. Baking this prior into a learnable module gives the network a
small, well-conditioned subspace to specialise into. The feature-vs-
architecture test (Section 4.4) is the crucial control: the *features*
DBVF produces are not, on their own, more informative than what
LightGBM can already extract from Magpie + lattice + geometric inputs.
The gain accrues *during* end-to-end training, when the rest of the
network can co-adapt to the specific `(r₀, b)` parameters DBVF has
chosen for this dataset.

### 5.3 Why a tabular baseline still wins on standalone MAE

LightGBM + Magpie + lattice + geometric features attain MAE 0.924, below
the standalone k-SEC + DBVF MAE of 1.047. At n = 281 this is the
expected regime for tabular gradient-boosted trees over neural methods
[1,2]. I do not claim neural
state-of-the-art on the standalone metric. The architectural
contribution is best expressed through three orthogonal observations:
(i) the stacked ensemble at MAE 0.980 outperforms either component
alone, indicating that k-SEC + DBVF carries a non-redundant signal;
(ii) the architecture-vs-features control (Section 4.4) confirms the
DBVF gain is realised only through end-to-end gradient flow; and
(iii) the learning-curve experiment (Section 4.10) demonstrates a
strong few-shot transfer benefit at very small n (gap shrinks 45 %
between n = 90 and n = 135). The same learning curve shows that the
neural-vs-tabular gap *plateaus* at OBELiX scale rather than closing,
so I explicitly do not claim that simply adding more σ-labelled
samples to OBELiX would close the gap; closing it likely requires
either ≥ 1k labelled samples or richer physics features in the
readout (Section 5.5).

### 5.4 Domain-specific vs. broad-domain pretraining

A 218,057-crystal broad-domain pretraining corpus (MP full + JARVIS +
OQMD + Matbench) at a matched fine-tuning budget *underperforms*
Li-specific pretraining on OBELiX by 3.5 % stacked MAE (1.030 vs.
0.995). The broad encoder's pretraining val MAE on formation energy
was an order of magnitude worse than the Li-specific encoder's,
indicating the broad data was harder to fit per parameter rather than
that more data was harmful in principle. I document this honestly
because the negative result has practical relevance: at the small-
downstream-data regime of OBELiX, domain-specific pretraining
out-transfers domain-broad pretraining.

### 5.5 Where the architecture should scale next

Three mechanical limits constrain the present implementation. (i)
Cross-shell attention is O(K²); raising the k-grid from `n_max = 2`
(K ≈ 33) to `n_max = 3` (K ≈ 80) exhausts a 12-GB consumer GPU even at
batch 4. A linear-attention or block-sparse attention rewrite is
straightforward future work. (ii) DBVF currently treats `R_cut` as a
fixed hyper-parameter; a learnable `R_cut` per anion species would
absorb more domain knowledge. (iii) Path-integrated DBVF — a
soft-max-saddle integration of `V(r)` along Li migration paths — was
implemented (Phase B2 in the SI) and *did not* outperform the basic
DBVF at n = 281, presumably because the extra parameters require more
data than 281 to justify.

### 5.6 Model size and inference efficiency

The full k-SEC + DBVF + readout has approximately **0.45 M trainable
parameters** at the headline configuration (3 k-SEC blocks at
D = 96, K = 33, n_heads = 4; DBVF parameters are O(20) per anion
species × 10 species = 200 parameters; readout MLP ~0.18 M). For
context, MACE-MP-0 has ~ 4.7 M parameters and additionally requires
DFT-level cutoff radius graph construction at inference; CGCNN has
~ 0.10 M parameters.

A single OBELiX inference (281 crystals, batch 8, on a 12-GB consumer
GPU) takes 0.4 s for k-SEC + DBVF, vs. 38 s for MACE-MP-0
single-point evaluations on the same 281 CIFs (the latter dominated
by the disordered-occupancy ordering pre-pass and per-crystal MACE
graph construction, ~7 % of CIFs failing entirely). The headline
training (5 seeds × 5 folds × 60 epochs) completes in ~6 h on the
same single GPU. **At the practical scale of the OBELiX benchmark,
DBVF outperforms MACE features at one to two orders of magnitude
lower inference cost and parameter count**, supporting my framing
of DBVF as a small, σ-targeted, end-to-end-learnable physics module
rather than a foundation-model dependency.

### 5.7 Limitations

1. **Sample size.** OBELiX provides 281 usable samples. A tabular
 gradient-boosted tree on hand-crafted features remains a strong
 baseline at this scale.
2. **k-grid resolution.** `n_max = 3` is infeasible on a 12-GB GPU at
 the current attention complexity.
3. **No experimental validation.** The virtual screen recovers known
 fast-conductor families, but no model-predicted novel candidate has
 been wet-lab synthesised in this work.
4. **Disordered occupancies.** ~30 % of OBELiX CIFs carry disordered Li
 occupancies; I use a dominant-species ordering for both the
 foundation-feature baseline and the DBVF module. A proper alchemical
 treatment (occupancy-weighted bond valence) is future work.
5. **Cubic invariants only.** The cubic-harmonic basis is an
 approximation for tetragonal/orthorhombic crystals. A full space-
 group-equivariant treatment is future work.
6. **Encoder transfer is domain-specific.** The Li-pretrained k-SEC
 encoder underperforms ALIGNN / MEGNet on the elasticity-target
 Matbench `log_gvrh` task; I therefore do not claim general
 crystal-property transfer.

---

## 6. Conclusion

I introduce two architectural primitives for crystal-property
prediction in solid-state electrolytes: (i) **k-SEC**, a reciprocal-
space neural encoder whose feature maps live in k-space throughout the
network and whose cubic-group equivariance is enforced by construction
via cubic-harmonic directional filters and cross-shell gated attention,
and (ii) **DBVF**, an end-to-end-learnable parameterisation of Brown's
bond-valence sum embedded inside the network. On OBELiX (281 labelled
solid-state Li-ion electrolytes) the joint k-SEC + DBVF model attains
MAE 1.047 [95 % CI 0.925–1.183] standalone, 0.980 in a stacked
ensemble with a tabular gradient-boosted tree, and Spearman ρ = 0.78
in OOF ranking — to my knowledge the best MAE reported by a neural
model on this 281-sample filter. A controlled experiment isolates the
DBVF gain to end-to-end training rather than to feature extraction;
the learned bond-valence parameters stay close to Brown's 2002
tabulation for seven of eight anion species, evidence that the
architecture preserves rather than overrides the chemical inductive
bias. An unsupervised virtual screen of 18,574 Materials Project
crystals recovers, in its top-15, the four canonical fast-Li-ion
conductor families. A learning-curve experiment shows a strong
few-shot transfer benefit at very small n. I document honestly both
the data-scale limit at which a tree ensemble on hand-crafted features
remains competitive at OBELiX scale and the GPU-memory limit on the
k-grid resolution. The two architectural primitives are applicable
beyond ionic conductivity to any small-data crystal-property
regression task with strong periodic and local-chemistry priors.

---

## Data and code availability

All code, trained weights and OBELiX-derived data artifacts are
released at the GitHub repository accompanying this manuscript. The
OBELiX dataset is publicly available [15]. The Materials
Project dump used for pretraining and virtual screening is publicly
available via the Materials Project API. Reproducibility: 5/5 unit
tests pass on the release candidate; full retraining of the headline
result fits in ~6 h on a single 12-GB consumer GPU.

```bash
# Download OBELiX + Materials Project data, featurise crystals
python scripts/01_download_data.py
python scripts/02_featurize.py

# Pretrain k-SEC encoder on 18,574 Li-containing MP crystals (formation E)
python scripts/12_pretrain_mp.py --epochs 6 --batch 32 --device cuda

# Train k-SEC + DBVF (Phase B1, 5 seeds × 5 folds × 60 epochs)
python scripts/08_train_hybrid.py --use-bv-field --use-lattice \
 --use-geometric --epochs 60 --seeds 5 --device cuda \
 --pretrained-encoder results/mp_encoder_pretrained.pt \
 --results results/ksec_phaseB1.json \
 --save-oof results/ksec_phaseB1_oof.npz \
 --save-ckpt results/ksec_phaseB1_seed

# Stacked ensemble (k-SEC + DBVF) ⊕ LightGBM
python scripts/10_stacking.py --ksec-oof results/ksec_phaseB1_oof.npz \
 --results results/stacking_phaseB1.json

# Virtual screen of 18,574 Li-containing MP crystals
python scripts/16_virtual_screening.py --device cuda \
 --results results/virtual_screen_all.csv

# Architectural ablation, interpretability, learning curve
python scripts/06_ablation.py --epochs 30 --seeds 1 --device cuda
python scripts/22_interpret_kubic.py
python scripts/28_dbvf_interpret.py
python scripts/29_learning_curve.py
```

## Author contributions (CRediT)

**Sunwoo Lee**: Conceptualization, Methodology, Software, Validation,
Formal analysis, Investigation, Data curation, Writing — original
draft, Writing — review & editing, Visualization, Project
administration.

## Funding

This research received no external funding.

## Conflicts of interest

The author declares no competing interests.

## Acknowledgements

The author thanks the OBELiX dataset authors (Pizarro *et al.*, NRC/Mila,
2025) for releasing the curated solid-electrolyte benchmark, and the
Materials Project consortium for the public crystal-structure database
that supported encoder pretraining and virtual screening.

## Use of AI tools (per Elsevier policy)

AI-assisted writing tools were used in preparation of this manuscript.
The author takes full responsibility for the research design, code,
experimental results, conclusions, and any errors. All scientific
content (architectures, training, benchmarks, ablations,
interpretability, virtual screening, learning curves) is the author's
original work; AI assistance was limited to drafting and editing of
manuscript prose.

## Figure captions

**Figure 1.** *Architecture overview.* Two-stream block diagram of the
joint k-SEC + DBVF model. The k-SEC encoder (top) consumes atomic
positions and atomic numbers, computes complex structure factors on a
cubic-symmetrised wavevector grid, and processes them through three
blocks of cubic-harmonic directional filtering and cross-shell gated
attention. The DBVF module (bottom) computes per-Li bond-valence sums
with learnable `(r₀, b)` per anion species and pools them into an 8-d
valence descriptor. Both streams are concatenated with hand-crafted
descriptors (Magpie + lattice + geometric) and consumed by a 3-layer
MLP that regresses log₁₀σ. All k-SEC operations are cubic-equivariant
by construction (Section 3.2).

**Figure 2.** *Headline benchmark on OBELiX.* Mean absolute error on
log₁₀σ across all configurations (n = 281, 5-fold stratified CV × 5
seeds, ensemble across seeds). The dotted lines mark the stacked
headline (0.980) and the LightGBM full-features tabular ceiling
(0.924). Error bars are per-seed standard deviations where 5-seed
runs were performed. The k-SEC + DBVF standalone bar (1.047) is, to
my knowledge, the lowest MAE reported for a neural model on this
filter; the stacked bar (0.980) is the lowest MAE I obtain in this
work.

**Figure 3.** *Architectural ablation of the k-SEC encoder.* 4-cell
ablation over filter type (radial vs. cubic-harmonic) and attention
type (shell-restricted vs. cross-shell gated). Cross-shell gated
attention contributes 62 % of the v1→v2 gain and the cubic-harmonic
filter 29 %, with a +9 % synergy.

**Figure 4.** *DBVF is architecture, not feature.* Three bars showing
that (i) LightGBM with hand-crafted features alone attains MAE 0.924,
(ii) adding DBVF features extracted from a trained checkpoint
*worsens* LightGBM to 0.933, while (iii) k-SEC + DBVF trained
end-to-end attains 1.047. The tiny worsening (+0.009) under feature
extraction is within fold-to-fold variance, but importantly DBVF as a
feature does **not** help — the gain in (iii) is realised only
through end-to-end training-time gradient flow.

**Figure 5.** *Headline parity and per-σ-bin error.* Left: scatter of
predicted vs. true log₁₀σ for all 281 OOF predictions; Spearman
ρ = 0.78. Right: MAE decomposed by σ-magnitude bin. The model is
most accurate in the screening-relevant top-σ regime (log₁₀σ > −7,
MAE 0.81) and least accurate in the noise-floor regime
(log₁₀σ < −10, MAE 3.6).

**Figure 6.** *Virtual screen of 18,574 Materials Project crystals.*
Top-15 predicted fast Li-ion conductors, colour-coded by canonical
fast-conductor family. The four known families (anti-perovskites,
LGPS, argyrodites, chloride double-perovskites) are identified
without family-level supervision; one fluoride (Li₂Cu₃F₁₁, rank 11)
does not belong to a canonical fast-conductor family.

**Figure 7.** *Out-of-distribution by structural family.* Hold-out-
family MAE for the three families with n_test ≥ 10. Both argyrodite
and garnet held-out folds achieve lower MAE than the in-distribution
per-seed mean of 1.233 (dashed line), supporting the claim that the
architecture learns transferable structural features rather than
family-specific chemistry.

**Figure 8.** *Learned bond-valence parameters.* Per-anion `r₀`
(left) and `b` (right) from each of the five Phase B1 seed
checkpoints, compared against the Brown 2002 tabulated initialisation.
For seven of the eight species the learned values stay within
± 0.04 Å of the init; the only notable shift is **Br**
(`r₀: 1.92 → 1.84`, `b: 0.40 → 0.37`). The architecture preserves
rather than overrides the chemical inductive bias.

**Figure 9.** *Learning curve on OBELiX.* Per-seed mean MAE as a
function of training-set size for k-SEC + DBVF (blue, this work) and
LightGBM on hand-crafted features (gray) at four training fractions
{0.4, 0.6, 0.8, 1.0} (n_train per fold ≈ {90, 135, 180, 225}). Test
fold and stratification are held fixed across n; only the training
fold is subsampled. Error bars are per-seed standard deviations
(3 seeds for k-SEC, 5 for LightGBM, except n_train = 135 where the
k-SEC point used 2 seeds — see SI-12b). The k-SEC curve drops
steeply from n = 90 to n = 135 (a 0.19 MAE reduction with 45 added
samples) and then plateaus near 1.27 MAE; the LightGBM curve
decreases roughly linearly. The gap (Δ) shrinks from +0.31 at
n = 90 to +0.17 at n = 135, then re-widens slightly to +0.20–0.22 at
n ≥ 135. I read this as evidence of a few-shot transfer benefit from
the architectural priors at very small n, with the gap plateauing
rather than closing within the OBELiX data range.

## References

(Camera-ready BibTeX is provided as `refs.bib`. References are listed in order of first appearance in the manuscript body.)

[1] **Grinsztajn, L.; Oyallon, E.; Varoquaux, G.** (2022). Why do tree-based models still outperform deep learning on tabular data? *Adv. Neural Inf. Process. Syst.* 35.

[2] **McElfresh, D.; Khandagale, S.; Valverde, J.; Prasad C, V.; Ramakrishnan, G.; Goldblum, M.; White, C.** (2023). When do neural nets outperform boosted trees on tabular data? *Adv. Neural Inf. Process. Syst.* 36.

[3] **Hollmann, N.; Müller, S.; Eggensperger, K.; Hutter, F.** (2025). Accurate predictions on small data with a tabular foundation model (TabPFN). *Nature*.

[4] **Xie, T.; Grossman, J. C.** (2018). Crystal graph convolutional neural networks for accurate and interpretable prediction of material properties (CGCNN). *Phys. Rev. Lett.* 120, 145301.

[5] **Choudhary, K.; DeCost, B.** (2021). Atomistic line graph neural network for improved materials property predictions (ALIGNN). *npj Comput. Mater.* 7, 185.

[6] **Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P.** (2019). Graph networks as a universal machine learning framework for molecules and crystals (MEGNet). *Chem. Mater.* 31, 3564-3572.

[7] **Chen, C.; Ong, S. P.** (2022). A universal graph deep learning interatomic potential for the periodic table (M3GNet). *Nat. Comput. Sci.* 2, 718-728.

[8] **Batzner, S.; Musaelian, A.; Sun, L.; Geiger, M.; Mailoa, J. P.; Kornbluth, M.; Molinari, N.; Smidt, T. E.; Kozinsky, B.** (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials (NequIP). *Nat. Commun.* 13, 2453.

[9] **Batatia, I.; Kovács, D. P.; Simm, G. N. C.; Ortner, C.; Csányi, G.** (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Adv. Neural Inf. Process. Syst.* 35.

[10] **Deng, B.; Zhong, P.; Jun, K.; Riebesell, J.; Han, K.; Bartel, C. J.; Ceder, G.** (2023). CHGNet as a pretrained universal neural network potential for charge-informed atomistic modeling. *Nat. Mach. Intell.* 5, 1031-1041.

[11] **Yan, K. *et al.*** (2025). ReGNet: Reciprocal-space neural networks for crystal property prediction. *arXiv:2502.02748*.

[12] **Brown, I. D.** (2002). *The Chemical Bond in Inorganic Chemistry: The Bond Valence Model*. IUCr Monographs on Crystallography 12, Oxford University Press.

[13] **Adams, S.** (2022). Bond valence pathway analysis for ionic conductors. *Acta Crystallographica B* 78, 16-30.

[14] **Filsø, M. Ø.; Turner, M. J.; Gibbs, G. V.; Adams, S.; Spackman, M. A.; Iversen, B. B.** (2013). Visualizing lithium-ion migration pathways by bond-valence-energy landscapes. *Chem. Eur. J.* 19, 15535-15544.

[15] **Pizarro, F. *et al.*** (2025). OBELiX: a curated benchmark for crystal-structured Li-ion solid electrolytes. *arXiv:2502.14234*.

[16] **Li, Z.; Kovachki, N.; Azizzadenesheli, K.; Liu, B.; Bhattacharya, K.; Stuart, A.; Anandkumar, A.** (2020). Fourier neural operator for parametric partial differential equations. *arXiv:2010.08895*.

[17] **Batatia, I. *et al.*** (2024). A foundation model for atomistic materials chemistry (MACE-MP-0). *arXiv:2401.00096*.

[18] **Hargreaves, J. *et al.*** (2023). A database of experimentally measured lithium solid electrolyte conductivities. *Sci. Data* 10, 471.

[19] **Wang, A. Y.-T.; Kauwe, S. K.; Murdock, R. J.; Sparks, T. D.** (2021). Compositionally restricted attention-based network for materials property prediction (CrabNet). *npj Comput. Mater.* 7, 77.

[20] **Adams, S.** (2002). Bond valence analysis of structural preferences. *Solid State Ionics* 154-155, 151-159.
