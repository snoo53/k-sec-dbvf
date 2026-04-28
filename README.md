# k-SEC + DBVF

Code, data and trained-model artifacts for:

> **Cubic-Equivariant k-Space Convolution and a Differentiable Bond-Valence
> Field for Ionic-Conductivity Prediction in Solid-State Electrolytes.**
> Sunwoo Lee, 2026.

| Resource | Status |
|---|---|
| Manuscript submission | *Journal of Materials Science & Technology* (Elsevier), submitted April 2026, manuscript ID **J-MST-D-26-03318** |
| Preprint | ChemRxiv (DOI to be added once posted) |
| Code license | MIT (this repository) |
| Manuscript license | CC BY 4.0 |

## What this is

Two architectural primitives for crystal-property prediction in solid-state
lithium-ion battery electrolytes:

1. **k-SEC** (k-Space Equivariant Convolution) — a neural encoder whose
   feature maps live in reciprocal space throughout the network. Cubic-group
   (O_h) equivariance is enforced *by construction* through:
   - **Cubic-harmonic directional filters** `W(|k|, K_0, K_4a, K_4b, K_6a, K_6b)`
     — the argument is a 6-d encoding of k that is invariant under O_h rotations.
   - **Cross-shell gated attention** — full K×K attention modulated by a
     gate that depends only on wavevector magnitudes (cubic-scalar invariants),
     preserving equivariance while permitting umklapp-style cross-shell coupling.

2. **DBVF** (Differentiable Bond-Valence Field) — Brown's bond-valence sum
   embedded inside the network as a learnable module. The per-anion `(r_0, b)`
   parameters are trainable model parameters (initialised from Brown 2002 and
   updated end-to-end by gradient descent). To my knowledge, the first
   architectural treatment of bond-valence parameters in a neural network.

## Headline result

On the OBELiX benchmark (281 labelled solid-state Li-ion electrolytes,
5-fold cross-validation × 5 seeds), the joint k-SEC + DBVF model attains:

- **MAE 1.047** (95 % bootstrap CI 0.925–1.183) on log₁₀σ standalone
- **MAE 0.980** in a stacked ensemble with a tabular gradient-boosted-tree baseline
- **Spearman ρ = 0.78** in out-of-fold ranking
- **Top-10 precision 70 %** — a 19.7× lift over a random ranker

An unsupervised virtual screen of 18,574 Li-containing Materials Project
crystals identifies all four known fast-Li-ion conductor families
(anti-perovskites, LGPS, argyrodites, chloride double-perovskites) in its
top-15 — without family-level supervision.

A controlled experiment confirms that the DBVF gain is realised only through
end-to-end training: feeding the trained DBVF features to a gradient-boosted
tree does *not* improve it.

## Repository contents

```
src/ionpath/        — model, training, and analysis code (PyTorch)
scripts/            — driver scripts (data fetch, training, evaluation, plotting)
results/            — Phase B1 trained-model checkpoints and OOF predictions
                      (large data caches and generic .pt files are gitignored;
                       headline weights and JSON metric files are committed)
figs/               — manuscript figures (300 dpi PNGs)
JMST-DRAFT.md       — full manuscript (markdown source)
JMST-SI.md          — supplementary information
JMST-COVER-LETTER.md — submission cover letter
RESULTS-kSEC.md     — running results notes (development context)
ChemRxiv-Manuscript.pdf — combined preprint PDF (manuscript + figures + SI)
refs.bib            — BibTeX references
applications/       — build pipeline that produced submission artifacts
                      (manuscript .docx, anonymised version, figure renaming,
                       Elsevier DOCI template fill, etc.)
```

## Reproducing the headline result

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch OBELiX + Materials Project data
#    (NOTE: requires a Materials Project API key in .env as MP_API_KEY=...)
python scripts/01_download_data.py
python scripts/02_featurize.py

# 3. Pretrain the k-SEC encoder on 18,574 Li-containing MP crystals
#    (~2 hours on a 12-GB consumer GPU)
python scripts/12_pretrain_mp.py --epochs 6 --batch 32 --device cuda

# 4. Train the headline k-SEC + DBVF model (5 seeds × 5 folds × 60 epochs,
#    ~6 hours on a 12-GB consumer GPU)
python scripts/08_train_hybrid.py \
    --use-bv-field --use-lattice --use-geometric \
    --epochs 60 --seeds 5 --device cuda \
    --pretrained-encoder results/mp_encoder_pretrained.pt \
    --results results/ksec_phaseB1.json \
    --save-oof results/ksec_phaseB1_oof.npz \
    --save-ckpt results/ksec_phaseB1_seed

# 5. Stack with LightGBM (ridge meta-learner)
python scripts/10_stacking.py \
    --ksec-oof results/ksec_phaseB1_oof.npz \
    --results results/stacking_phaseB1.json

# 6. Virtual screen the 18,574 Li-containing MP crystals
python scripts/16_virtual_screening.py --device cuda \
    --results results/virtual_screen_all.csv

# 7. Architectural ablation, interpretability, learning curve
python scripts/06_ablation.py --epochs 30 --seeds 1 --device cuda
python scripts/22_interpret_kubic.py
python scripts/28_dbvf_interpret.py
python scripts/29_learning_curve.py
```

The committed checkpoints in `results/ksec_phaseB1_seed{0..4}.pt` reproduce
the headline numbers without retraining.

## Data and licences

- **OBELiX benchmark** ([Pizarro *et al.*, 2025](https://arxiv.org/abs/2502.14234)) —
  the 281-sample solid-electrolyte benchmark used here. Fetched by
  `scripts/01_download_data.py`; not redistributed in this repo per the
  upstream license.
- **Materials Project** — used for encoder pretraining (18,574 Li-containing
  crystals) and the virtual screen (18,574 crystals reranked). Requires
  a free Materials Project API key.
- **Brown 2002 bond-valence parameters** — used as initialisation for DBVF,
  embedded in `src/ionpath/models/bond_valence_field.py`.

## Citation

If you use this code or build on the architecture, please cite the
manuscript:

```bibtex
@article{Lee2026kSEC,
  author  = {Lee, Sunwoo},
  title   = {Cubic-Equivariant {k}-Space Convolution and a Differentiable
             Bond-Valence Field for Ionic-Conductivity Prediction in
             Solid-State Electrolytes},
  journal = {Journal of Materials Science \& Technology},
  year    = {2026},
  note    = {Submitted (manuscript ID J-MST-D-26-03318); preprint on
             ChemRxiv}
}
```

## Honest limitations

The standalone neural model attains MAE 1.047, while a gradient-boosted tree
on hand-crafted features attains MAE 0.924 on the same OBELiX 281-sample
filter. At this benchmark scale, tabular methods remain competitive; a
learning-curve experiment in §4.10 of the manuscript shows the gap shrinks
sharply between n_train = 90 → 135 (a 45 % reduction) and then plateaus.
Closing the residual gap likely requires more labelled σ samples or
richer DFT-derived physics features in the readout. Negative results
(heteroscedastic loss, percolation features, path-integrated DBVF,
dual-stream BatteryNet) are documented in the SI rather than buried.

## Use of AI tools

AI-assisted writing tools were used in preparation of the manuscript prose.
The author takes full responsibility for the research design, code,
experimental results, and conclusions. All scientific content
(architectures, training, benchmarks, ablations, interpretability,
virtual screening, learning curves) is the author's original work; AI
assistance was limited to drafting and editing of manuscript prose.

## Contact

Sunwoo Lee — Independent researcher, South Korea
ORCID: [0009-0004-9159-367X](https://orcid.org/0009-0004-9159-367X)
Email: lee.11539@buckeyemail.osu.edu

For questions about the architecture, training, or reproducing results,
open an issue on this repository.
