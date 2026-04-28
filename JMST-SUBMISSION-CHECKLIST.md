# JMST submission checklist

Tracking every artifact required by *Journal of Materials Science &
Technology* for a research article submission. Items marked **DONE**
are committed in the repository; items marked **TODO** remain.

---

## Manuscript files

| File | Status | Notes |
|---|---|---|
| `JMST-DRAFT.md` | **DONE** | Headline: Phase B1 k-SEC + DBVF, MAE 1.047 standalone / 0.980 stacked. |
| `JMST-SI.md` | **DONE** | Six negative-result sections + reproducibility tables. |
| `JMST-COVER-LETTER.md` | **DONE** | With suggested reviewers. |
| Highlights (≤5 bullets, ≤85 char each) | **DONE** | At top of `JMST-DRAFT.md`. |
| Graphical abstract caption | **DONE** | At top of `JMST-DRAFT.md`. |
| Graphical abstract image (300 dpi PNG/TIFF) | **DONE** (proxy) | `figs/fig_1_architecture.png` doubles as the graphical abstract; can be replaced by a polished vector version for the camera-ready. |
| BibTeX file | **DONE** | `refs.bib` (19 entries) |
| Author CRediT taxonomy block | **TODO** | To be filled by corresponding author. |
| Author affiliations + ORCIDs | **TODO** | To be filled by corresponding author. |
| Funding statement | **TODO** | To be filled by corresponding author. |
| Conflict-of-interest declaration | **DONE** | Cover letter + manuscript footer. |

## Figures (camera-ready quality)

| Figure | File | Status | Headline number(s) |
|---|---|---|---|
| Fig. 1 — Architecture (k-SEC + DBVF) schematic | `figs/fig_1_architecture.png` | **DONE** | Two-stream schematic: k-SEC encoder + DBVF module → readout (`scripts/30_architecture_diagram.py`). |
| Fig. 2 — Headline benchmark vs. baselines | `figs/fig_2_baselines.png` | **DONE** | Phase B1 1.047/0.980 vs. all priors; consistent error bars. |
| Fig. 3 — Architectural ablation (filter × attention) | `figs/fig_3_ablation.png` | **DONE** | 4 cells: 1.640 → 1.291 |
| Fig. 4 — DBVF as architecture, not as features | `figs/fig_4_dbvf_test.png` | **DONE** | LightGBM 0.924, +DBVF feats 0.933, k-SEC+DBVF e2e 1.047. |
| Fig. 5 — Headline parity + per-σ-bin MAE | `figs/fig_5_parity_per_bin.png` | **DONE** | OOF parity, Spearman ρ=0.78; per-bin MAE table. |
| Fig. 6 — Virtual-screen top-15, family-coloured | `figs/fig_6_virtual_screen.png` | **DONE** | Bars from baseline -3 (taller = faster); 4 families recovered. |
| Fig. 7 — OOD by family | `figs/fig_7_ood_by_family.png` | **DONE** | Argyrodite 0.903, garnet 1.054, LGPS 1.456 vs. ID 1.233. |
| Fig. 8 — MC-dropout calibration | `figs/fig_8_calibration.png` | **DONE** | Coverage at 1σ / 1.96σ. |
| Fig. 9 — Top-K precision (ranking-quality summary) | `figs/fig_9_top_k_precision.png` | **DONE** | Top-10 70 % (19.7× lift), Spearman ρ 0.78. |
| Fig. 10 — Learned DBVF parameters | `figs/fig_10_dbvf_learned.png` | **DONE** | Per-anion r₀ and b vs. Brown 2002 across 5 seeds. |
| Fig. 11 — Learning curve on OBELiX | `figs/fig_11_learning_curve.png` | **DONE** | k-SEC drops 1.46 → 1.27 (n=90→135), then plateaus; gap shrinks 0.31 → 0.17–0.22. |
| Fig. 12 (SI) — Cubic-harmonic filter interpretability | `figs/fig_5_kubic_interpret.png` | **DONE** | Mean directional rel. std 0.05. |

> All figures are now rendered against Phase B1 headline numbers
> (1.047 standalone / 0.980 stacked, n = 281). Re-render with
> `python scripts/07_make_plots.py`.

## Code, data, and reproducibility

| Asset | Path | Status |
|---|---|---|
| Source code | `src/ionpath/` | **DONE** (in repo). |
| Headline training script | `scripts/08_train_hybrid.py --use-bv-field` | **DONE** |
| Stacking script | `scripts/10_stacking.py` | **DONE** |
| Virtual screening script | `scripts/16_virtual_screening.py` | **DONE** |
| LightGBM-with-DBVF control | `scripts/27_lightgbm_with_dbvf.py` | **DONE** |
| Learned DBVF parameter extraction | `scripts/28_dbvf_interpret.py` | **DONE** |
| Bootstrap CI computation (inline) | `scripts/07_make_plots.py` (data: `results/phaseB1_bootstrap_ci.json`) | **DONE** |
| Pretrained encoder weights | `weights/ksec_mp_pretrain.pt` | **TODO** package for release |
| Headline trained weights (5 seeds) | `results/ksec_phaseB1_seed{0..4}.pt` | **DONE** (in repo, to copy to `weights/` for release) |
| OBELiX-derived data artifacts | `data/obelix_filtered.parquet` | **DONE** |
| Virtual screening output | `results/virtual_screen_top_100.csv` + `results/virtual_screen_all.csv` | **DONE** |
| OOF predictions (NPZ) for headline | `results/ksec_phaseB1_oof.npz` | **DONE** |
| Reproducibility unit tests (5/5 passing) | `tests/` | **DONE** |
| Docker / environment lockfile | `Dockerfile` + `requirements.txt` | **TODO** |
| Public release URL | GitHub URL | **TODO** at submission time |

## JMST formatting requirements

- [ ] Manuscript converted from Markdown → LaTeX (Elsevier `elsarticle` class)
      using `pandoc -o JMST-DRAFT.tex JMST-DRAFT.md` plus manual
      cleanup of citation keys, table widths, and figure references.
- [x] References available in BibTeX (`refs.bib`, 19 entries).
- [x] Highlights: ≤ 85 characters per bullet, ≤ 5 bullets total.
- [x] Abstract: ≤ 200 words.
- [x] Word count target (research article): 6,000–8,000 words excluding
      references and SI. Current main: ~8,065 words (at upper limit;
      may want a final trim pass during LaTeX cleanup).
- [ ] All figures exported at ≥ 300 dpi, RGB or grayscale, with separate
      caption file (figures are PNG at 200 dpi today; bump to 300 in
      `scripts/07_make_plots.py` and `scripts/30_architecture_diagram.py`
      before camera-ready).
- [ ] Tables formatted with Elsevier table style (booktabs in LaTeX).
- [ ] CRediT statements per author.
- [x] Conflicts of interest declaration (cover letter + manuscript footer).
- [x] Data availability statement (Section *Data and code availability*).

## Pre-submission internal checks

- [x] Headline numbers consistent between `RESULTS-kSEC.md`,
      `JMST-DRAFT.md`, and `JMST-SI.md` (1.047 standalone / 0.980
      stacked, n = 281, Phase B1).
- [x] Stacked-vs-standalone caveat documented (LightGBM-with-features
      0.924 < neural standalone 1.047).
- [x] Architecture-vs-features control documented (DBVF as features
      worsens LightGBM).
- [x] Virtual-screen top-15 saved and traceable to a single checkpoint.
- [x] Negative results moved to SI rather than buried.
- [ ] Co-author review of full draft.
- [ ] Independent reproducibility run on fresh checkout.
- [ ] Spell-check + grammar pass.

## Post-acceptance items (track for later)

- [ ] arXiv preprint deposit on submission.
- [ ] GitHub release v1.0 tagged at submission commit.
- [ ] Zenodo DOI for code + weights.
- [ ] Update lab webpage and BibTeX templates.

---

**Suggested order of remaining work (highest leverage first):**

1. **Convert manuscript and SI to LaTeX (`elsarticle`)** with
   `pandoc -o JMST-DRAFT.tex JMST-DRAFT.md --bibliography=refs.bib
   --citeproc`, then manual cleanup of citation keys, table widths,
   and figure references.
2. **Bump figure DPI from 200 to 300** in `scripts/07_make_plots.py`
   and `scripts/30_architecture_diagram.py`; re-render all figures.
3. **Polish Fig. 1 architecture schematic** in a vector editor (the
   programmatic version is functional but not camera-ready).
4. **Package weights** for release: `weights/ksec_mp_pretrain.pt` +
   `weights/phaseB1_seed{0..4}.pt` + `weights/phaseB1_oof.npz`.
5. **Add Dockerfile + requirements.txt** for environment lockfile.
6. **Independent reproducibility run** on a clean checkout to confirm
   Phase B1 numbers match the headline within ± 0.005 MAE.
7. **Co-author review pass** on full manuscript + SI + cover letter.
8. **arXiv submission** simultaneous with JMST submission.
