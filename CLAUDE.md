# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code and submission pipeline for the manuscript *Cubic-Equivariant k-Space Convolution and a Differentiable Bond-Valence Field for Ionic-Conductivity Prediction in Solid-State Electrolytes* (Sunwoo Lee, 2026).

- Submitted to *Journal of Materials Science & Technology* (manuscript ID `J-MST-D-26-03318`)
- Preprint: ChemRxiv, DOI `10.26434/chemrxiv.15002591/v1`
- Public mirror: `github.com/snoo53/k-sec-dbvf`

`README.md` covers the headline numbers, model description, citation, and the full reproduction recipe. This file documents the things you have to read multiple files to learn.

## Two pipelines, one repo — keep them separate

This repo mixes **research code** and a **document-build pipeline**. They share no imports and serve different purposes; conflating them is the most common mistake.

| Pipeline | Lives in | Purpose | Entry point |
|---|---|---|---|
| Research | `src/ionpath/`, `scripts/`, `tests/`, `configs/`, `data/`, `results/`, `figs/` | Train and evaluate the k-SEC + DBVF model | `scripts/08_train_hybrid.py` and friends |
| Documents | `applications/`, `JMST-*.md`, `refs.bib`, `ChemRxiv-Manuscript.pdf` | Produce manuscript .docx, anonymous version, cover letter, figure renaming, ChemRxiv preprint PDF, JHU/transfer-app templates | `applications/build_jmst_submission.py`, `applications/build_chemrxiv_pdf.py` |

If you're asked to "fix the manuscript," the work is almost always in `JMST-DRAFT.md` plus a rebuild via `applications/build_jmst_submission.py`. If you're asked to "improve the model," the work is in `src/ionpath/models/` plus a rerun of the relevant `scripts/`.

**Don't edit `JMST-DRAFT.md`, `JMST-SI.md`, or `JMST-COVER-LETTER.md` without an explicit user request.** The manuscript is submitted (J-MST-D-26-03318); changes propagate into rebuilds via `applications/build_jmst_submission.py` and could desync the submitted version from the repo if rebuilt and committed casually.

### Numbered-script convention

`scripts/NN_<name>.py` are the only intended entry points. Numbering is roughly chronological in the project's research narrative, not topological. Functional groupings:

| Range | Purpose |
|---|---|
| 01–02 | OBELiX download + featurize |
| 03, 08 | Train k-SEC standalone (03) and hybrid k-SEC + DBVF (08, headline) |
| 04–07 | OOD-by-family, MC dropout, ablation, plotting |
| 09, 12, 21 | Pretraining drivers (Magpie, MP, mega-pretrain) |
| 10 | LightGBM stacking on k-SEC OOF |
| 11, 17–19, 23 | External fetchers (MP, JARVIS, OQMD, MatBench, AFLOW) |
| 13 | LLM literature mining (uses `anthropic`) |
| 14, 15 | MatBench eval, Arrhenius multi-task |
| 16 | Virtual screening on 18,574 Li-containing MP crystals |
| 22, 28 | Interpretability (Kubic filter, DBVF parameters) |
| 24–27 | MACE-based barriers/energy + LightGBM (negative-result territory) |
| 29 | Learning-curve experiment |
| 30 | Architecture diagram for the manuscript |

### Model components in `src/ionpath/models/`

| File | Exports | Role |
|---|---|---|
| `kspace_conv.py` | `KSECNet` | Headline k-space encoder (Kubic-harmonic filters + cross-shell gated attention) |
| `bond_valence_field.py` | `LearnableBVParams`, `compute_bv_features` | DBVF — Brown 2002 (r₀, b) as trainable params |
| `path_bv_field.py` | `LearnablePathBVParams`, `compute_path_bv_features` | Phase B2 path-integrated DBVF (SI negative result) |
| `mpnn_encoder.py` | `MPNNEncoder` | Auxiliary real-space stream |
| `cross_attention_bridge.py` | `CrossAttentionBridge`, `pad_atoms` | Couples k-space and real-space when both enabled |

Composition happens in `scripts/08_train_hybrid.py` via `--use-bv-field`, `--use-lattice`, `--use-geometric`, `--hetero` flags.

## Phase naming — the foot-gun

Experiments are labeled by phase across `RESULTS-kSEC.md`, `IMPROVEMENT-PLAN.md`, `PIVOT-START.md`, the SI, and result filenames. **Be careful which phase is the headline.**

- **Phase A1–A4** — early architectural ablations (heteroscedastic NLL, percolation features, n_max grid, etc.) — most are negative results; they live in the SI as honest characterization.
- **Phase B1** — the **headline result** (k-SEC + DBVF). Anything matching `phaseB1` or `ksec_phaseB1*` is the submitted model. Checkpoints: `results/ksec_phaseB1_seed{0..4}.pt`. Metrics: `results/ksec_phaseB1.json`. OOF predictions: `results/ksec_phaseB1_oof.npz`.
- **Phase B2** — path-integrated DBVF (negative result; SI)
- **Phase B3** — dual-stream **BatteryNet** (negative result; SI). The name "BatteryNet" refers to *this negative baseline*, not the headline. Don't confuse `scripts/15_*` exploration code with the submitted model.

When a commit message, log file, or filename references a phase, that's how to read it.

## Setup quirks

- Python ≥ 3.10, PyTorch ≥ 2.2.
- `requirements.txt` is the canonical install path. `pyproject.toml` declares `ionpath` as a package with optional extras (`materials`, `graph`, `boost`, `viz`, `dev`), but **no workflow actually installs it** — tests prepend `src/` to `sys.path` ([tests/test_smoke.py:8](tests/test_smoke.py#L8)) and scripts run from the repo root. Don't `pip install -e .` expecting it to be wired up.
- **Materials Project API key required**: `echo "MP_API_KEY=<your key>" > .env` before any script that fetches MP data (`scripts/11_fetch_mp.py`, `scripts/12_pretrain_mp.py`, `scripts/16_virtual_screening.py`).
- Training/pretraining expect a CUDA GPU (~12 GB VRAM); CPU runs the smoke tests only.

## Common commands

```bash
# Smoke tests — fast, no data download. Five tests covering Fourier basis,
# CIF→graph conversion, KSECNet forward/backward, and translation invariance.
pytest tests/test_smoke.py
pytest tests/test_smoke.py::test_ksec_forward_backward    # single test
python tests/test_smoke.py     # also works (the file has a bare-main runner)

# Lint
ruff check .

# Reproduce headline numbers without retraining (uses committed checkpoints)
python scripts/10_stacking.py --ksec-oof results/ksec_phaseB1_oof.npz \
    --results results/stacking_phaseB1.json

# Full retraining recipe — see README.md ("Reproducing the headline result")
# for the canonical 7-step pipeline (~8 hours on a 12 GB GPU).

# Rebuild the JMST submission package (Manuscript.docx + anonymous version
# + cover letter + DOCI + highlights + renamed figures, zipped)
python applications/build_jmst_submission.py
```

## Repository debris worth knowing about

These files exist alongside source and can confuse a fresh search:

- `JMST-DRAFT.md.bak`, `JMST-DRAFT.md.bak2`, `JMST-SI.md.bak` — pre-edit snapshots. Safe to ignore. The active manuscript source is `JMST-DRAFT.md` (no suffix).
- `*.log` at the repo root (e.g. `phaseB1.log`, `hybrid_*.log`, `mp_broad_pretrain.log`, `mace_barriers.log`) — training-run stdout captures from earlier sessions. Not load-bearing.
- `notebooks/` and `checkpoints/` — exploratory work and throwaway pretraining checkpoints. The canonical headline checkpoints live in `results/ksec_phaseB1_seed{0..4}.pt`.
- `applications/jmst_submission/` — the most-recent build output, regenerated each time `build_jmst_submission.py` runs. Don't edit files here directly; they're overwritten.
- `JMST-Submission-Package.zip` at the repo root is the latest produced artifact, also regenerated.
- `applications/build_jhu_*`, `applications/jhu_research_*`, `applications/stanford_additional_info.txt` — **personal transfer-application materials**, not manuscript code. Several entries in this group are gitignored under `.gitignore`'s "Private" section. Treat as out-of-scope unless the user explicitly asks about them.

## DBVF and Brown 2002

The Differentiable Bond-Valence Field treats Brown's 2002 tabulated `(r₀, b)` parameters as **trainable** PyTorch parameters, initialised from Brown's table and updated end-to-end. The initialisation values and the implementation both live in `src/ionpath/models/bond_valence_field.py`. The "first treatment of Brown's tabulated parameters as trainable parameters of a neural network" is the novelty claim in the manuscript and is true to the author's knowledge — be precise about this if you need to write about it.

## Author affiliation in artifacts

The manuscript and submitted artifacts list **"Independent researcher, South Korea"** as the affiliation, with `lee.11539@buckeyemail.osu.edu` as correspondence. ORCID `0009-0004-9159-367X`. If you regenerate any title page, cover letter, or DOCI declaration, these are the canonical values; they're hard-coded in `applications/build_title_page.py`, `applications/build_cover_pdf.py`, and `applications/build_coi_declaration.py`.

A separate prior project (Aim Materials) uses **The Ohio State University** affiliation; that project is in a different repo (`~/Desktop/Old Projects/aim-materials/`), not here. Don't cross-contaminate the affiliation strings.
