# Post-reboot start-here checklist

This file tells future-you (or future-me) exactly what to do next on the
Nature-tier pivot, in order. See [ROADMAP.md](ROADMAP.md) for the full plan.

## Updates from post-reboot session (2026-04-24 late evening)

**New compute launched:**
- `mp_broad_encoder_pretrained.pt` training is in progress on the full 50k
  all-element MP set (30 epochs, ~5–8 h on consumer GPU). When finished,
  every downstream task should load this instead of the Li-only encoder.

**Data scaled:**
- `data/raw/mp_broad.jsonl.gz` — 50,000 all-element MP crystals (19.5 MB)
- `data/cache/mp_broad_parsed.pkl` — 50k parsed pymatgen Structures (37 MB)

**Scripts made runnable:**
- `scripts/15_arrhenius_multitask.py` — **fully implemented** (was
  skeleton). Smoke-tested at 1 epoch CPU. Combines 281 OBELiX CIFs +
  750 Hargreaves composition-only entries via Arrhenius physics into
  1,031 effective training samples — 3.7× scale-up.
- `scripts/14_matbench_eval.py` — switched from `matbench` package to
  `matminer.datasets.load_dataset` (avoids the Python 3.11 install issue).
  All 13 matbench tasks accessible.
- `scripts/08_train_hybrid.py` — added `--save-ckpt` flag so final models
  can be reloaded by `16_virtual_screening.py`.

**Dependencies:**
- `anthropic` installed (for 13_literature_mine.py).
- `matminer` installed (replaces matbench package for dataset loading).
- `matbench` package still fails on Python 3.11 — not needed with matminer.

---

## State at end of previous session (2026-04-24 evening)

**Data on disk:**
- `data/cache/crystals.pkl` — 562 OBELiX entries, 285 with CIFs, Magpie + lattice + geometric populated
- `data/cache/mp_parsed.pkl` — 18,574 Li-containing MP crystals, parsed
- `data/raw/LiIonDatabase.csv` — Hargreaves 820 entries
- `data/raw/mp_li.jsonl.gz` — 18,574 MP Li, raw download (9.2 MB)
- `data/raw/mp_broad.jsonl.gz` — should be written in background (50k all-element MP, ~250 MB)

**Trained weights:**
- `results/mp_encoder_pretrained.pt` — MP-pretrained k-SEC encoder (2.4 MB, val Ef 0.072 eV/atom)
- `results/magpie_pretrained.pt` — Hargreaves Magpie pretrain (90 KB, val MAE 0.986)

**Benchmark results on OBELiX filtered (n=281):**
- Stacked (k-SEC + LightGBM): **MAE 0.995, R² 0.625, AUC 0.897** ← current best
- LightGBM alone: 0.999
- k-SEC + MP + all features: 1.103

## Immediate next steps (first week post-reboot)

### Day 1 — Verify broad MP fetch completed

```bash
ls -la data/raw/mp_broad.jsonl.gz    # should be ~100-250 MB
zcat data/raw/mp_broad.jsonl.gz | wc -l    # should be ~50000
```

If not done, rerun:
```bash
python scripts/11_fetch_mp.py --phase broad --limit 50000 --out data/raw/mp_broad.jsonl.gz
```

### Day 2 — Re-pretrain MP encoder on broader 50k set  [ALREADY LAUNCHED]

Running from the current session — check `mp_broad_pretrain.log` and
`results/mp_broad_encoder_pretrained.pt` for output. Expected: 5–8 h
GPU (50k samples). Target val_MAE_Ef < 0.1, val_MAE_Eg < 0.4.

If the run was killed mid-session, re-launch:
```bash
python scripts/12_pretrain_mp.py \
    --epochs 30 --batch-size 32 --device cuda \
    --input data/raw/mp_broad.jsonl.gz \
    --cache data/cache/mp_broad_parsed.pkl \
    --out results/mp_broad_encoder_pretrained.pt
```

### Day 3–4 — WP1: Unify σ data  [scripts ready]

1. Hargreaves + OBELiX Arrhenius multi-task (fully implemented, smoke-tested):
```bash
python scripts/15_arrhenius_multitask.py --epochs 80 --seeds 5 --device cuda \
    --pretrained-encoder results/mp_broad_encoder_pretrained.pt \
    --results results/ksec_arrhenius.json
```
Expected: 1,031 training samples, 5-fold CV, 5 seeds. Target per-seed
MAE ≤1.15, ensemble ≤0.95. Runtime ~2–4 h.

2. LLM literature mining (need ~30 post-2023 SSE papers as plain text in `data/raw/papers/`):
```bash
ANTHROPIC_API_KEY=sk-... python scripts/13_literature_mine.py \
    --input-dir data/raw/papers
```

### Day 5 — WP2: First Matbench benchmark

```bash
python scripts/14_matbench_eval.py \
    --task matbench_mp_gap --epochs 40 --device cuda \
    --pretrained-encoder results/mp_broad_encoder_pretrained.pt
```

Expected: 5-fold Matbench CV, log results to JSON, record to matbench leaderboard format.

## Reference: directory layout after pivot

```
battery/
├── ROADMAP.md              — 7-month plan
├── PIVOT-START.md          — this file
├── RESULTS-kSEC.md         — final numbers from pre-pivot session
├── IMPROVEMENT-PLAN.md     — (older, subsumed by ROADMAP)
├── scripts/
│   ├── 01-10*.py           — original k-SEC pipeline
│   ├── 11_fetch_mp.py      — MP API fetcher
│   ├── 12_pretrain_mp.py   — MP encoder pretrain
│   ├── 13_literature_mine.py     — LLM σ extraction (WP1)
│   ├── 14_matbench_eval.py       — Matbench benchmark harness (WP2)
│   ├── 15_arrhenius_multitask.py — unified σ training (WP1+WP3)
│   └── 16_virtual_screening.py   — MP ranking by predicted σ (WP5)
└── data/
    ├── raw/                — LiIonDatabase.csv, mp_li.jsonl.gz, mp_broad.jsonl.gz,
    │                         papers/ (user-supplied for WP1 mining)
    └── cache/              — crystals.pkl, mp_parsed.pkl, mp_broad_parsed.pkl
```

## Important notes

1. **matbench package install fails on Python 3.11** (`distutils.msvccompiler`
   removed). Three workarounds:
   - Use `matminer.datasets.get_all_dataset_info()` to pull the raw dataset
     from Figshare and handle the 5-fold split manually (safest).
   - Install in a separate conda env with Python 3.10.
   - Try `pip install "numpy<1.26" matbench` to pin old numpy.
   Don't waste time on this until WP2 (month 3).

2. **Not in-session**: WP3 CHGNet/MACE migration-barrier computation requires
   the `mace-torch` or `chgnet` package + a cheap NEB implementation. Budget
   a full day to set that up when you get to WP3.

2. **LLM API budget**: expect ~$100–300 in Claude credits for WP1 literature
   mining. 30 papers × ~50k tokens each × Sonnet pricing.

3. **Compute budget**: ~200 GPU-days total across 7 months. Consumer RTX 4070
   SUPER is enough; a cloud GPU burst for Matbench MP-E-form (106k samples) may
   be worth it to save wall-time.

4. **Keep the old results**: `results/ksec_final.json`, `ksec_final_oof.npz`,
   `stacking.json` are the "baseline before pivot" for the paper. Don't
   overwrite; future experiments write to new result files.

5. **Memory**: the pre-pivot 14 GB RAM bloat bug is fixed in
   `scripts/08_train_hybrid.py` (gc + empty_cache between folds). Further
   scripts should copy that pattern.
