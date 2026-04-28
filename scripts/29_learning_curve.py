"""Learning-curve experiment for the JMST submission.

Runs k-SEC + DBVF (Phase B1 config) at training fractions
{0.4, 0.6, 0.8, 1.0} with 3 seeds × 5 folds, and LightGBM at the same
fractions with 5 seeds. Plots MAE-vs-n_train for both, with per-seed
standard deviations as error bars.

The fraction = 1.0 point for k-SEC + DBVF reuses the existing
ksec_phaseB1.json (5-seed result) so we don't retrain it.

Usage:
    python scripts/29_learning_curve.py             # run full experiment
    python scripts/29_learning_curve.py --plot-only # re-plot from JSONs
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("learning_curve")


FRACTIONS = [0.4, 0.6, 0.8]   # 1.0 reuses the existing Phase B1 result
KSEC_SEEDS = 3                # 3 seeds for the curve points (5 already exists at 1.0)
LGBM_SEEDS = 5
N_FOLDS = 5
EPOCHS = 60


def run_ksec_at_fraction(frac: float) -> dict:
    """Run k-SEC + DBVF training at the given fraction; returns per-fraction summary."""
    out_path = RESULTS / f"learning_curve_ksec_f{frac:.1f}.json"
    if out_path.exists():
        log.info("ksec frac=%.1f: cached at %s", frac, out_path)
        with open(out_path) as f:
            return json.load(f)

    log.info("Running k-SEC + DBVF at fraction %.1f (%d seeds)...", frac, KSEC_SEEDS)
    cmd = [
        sys.executable, str(ROOT / "scripts/08_train_hybrid.py"),
        "--use-bv-field", "--use-lattice", "--use-geometric",
        "--seeds", str(KSEC_SEEDS),
        "--epochs", str(EPOCHS),
        "--batch-size", "8",
        "--train-fraction", str(frac),
        "--results", str(out_path),
        "--pretrained-encoder", "results/mp_encoder_pretrained.pt",
    ]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        log.error("k-SEC training failed at fraction %.1f: %s", frac, e)
        raise
    log.info("ksec frac=%.1f done in %.1f min", frac, (time.time() - t0) / 60)
    with open(out_path) as f:
        return json.load(f)


def lightgbm_oof(X, y, eligible, n_seeds=5, train_fraction=1.0):
    """Run LightGBM OOF, optionally subsampling each fold's training set."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    n = len(eligible)
    pred_per_seed = np.full((n_seeds, len(y)), np.nan, dtype=np.float32)
    y_eligible = y[eligible]
    qs = np.quantile(y_eligible, np.linspace(0, 1, 11))
    qs[0] -= 1e-6; qs[-1] += 1e-6
    bins = np.clip(np.digitize(y_eligible, qs) - 1, 0, 9)

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr_idx, te_idx) in enumerate(skf.split(eligible, bins)):
            train_eligible = eligible[tr_idx]
            test_eligible = eligible[te_idx]
            # Subsample train at the requested fraction (stratified)
            if train_fraction < 1.0:
                target = max(8, int(round(len(train_eligible) * train_fraction)))
                rng = np.random.default_rng(300 + seed * 7 + k)
                y_train = y[train_eligible]
                qs2 = np.quantile(y_train, np.linspace(0, 1, 6))
                qs2[0] -= 1e-6; qs2[-1] += 1e-6
                tb = np.clip(np.digitize(y_train, qs2) - 1, 0, 4)
                kept = []
                per_bin = max(1, target // 5)
                for b in range(5):
                    pool = np.where(tb == b)[0]
                    if len(pool) == 0:
                        continue
                    take = min(len(pool), per_bin)
                    kept.extend(rng.choice(pool, size=take, replace=False).tolist())
                if len(kept) < target:
                    remaining = list(set(range(len(train_eligible))) - set(kept))
                    extra = rng.choice(remaining, size=min(target - len(kept), len(remaining)), replace=False)
                    kept.extend(extra.tolist())
                train_eligible = train_eligible[np.array(sorted(kept), dtype=np.int64)]

            params = dict(
                num_leaves=31, learning_rate=0.05, n_estimators=300,
                min_child_samples=5, verbose=-1, random_state=seed,
            )
            m = lgb.LGBMRegressor(**params)
            m.fit(X[train_eligible], y[train_eligible])
            pred_per_seed[seed, test_eligible] = m.predict(X[test_eligible])
    return pred_per_seed


def run_lightgbm_at_fraction(frac: float) -> dict:
    out_path = RESULTS / f"learning_curve_lgbm_f{frac:.1f}.json"
    if out_path.exists():
        log.info("lgbm frac=%.1f: cached at %s", frac, out_path)
        with open(out_path) as f:
            return json.load(f)

    log.info("Running LightGBM at fraction %.1f (%d seeds)...", frac, LGBM_SEEDS)
    with open(ROOT / "data/cache/crystals.pkl", "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(ROOT / "data/cache/labels.npz", allow_pickle=True)
    log_sigma = z["log_sigma"].astype(np.float32)
    mask = z["mask"]
    has_cg = np.array([c is not None for c in crystals])
    eligible = np.where((mask > 0) & has_cg & (log_sigma > -15.0))[0]

    # Magpie + lattice + geometric features (same as the LightGBM full-features baseline)
    from ionpath.data.magpie import featurize_composition
    GEO_DIM = 25
    feat_dim = 132 + 8 + GEO_DIM
    X = np.zeros((len(crystals), feat_dim), dtype=np.float32)
    for i, c in enumerate(crystals):
        if c is None:
            continue
        if getattr(c, "magpie", None) is not None:
            X[i, :132] = c.magpie
        else:
            X[i, :132] = featurize_composition(c.composition)
        if getattr(c, "lattice", None) is not None:
            X[i, 132:140] = c.lattice
        if getattr(c, "geometric", None) is not None:
            X[i, 140:140 + GEO_DIM] = c.geometric

    pred = lightgbm_oof(X, log_sigma, eligible, n_seeds=LGBM_SEEDS, train_fraction=frac)
    ens = np.nanmean(pred, axis=0)
    err = np.abs(ens[eligible] - log_sigma[eligible])
    mae_ens = float(err.mean())
    # Per-seed MAE
    per_seed_mae = []
    for s in range(LGBM_SEEDS):
        e = np.abs(pred[s, eligible] - log_sigma[eligible])
        e = e[np.isfinite(e)]
        if len(e):
            per_seed_mae.append(float(e.mean()))
    n_train = int(round(len(eligible) * 4 / 5 * frac))
    out = dict(
        fraction=frac, n_train_per_fold=n_train,
        mae_ensemble=mae_ens,
        mae_per_seed_mean=float(np.mean(per_seed_mae)),
        mae_per_seed_std=float(np.std(per_seed_mae)),
        per_seed_mae=per_seed_mae,
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("lgbm frac=%.1f n_train≈%d ensemble MAE=%.3f per-seed=%.3f±%.3f",
             frac, n_train, mae_ens, out["mae_per_seed_mean"], out["mae_per_seed_std"])
    return out


def parse_ksec_phase_b1_baseline() -> dict:
    """Use the existing 5-seed Phase B1 result as the fraction=1.0 point."""
    with open(RESULTS / "ksec_phaseB1.json") as f:
        b1 = json.load(f)
    per_seed = [s["mae_mean"] for s in b1["per_seed"]]
    return dict(
        fraction=1.0, n_train_per_fold=int(round(281 * 4 / 5)),
        mae_ensemble=b1["seed_ensemble"]["mae"],
        mae_per_seed_mean=float(np.mean(per_seed)),
        mae_per_seed_std=float(np.std(per_seed)),
        per_seed_mae=per_seed,
    )


def parse_ksec_run(run_json: dict) -> dict:
    per_seed = [s["mae_mean"] for s in run_json["per_seed"]]
    cfg = run_json.get("config", {})
    n_train = cfg.get("n_train_per_fold")
    # Partial files (e.g. f=0.6 with only 2 seeds) lack seed_ensemble; skip it
    ens = run_json.get("seed_ensemble", {}).get("mae")
    return dict(
        fraction=cfg.get("train_fraction", None),
        n_train_per_fold=n_train,
        mae_ensemble=ens,
        mae_per_seed_mean=float(np.mean(per_seed)),
        mae_per_seed_std=float(np.std(per_seed)),
        per_seed_mae=per_seed,
        n_seeds=len(per_seed),
    )


def plot_learning_curve(ksec_pts, lgbm_pts):
    import matplotlib.pyplot as plt
    ksec_pts = sorted(ksec_pts, key=lambda d: d["n_train_per_fold"])
    lgbm_pts = sorted(lgbm_pts, key=lambda d: d["n_train_per_fold"])
    xs_k = [d["n_train_per_fold"] for d in ksec_pts]
    ys_k = [d["mae_per_seed_mean"] for d in ksec_pts]
    es_k = [d["mae_per_seed_std"] for d in ksec_pts]
    xs_l = [d["n_train_per_fold"] for d in lgbm_pts]
    ys_l = [d["mae_per_seed_mean"] for d in lgbm_pts]
    es_l = [d["mae_per_seed_std"] for d in lgbm_pts]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(xs_k, ys_k, yerr=es_k, marker="o", color="#1f77b4",
                capsize=3, label="k-SEC + DBVF (this work)")
    ax.errorbar(xs_l, ys_l, yerr=es_l, marker="s", color="#888888",
                capsize=3, label="LightGBM + Magpie + lattice + geometric")
    for x, y in zip(xs_k, ys_k):
        ax.text(x, y - 0.04, f"{y:.2f}", ha="center", fontsize=8, color="#1f77b4")
    for x, y in zip(xs_l, ys_l):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=8, color="#444444")
    ax.set_xlabel("training samples per fold (n)")
    ax.set_ylabel("per-seed MAE on log₁₀ σ (5-fold CV)")
    ax.set_title("Learning curve on OBELiX — does the gap to LightGBM shrink with n?")
    ax.legend(loc="upper right")
    ax.set_ylim(0.6, max(ys_k + ys_l) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_11_learning_curve.png", dpi=200)
    plt.close(fig)
    log.info("wrote %s", FIGS / "fig_11_learning_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip experiments; just re-render the plot from cached JSONs")
    args = ap.parse_args()

    if not args.plot_only:
        for frac in FRACTIONS:
            run_ksec_at_fraction(frac)
            run_lightgbm_at_fraction(frac)
        # 1.0 point: existing Phase B1 result for k-SEC; rerun LightGBM if needed
        run_lightgbm_at_fraction(1.0)

    # Aggregate
    ksec_pts = [parse_ksec_phase_b1_baseline()]
    for frac in FRACTIONS:
        path = RESULTS / f"learning_curve_ksec_f{frac:.1f}.json"
        if path.exists():
            with open(path) as f:
                ksec_pts.append(parse_ksec_run(json.load(f)))
    lgbm_pts = []
    for frac in FRACTIONS + [1.0]:
        path = RESULTS / f"learning_curve_lgbm_f{frac:.1f}.json"
        if path.exists():
            with open(path) as f:
                lgbm_pts.append(json.load(f))

    summary = dict(ksec=ksec_pts, lgbm=lgbm_pts)
    with open(RESULTS / "learning_curve_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("saved learning_curve_summary.json")

    if not lgbm_pts:
        log.warning("no LightGBM points — plot skipped")
        return
    plot_learning_curve(ksec_pts, lgbm_pts)


if __name__ == "__main__":
    main()
