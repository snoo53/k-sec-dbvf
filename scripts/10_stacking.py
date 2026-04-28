"""Stacking: combine k-SEC Hybrid OOF predictions with LightGBM+Magpie
OOF predictions via a ridge meta-learner.

Produces:
  - LightGBM+Magpie OOF (same CV folds/seeds as k-SEC hybrid)
  - k-SEC Hybrid OOF (loaded from --ksec-oof .npz)
  - Stacked meta-learner OOF
  - Final metrics for all three

Usage:
    python scripts/10_stacking.py \
        --ksec-oof results/ksec_hybrid_frozen_oof.npz \
        --results results/stacking.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.data.magpie import featurize_composition

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def stratified_folds(y, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    q = np.quantile(y, np.linspace(0, 1, 11)); q[0] -= 1e-6; q[-1] += 1e-6
    b = np.clip(np.digitize(y, q) - 1, 0, 9)
    folds = [[] for _ in range(n_folds)]
    for bi in range(10):
        idx = np.where(b == bi)[0]
        rng.shuffle(idx)
        for j, k in enumerate(idx):
            folds[j % n_folds].append(int(k))
    return folds


def lightgbm_oof(X, y, eligible, n_seeds=5):
    """LightGBM+Magpie OOF predictions using the same 5-seed × 5-fold splits
    as hybrid k-SEC. Averaged across seeds.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("Install lightgbm: pip install lightgbm")
        raise
    all_preds = []
    for seed in range(n_seeds):
        folds = stratified_folds(y[eligible], seed=seed * 7)
        folds = [[int(eligible[i]) for i in f] for f in folds]
        pred_seed = np.full_like(y, np.nan, dtype=np.float32)
        for k in range(5):
            test = np.array(folds[k], dtype=np.int64)
            train = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05,
                num_leaves=31, min_child_samples=5, verbose=-1,
                random_state=seed * 31 + 1,
            )
            m.fit(X[train], y[train])
            pred_seed[test] = m.predict(X[test])
        all_preds.append(pred_seed)
    return np.stack(all_preds, axis=0)                 # (n_seeds, N)


def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(mae=float("nan"), rmse=float("nan"), r2=float("nan"))
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2)


def classification_auc(y_true_log, scores, threshold=-4.0):
    y = (y_true_log >= threshold).astype(np.float64)
    order = np.argsort(scores)
    yt = y[order]
    n_pos = yt.sum(); n_neg = yt.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, yt.size + 1, dtype=np.float64)
    return float((np.sum(ranks * yt) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crystals", default="data/cache/crystals.pkl")
    p.add_argument("--labels", default="data/cache/labels.npz")
    p.add_argument("--ksec-oof", required=True,
                   help="NPZ file containing 'ensemble' (N,) and 'per_seed' (S,N) from hybrid training")
    p.add_argument("--results", default="results/stacking.json")
    p.add_argument("--n-seeds", type=int, default=5)
    args = p.parse_args()

    with open(args.crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(args.labels, allow_pickle=True)
    log_sigma = z["log_sigma"].astype(np.float32)
    mask = z["mask"]
    has_cg = np.array([c is not None for c in crystals])
    # Match training's filter: log_sigma > -15 (physical detection-limit) and has CIF
    eligible = np.where((mask > 0) & has_cg & (log_sigma > -15.0))[0]
    log.info("Eligible: %d", len(eligible))

    # Magpie features for LightGBM
    X = np.zeros((len(crystals), 132), dtype=np.float32)
    for i, c in enumerate(crystals):
        if c is None:
            continue
        if getattr(c, "magpie", None) is not None:
            X[i] = c.magpie
        else:
            X[i] = featurize_composition(c.composition)

    # Train LightGBM OOF
    log.info("Training LightGBM OOF with %d seeds × 5 folds...", args.n_seeds)
    lgb_per_seed = lightgbm_oof(X, log_sigma, eligible, n_seeds=args.n_seeds)
    lgb_ens = np.nanmean(lgb_per_seed, axis=0)
    lgb_metrics = regression_metrics(log_sigma[eligible], lgb_ens[eligible])
    lgb_auc = classification_auc(log_sigma[eligible], lgb_ens[eligible])
    log.info("LightGBM+Magpie ensemble  MAE=%.3f  R²=%.3f  AUC=%.3f",
             lgb_metrics["mae"], lgb_metrics["r2"], lgb_auc)

    # Load hybrid k-SEC OOF
    oof = np.load(args.ksec_oof, allow_pickle=True)
    ksec_per_seed = oof["per_seed"]
    ksec_ens = oof["ensemble"]
    ksec_metrics = regression_metrics(log_sigma[eligible], ksec_ens[eligible])
    ksec_auc = classification_auc(log_sigma[eligible], ksec_ens[eligible])
    log.info("k-SEC hybrid ensemble (from %s)  MAE=%.3f  R²=%.3f  AUC=%.3f",
             args.ksec_oof, ksec_metrics["mae"], ksec_metrics["r2"], ksec_auc)

    # Stack: ridge meta-learner on (ksec_ens, lgb_ens) OOF
    # We use nested 5-fold CV on eligible samples to honest-OOF the meta-learner.
    stack_X = np.stack([ksec_ens[eligible], lgb_ens[eligible]], axis=1)  # (N, 2)
    stack_y = log_sigma[eligible]
    folds = stratified_folds(stack_y, seed=0)
    stack_pred = np.full(len(eligible), np.nan, dtype=np.float32)
    ridge_coefs = []
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        raise
    for k in range(5):
        test = np.array(folds[k], dtype=np.int64)
        train = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
        m = Ridge(alpha=1.0)
        m.fit(stack_X[train], stack_y[train])
        stack_pred[test] = m.predict(stack_X[test])
        ridge_coefs.append(dict(coef=m.coef_.tolist(), intercept=float(m.intercept_)))
    stack_metrics = regression_metrics(stack_y, stack_pred)
    stack_auc = classification_auc(stack_y, stack_pred)
    log.info("Stacked (ridge) ensemble  MAE=%.3f  R²=%.3f  AUC=%.3f",
             stack_metrics["mae"], stack_metrics["r2"], stack_auc)

    out = dict(
        lightgbm=dict(**lgb_metrics, auc=lgb_auc),
        ksec_hybrid=dict(**ksec_metrics, auc=ksec_auc),
        stacked_ridge=dict(**stack_metrics, auc=stack_auc,
                            per_fold_coefs=ridge_coefs),
        config=dict(n_seeds=args.n_seeds,
                     ksec_oof_source=str(args.ksec_oof)),
    )
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2, default=float))
    log.info("saved to %s", args.results)


if __name__ == "__main__":
    main()
