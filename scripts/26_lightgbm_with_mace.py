"""Critical fairness test: how much of Phase A4's win is from k-SEC vs
just from giving LightGBM the MACE features?

Train LightGBM on (Magpie + lattice + geometric + MACE), 5-fold CV with
the same splits as the rest of the project. Report MAE/R²/AUC.

If this beats stacked k-SEC (0.986), the architecture is redundant.
If it loses to k-SEC standalone (1.079), MACE alone isn't enough — the
k-space features are doing real work.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return dict(mae=mae, r2=r2)


def auc(y_true_log, scores, threshold=-4.0):
    y = (y_true_log >= threshold).astype(np.float64)
    order = np.argsort(scores); yt = y[order]
    n_pos = yt.sum(); n_neg = yt.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, yt.size + 1, dtype=np.float64)
    return float((np.sum(ranks * yt) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def lightgbm_oof(X, y, eligible, n_seeds=5):
    import lightgbm as lgb
    all_preds = []
    for seed in range(n_seeds):
        folds = stratified_folds(y[eligible], seed=seed * 7)
        folds = [[int(eligible[i]) for i in f] for f in folds]
        pred = np.full_like(y, np.nan, dtype=np.float32)
        for k in range(5):
            test = np.array(folds[k], dtype=np.int64)
            train = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05,
                num_leaves=31, min_child_samples=5, verbose=-1,
                random_state=seed * 31 + 1,
            )
            m.fit(X[train], y[train])
            pred[test] = m.predict(X[test])
        all_preds.append(pred)
    return np.nanmean(np.stack(all_preds, axis=0), axis=0)


def main():
    with open("data/cache/crystals.pkl", "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load("data/cache/labels.npz", allow_pickle=True)
    log_sigma = z["log_sigma"].astype(np.float32)
    mask = z["mask"]

    has_cg = np.array([c is not None for c in crystals])
    eligible = np.where((mask > 0) & has_cg & (log_sigma > -15.0))[0]
    log.info("Eligible: %d", len(eligible))

    # Three feature configurations
    feature_sets = {}

    # (1) Magpie alone (the original tabular ceiling)
    X1 = np.zeros((len(crystals), 132), dtype=np.float32)
    for i, c in enumerate(crystals):
        if c is not None and c.magpie is not None:
            X1[i] = c.magpie
    feature_sets["LightGBM + Magpie"] = X1

    # (2) Magpie + lattice + geometric (richer tabular)
    X2 = np.zeros((len(crystals), 132 + 8 + 25), dtype=np.float32)
    for i, c in enumerate(crystals):
        if c is None:
            continue
        if c.magpie is not None:
            X2[i, :132] = c.magpie
        if c.lattice_feats is not None:
            X2[i, 132:140] = c.lattice_feats
        if c.geometric is not None:
            X2[i, 140:165] = c.geometric
    feature_sets["LightGBM + Magpie + lattice + geometric"] = X2

    # (3) Magpie + lattice + geometric + MACE (4-dim)
    X3 = np.zeros((len(crystals), 132 + 8 + 25 + 4), dtype=np.float32)
    for i, c in enumerate(crystals):
        if c is None:
            continue
        if c.magpie is not None:
            X3[i, :132] = c.magpie
        if c.lattice_feats is not None:
            X3[i, 132:140] = c.lattice_feats
        if c.geometric is not None:
            X3[i, 140:165] = c.geometric
        if getattr(c, "mace", None) is not None:
            X3[i, 165:169] = c.mace
        else:
            X3[i, 165:169] = [-7.5, 0.0, 0.0, 0.0]
    feature_sets["LightGBM + ALL (Magpie + lattice + geometric + MACE)"] = X3

    results = {}
    for name, X in feature_sets.items():
        log.info("training: %s  (X shape=%s)", name, X.shape)
        ens = lightgbm_oof(X, log_sigma, eligible)
        m = regression_metrics(log_sigma[eligible], ens[eligible])
        a = auc(log_sigma[eligible], ens[eligible])
        log.info("  %s  MAE=%.3f  R²=%.3f  AUC=%.3f", name, m["mae"], m["r2"], a)
        results[name] = dict(**m, auc=a)

    # Reference numbers from Phase A4 (loaded from earlier OOF)
    oof = np.load("results/ksec_phaseA4_oof.npz", allow_pickle=True)
    ksec_ens = oof["ensemble"]
    ksec_m = regression_metrics(log_sigma[eligible], ksec_ens[eligible])
    ksec_a = auc(log_sigma[eligible], ksec_ens[eligible])
    results["k-SEC Phase A4 (standalone)"] = dict(**ksec_m, auc=ksec_a)

    log.info("\n=== Final comparison ===")
    log.info("Configuration                                                      MAE     R²    AUC")
    log.info("-" * 90)
    for name, m in results.items():
        log.info("%-65s  %5.3f  %5.3f  %5.3f", name, m["mae"], m["r2"], m["auc"])

    Path("results/lightgbm_with_mace.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
