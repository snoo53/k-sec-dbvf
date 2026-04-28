"""Brutal honest test: extract DBVF features from a trained Phase B1
checkpoint and feed them to LightGBM. If LightGBM with DBVF features
also wins, the k-SEC architecture is still empirically redundant.

This is the analogue of script 26 but using our novel DBVF module
instead of the established MACE potential.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.models import KSECNet
from ionpath.models.bond_valence_field import compute_bv_features

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
    return dict(mae=mae, r2=1.0 - ss_res / max(ss_tot, 1e-12))


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
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=5, verbose=-1, random_state=seed * 31 + 1,
            )
            m.fit(X[train], y[train])
            pred[test] = m.predict(X[test])
        all_preds.append(pred)
    return np.nanmean(np.stack(all_preds, axis=0), axis=0)


def extract_dbvf_features(crystals, ckpt_paths, device="cuda"):
    """For each (crystal, checkpoint), run the model up to the DBVF
    output and capture the 8-dim BV-feature vector. Average across
    checkpoints for an ensembled DBVF feature per crystal.
    """
    n = len(crystals)
    accum = np.zeros((n, 8), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = KSECNet(
            feature_dim=96, num_blocks=3, n_max=2, dropout=0.0,
            tabular_dim=132, lattice_dim=8, geometric_dim=25,
            mace_dim=0, bv_field=True,
        ).to(device)
        m.load_state_dict(ckpt["state"], strict=False)
        m.eval()
        for i, c in enumerate(crystals):
            if c is None:
                continue
            with torch.no_grad():
                atom_z = torch.from_numpy(c.atom_z).long().to(device)
                fp = torch.from_numpy(c.frac_pos).float().to(device)
                bi = torch.zeros(len(c.atom_z), dtype=torch.long, device=device)
                cell = torch.from_numpy(c.cell).float().to(device)[None]
                feats = compute_bv_features(
                    m.bv_params, atom_z, fp, cell, bi,
                    num_graphs=1, mobile_z=3, cutoff=4.0,
                )
            accum[i] += feats[0].cpu().numpy()
            counts[i] += 1
        del m
    counts_safe = np.maximum(counts, 1).reshape(-1, 1)
    return accum / counts_safe, counts > 0


def main():
    with open("data/cache/crystals.pkl", "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load("data/cache/labels.npz", allow_pickle=True)
    log_sigma = z["log_sigma"].astype(np.float32)
    mask = z["mask"]
    has_cg = np.array([c is not None for c in crystals])
    eligible = np.where((mask > 0) & has_cg & (log_sigma > -15.0))[0]
    log.info("Eligible: %d", len(eligible))

    log.info("extracting DBVF features from 5 trained checkpoints...")
    ckpt_paths = [f"results/ksec_phaseB1_seed{s}.pt" for s in range(5)]
    dbvf_feats, valid = extract_dbvf_features(crystals, ckpt_paths)
    log.info("DBVF features extracted: shape=%s, valid=%d", dbvf_feats.shape, valid.sum())

    # Build feature sets
    feature_sets = {}

    # Magpie + lattice + geometric (no MACE, no DBVF)
    X_base = np.zeros((len(crystals), 132 + 8 + 25), dtype=np.float32)
    for i, c in enumerate(crystals):
        if c is None:
            continue
        if c.magpie is not None:
            X_base[i, :132] = c.magpie
        if c.lattice_feats is not None:
            X_base[i, 132:140] = c.lattice_feats
        if c.geometric is not None:
            X_base[i, 140:165] = c.geometric
    feature_sets["LightGBM + Magpie + lattice + geometric (no DBVF)"] = X_base

    # +DBVF features
    X_dbvf = np.concatenate([X_base, dbvf_feats], axis=1)
    feature_sets["LightGBM + ALL + DBVF (8 features from our trained module)"] = X_dbvf

    results = {}
    for name, X in feature_sets.items():
        log.info("training: %s  X.shape=%s", name, X.shape)
        ens = lightgbm_oof(X, log_sigma, eligible)
        m = regression_metrics(log_sigma[eligible], ens[eligible])
        a = auc(log_sigma[eligible], ens[eligible])
        log.info("  %s  MAE=%.3f  R²=%.3f  AUC=%.3f", name, m["mae"], m["r2"], a)
        results[name] = dict(**m, auc=a)

    # Reference: Phase B1 standalone
    oof = np.load("results/ksec_phaseB1_oof.npz", allow_pickle=True)
    ksec_ens = oof["ensemble"]
    ksec_m = regression_metrics(log_sigma[eligible], ksec_ens[eligible])
    ksec_a = auc(log_sigma[eligible], ksec_ens[eligible])
    results["k-SEC Phase B1 (DBVF) standalone (5-seed ensemble)"] = dict(**ksec_m, auc=ksec_a)

    log.info("\n=== Final comparison ===")
    log.info("%-65s  %5s  %5s  %5s", "Configuration", "MAE", "R²", "AUC")
    log.info("-" * 90)
    for name, m in results.items():
        log.info("%-65s  %5.3f  %5.3f  %5.3f", name, m["mae"], m["r2"], m["auc"])

    Path("results/lightgbm_with_dbvf.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
