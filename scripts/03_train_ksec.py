"""Train k-SEC (k-Space Equivariant Convolutional Network) on OBELiX.

Usage:
    python scripts/03_train_ksec.py --epochs 80 --seeds 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.models import KSECNet

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


def build_inputs(crystals, idx, device):
    atom_z, frac_pos, batch_idx = [], [], []
    for b, gi in enumerate(idx):
        cg = crystals[gi]
        atom_z.append(cg.atom_z)
        frac_pos.append(cg.frac_pos.astype(np.float32))
        batch_idx.append(np.full(len(cg.atom_z), b, dtype=np.int64))
    return (
        torch.from_numpy(np.concatenate(atom_z)).long().to(device),
        torch.from_numpy(np.concatenate(frac_pos)).float().to(device),
        torch.from_numpy(np.concatenate(batch_idx)).long().to(device),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return dict(mae=float("nan"), rmse=float("nan"), r2=float("nan"), n=0)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2, n=int(y_true.size))


def classification_auc(y_true_log: np.ndarray, scores: np.ndarray, threshold: float = -4.0) -> float:
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
    p.add_argument("--results", default="results/ksec.json")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    log.info("device=%s  epochs=%d  seeds=%d  feat_dim=%d  n_max=%d",
             args.device, args.epochs, args.seeds, args.feature_dim, args.n_max)

    with open(args.crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(args.labels, allow_pickle=True)
    log_sigma = z["log_sigma"]
    mask = z["mask"]
    has_cg = np.array([c is not None for c in crystals])
    eligible = np.where((mask > 0) & has_cg)[0]
    log.info("Eligible samples (labelled + CIF): %d", len(eligible))

    seed_rows = []
    for seed in range(args.seeds):
        torch.manual_seed(seed * 31 + 1); np.random.seed(seed * 31 + 1)
        folds = stratified_folds(log_sigma[eligible], seed=seed * 7)
        folds = [[int(eligible[i]) for i in f] for f in folds]

        fold_rows = []
        all_pred = np.full_like(log_sigma, np.nan, dtype=np.float32)
        for k in range(5):
            test = np.array(folds[k], dtype=np.int64)
            train_all = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
            rng = np.random.default_rng(100 + seed * 7 + k)
            val_size = max(8, int(0.1 * len(train_all)))
            val = rng.choice(train_all, size=val_size, replace=False)
            train = np.setdiff1d(train_all, val)

            m = KSECNet(
                feature_dim=args.feature_dim,
                num_blocks=args.num_blocks,
                n_max=args.n_max,
            ).to(args.device)
            m.set_target_shift(float(log_sigma[train].mean()))
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

            best_val = float("inf"); best_state = None
            t0 = time.time()
            for ep in range(args.epochs):
                m.train()
                rng2 = np.random.default_rng(ep + seed * 31 + k)
                order = rng2.permutation(train)
                for s in range(0, len(order), args.batch_size):
                    idx = order[s:s + args.batch_size]
                    atom_z, frac_pos, batch_idx = build_inputs(crystals, idx, args.device)
                    pred = m.forward_structure(atom_z, frac_pos, batch_idx, num_graphs=len(idx))
                    target = torch.from_numpy(log_sigma[idx]).float().to(args.device)
                    loss = ((pred - target) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                sched.step()
                m.eval()
                with torch.no_grad():
                    atom_z, frac_pos, batch_idx = build_inputs(crystals, val, args.device)
                    pred = m.forward_structure(atom_z, frac_pos, batch_idx, num_graphs=len(val))
                    val_target = torch.from_numpy(log_sigma[val]).float().to(args.device)
                    val_mae = float((pred - val_target).abs().mean())
                if val_mae < best_val - 1e-4:
                    best_val = val_mae
                    best_state = {kk: vv.clone() for kk, vv in m.state_dict().items()}
                if ep % 10 == 0 or ep == args.epochs - 1:
                    log.info("s%d-f%d ep %02d  val_mae=%.3f  best=%.3f", seed, k, ep, val_mae, best_val)

            if best_state: m.load_state_dict(best_state)
            m.eval()
            with torch.no_grad():
                atom_z, frac_pos, batch_idx = build_inputs(crystals, test, args.device)
                pred = m.forward_structure(atom_z, frac_pos, batch_idx, num_graphs=len(test)).cpu().numpy()
            all_pred[test] = pred
            target = log_sigma[test]
            reg = regression_metrics(target, pred)
            auc = classification_auc(target, pred)
            fold_rows.append(dict(
                seed=seed, fold=k, wall_s=time.time() - t0, best_val_mae=best_val,
                mae=reg["mae"], rmse=reg["rmse"], r2=reg["r2"], auc=auc,
            ))
            log.info("s%d-f%d  MAE=%.3f  R²=%.3f  AUC=%.3f", seed, k, reg["mae"], reg["r2"], auc)

        maes = [r["mae"] for r in fold_rows]
        r2s = [r["r2"] for r in fold_rows]
        aucs = [r["auc"] for r in fold_rows]
        seed_rows.append(dict(
            seed=seed,
            mae_mean=float(np.mean(maes)), mae_std=float(np.std(maes)),
            r2_mean=float(np.mean(r2s)),
            auc_mean=float(np.mean(aucs)),
            fold_rows=fold_rows,
        ))
        log.info("==== seed %d  MAE=%.3f±%.3f  R²=%.3f  AUC=%.3f ====",
                 seed, seed_rows[-1]["mae_mean"], seed_rows[-1]["mae_std"],
                 seed_rows[-1]["r2_mean"], seed_rows[-1]["auc_mean"])

    mae_all = [s["mae_mean"] for s in seed_rows]
    out = dict(
        model="k-SEC (k-Space Equivariant Convolutional Net)",
        config=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    feature_dim=args.feature_dim, num_blocks=args.num_blocks,
                    n_max=args.n_max, seeds=args.seeds),
        per_seed=seed_rows,
        aggregate=dict(
            mae_mean=float(np.mean(mae_all)), mae_std=float(np.std(mae_all)),
        ),
    )
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))
    log.info("FINAL  MAE=%.3f±%.3f", np.mean(mae_all), np.std(mae_all))


if __name__ == "__main__":
    main()
