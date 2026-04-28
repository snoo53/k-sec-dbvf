"""MC-dropout uncertainty quantification for k-SEC.

Train once per fold, then at test time run N forward passes with dropout
active. Report (mean, std) per sample and calibration metrics.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crystals", default="data/cache/crystals.pkl")
    p.add_argument("--labels", default="data/cache/labels.npz")
    p.add_argument("--results", default="results/mc_dropout.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-mc-samples", type=int, default=30)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    with open(args.crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(args.labels, allow_pickle=True)
    log_sigma = z["log_sigma"]
    mask_arr = z["mask"]

    has_cg = np.array([c is not None for c in crystals])
    eligible = np.where((mask_arr > 0) & has_cg)[0]
    log.info("Eligible: %d", len(eligible))

    torch.manual_seed(0); np.random.seed(0)
    folds = stratified_folds(log_sigma[eligible], seed=0)
    folds = [[int(eligible[i]) for i in f] for f in folds]

    all_mean, all_std, all_target = [], [], []
    for k in range(5):
        test = np.array(folds[k], dtype=np.int64)
        train_all = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
        rng = np.random.default_rng(10 + k)
        val = rng.choice(train_all, size=max(8, int(0.1 * len(train_all))), replace=False)
        train = np.setdiff1d(train_all, val)

        m = KSECNet(feature_dim=96, num_blocks=3, n_max=2, dropout=args.dropout).to(args.device)
        m.set_target_shift(float(log_sigma[train].mean()))
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_val = float("inf"); best_state = None
        t0 = time.time()
        for ep in range(args.epochs):
            m.train()
            order = rng.permutation(train)
            for s in range(0, len(order), args.batch_size):
                idx = order[s:s + args.batch_size]
                az, fp, bi = build_inputs(crystals, idx, args.device)
                pred = m.forward_structure(az, fp, bi, num_graphs=len(idx))
                target = torch.from_numpy(log_sigma[idx]).float().to(args.device)
                loss = ((pred - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                az, fp, bi = build_inputs(crystals, val, args.device)
                pred = m.forward_structure(az, fp, bi, num_graphs=len(val))
                val_mae = float((pred - torch.from_numpy(log_sigma[val]).float().to(args.device)).abs().mean())
            if val_mae < best_val - 1e-4:
                best_val = val_mae
                best_state = {kk: vv.clone() for kk, vv in m.state_dict().items()}
        if best_state: m.load_state_dict(best_state)

        # MC-dropout inference
        az, fp, bi = build_inputs(crystals, test, args.device)
        with torch.no_grad():
            mean, std = m.forward_mc_dropout(az, fp, bi, num_graphs=len(test),
                                              n_samples=args.n_mc_samples)
        log.info("fold %d  MAE=%.3f  mean σ_MC=%.3f  (%.0fs)",
                 k, float((mean.cpu() - torch.from_numpy(log_sigma[test])).abs().mean()),
                 float(std.mean()), time.time() - t0)
        all_mean.append(mean.cpu().numpy())
        all_std.append(std.cpu().numpy())
        all_target.append(log_sigma[test])

    all_mean = np.concatenate(all_mean)
    all_std = np.concatenate(all_std)
    all_target = np.concatenate(all_target)

    # Coverage at various confidence levels
    coverage = {}
    for z in (1.0, 1.96):
        lo = all_mean - z * all_std
        hi = all_mean + z * all_std
        coverage[f"{z}σ"] = float(np.mean((all_target >= lo) & (all_target <= hi)))

    mae = float(np.mean(np.abs(all_mean - all_target)))
    mean_sigma = float(all_std.mean())
    log.info("AGG  MAE=%.3f  mean σ_MC=%.3f  coverage@1σ=%.3f  coverage@1.96σ=%.3f",
             mae, mean_sigma, coverage["1.0σ"], coverage["1.96σ"])

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(dict(
        config=dict(epochs=args.epochs, dropout=args.dropout,
                    n_mc_samples=args.n_mc_samples),
        aggregate_mae=mae,
        mean_uncertainty=mean_sigma,
        coverage=coverage,
    ), indent=2))


if __name__ == "__main__":
    main()
