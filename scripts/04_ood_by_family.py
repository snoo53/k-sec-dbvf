"""Out-of-distribution evaluation: train k-SEC on all structural families
except one, test on the held-out family. Repeat for each family with
≥ 5 samples.

JMST reviewers ask for this: does the model generalize to unseen chemistry?
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


def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(mae=float("nan"), r2=float("nan"), n=0)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return dict(mae=mae, r2=r2, n=int(yt.size))


def train_one_holdout(
    crystals, log_sigma, mask, train_idx, val_idx, test_idx,
    epochs, lr, batch_size, device, feature_dim=96, num_blocks=3,
    n_max=2, dropout=0.15, seed=42,
):
    torch.manual_seed(seed)
    m = KSECNet(feature_dim=feature_dim, num_blocks=num_blocks,
                n_max=n_max, dropout=dropout).to(device)
    m.set_target_shift(float(log_sigma[train_idx].mean()))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf"); best_state = None
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        m.train()
        order = rng.permutation(train_idx)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            az, fp, bi = build_inputs(crystals, idx, device)
            pred = m.forward_structure(az, fp, bi, num_graphs=len(idx))
            target = torch.from_numpy(log_sigma[idx]).float().to(device)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            az, fp, bi = build_inputs(crystals, val_idx, device)
            pred = m.forward_structure(az, fp, bi, num_graphs=len(val_idx))
            val_mae = float((pred - torch.from_numpy(log_sigma[val_idx]).float().to(device)).abs().mean())
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    if best_state: m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        az, fp, bi = build_inputs(crystals, test_idx, device)
        pred = m.forward_structure(az, fp, bi, num_graphs=len(test_idx)).cpu().numpy()
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crystals", default="data/cache/crystals.pkl")
    p.add_argument("--labels", default="data/cache/labels.npz")
    p.add_argument("--results", default="results/ood_by_family.json")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-family-size", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    with open(args.crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(args.labels, allow_pickle=True)
    log_sigma = z["log_sigma"]
    mask_arr = z["mask"]
    families = z["families"]

    has_cg = np.array([c is not None for c in crystals])
    eligible = np.where((mask_arr > 0) & has_cg)[0]
    fam_eligible = families[eligible]
    log.info("Eligible: %d samples", len(eligible))

    unique_families, counts = np.unique(fam_eligible, return_counts=True)
    families_to_hold = [f for f, c in zip(unique_families, counts) if c >= args.min_family_size]
    log.info("Families ≥%d samples: %d (%s)", args.min_family_size, len(families_to_hold), families_to_hold)

    rows = []
    for fam in families_to_hold:
        test_global = eligible[fam_eligible == fam]
        train_pool = np.setdiff1d(eligible, test_global)
        rng = np.random.default_rng(42)
        val = rng.choice(train_pool, size=max(8, int(0.1 * len(train_pool))), replace=False)
        train = np.setdiff1d(train_pool, val)

        t0 = time.time()
        pred = train_one_holdout(
            crystals, log_sigma, mask_arr, train, val, test_global,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            device=args.device,
        )
        metrics = regression_metrics(log_sigma[test_global], pred)
        rows.append(dict(family=str(fam), n_test=int(len(test_global)),
                          **metrics, wall_s=round(time.time() - t0, 1)))
        log.info("family=%-15s  n=%3d  MAE=%.3f  R²=%.3f  (%.0fs)",
                 fam, len(test_global), metrics["mae"], metrics["r2"], time.time() - t0)
        # Save incremental results so a kill doesn't lose everything
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results).write_text(json.dumps(dict(
            config=dict(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size),
            per_family=rows,
            mean_ood_mae=float(np.mean([r["mae"] for r in rows])),
            complete=False,
        ), indent=2))

    mean_mae = float(np.mean([r["mae"] for r in rows]))
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(dict(
        config=dict(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size),
        per_family=rows,
        mean_ood_mae=mean_mae,
    ), indent=2))
    log.info("Mean OOD MAE across %d families: %.3f", len(rows), mean_mae)


if __name__ == "__main__":
    main()
