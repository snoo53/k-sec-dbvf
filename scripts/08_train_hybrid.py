"""Hybrid k-SEC: k-space features + Magpie composition features at readout,
with optional heteroscedastic loss and multi-seed ensembling.

Target: beat LightGBM+Magpie (MAE 1.099 on log10 σ, OBELiX 285-sample CIF subset).

Usage:
    python scripts/08_train_hybrid.py --epochs 60 --seeds 5 --device cuda \
        --hetero --results results/ksec_hybrid.json
"""

from __future__ import annotations

import argparse
import gc
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
from ionpath.data.geometric import GEOMETRIC_FEATURE_DIM as GEOMETRIC_DIM

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


def build_inputs(crystals, idx, device, use_magpie=True, use_lattice=False, use_geometric=False, use_mace=False, use_cell=False):
    atom_z, frac_pos, batch_idx, magpie, lattice, geometric, mace, cells = [], [], [], [], [], [], [], []
    for b, gi in enumerate(idx):
        cg = crystals[gi]
        atom_z.append(cg.atom_z)
        frac_pos.append(cg.frac_pos.astype(np.float32))
        batch_idx.append(np.full(len(cg.atom_z), b, dtype=np.int64))
        if use_cell:
            cells.append(cg.cell.astype(np.float32))
        if use_magpie:
            magpie.append(cg.magpie.astype(np.float32) if cg.magpie is not None
                          else np.zeros(132, dtype=np.float32))
        if use_lattice:
            lf = getattr(cg, "lattice_feats", None)
            lattice.append(lf.astype(np.float32) if lf is not None
                           else np.zeros(8, dtype=np.float32))
        if use_geometric:
            gf = getattr(cg, "geometric", None)
            geometric.append(gf.astype(np.float32) if gf is not None
                             else np.zeros(GEOMETRIC_DIM, dtype=np.float32))
        if use_mace:
            mf = getattr(cg, "mace", None)
            mace.append(mf.astype(np.float32) if mf is not None
                        else np.array([-7.5, 0.0, 0.0, 0.0], dtype=np.float32))  # imputed mean + invalid mask
    out = (
        torch.from_numpy(np.concatenate(atom_z)).long().to(device),
        torch.from_numpy(np.concatenate(frac_pos)).float().to(device),
        torch.from_numpy(np.concatenate(batch_idx)).long().to(device),
    )
    tab = torch.from_numpy(np.stack(magpie, axis=0)).float().to(device) if use_magpie else None
    lat = torch.from_numpy(np.stack(lattice, axis=0)).float().to(device) if use_lattice else None
    geo = torch.from_numpy(np.stack(geometric, axis=0)).float().to(device) if use_geometric else None
    mc = torch.from_numpy(np.stack(mace, axis=0)).float().to(device) if use_mace else None
    cl = torch.from_numpy(np.stack(cells, axis=0)).float().to(device) if use_cell else None
    return out + (tab, lat, geo, mc, cl)


def regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(mae=float("nan"), rmse=float("nan"), r2=float("nan"), n=0)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2, n=int(yt.size))


def classification_auc(y_true_log, scores, threshold=-4.0):
    y = (y_true_log >= threshold).astype(np.float64)
    order = np.argsort(scores)
    yt = y[order]
    n_pos = yt.sum(); n_neg = yt.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, yt.size + 1, dtype=np.float64)
    return float((np.sum(ranks * yt) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class HybridHead(torch.nn.Module):
    """Wraps KSECNet to produce (mu, log_var) for heteroscedastic loss."""

    def __init__(self, ksec: KSECNet):
        super().__init__()
        self.ksec = ksec
        # Tap off an extra head: use the final readout's hidden size
        h = ksec.readout[0].in_features
        self.log_var_head = torch.nn.Sequential(
            torch.nn.Linear(h, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, atom_z, frac_pos, batch_idx, num_graphs, tabular, lattice_feats=None, geometric=None):
        # Call parent forward but also capture pre-readout hidden
        import math
        z = self.ksec.embed(atom_z)
        z_complex = torch.complex(z, torch.zeros_like(z))
        phases = -2.0 * math.pi * (frac_pos @ self.ksec.k_points.T)
        exp_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        contrib = z_complex.unsqueeze(1) * exp_phases.unsqueeze(-1)
        F = torch.zeros(num_graphs, self.ksec.K, z.shape[-1],
                        dtype=contrib.dtype, device=contrib.device)
        F.index_add_(0, batch_idx, contrib)
        counts = torch.zeros(num_graphs, device=z.device)
        counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        F = F / counts.clamp(min=1.0).view(-1, 1, 1)
        for block in self.ksec.blocks:
            F = block(F, self.ksec.k_mags, self.ksec.kubics)
        h = torch.cat([F.real.mean(dim=1), F.imag.mean(dim=1)], dim=-1)
        if self.ksec.tabular_proj is not None:
            t = self.ksec.tabular_norm(tabular)
            t = self.ksec.tabular_proj(t)
            h = torch.cat([h, t], dim=-1)
        if self.ksec.lattice_proj is not None:
            l = self.ksec.lattice_norm(lattice_feats)
            l = self.ksec.lattice_proj(l)
            h = torch.cat([h, l], dim=-1)
        if self.ksec.geometric_proj is not None:
            g = self.ksec.geometric_norm(geometric)
            g = self.ksec.geometric_proj(g)
            h = torch.cat([h, g], dim=-1)
        mu = self.ksec.readout(h).squeeze(-1) + self.ksec.log_sigma_shift
        log_var = self.log_var_head(h).squeeze(-1)
        # clamp log-var to a sane range
        log_var = log_var.clamp(-6.0, 4.0)
        return mu, log_var


def heteroscedastic_nll(mu, log_var, target):
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (mu - target) ** 2 * inv_var).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crystals", default="data/cache/crystals.pkl")
    p.add_argument("--labels", default="data/cache/labels.npz")
    p.add_argument("--results", default="results/ksec_hybrid.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--hetero", action="store_true",
                   help="Use heteroscedastic NLL instead of MSE")
    p.add_argument("--no-magpie", action="store_true",
                   help="Disable Magpie tabular features (ablation)")
    p.add_argument("--use-lattice", action="store_true",
                   help="Enable lattice-matrix features at readout (8-dim)")
    p.add_argument("--use-geometric", action="store_true",
                   help=f"Enable BV/geometric features at readout ({GEOMETRIC_DIM}-dim)")
    p.add_argument("--use-mace", action="store_true",
                   help="Enable MACE-MP-0 auxiliary features (4-dim: E/atom, E/Li, F_rms, valid)")
    p.add_argument("--use-bv-field", action="store_true",
                   help="Enable Differentiable Bond-Valence Field (NOVEL: learnable r0/b parameters)")
    p.add_argument("--use-path-bv-field", action="store_true",
                   help="Enable Site-resolved DBVF with path-integration (NOVEL: differentiable migration-saddle)")
    p.add_argument("--dual-stream", action="store_true",
                   help="BatteryNet: enable real-space MPNN + cross-attention bridge to k-SEC (NOVEL)")
    p.add_argument("--min-log-sigma", type=float, default=-15.0,
                   help="Drop samples with log_sigma below this (physical detection-limit filter)")
    p.add_argument("--require-geometric-valid", action="store_true",
                   help="Drop samples whose geometric featurization failed (feat_valid != 1)")
    p.add_argument("--pretrained-magpie", default=None,
                   help="Path to pretrained Magpie head weights (e.g., results/magpie_pretrained.pt)")
    p.add_argument("--pretrained-encoder", default=None,
                   help="Path to MP-pretrained encoder weights (load embed + blocks only; readout stays fresh)")
    p.add_argument("--freeze-magpie", action="store_true",
                   help="Freeze tabular_norm + tabular_proj during fine-tuning (requires --pretrained-magpie)")
    p.add_argument("--save-oof", default=None,
                   help="Path to .npz to save out-of-fold ensemble predictions (for stacking)")
    p.add_argument("--save-ckpt", default=None,
                   help="Path prefix to save per-seed final checkpoints (e.g. results/final_ckpt_s)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Subsample each fold's training set to this fraction (0,1]. "
                        "Test fold is unchanged so MAE remains comparable across n. "
                        "Used for learning-curve experiments.")
    args = p.parse_args()

    use_magpie = not args.no_magpie
    use_lattice = args.use_lattice
    log.info("device=%s  epochs=%d  seeds=%d  hetero=%s  magpie=%s  lattice=%s",
             args.device, args.epochs, args.seeds, args.hetero, use_magpie, use_lattice)

    with open(args.crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(args.labels, allow_pickle=True)
    log_sigma = z["log_sigma"]
    mask = z["mask"]
    has_cg = np.array([c is not None for c in crystals])
    has_magpie = np.array([c is not None and getattr(c, "magpie", None) is not None
                           for c in crystals])
    required = has_cg & (has_magpie if use_magpie else has_cg)

    # Physical + quality filtration
    n0 = int(((mask > 0) & required).sum())
    log_sigma_ok = log_sigma > args.min_log_sigma
    required = required & log_sigma_ok
    n1 = int(((mask > 0) & required).sum())

    if args.require_geometric_valid:
        geo_valid = np.array([
            c is not None and getattr(c, "geometric", None) is not None
            and float(c.geometric[-1]) == 1.0 for c in crystals])
        required = required & geo_valid
    n2 = int(((mask > 0) & required).sum())

    eligible = np.where((mask > 0) & required)[0]
    log.info("Eligible: %d  (log_sigma>%.1f filter dropped %d, geometric-valid filter dropped %d)",
             len(eligible), args.min_log_sigma, n0 - n1, n1 - n2)

    use_geometric = args.use_geometric
    use_mace = args.use_mace
    use_bv = args.use_bv_field
    use_path_bv = args.use_path_bv_field
    use_dual = args.dual_stream
    tab_dim = 132 if use_magpie else 0
    lat_dim = 8 if use_lattice else 0
    geo_dim = GEOMETRIC_DIM if use_geometric else 0
    mace_dim = 4 if use_mace else 0
    seed_rows = []
    all_preds_per_seed = []
    for seed in range(args.seeds):
        torch.manual_seed(seed * 31 + 1); np.random.seed(seed * 31 + 1)
        folds = stratified_folds(log_sigma[eligible], seed=seed * 7)
        folds = [[int(eligible[i]) for i in f] for f in folds]
        all_pred = np.full_like(log_sigma, np.nan, dtype=np.float32)

        fold_rows = []
        for k in range(5):
            test = np.array(folds[k], dtype=np.int64)
            train_all = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
            rng = np.random.default_rng(100 + seed * 7 + k)
            val_size = max(8, int(0.1 * len(train_all)))
            val = rng.choice(train_all, size=val_size, replace=False)
            train = np.setdiff1d(train_all, val)
            # Learning-curve subsampling: keep only a fraction of the training set.
            # Test fold is preserved so test-MAE is directly comparable across n.
            if args.train_fraction < 1.0:
                target_n = max(8, int(round(len(train) * args.train_fraction)))
                rng_sub = np.random.default_rng(200 + seed * 7 + k)
                # Stratify the subsample on log_sigma quintiles so small training
                # sets keep coverage of the target distribution.
                y_train = log_sigma[train]
                qs = np.quantile(y_train, np.linspace(0, 1, 6))
                qs[0] -= 1e-6; qs[-1] += 1e-6
                bins = np.clip(np.digitize(y_train, qs) - 1, 0, 4)
                kept = []
                per_bin = max(1, target_n // 5)
                for b in range(5):
                    pool = np.where(bins == b)[0]
                    if len(pool) == 0:
                        continue
                    take = min(len(pool), per_bin)
                    kept.extend(rng_sub.choice(pool, size=take, replace=False).tolist())
                # Top up to target_n if rounding left us short
                if len(kept) < target_n:
                    remaining = list(set(range(len(train))) - set(kept))
                    extra = rng_sub.choice(remaining, size=min(target_n - len(kept), len(remaining)), replace=False)
                    kept.extend(extra.tolist())
                train = train[np.array(sorted(kept), dtype=np.int64)]

            base = KSECNet(
                feature_dim=args.feature_dim, num_blocks=args.num_blocks,
                n_max=args.n_max, dropout=args.dropout,
                tabular_dim=tab_dim, lattice_dim=lat_dim,
                geometric_dim=geo_dim, mace_dim=mace_dim,
                bv_field=use_bv, path_bv_field=use_path_bv,
                dual_stream=use_dual,
            ).to(args.device)
            base.set_target_shift(float(log_sigma[train].mean()))

            if args.pretrained_magpie:
                pre = torch.load(args.pretrained_magpie, map_location=args.device, weights_only=False)
                missing, unexpected = base.load_state_dict(pre["state"], strict=False)
                if seed == 0 and k == 0:
                    log.info("loaded Magpie pretrain  val_mae=%.3f  missing=%d unexpected=%d",
                             pre.get("val_mae", float("nan")), len(missing), len(unexpected))
            if args.pretrained_encoder:
                enc = torch.load(args.pretrained_encoder, map_location=args.device, weights_only=False)
                enc_state = {k: v for k, v in enc["state"].items()
                             if k.startswith("embed.") or k.startswith("blocks.")}
                missing, unexpected = base.load_state_dict(enc_state, strict=False)
                loaded = len(enc_state)
                if seed == 0 and k == 0:
                    log.info("loaded MP encoder (%d tensors: embed+blocks)  val_score=%.3f  missing=%d unexpected=%d",
                             loaded, enc.get("val_score", float("nan")), len(missing), len(unexpected))
            if args.freeze_magpie:
                if not args.pretrained_magpie:
                    raise ValueError("--freeze-magpie requires --pretrained-magpie")
                n_frozen = 0
                for pn, p_ in base.named_parameters():
                    if pn.startswith("tabular_norm.") or pn.startswith("tabular_proj."):
                        p_.requires_grad_(False); n_frozen += 1
                if seed == 0 and k == 0:
                    log.info("froze %d Magpie parameters", n_frozen)
            m = HybridHead(base).to(args.device) if args.hetero else base

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
                    az, fp, bi, tab, lat, geo, mc, cl = build_inputs(crystals, idx, args.device, use_magpie, use_lattice, use_geometric, use_mace, use_bv or use_path_bv or use_dual)
                    target = torch.from_numpy(log_sigma[idx]).float().to(args.device)
                    if args.hetero:
                        mu, lv = m(az, fp, bi, num_graphs=len(idx), tabular=tab, lattice_feats=lat, geometric=geo, mace=mc, cell=cl)
                        loss = heteroscedastic_nll(mu, lv, target)
                    else:
                        pred = m.forward_structure(az, fp, bi, num_graphs=len(idx), tabular=tab, lattice_feats=lat, geometric=geo, mace=mc, cell=cl)
                        loss = ((pred - target) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                sched.step()
                m.eval()
                with torch.no_grad():
                    az, fp, bi, tab, lat, geo, mc, cl = build_inputs(crystals, val, args.device, use_magpie, use_lattice, use_geometric, use_mace, use_bv or use_path_bv or use_dual)
                    if args.hetero:
                        mu, _ = m(az, fp, bi, num_graphs=len(val), tabular=tab, lattice_feats=lat, geometric=geo, mace=mc, cell=cl)
                        pred = mu
                    else:
                        pred = m.forward_structure(az, fp, bi, num_graphs=len(val), tabular=tab, lattice_feats=lat, geometric=geo, mace=mc, cell=cl)
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
                az, fp, bi, tab, lat, geo, mc, cl = build_inputs(crystals, test, args.device, use_magpie, use_lattice, use_geometric, use_mace, use_bv or use_path_bv or use_dual)
                if args.hetero:
                    mu, _ = m(az, fp, bi, num_graphs=len(test), tabular=tab, lattice_feats=lat, geometric=geo, mace=mc, cell=cl)
                    pred = mu.cpu().numpy()
                else:
                    pred = m.forward_structure(az, fp, bi, num_graphs=len(test), tabular=tab, lattice_feats=lat, geometric=geo, mace=mc, cell=cl).cpu().numpy()
            all_pred[test] = pred
            target = log_sigma[test]
            reg = regression_metrics(target, pred)
            auc = classification_auc(target, pred)
            fold_rows.append(dict(seed=seed, fold=k, wall_s=time.time() - t0,
                                   best_val_mae=best_val, **reg, auc=auc))
            log.info("s%d-f%d  MAE=%.3f  R²=%.3f  AUC=%.3f", seed, k, reg["mae"], reg["r2"], auc)

            # Capture last-fold state for checkpoint save (BEFORE the del cleanup).
            # Only kept on the last fold of the seed so we don't waste memory.
            if args.save_ckpt and k == 4:
                last_fold_state = {kk: vv.clone().cpu() for kk, vv in m.state_dict().items()}

            # Explicit cleanup to prevent RAM bloat across folds
            del m, base, opt, sched, best_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        all_preds_per_seed.append(all_pred.copy())
        log.info("==== seed %d  MAE=%.3f±%.3f  R²=%.3f  AUC=%.3f ====",
                 seed, seed_rows[-1]["mae_mean"], seed_rows[-1]["mae_std"],
                 seed_rows[-1]["r2_mean"], seed_rows[-1]["auc_mean"])

        # Save per-seed final model checkpoint (for virtual screening)
        if args.save_ckpt and 'last_fold_state' in dir():
            ckpt_path = Path(f"{args.save_ckpt}{seed}.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state=last_fold_state,
                            config=dict(feature_dim=args.feature_dim,
                                         num_blocks=args.num_blocks,
                                         n_max=args.n_max,
                                         tabular_dim=tab_dim,
                                         lattice_dim=lat_dim,
                                         geometric_dim=geo_dim)),
                       ckpt_path)
            log.info("  saved checkpoint seed=%d → %s", seed, ckpt_path)
            del last_fold_state

        # Incremental save after each seed so kills don't lose progress
        if args.save_oof:
            Path(args.save_oof).parent.mkdir(parents=True, exist_ok=True)
            stacked_partial = np.stack(all_preds_per_seed, axis=0)
            ens_partial = np.nanmean(stacked_partial, axis=0)
            np.savez(args.save_oof,
                     per_seed=stacked_partial,
                     ensemble=ens_partial,
                     log_sigma=log_sigma,
                     mask=mask.astype(np.float32),
                     n_seeds_completed=len(all_preds_per_seed))
            log.info("checkpoint: saved OOF for %d seed(s) to %s",
                     len(all_preds_per_seed), args.save_oof)
        # Also save partial results json after each seed
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        partial_mae_all = [s["mae_mean"] for s in seed_rows]
        partial_out = dict(
            model="k-SEC Hybrid (partial)",
            config=dict(epochs=args.epochs, seeds_completed=len(seed_rows),
                        seeds_planned=args.seeds),
            per_seed=seed_rows,
            aggregate=dict(
                mae_mean=float(np.mean(partial_mae_all)),
                mae_std=float(np.std(partial_mae_all)),
            ),
        )
        Path(args.results).write_text(json.dumps(partial_out, indent=2))
        gc.collect()

    # Seed-ensembled predictions: average across seeds per sample
    stacked = np.stack(all_preds_per_seed, axis=0)
    ens_pred = np.nanmean(stacked, axis=0)
    ens_mask = np.isfinite(ens_pred) & (mask > 0)
    ens_reg = regression_metrics(log_sigma[ens_mask], ens_pred[ens_mask])
    ens_auc = classification_auc(log_sigma[ens_mask], ens_pred[ens_mask])

    mae_all = [s["mae_mean"] for s in seed_rows]
    out = dict(
        model="k-SEC Hybrid (Kubic + cross-shell + Magpie readout)",
        config=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    feature_dim=args.feature_dim, num_blocks=args.num_blocks,
                    n_max=args.n_max, seeds=args.seeds,
                    hetero=args.hetero, use_magpie=use_magpie,
                    dropout=args.dropout,
                    train_fraction=args.train_fraction,
                    n_train_per_fold=int(round(len(eligible) * 4 / 5 * args.train_fraction))),
        per_seed=seed_rows,
        aggregate=dict(
            mae_mean=float(np.mean(mae_all)), mae_std=float(np.std(mae_all)),
        ),
        seed_ensemble=dict(
            mae=ens_reg["mae"], rmse=ens_reg["rmse"],
            r2=ens_reg["r2"], auc=ens_auc,
        ),
    )
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    if args.save_oof:
        Path(args.save_oof).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_oof,
                 per_seed=np.stack(all_preds_per_seed, axis=0),  # (n_seeds, N)
                 ensemble=ens_pred,                               # (N,)
                 log_sigma=log_sigma,
                 mask=mask.astype(np.float32))
        log.info("saved OOF preds to %s", args.save_oof)

    log.info("PER-SEED MAE=%.3f±%.3f   ENSEMBLE MAE=%.3f  R²=%.3f  AUC=%.3f",
             np.mean(mae_all), np.std(mae_all),
             ens_reg["mae"], ens_reg["r2"], ens_auc)


if __name__ == "__main__":
    main()
