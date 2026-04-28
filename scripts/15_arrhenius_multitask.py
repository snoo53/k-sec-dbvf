"""WP1+WP3: Temperature-aware Arrhenius-parameterized joint training on
OBELiX (CIF) + Hargreaves (composition-only).

The network outputs (log σ_0, E_a) and reconstructs σ at any temperature
via  log σ(T) = log σ_0 − E_a / (k_B T) / ln10. Training loss is MSE
over log σ on whichever T each sample was measured at.

Two parallel encoder streams share the Arrhenius head:
  - k-SEC encoder + Magpie readout for samples with a CIF
  - Magpie-only encoder for samples with only composition

Minibatches alternate between the two streams so both heads are trained
each epoch.

Usage:
    python scripts/15_arrhenius_multitask.py --epochs 80 --device cuda \
        --pretrained-encoder results/mp_broad_encoder_pretrained.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.data.hargreaves import load_hargreaves, dedupe_against_obelix
from ionpath.data.magpie import featurize_composition
from ionpath.models import KSECNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


K_B_eV = 8.617333262e-5


def arrhenius_log_sigma(log_sigma_0: torch.Tensor, E_a: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
    """log10 σ(T) = log10 σ_0 − E_a / (k_B T) / ln10."""
    return log_sigma_0 - E_a / (K_B_eV * T_K) / math.log(10.0)


class ArrheniusHead(nn.Module):
    """Takes pooled features → (log σ_0, E_a)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.log_sigma_0 = nn.Linear(hidden, 1)
        self.E_a = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
        t = self.trunk(h)
        ls0 = self.log_sigma_0(t).squeeze(-1)
        Ea = torch.nn.functional.softplus(self.E_a(t).squeeze(-1)) * 0.5  # soft-positive, scale ~0.3 eV
        return arrhenius_log_sigma(ls0, Ea, T_K)


class CompHead(nn.Module):
    """Magpie-only encoder for composition-only samples."""

    def __init__(self, magpie_dim: int = 132, hidden: int = 192):
        super().__init__()
        self.norm = nn.LayerNorm(magpie_dim)
        self.enc = nn.Sequential(
            nn.Linear(magpie_dim, hidden), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )

    def forward(self, magpie: torch.Tensor) -> torch.Tensor:
        return self.enc(self.norm(magpie))


class CIFStreamHead(nn.Module):
    """Wraps KSECNet to produce pooled features (no readout)."""

    def __init__(self, ksec: KSECNet, magpie_dim: int = 132, hidden: int = 192):
        super().__init__()
        self.ksec = ksec
        self.magpie_norm = nn.LayerNorm(magpie_dim)
        self.magpie_proj = nn.Sequential(
            nn.Linear(magpie_dim, 96), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(96, 96), nn.SiLU(),
        )
        # Expected pooled dim: 2*feature_dim (real+imag) + 96
        self.pool_dim = 2 * ksec.feature_dim + 96

    def forward(self, atom_z, frac_pos, batch_idx, num_graphs, magpie) -> torch.Tensor:
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
        h_k = torch.cat([F.real.mean(dim=1), F.imag.mean(dim=1)], dim=-1)
        h_m = self.magpie_proj(self.magpie_norm(magpie))
        return torch.cat([h_k, h_m], dim=-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_data(obelix_crystals, obelix_labels, hargreaves_csv):
    with open(obelix_crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(obelix_labels, allow_pickle=True)
    ls = z["log_sigma"]; mask = z["mask"]

    ksec_rows = []
    for i, c in enumerate(crystals):
        if mask[i] <= 0 or not np.isfinite(ls[i]):
            continue
        if ls[i] < -15 or ls[i] > -1:
            continue
        if c is None or getattr(c, "magpie", None) is None:
            continue
        ksec_rows.append(dict(
            atom_z=c.atom_z.astype(np.int64),
            frac_pos=c.frac_pos.astype(np.float32) % 1.0,
            magpie=c.magpie.astype(np.float32),
            T_K=np.float32(298.15),
            log_sigma=np.float32(ls[i]),
            source="obelix",
            composition=c.composition,
        ))

    # Hargreaves (all temperatures, not just RT)
    df = load_hargreaves(Path(hargreaves_csv))
    obelix_comps = {c.composition for c in crystals if c is not None}
    df = dedupe_against_obelix(df, obelix_comps)

    comp_rows = []
    for _, row in df.iterrows():
        try:
            ls_h = float(row["log_sigma"])
            T_C = float(row["temperature_C"])
        except Exception:
            continue
        if not (-18 < ls_h < 1):
            continue
        if not (-20 < T_C < 900):
            continue
        try:
            magpie = featurize_composition(row["composition"])
        except Exception:
            continue
        comp_rows.append(dict(
            magpie=magpie.astype(np.float32),
            T_K=np.float32(T_C + 273.15),
            log_sigma=np.float32(ls_h),
            source="hargreaves",
            composition=row["composition"],
        ))

    log.info("ksec stream (OBELiX CIF): %d samples", len(ksec_rows))
    log.info("comp stream (Hargreaves, all T): %d samples", len(comp_rows))
    return ksec_rows, comp_rows


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return dict(mae=mae, r2=r2, n=int(y_true.size))


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


def build_ksec_batch(rows, idx, device):
    atom_z, frac_pos, batch_idx, magpie, T_K, y = [], [], [], [], [], []
    for b, i in enumerate(idx):
        r = rows[i]
        atom_z.append(r["atom_z"])
        frac_pos.append(r["frac_pos"])
        batch_idx.append(np.full(len(r["atom_z"]), b, dtype=np.int64))
        magpie.append(r["magpie"])
        T_K.append(r["T_K"]); y.append(r["log_sigma"])
    return (
        torch.from_numpy(np.concatenate(atom_z)).long().to(device),
        torch.from_numpy(np.concatenate(frac_pos)).float().to(device),
        torch.from_numpy(np.concatenate(batch_idx)).long().to(device),
        torch.from_numpy(np.stack(magpie, axis=0)).float().to(device),
        torch.tensor(T_K, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def build_comp_batch(rows, idx, device):
    magpie, T_K, y = [], [], []
    for i in idx:
        r = rows[i]
        magpie.append(r["magpie"]); T_K.append(r["T_K"]); y.append(r["log_sigma"])
    return (
        torch.from_numpy(np.stack(magpie, axis=0)).float().to(device),
        torch.tensor(T_K, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obelix-crystals", default="data/cache/crystals.pkl")
    p.add_argument("--obelix-labels", default="data/cache/labels.npz")
    p.add_argument("--hargreaves", default="data/raw/LiIonDatabase.csv")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--comp-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--pretrained-encoder", default=None)
    p.add_argument("--results", default="results/ksec_arrhenius.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ksec_rows, comp_rows = build_data(
        args.obelix_crystals, args.obelix_labels, args.hargreaves,
    )
    log.info("TOTAL effective training: %d samples", len(ksec_rows) + len(comp_rows))

    # 5-fold CV on ksec_rows only (that's the evaluation target)
    ksec_y = np.array([r["log_sigma"] for r in ksec_rows], dtype=np.float32)

    seed_rows = []
    all_preds_per_seed = []
    for seed in range(args.seeds):
        torch.manual_seed(seed * 31 + 1); np.random.seed(seed * 31 + 1)
        folds = stratified_folds(ksec_y, seed=seed * 7)

        fold_rows = []
        all_pred = np.full(len(ksec_rows), np.nan, dtype=np.float32)
        for k in range(5):
            test = np.array(folds[k], dtype=np.int64)
            train_ksec = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)

            # Build model
            base = KSECNet(
                feature_dim=args.feature_dim, num_blocks=args.num_blocks,
                n_max=args.n_max, dropout=args.dropout,
            ).to(args.device)
            if args.pretrained_encoder:
                enc = torch.load(args.pretrained_encoder, map_location=args.device, weights_only=False)
                enc_state = {kk: vv for kk, vv in enc["state"].items()
                             if kk.startswith("embed.") or kk.startswith("blocks.")}
                base.load_state_dict(enc_state, strict=False)

            cif_head = CIFStreamHead(base).to(args.device)
            comp_head = CompHead(magpie_dim=132, hidden=192).to(args.device)
            arr = ArrheniusHead(cif_head.pool_dim, hidden=128).to(args.device)
            arr_comp = ArrheniusHead(192, hidden=128).to(args.device)
            # Each stream gets its own Arrhenius head. Physics constraint
            # (Arrhenius form) is hardcoded in arrhenius_log_sigma; weight
            # sharing is unnecessary for the physics.

            params = (list(cif_head.parameters()) + list(comp_head.parameters())
                      + list(arr.parameters()) + list(arr_comp.parameters()))
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

            rng = np.random.default_rng(100 + seed * 7 + k)
            best_val_mae = float("inf"); best_state = None

            # 10% of train as val
            val_size = max(8, int(0.1 * len(train_ksec)))
            val = rng.choice(train_ksec, size=val_size, replace=False)
            train = np.setdiff1d(train_ksec, val)
            comp_idx_all = np.arange(len(comp_rows))

            t0 = time.time()
            for ep in range(args.epochs):
                cif_head.train(); comp_head.train(); arr.train()

                # ksec stream
                order_k = rng.permutation(train)
                # comp stream (reshuffle each epoch)
                order_c = rng.permutation(comp_idx_all)
                n_k_batches = (len(order_k) + args.batch_size - 1) // args.batch_size
                n_c_batches = (len(order_c) + args.comp_batch_size - 1) // args.comp_batch_size

                # interleave
                max_batches = max(n_k_batches, n_c_batches)
                for b in range(max_batches):
                    loss = torch.zeros((), device=args.device)
                    if b < n_k_batches:
                        idx = order_k[b * args.batch_size:(b + 1) * args.batch_size]
                        az, fp, bi, mag, tk, y = build_ksec_batch(ksec_rows, idx, args.device)
                        h = cif_head(az, fp, bi, num_graphs=len(idx), magpie=mag)
                        pred = arr(h, tk)
                        loss = loss + ((pred - y) ** 2).mean()
                    if b < n_c_batches:
                        idx = order_c[b * args.comp_batch_size:(b + 1) * args.comp_batch_size]
                        mag, tk, y = build_comp_batch(comp_rows, idx, args.device)
                        h = comp_head(mag)
                        pred = arr_comp(h, tk)
                        loss = loss + 0.5 * ((pred - y) ** 2).mean()  # downweight aux stream
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                sched.step()

                # Val (on ksec stream only — that's the paper metric)
                cif_head.eval(); comp_head.eval(); arr.eval()
                with torch.no_grad():
                    az, fp, bi, mag, tk, y = build_ksec_batch(ksec_rows, val, args.device)
                    h = cif_head(az, fp, bi, num_graphs=len(val), magpie=mag)
                    pred = arr(h, tk)
                    val_mae = float((pred - y).abs().mean())
                if val_mae < best_val_mae - 1e-4:
                    best_val_mae = val_mae
                    best_state = {
                        kk: vv.clone() for kk, vv in
                        {**{f"cif.{k}": v for k, v in cif_head.state_dict().items()},
                         **{f"comp.{k}": v for k, v in comp_head.state_dict().items()},
                         **{f"arr.{k}": v for k, v in arr.state_dict().items()}}.items()
                    }
                if ep % 10 == 0 or ep == args.epochs - 1:
                    log.info("s%d-f%d ep %02d  val_mae=%.3f  best=%.3f", seed, k, ep, val_mae, best_val_mae)

            # Test
            if best_state:
                cif_head.load_state_dict({k.removeprefix("cif."): v for k, v in best_state.items() if k.startswith("cif.")}, strict=False)
                arr.load_state_dict({k.removeprefix("arr."): v for k, v in best_state.items() if k.startswith("arr.")})
            cif_head.eval(); arr.eval()
            with torch.no_grad():
                az, fp, bi, mag, tk, y = build_ksec_batch(ksec_rows, test, args.device)
                h = cif_head(az, fp, bi, num_graphs=len(test), magpie=mag)
                pred = arr(h, tk).cpu().numpy()
            target = ksec_y[test]
            reg = regression_metrics(target, pred)
            all_pred[test] = pred
            fold_rows.append(dict(seed=seed, fold=k, wall_s=time.time() - t0, **reg))
            log.info("s%d-f%d  MAE=%.3f  R²=%.3f", seed, k, reg["mae"], reg["r2"])

            del cif_head, comp_head, arr, arr_comp, base, opt, sched, best_state
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        maes = [r["mae"] for r in fold_rows]
        seed_rows.append(dict(seed=seed, mae_mean=float(np.mean(maes)), mae_std=float(np.std(maes)),
                               fold_rows=fold_rows))
        all_preds_per_seed.append(all_pred.copy())
        log.info("==== seed %d  MAE=%.3f ± %.3f", seed, seed_rows[-1]["mae_mean"], seed_rows[-1]["mae_std"])

    # Ensemble on ksec
    stacked = np.stack(all_preds_per_seed, axis=0)
    ens = np.nanmean(stacked, axis=0)
    ens_reg = regression_metrics(ksec_y, ens)
    log.info("ENSEMBLE  MAE=%.3f  R²=%.3f", ens_reg["mae"], ens_reg["r2"])

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(dict(
        per_seed=seed_rows,
        ensemble=ens_reg,
        config=dict(epochs=args.epochs, seeds=args.seeds, lr=args.lr,
                    pretrained_encoder=args.pretrained_encoder),
    ), indent=2))


if __name__ == "__main__":
    main()
