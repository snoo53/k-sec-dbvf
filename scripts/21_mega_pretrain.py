"""Mega-pretrain k-SEC encoder on the unified 218k+ crystal dataset.

Reads: data/cache/unified_pretrain.parquet
Writes: results/mp_mega_encoder.pt (embed + blocks weights)

Multi-task: formation energy AND bandgap regression, z-score normalized.

Usage:
    python scripts/21_mega_pretrain.py --epochs 15 --batch-size 64 --device cuda \\
        --input data/cache/unified_pretrain.parquet \\
        --out results/mp_mega_encoder.pt
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.models import KSECNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_cif_to_arrays(cif: str, max_nsites: int = 80):
    """Parse CIF → (atom_z, frac_pos). Return None if too many sites or invalid."""
    if cif is None or not cif.strip():
        return None
    try:
        from pymatgen.core import Structure
    except ImportError:
        sys.exit("pip install pymatgen")
    try:
        s = Structure.from_str(cif, fmt="cif")
    except Exception:
        return None
    if len(s) > max_nsites or len(s) < 1:
        return None
    from ionpath.data.featurize import _Z, _sym
    atom_z = np.array([_Z.get(_sym(site), 0) for site in s], dtype=np.int64)
    frac_pos = np.array([site.frac_coords for site in s], dtype=np.float32) % 1.0
    return atom_z, frac_pos


class MegaPretrainHead(nn.Module):
    """k-SEC encoder + two-head regression (Ef, Eg)."""

    def __init__(self, ksec: KSECNet, hidden: int = 256):
        super().__init__()
        self.ksec = ksec
        dim = 2 * ksec.feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.ef_head = nn.Linear(hidden, 1)
        self.eg_head = nn.Linear(hidden, 1)

    def forward(self, atom_z, frac_pos, batch_idx, num_graphs):
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
        t = self.trunk(h)
        return self.ef_head(t).squeeze(-1), self.eg_head(t).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/cache/unified_pretrain.parquet")
    p.add_argument("--cache", default="data/cache/unified_parsed.pkl",
                   help="Pickled list of parsed (atom_z, frac_pos, ef_z, eg_z, ef_mask, eg_mask)")
    p.add_argument("--out", default="results/mp_mega_encoder.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-nsites", type=int, default=80)
    p.add_argument("--subsample", type=int, default=None,
                   help="Randomly subsample training set to this many rows (after filter)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists():
        log.info("loading cached parsed data from %s", cache_path)
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
    else:
        log.info("reading %s", args.input)
        df = pd.read_parquet(args.input)
        log.info("rows=%d  keeping only those with a parseable CIF", len(df))

        # Keep only entries with a CIF
        df = df[df["structure_cif"].notna() & (df["structure_cif"] != "")].reset_index(drop=True)
        log.info("after CIF filter: %d rows", len(df))

        # Parse CIFs
        parsed = []
        t0 = time.time()
        for i, row in df.iterrows():
            res = parse_cif_to_arrays(row["structure_cif"], args.max_nsites)
            if res is None:
                continue
            atom_z, frac_pos = res
            parsed.append(dict(
                atom_z=atom_z,
                frac_pos=frac_pos,
                ef=float(row["formation_energy_per_atom"]) if pd.notna(row["formation_energy_per_atom"]) else None,
                eg=float(row["band_gap"]) if pd.notna(row["band_gap"]) else None,
                source=row["source"],
            ))
            if (i + 1) % 10000 == 0:
                log.info("  parsed %d/%d  kept=%d  (%.0fs)", i + 1, len(df), len(parsed), time.time() - t0)
        log.info("parsed %d structures in %.0fs", len(parsed), time.time() - t0)

        # Compute z-score stats
        ef_values = np.array([r["ef"] for r in parsed if r["ef"] is not None], dtype=np.float32)
        eg_values = np.array([r["eg"] for r in parsed if r["eg"] is not None], dtype=np.float32)
        ef_mean, ef_sd = float(ef_values.mean()), float(ef_values.std() + 1e-8)
        eg_mean, eg_sd = float(eg_values.mean()), float(eg_values.std() + 1e-8)

        data = dict(parsed=parsed, ef_mean=ef_mean, ef_sd=ef_sd, eg_mean=eg_mean, eg_sd=eg_sd)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(data, fh)
        log.info("cached parse to %s", cache_path)

    parsed = data["parsed"]
    ef_mean, ef_sd = data["ef_mean"], data["ef_sd"]
    eg_mean, eg_sd = data["eg_mean"], data["eg_sd"]
    # Filter at load-time so cache can hold wider data but training uses a subset
    n_before = len(parsed)
    parsed = [r for r in parsed if len(r["atom_z"]) <= args.max_nsites]
    log.info("n=%d → %d after max_nsites=%d filter", n_before, len(parsed), args.max_nsites)
    if args.subsample and len(parsed) > args.subsample:
        import random
        random.Random(42).shuffle(parsed)
        parsed = parsed[: args.subsample]
        log.info("subsampled to %d rows", len(parsed))
    log.info("Ef μ=%.3f σ=%.3f   Eg μ=%.3f σ=%.3f", ef_mean, ef_sd, eg_mean, eg_sd)

    # Train/val split
    rng = np.random.default_rng(42)
    idx = np.arange(len(parsed)); rng.shuffle(idx)
    n_val = int(args.val_frac * len(parsed))
    val_idx = idx[:n_val]; train_idx = idx[n_val:]
    log.info("train=%d  val=%d", len(train_idx), len(val_idx))

    # Build model
    m = KSECNet(feature_dim=args.feature_dim, num_blocks=args.num_blocks,
                 n_max=args.n_max, dropout=args.dropout).to(args.device)
    head = MegaPretrainHead(m).to(args.device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val = float("inf")
    best_state = None

    def build_batch(indices):
        atom_z, frac_pos, batch_idx = [], [], []
        ef_z, eg_z, ef_mask, eg_mask = [], [], [], []
        for bi, i in enumerate(indices):
            r = parsed[i]
            atom_z.append(r["atom_z"])
            frac_pos.append(r["frac_pos"])
            batch_idx.append(np.full(len(r["atom_z"]), bi, dtype=np.int64))
            if r["ef"] is not None:
                ef_z.append((r["ef"] - ef_mean) / ef_sd); ef_mask.append(1.0)
            else:
                ef_z.append(0.0); ef_mask.append(0.0)
            if r["eg"] is not None:
                eg_z.append((r["eg"] - eg_mean) / eg_sd); eg_mask.append(1.0)
            else:
                eg_z.append(0.0); eg_mask.append(0.0)
        return (
            torch.from_numpy(np.concatenate(atom_z)).long().to(args.device),
            torch.from_numpy(np.concatenate(frac_pos)).float().to(args.device),
            torch.from_numpy(np.concatenate(batch_idx)).long().to(args.device),
            torch.tensor(ef_z, dtype=torch.float32, device=args.device),
            torch.tensor(eg_z, dtype=torch.float32, device=args.device),
            torch.tensor(ef_mask, dtype=torch.float32, device=args.device),
            torch.tensor(eg_mask, dtype=torch.float32, device=args.device),
        )

    for ep in range(args.epochs):
        head.train()
        t0 = time.time()
        rng = np.random.default_rng(ep + 1)
        order = rng.permutation(train_idx)
        train_loss_sum = 0.0; nb = 0
        for bstart in range(0, len(order), args.batch_size):
            bs = order[bstart:bstart + args.batch_size]
            az, fp, bi, ef_z, eg_z, ef_m, eg_m = build_batch(bs)
            ef_pred, eg_pred = head(az, fp, bi, num_graphs=len(bs))
            loss_ef = ((ef_pred - ef_z) ** 2 * ef_m).sum() / ef_m.sum().clamp(min=1)
            loss_eg = ((eg_pred - eg_z) ** 2 * eg_m).sum() / eg_m.sum().clamp(min=1)
            loss = loss_ef + loss_eg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
            train_loss_sum += float(loss); nb += 1
        sched.step()

        # Val
        head.eval()
        val_ef_err = []; val_eg_err = []
        with torch.no_grad():
            for bstart in range(0, len(val_idx), args.batch_size):
                bs = val_idx[bstart:bstart + args.batch_size]
                az, fp, bi, ef_z, eg_z, ef_m, eg_m = build_batch(bs)
                ef_pred, eg_pred = head(az, fp, bi, num_graphs=len(bs))
                val_ef_err.append(((ef_pred - ef_z).abs() * ef_m).cpu().numpy())
                val_eg_err.append(((eg_pred - eg_z).abs() * eg_m).cpu().numpy())
        val_ef = np.concatenate(val_ef_err); val_eg = np.concatenate(val_eg_err)
        val_ef_mae = float(val_ef[val_ef > 0].mean() if (val_ef > 0).any() else 0)
        val_eg_mae = float(val_eg[val_eg > 0].mean() if (val_eg > 0).any() else 0)
        val_score = val_ef_mae + val_eg_mae

        log.info("ep %02d  train_loss=%.4f  val_MAE_Ef=%.3f (z)  val_MAE_Eg=%.3f (z)  (%.0fs)",
                 ep, train_loss_sum / max(nb, 1), val_ef_mae, val_eg_mae, time.time() - t0)

        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.clone().cpu() for k, v in m.state_dict().items()}

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is None:
        best_state = m.state_dict()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state=best_state,
        val_score=best_val,
        ef_mean=ef_mean, ef_sd=ef_sd,
        eg_mean=eg_mean, eg_sd=eg_sd,
        val_mae_ef=val_ef_mae * ef_sd,
        val_mae_eg=val_eg_mae * eg_sd,
    ), args.out)
    log.info("saved mega encoder to %s (best val score=%.3f)", args.out, best_val)


if __name__ == "__main__":
    main()
