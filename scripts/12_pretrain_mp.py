"""Pretrain k-SEC's crystal encoder on Materials Project bandgap +
formation energy (multi-task regression).

The goal is to give the k-space encoder 50× more gradient signal than
the 285-sample OBELiX CV provides, so the downstream σ fine-tuning
starts from a much richer representation.

Saves encoder weights (embed + blocks) to a .pt so
`scripts/08_train_hybrid.py` can load them via --pretrained-encoder.

Usage:
    python scripts/12_pretrain_mp.py \
        --input data/raw/mp_li.jsonl.gz \
        --epochs 30 --device cuda \
        --out results/mp_encoder_pretrained.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.data.featurize import CrystalGraph, build_crystal_graph
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


def load_mp_jsonl(path: Path, max_nsites: int = 80, cache_path: Path | None = None):
    """Parse MP jsonl.gz into CrystalGraphs + target arrays.

    Supports a binary pickle cache to skip re-parsing on subsequent runs.
    """
    if cache_path and cache_path.exists():
        log.info("loading cached MP crystals from %s", cache_path)
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    log.info("parsing %s (this can take several minutes for large files)...", path)
    crystals, ef, eg = [], [], []
    t0 = time.time()
    n_parsed = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["nsites"] > max_nsites:
                continue
            cg = build_crystal_graph(rec["cif"], mobile_ion="Li", with_magpie=False)
            if cg is None:
                continue
            crystals.append(cg)
            ef.append(float(rec["formation_energy_per_atom"]))
            eg.append(float(rec["band_gap"]))
            n_parsed += 1
            if n_parsed % 1000 == 0:
                log.info("  parsed %d  (%.1fs)", n_parsed, time.time() - t0)
    ef = np.array(ef, dtype=np.float32)
    eg = np.array(eg, dtype=np.float32)
    log.info("parsed %d structures in %.1fs", len(crystals), time.time() - t0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump((crystals, ef, eg), fh)
        log.info("cached to %s", cache_path)
    return crystals, ef, eg


class PretrainHead(torch.nn.Module):
    """k-SEC base + dual output head for (E_formation, bandgap)."""

    def __init__(self, base: KSECNet):
        super().__init__()
        self.base = base
        # Two-target readout replaces the single-target one
        in_features = base.readout[0].in_features
        self.dual_readout = torch.nn.Sequential(
            torch.nn.Linear(in_features, 192), torch.nn.SiLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(192, 192), torch.nn.SiLU(),
            torch.nn.Linear(192, 2),        # [E_f, Eg]
        )

    def forward(self, atom_z, frac_pos, batch_idx, num_graphs):
        import math
        z = self.base.embed(atom_z)
        z_complex = torch.complex(z, torch.zeros_like(z))
        phases = -2.0 * math.pi * (frac_pos @ self.base.k_points.T)
        exp_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        contrib = z_complex.unsqueeze(1) * exp_phases.unsqueeze(-1)
        F = torch.zeros(num_graphs, self.base.K, z.shape[-1],
                        dtype=contrib.dtype, device=contrib.device)
        F.index_add_(0, batch_idx, contrib)
        counts = torch.zeros(num_graphs, device=z.device)
        counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        F = F / counts.clamp(min=1.0).view(-1, 1, 1)
        for block in self.base.blocks:
            F = block(F, self.base.k_mags, self.base.kubics)
        h = torch.cat([F.real.mean(dim=1), F.imag.mean(dim=1)], dim=-1)
        return self.dual_readout(h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/mp_li.jsonl.gz")
    p.add_argument("--cache", default="data/cache/mp_parsed.pkl")
    p.add_argument("--out", default="results/mp_encoder_pretrained.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-nsites", type=int, default=80)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    crystals, ef, eg = load_mp_jsonl(
        Path(args.input), max_nsites=args.max_nsites, cache_path=Path(args.cache)
    )
    n = len(crystals)
    log.info("n=%d  Ef range [%.3f, %.3f]  Eg range [%.3f, %.3f]",
             n, ef.min(), ef.max(), eg.min(), eg.max())

    # Target normalization to unit variance (makes MSE scale-invariant)
    ef_mu, ef_sd = float(ef.mean()), float(ef.std() + 1e-6)
    eg_mu, eg_sd = float(eg.mean()), float(eg.std() + 1e-6)
    y = np.stack([(ef - ef_mu) / ef_sd, (eg - eg_mu) / eg_sd], axis=1)

    rng = np.random.default_rng(0)
    idx = np.arange(n); rng.shuffle(idx)
    n_val = max(int(args.val_frac * n), 200)
    val, train = idx[:n_val], idx[n_val:]
    log.info("train=%d  val=%d", len(train), len(val))

    base = KSECNet(
        feature_dim=args.feature_dim, num_blocks=args.num_blocks,
        n_max=args.n_max, dropout=args.dropout,
    ).to(args.device)
    model = PretrainHead(base).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf"); best_state = None
    for ep in range(args.epochs):
        model.train()
        rng2 = np.random.default_rng(ep + 1)
        order = rng2.permutation(train)
        total_loss, n_batches = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            ids = order[s:s + args.batch_size]
            az, fp, bi = build_inputs(crystals, ids, args.device)
            target = torch.from_numpy(y[ids]).float().to(args.device)
            pred = model(az, fp, bi, num_graphs=len(ids))
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_loss += float(loss); n_batches += 1
        sched.step()

        # validation
        model.eval()
        val_mae_ef, val_mae_eg = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(val), args.batch_size):
                ids = val[s:s + args.batch_size]
                az, fp, bi = build_inputs(crystals, ids, args.device)
                target = torch.from_numpy(y[ids]).float().to(args.device)
                pred = model(az, fp, bi, num_graphs=len(ids))
                val_mae_ef += float((pred[:, 0] - target[:, 0]).abs().sum())
                val_mae_eg += float((pred[:, 1] - target[:, 1]).abs().sum())
        val_mae_ef /= len(val)
        val_mae_eg /= len(val)
        val_score = val_mae_ef + val_mae_eg
        log.info("ep %02d  train_loss=%.4f  val_MAE_Ef=%.3f (z)  val_MAE_Eg=%.3f (z)",
                 ep, total_loss / max(n_batches, 1), val_mae_ef, val_mae_eg)
        if val_score < best_val:
            best_val = val_score
            # Save only base encoder state (embed + blocks), not the dual readout
            best_state = {k: v.cpu().clone() for k, v in base.state_dict().items()}

    # Save pretrained base state
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state=best_state,
        val_score=best_val,
        val_mae_ef=val_mae_ef * ef_sd,   # approximate unscaled MAE
        val_mae_eg=val_mae_eg * eg_sd,
        config=dict(
            feature_dim=args.feature_dim, num_blocks=args.num_blocks,
            n_max=args.n_max, n_train=len(train), n_val=len(val),
            target_norm=dict(ef_mu=ef_mu, ef_sd=ef_sd, eg_mu=eg_mu, eg_sd=eg_sd),
        ),
    ), args.out)
    log.info("saved pretrained encoder to %s (val MAE_Ef=%.3f eV/atom, val MAE_Eg=%.3f eV)",
             args.out, val_mae_ef * ef_sd, val_mae_eg * eg_sd)


if __name__ == "__main__":
    main()
