"""Ablation study: Kubic filter vs. radial, cross-shell vs. shell-restricted.

We evaluate four architectures on the same 5-fold CV, 2 seeds, to isolate
each component's contribution.

  A. Full v2            = Kubic filter + cross-shell gated attention
  B. Kubic only         = Kubic filter + shell-restricted attention
  C. Cross-shell only   = Radial filter + cross-shell gated attention
  D. v1 baseline        = Radial filter + shell-restricted attention

The full network is patched at construction with different block variants.
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.models.kspace_conv import (
    CrossShellGatedAttention,
    KSECBlock,
    KSECNet,
    KubicHarmonicFilter,
    _CLN,
    _kubic_invariants,
)
from ionpath.utils.wyckoff_fourier import generate_wyckoff_wavevectors, precompute_orbits

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablated components
# ---------------------------------------------------------------------------


class RadialFilter(nn.Module):
    """v1-style filter: W(|k|) only, no directional invariants."""

    def __init__(self, feature_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gain = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * feature_dim),
        )
        self.bias = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * feature_dim),
        )

    def forward(self, H, k_mags, kubics):
        k_in = k_mags.unsqueeze(-1)
        D = H.shape[-1]
        g = self.gain(k_in); b = self.bias(k_in)
        wr, wi = g[..., :D], g[..., D:]
        br, bi = b[..., :D], b[..., D:]
        W = torch.complex(wr, wi).unsqueeze(0)
        B = torch.complex(br, bi).unsqueeze(0)
        return H * W + B


class ShellRestrictedAttention(nn.Module):
    """v1-style attention: only within k-shells of equal |k|."""

    def __init__(self, feature_dim: int, n_heads: int = 4):
        super().__init__()
        self.h = n_heads
        self.dh = feature_dim // n_heads
        self.feature_dim = feature_dim
        self.qkv = nn.Linear(2 * feature_dim, 6 * feature_dim)
        self.o = nn.Linear(2 * feature_dim, 2 * feature_dim)

    def forward(self, H, k_mags):
        D = self.feature_dim
        B, K, _ = H.shape
        x = torch.cat([H.real, H.imag], dim=-1)
        qkv = self.qkv(x)
        qr, qi, kr, ki, vr, vi = qkv.chunk(6, dim=-1)
        q = torch.complex(qr, qi).view(B, K, self.h, self.dh)
        k = torch.complex(kr, ki).view(B, K, self.h, self.dh)
        v = torch.complex(vr, vi).view(B, K, self.h, self.dh)
        # shell-mask: same |k| within tolerance 1e-3
        km = k_mags
        shell = (km.unsqueeze(0) - km.unsqueeze(-1)).abs() < 1e-3         # (K, K)
        scores_r = torch.einsum("bkhd,bjhd->bkjh", q.real, k.real) + \
                   torch.einsum("bkhd,bjhd->bkjh", q.imag, k.imag)
        scores = scores_r / (self.dh ** 0.5)
        scores = scores.masked_fill(~shell.unsqueeze(0).unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(scores, dim=2)
        or_ = torch.einsum("bkjh,bjhd->bkhd", alpha, v.real)
        oi_ = torch.einsum("bkjh,bjhd->bkhd", alpha, v.imag)
        out = torch.complex(or_, oi_).reshape(B, K, D)
        y = self.o(torch.cat([out.real, out.imag], dim=-1))
        yr, yi = y.chunk(2, dim=-1)
        return torch.complex(yr, yi) + H


class PatchedBlock(nn.Module):
    """A k-SEC block with configurable filter and attention types."""

    def __init__(self, feature_dim: int, n_heads: int, dropout: float,
                 filter_type: str, attn_type: str):
        super().__init__()
        if filter_type == "kubic":
            self.filt = KubicHarmonicFilter(feature_dim, dropout=dropout)
        elif filter_type == "radial":
            self.filt = RadialFilter(feature_dim, dropout=dropout)
        else:
            raise ValueError(filter_type)
        if attn_type == "cross_shell":
            self.attn = CrossShellGatedAttention(feature_dim, n_heads)
            self._attn_signature = "cross_shell"
        elif attn_type == "shell_restricted":
            self.attn = ShellRestrictedAttention(feature_dim, n_heads)
            self._attn_signature = "shell_restricted"
        else:
            raise ValueError(attn_type)
        self.norm_in = _CLN(feature_dim)
        self.norm_attn = _CLN(feature_dim)

    def forward(self, H, k_mags, kubics):
        H = self.norm_in(H)
        H = self.filt(H, k_mags, kubics)
        gate = torch.sigmoid(H.abs() - 1.0)
        H = H * gate
        H = self.norm_attn(H)
        H = self.attn(H, k_mags)                                          # both take (H, k_mags)
        return H


class PatchedKSECNet(nn.Module):
    def __init__(self, filter_type: str, attn_type: str, **kw):
        super().__init__()
        # Minimal re-implementation of KSECNet with configurable block
        feature_dim = kw.get("feature_dim", 96)
        num_blocks = kw.get("num_blocks", 3)
        n_heads = kw.get("n_heads", 4)
        n_max = kw.get("n_max", 2)
        readout_hidden = kw.get("readout_hidden", 192)
        dropout = kw.get("dropout", 0.15)
        num_species = kw.get("num_species", 100)

        self.embed = nn.Embedding(num_species, feature_dim)
        self.feature_dim = feature_dim

        wv_np = generate_wyckoff_wavevectors(n_max=n_max)
        orbits = precompute_orbits(wv_np)
        all_k = []
        for orbit in orbits:
            for k_vec in orbit:
                all_k.append(k_vec.numpy())
        all_k = np.array(all_k, dtype=np.float32)
        all_k = np.concatenate([np.zeros((1, 3), dtype=np.float32), all_k], axis=0)
        k_mags = np.linalg.norm(all_k, axis=-1)
        self.register_buffer("k_points", torch.from_numpy(all_k).float(), persistent=False)
        self.register_buffer("k_mags", torch.from_numpy(k_mags).float(), persistent=False)
        kubics = _kubic_invariants(torch.from_numpy(all_k).float())
        kubics[0] = 0.0
        self.register_buffer("kubics", kubics, persistent=False)
        self.K = all_k.shape[0]

        self.blocks = nn.ModuleList([
            PatchedBlock(feature_dim, n_heads, dropout, filter_type, attn_type)
            for _ in range(num_blocks)
        ])

        self.readout = nn.Sequential(
            nn.Linear(2 * feature_dim, readout_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(readout_hidden, readout_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(readout_hidden, 1),
        )
        self.log_sigma_shift = nn.Parameter(torch.tensor(-5.0))

    def set_target_shift(self, mean: float):
        import torch
        with torch.no_grad():
            self.log_sigma_shift.copy_(torch.tensor(float(mean)))

    def forward_structure(self, atom_z, frac_pos, batch_idx, num_graphs):
        import math
        z = self.embed(atom_z)
        z_complex = torch.complex(z, torch.zeros_like(z))
        phases = -2.0 * math.pi * (frac_pos @ self.k_points.T)
        exp_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        contrib = z_complex.unsqueeze(1) * exp_phases.unsqueeze(-1)
        F = torch.zeros(num_graphs, self.K, z.shape[-1],
                        dtype=contrib.dtype, device=contrib.device)
        F.index_add_(0, batch_idx, contrib)
        counts = torch.zeros(num_graphs, device=z.device)
        counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        F = F / counts.clamp(min=1.0).view(-1, 1, 1)
        for block in self.blocks:
            F = block(F, self.k_mags, self.kubics)
        F_re = F.real.mean(dim=1); F_im = F.imag.mean(dim=1)
        h = torch.cat([F_re, F_im], dim=-1)
        y = self.readout(h).squeeze(-1)
        return y + self.log_sigma_shift


# ---------------------------------------------------------------------------
# Training + CV
# ---------------------------------------------------------------------------


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


def run_one_config(name, filter_type, attn_type, crystals, log_sigma, mask, args):
    torch.manual_seed(0); np.random.seed(0)
    eligible = np.where((mask > 0) & np.array([c is not None for c in crystals]))[0]
    mae_seeds = []
    r2_seeds = []
    for seed in range(args.seeds):
        torch.manual_seed(seed * 31 + 1); np.random.seed(seed * 31 + 1)
        folds = stratified_folds(log_sigma[eligible], seed=seed * 7)
        folds = [[int(eligible[i]) for i in f] for f in folds]
        fold_maes = []
        fold_r2s = []
        for k in range(5):
            test = np.array(folds[k], dtype=np.int64)
            train_all = np.array([i for j in range(5) if j != k for i in folds[j]], dtype=np.int64)
            rng = np.random.default_rng(100 + seed * 7 + k)
            val = rng.choice(train_all, size=max(8, int(0.1 * len(train_all))), replace=False)
            train = np.setdiff1d(train_all, val)

            m = PatchedKSECNet(filter_type=filter_type, attn_type=attn_type,
                                feature_dim=args.feature_dim,
                                num_blocks=args.num_blocks,
                                dropout=args.dropout).to(args.device)
            m.set_target_shift(float(log_sigma[train].mean()))
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            best_val = float("inf"); best_state = None
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
            m.eval()
            with torch.no_grad():
                az, fp, bi = build_inputs(crystals, test, args.device)
                pred = m.forward_structure(az, fp, bi, num_graphs=len(test)).cpu().numpy()
            target = log_sigma[test]
            err = pred - target
            mae = float(np.mean(np.abs(err)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((target - target.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            fold_maes.append(mae); fold_r2s.append(r2)
            log.info("[%s] seed=%d fold=%d  MAE=%.3f  R²=%.3f", name, seed, k, mae, r2)
        mae_seeds.append(float(np.mean(fold_maes)))
        r2_seeds.append(float(np.mean(fold_r2s)))
    return dict(
        mae_mean=float(np.mean(mae_seeds)),
        mae_std=float(np.std(mae_seeds)),
        r2_mean=float(np.mean(r2_seeds)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crystals", default="data/cache/crystals.pkl")
    p.add_argument("--labels", default="data/cache/labels.npz")
    p.add_argument("--results", default="results/ablation.json")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    with open(args.crystals, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(args.labels, allow_pickle=True)
    log_sigma = z["log_sigma"]; mask_arr = z["mask"]

    configs = [
        ("Full v2 (Kubic + cross-shell)",   "kubic",   "cross_shell"),
        ("Kubic + shell-restricted",         "kubic",   "shell_restricted"),
        ("Radial + cross-shell",             "radial",  "cross_shell"),
        ("v1 baseline (Radial + shell-restr.)", "radial",  "shell_restricted"),
    ]

    results = {}
    for name, ftype, atype in configs:
        t0 = time.time()
        log.info("=== running config: %s ===", name)
        metrics = run_one_config(name, ftype, atype, crystals, log_sigma, mask_arr, args)
        metrics["wall_s"] = time.time() - t0
        results[name] = metrics
        log.info("[%s] MAE=%.3f ± %.3f  R²=%.3f  (%.0fs)",
                 name, metrics["mae_mean"], metrics["mae_std"], metrics["r2_mean"],
                 metrics["wall_s"])

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(results, indent=2))
    log.info("")
    log.info("Final ablation table:")
    for name, v in sorted(results.items(), key=lambda kv: kv[1]["mae_mean"]):
        log.info("  %-45s  MAE=%.3f ± %.3f  R²=%.3f",
                 name, v["mae_mean"], v["mae_std"], v["r2_mean"])


if __name__ == "__main__":
    main()
