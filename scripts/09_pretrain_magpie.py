"""Pretrain the Magpie (tabular) branch of hybrid k-SEC.

Uses:
  - Hargreaves 2023 near-RT entries (after dedup vs. OBELiX): ~415 samples
  - OBELiX composition-only entries (labelled but no CIF): ~277 samples

Total: ~692 composition+log_sigma samples that are DISJOINT from the 285
OBELiX CIF-bearing samples used for hybrid CV evaluation. Thus the
pretrained Magpie branch does not leak test-fold information.

Produces: results/magpie_pretrained.pt with keys
  - tabular_norm.*
  - tabular_proj.*
that can be loaded into KSECNet's hybrid head.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.data.hargreaves import load_hargreaves, near_rt, dedupe_against_obelix
from ionpath.data.magpie import featurize_composition, magpie_feature_dim

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class MagpieRegressor(nn.Module):
    """Mirror KSECNet's tabular branch + a throwaway readout."""

    def __init__(self, tabular_dim: int = 132, tabular_hidden: int = 96,
                 readout_hidden: int = 192, dropout: float = 0.15):
        super().__init__()
        self.tabular_norm = nn.LayerNorm(tabular_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(tabular_hidden, tabular_hidden), nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(tabular_hidden, readout_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(readout_hidden, readout_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, x):
        h = self.tabular_norm(x)
        h = self.tabular_proj(h)
        return self.readout(h).squeeze(-1)


def build_training_set(crystals_path: str, labels_path: str, skip_cif_bearing: bool = True):
    """Return (X, y, source_tag). X is (N, 132) Magpie. y is (N,) log_sigma.

    Composition sources:
      - Hargreaves near-RT, deduped against OBELiX compositions
      - OBELiX labelled entries without a CIF (composition-only)

    The 285 OBELiX CIF-bearing samples are EXCLUDED to avoid leakage into
    downstream CV evaluation of hybrid k-SEC.
    """
    # OBELiX composition-only entries
    with open(crystals_path, "rb") as fh:
        crystals = pickle.load(fh)
    z = np.load(labels_path, allow_pickle=True)
    log_sigma = z["log_sigma"]
    mask = z["mask"]
    # record their compositions too (CIF-bearing) so Hargreaves can be deduped
    obelix_all_comps = set()
    obelix_comp_only = []  # (composition_str, log_sigma) for labelled composition-only
    for i, c in enumerate(crystals):
        if mask[i] <= 0:
            continue
        if c is not None and getattr(c, "composition", None):
            obelix_all_comps.add(c.composition)
            if skip_cif_bearing:
                continue  # skip for pretraining
            obelix_comp_only.append((c.composition, float(log_sigma[i])))
        # If c is None (no CIF parsed) we don't have a composition string cached.

    # For OBELiX records with no CIF (and thus no CrystalGraph), we need the
    # composition from records.jsonl. Load that.
    rec_path = Path(labels_path).parents[1] / "processed" / "records.jsonl"
    if rec_path.exists():
        with open(rec_path, "r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("log_sigma") is None:
                    continue
                if r.get("cif"):
                    obelix_all_comps.add(r.get("composition", ""))
                    continue
                # composition-only entry
                comp = r.get("composition", "")
                if not comp:
                    continue
                obelix_all_comps.add(comp)
                obelix_comp_only.append((comp, float(r["log_sigma"])))

    # Hargreaves near-RT, deduped
    df = load_hargreaves()
    df_rt = near_rt(df)
    df_clean = dedupe_against_obelix(df_rt, obelix_all_comps)
    hargreaves_rows = list(zip(df_clean["composition"].tolist(),
                                df_clean["log_sigma"].astype(float).tolist()))

    rows = [("hargreaves", c, y) for c, y in hargreaves_rows]
    rows += [("obelix_cmpOnly", c, y) for c, y in obelix_comp_only]
    log.info("pretraining set: Hargreaves=%d  OBELiX-compOnly=%d  total=%d",
             len(hargreaves_rows), len(obelix_comp_only), len(rows))

    # Featurize
    X = np.stack([featurize_composition(r[1]) for r in rows], axis=0).astype(np.float32)
    y = np.array([r[2] for r in rows], dtype=np.float32)
    src = np.array([r[0] for r in rows])
    return X, y, src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crystals", default="data/cache/crystals.pkl")
    p.add_argument("--labels", default="data/cache/labels.npz")
    p.add_argument("--out", default="results/magpie_pretrained.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    X, y, src = build_training_set(args.crystals, args.labels)
    log.info("X=%s  y=%s", X.shape, y.shape)

    torch.manual_seed(0); np.random.seed(0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    n_val = int(args.val_frac * len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    m = MagpieRegressor(tabular_dim=X.shape[1], dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf"); best_state = None
    for ep in range(args.epochs):
        m.train()
        order = rng.permutation(train_idx)
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            x = X_t[idx].to(args.device); t = y_t[idx].to(args.device)
            pred = m(x)
            loss = ((pred - t) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pred = m(X_t[val_idx].to(args.device))
            val_mae = float((pred - y_t[val_idx].to(args.device)).abs().mean())
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs - 1:
            log.info("ep %03d  val_mae=%.3f  best=%.3f", ep, val_mae, best_val)

    log.info("Best val MAE on pretraining holdout: %.3f", best_val)
    m.load_state_dict(best_state)

    # Save only the transferable sub-modules
    transfer = {}
    for k, v in m.state_dict().items():
        if k.startswith("tabular_norm.") or k.startswith("tabular_proj."):
            transfer[k] = v.cpu()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=transfer, val_mae=best_val,
                    n_train=int(len(train_idx)), n_val=int(len(val_idx)),
                    epochs=args.epochs), args.out)
    log.info("saved pretrained Magpie branch to %s (%d tensors)",
             args.out, len(transfer))


if __name__ == "__main__":
    main()
