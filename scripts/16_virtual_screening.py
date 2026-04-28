"""WP5: Virtual screening — run final k-SEC on all MP Li-containing crystals,
rank by predicted σ, produce candidate shortlist.

Reads:
  - data/cache/mp_parsed.pkl  (tuple: list[CrystalGraph], ef_array, eg_array)
  - results/ksec_ckpt_seed0.pt (saved by 08_train_hybrid.py --save-ckpt)

Outputs:
  - results/virtual_screen_top_100.csv  top 100 predicted Li conductors
  - results/virtual_screen_all.parquet  all rankings

Note: MP cache crystals were originally Li-containing (from scripts/11_fetch_mp.py
--phase li), so screening here is on already-Li-relevant candidates.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.models import KSECNet
from ionpath.data.magpie import featurize_composition
from ionpath.data.geometric import GEOMETRIC_FEATURE_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mp-cache", default="data/cache/mp_parsed.pkl")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-csv", default="results/virtual_screen_top_100.csv")
    p.add_argument("--out-parquet", default="results/virtual_screen_all.parquet")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    # Load MP cache (tuple of crystals, ef, eg)
    with open(args.mp_cache, "rb") as fh:
        cache = pickle.load(fh)
    crystals = cache[0] if isinstance(cache, tuple) else cache
    log.info("loaded %d MP Li crystals", len(crystals))

    # Make sure each crystal has Magpie + lattice + geometric features
    # (12_pretrain_mp.py only fills atom_z, frac_pos, cell — need to add the rest)
    log.info("ensuring features...")
    t0 = time.time()
    needs_magpie = sum(1 for c in crystals if getattr(c, "magpie", None) is None)
    needs_lat = sum(1 for c in crystals if getattr(c, "lattice_feats", None) is None)
    needs_geo = sum(1 for c in crystals if getattr(c, "geometric", None) is None)
    log.info("missing magpie=%d  lattice=%d  geometric=%d", needs_magpie, needs_lat, needs_geo)

    for i, c in enumerate(crystals):
        if c is None:
            continue
        if getattr(c, "magpie", None) is None:
            try:
                c.magpie = featurize_composition(c.composition)
            except Exception:
                c.magpie = np.zeros(132, dtype=np.float32)
        if getattr(c, "lattice_feats", None) is None:
            M = c.cell.astype(np.float32)
            a = float(np.linalg.norm(M[0])); b = float(np.linalg.norm(M[1])); cc = float(np.linalg.norm(M[2]))
            V = float(abs(np.linalg.det(M)))
            cosa = float(np.clip(np.dot(M[1], M[2]) / max(b * cc, 1e-9), -1, 1))
            cosb = float(np.clip(np.dot(M[0], M[2]) / max(a * cc, 1e-9), -1, 1))
            cosg = float(np.clip(np.dot(M[0], M[1]) / max(a * b, 1e-9), -1, 1))
            c.lattice_feats = np.array([
                a, b, cc,
                np.arccos(cosa), np.arccos(cosb), np.arccos(cosg),
                V, len(c.atom_z) / max(V, 1e-6),
            ], dtype=np.float32)
        if getattr(c, "geometric", None) is None:
            # Geometric features need the full pymatgen Structure; for screening
            # we use a zero-vector with feat_valid=0 to indicate "unavailable."
            c.geometric = np.zeros(GEOMETRIC_FEATURE_DIM, dtype=np.float32)
        if (i + 1) % 5000 == 0:
            log.info("  %d/%d featurized (%.1fs)", i + 1, len(crystals), time.time() - t0)
    log.info("featurization done in %.1fs", time.time() - t0)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    m = KSECNet(
        feature_dim=cfg.get("feature_dim", 96),
        num_blocks=cfg.get("num_blocks", 3),
        n_max=cfg.get("n_max", 2),
        dropout=0.0,
        tabular_dim=cfg.get("tabular_dim", 132),
        lattice_dim=cfg.get("lattice_dim", 8),
        geometric_dim=cfg.get("geometric_dim", 20),
    ).to(args.device)
    m.load_state_dict(ckpt["state"], strict=False)
    m.eval()
    log.info("model loaded from %s", args.checkpoint)

    # Inference
    log.info("running inference on %d candidates...", len(crystals))
    preds = []
    formulas = []
    nsites = []
    mids = []
    t0 = time.time()

    for bstart in range(0, len(crystals), args.batch_size):
        batch = crystals[bstart:bstart + args.batch_size]
        atom_z = np.concatenate([c.atom_z for c in batch])
        frac_pos = np.concatenate([c.frac_pos for c in batch]).astype(np.float32)
        bi = np.concatenate([np.full(len(c.atom_z), i, dtype=np.int64) for i, c in enumerate(batch)])
        magpie = np.stack([c.magpie for c in batch], axis=0).astype(np.float32)
        lat = np.stack([c.lattice_feats for c in batch], axis=0).astype(np.float32)
        geo = np.stack([c.geometric for c in batch], axis=0).astype(np.float32)

        with torch.no_grad():
            try:
                p_ = m.forward_structure(
                    torch.from_numpy(atom_z).long().to(args.device),
                    torch.from_numpy(frac_pos).float().to(args.device),
                    torch.from_numpy(bi).long().to(args.device),
                    num_graphs=len(batch),
                    tabular=torch.from_numpy(magpie).float().to(args.device),
                    lattice_feats=torch.from_numpy(lat).float().to(args.device),
                    geometric=torch.from_numpy(geo).float().to(args.device),
                ).cpu().numpy()
            except Exception as exc:
                log.warning("batch %d failed: %s", bstart, exc)
                p_ = np.full(len(batch), np.nan, dtype=np.float32)
        preds.extend(p_.tolist())
        formulas.extend([c.composition for c in batch])
        nsites.extend([len(c.atom_z) for c in batch])
        mids.extend([f"mp:{bstart + i}" for i in range(len(batch))])

        if (bstart + args.batch_size) % 1000 < args.batch_size:
            log.info("  %d/%d  (%.1fs)", bstart, len(crystals), time.time() - t0)

    # Save
    import pandas as pd
    df = pd.DataFrame(dict(
        mid=mids, formula=formulas, nsites=nsites, pred_log_sigma=preds,
    ))
    df = df.sort_values("pred_log_sigma", ascending=False).reset_index(drop=True)

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    # Save full ranking as CSV (avoids pyarrow extension type mismatch)
    full_csv = str(args.out_parquet).replace(".parquet", ".csv")
    df.to_csv(full_csv, index=False)
    df.head(args.top_k).to_csv(args.out_csv, index=False)
    args.out_parquet = full_csv  # for log message below
    log.info("\n%s", df.head(20).to_string())
    log.info("\nsaved full ranking to %s", args.out_parquet)
    log.info("saved top-%d to %s", args.top_k, args.out_csv)


if __name__ == "__main__":
    main()
