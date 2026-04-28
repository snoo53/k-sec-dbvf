"""WP2: Evaluate k-SEC on standard Matbench tasks.

Uses matminer's dataset loader (avoids the matbench-package install issue
on Python 3.11). Runs a simple 5-fold stratified CV; not a leaderboard
submission, but numerically comparable to the official leaderboard.

Supported tasks:
  - matbench_mp_e_form     (106k, formation energy per atom)
  - matbench_mp_gap        (106k, bandgap)
  - matbench_log_gvrh      (10k, log10 shear modulus)
  - matbench_log_kvrh      (10k, log10 bulk modulus)
  - matbench_dielectric    (4.7k, refractive index)
  - matbench_perovskites   (18k, formation energy of ABX3)
  - matbench_phonons       (1.2k, last phonon DOS peak)
  - matbench_jdft2d        (0.6k, exfoliation energy, 2D materials)

Usage:
    python scripts/14_matbench_eval.py --task matbench_mp_gap --epochs 40 --device cuda \\
        --pretrained-encoder results/mp_broad_encoder_pretrained.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Pre-import pymatgen so pickled Structure objects deserialize without
# triggering a (slow) lazy import inside subprocess on Windows.
from pymatgen.core import Structure  # noqa: F401

from ionpath.models import KSECNet
from ionpath.data.featurize import build_crystal_graph, CrystalGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


SUPPORTED = {
    "matbench_mp_e_form": "formation_energy_per_atom [eV/atom]",
    "matbench_mp_gap": "band_gap [eV]",
    "matbench_log_gvrh": "log10(G_VRH) [log10(GPa)]",
    "matbench_log_kvrh": "log10(K_VRH) [log10(GPa)]",
    "matbench_dielectric": "refractive_index",
}


def cg_from_pmg(s, with_magpie: bool = False) -> CrystalGraph | None:
    """Build a CrystalGraph from an already-parsed pymatgen Structure."""
    from ionpath.data.featurize import _Z, _sym
    try:
        atom_z = np.array([_Z.get(_sym(site), 0) for site in s], dtype=np.int64)
        frac_pos = np.array([site.frac_coords for site in s], dtype=np.float32) % 1.0
        cell = s.lattice.matrix.astype(np.float32)
        # lattice feats
        a, b, c = float(s.lattice.a), float(s.lattice.b), float(s.lattice.c)
        alpha, beta, gamma = (np.deg2rad(x) for x in (s.lattice.alpha, s.lattice.beta, s.lattice.gamma))
        V = float(s.lattice.volume)
        density = float(len(s) / max(V, 1e-6))
        lattice_feats = np.array([a, b, c, alpha, beta, gamma, V, density], dtype=np.float32)
        return CrystalGraph(
            atom_z=atom_z, frac_pos=frac_pos, cell=cell,
            composition=s.composition.reduced_formula,
            mobile_ion="Li",
            magpie=None,  # matbench tasks don't use Magpie
            lattice_feats=lattice_feats,
            geometric=None,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("cg build failed: %s", exc)
        return None


def build_inputs(crystals, idx, device):
    atom_z, frac_pos, batch_idx, lat = [], [], [], []
    for b, gi in enumerate(idx):
        cg = crystals[gi]
        atom_z.append(cg.atom_z)
        frac_pos.append(cg.frac_pos.astype(np.float32))
        batch_idx.append(np.full(len(cg.atom_z), b, dtype=np.int64))
        lat.append(cg.lattice_feats)
    return (
        torch.from_numpy(np.concatenate(atom_z)).long().to(device),
        torch.from_numpy(np.concatenate(frac_pos)).float().to(device),
        torch.from_numpy(np.concatenate(batch_idx)).long().to(device),
        torch.from_numpy(np.stack(lat, axis=0)).float().to(device),
    )


def main():
    print("MAIN_START", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(SUPPORTED.keys()))
    p.add_argument("--results", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--n-max", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap training set size (useful for rapid iteration)")
    p.add_argument("--pretrained-encoder", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    results_path = Path(args.results or f"results/matbench_{args.task}.json")

    # Prefer parsed pickle cache (fastest), then parquet, then matminer
    parsed_pkl = Path(f"data/cache/{args.task}_parsed.pkl")
    if parsed_pkl.exists():
        import pickle
        log.info("task=%s  loading parsed pickle: %s", args.task, parsed_pkl)
        with open(parsed_pkl, "rb") as fh:
            df = pickle.load(fh)
        print(f"PICKLE_LOADED n={len(df)}", flush=True)
    else:
        local_parquet = Path(f"data/raw/matbench/{args.task}.parquet")
        if local_parquet.exists():
            import pandas as pd
            log.info("task=%s  loading from local parquet: %s", args.task, local_parquet)
            df = pd.read_parquet(local_parquet)
        else:
            try:
                from matminer.datasets import load_dataset
            except ImportError:
                sys.exit("pip install matminer or pre-download to data/raw/matbench/")
            log.info("task=%s  loading via matminer ...", args.task)
            df = load_dataset(args.task)

    struct_col = "structure" if "structure" in df.columns else df.columns[0]
    target_col = [c for c in df.columns if c != struct_col][0]
    log.info("n=%d  target=%s", len(df), target_col)

    # Apply --limit BEFORE parsing CIFs (which is the slow step)
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        log.info("  (sampled to %d for quick eval)", args.limit)

    # If structures are CIF strings (from our pre-cached parquet), parse them
    if len(df) > 0 and isinstance(df[struct_col].iloc[0], str):
        log.info("converting %d cached CIF strings back to pymatgen Structure ...", len(df))
        from pymatgen.core import Structure
        import time as _time
        _t0 = _time.time()
        df[struct_col] = df[struct_col].apply(lambda s: Structure.from_str(s, fmt="cif") if s else None)
        df = df[df[struct_col].notna()].reset_index(drop=True)
        log.info("CIF parse done: n=%d in %.1fs", len(df), _time.time() - _t0)

    # Simple 5-fold CV (not the matbench-package official folds, but
    # numerically comparable). For leaderboard submission, use the matbench
    # package with its official fold split.
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(df))
    rng.shuffle(idx_all)
    folds_all = np.array_split(idx_all, 5)

    fold_scores = []
    for fold in range(5):
        test_idx = folds_all[fold]
        train_idx = np.concatenate([folds_all[k] for k in range(5) if k != fold])

        train_inputs = [df.iloc[i][struct_col] for i in train_idx]
        test_inputs = [df.iloc[i][struct_col] for i in test_idx]
        train_outputs = df.iloc[train_idx][target_col].values
        test_outputs = df.iloc[test_idx][target_col].values

        log.info("fold %d  n_train=%d  n_test=%d", fold, len(train_inputs), len(test_inputs))
        if args.limit and len(train_inputs) > args.limit:
            train_inputs = train_inputs[: args.limit]
            train_outputs = train_outputs[: args.limit]
            log.info("  (capped train to %d)", args.limit)

        # Parse all to CrystalGraphs
        train_cgs = [cg_from_pmg(s) for s in train_inputs]
        test_cgs = [cg_from_pmg(s) for s in test_inputs]
        train_keep = [i for i, c in enumerate(train_cgs) if c is not None]
        test_keep = [i for i, c in enumerate(test_cgs) if c is not None]
        train_cgs = [train_cgs[i] for i in train_keep]
        test_cgs = [test_cgs[i] for i in test_keep]
        y_train = np.array([train_outputs[i] for i in train_keep], dtype=np.float32)
        y_test = np.array([test_outputs[i] for i in test_keep], dtype=np.float32)

        crystals = train_cgs + test_cgs
        n_train = len(train_cgs)
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, n_train + len(test_cgs))

        # Train k-SEC
        m = KSECNet(
            feature_dim=args.feature_dim, num_blocks=args.num_blocks,
            n_max=args.n_max, dropout=args.dropout,
            tabular_dim=0, lattice_dim=8, geometric_dim=0,
        ).to(args.device)
        m.set_target_shift(float(y_train.mean()))

        if args.pretrained_encoder:
            enc = torch.load(args.pretrained_encoder, map_location=args.device, weights_only=False)
            enc_state = {kk: vv for kk, vv in enc["state"].items()
                         if kk.startswith("embed.") or kk.startswith("blocks.")}
            m.load_state_dict(enc_state, strict=False)
            log.info("  loaded MP encoder: %d tensors", len(enc_state))

        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        rng = np.random.default_rng(42 + fold)
        t0 = time.time()
        for ep in range(args.epochs):
            m.train()
            order = rng.permutation(n_train)
            for s in range(0, n_train, args.batch_size):
                idx = order[s:s + args.batch_size]
                az, fp, bi, lat = build_inputs(crystals, idx, args.device)
                pred = m.forward_structure(az, fp, bi, num_graphs=len(idx), lattice_feats=lat)
                target = torch.from_numpy(y_train[idx]).float().to(args.device)
                loss = ((pred - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sched.step()
            if ep % 5 == 0 or ep == args.epochs - 1:
                log.info("  fold=%d ep=%02d  train_loss=%.4f  (%.0fs)",
                         fold, ep, float(loss), time.time() - t0)

        m.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(test_cgs), args.batch_size):
                idx = test_indices[s:s + args.batch_size]
                az, fp, bi, lat = build_inputs(crystals, idx, args.device)
                p_ = m.forward_structure(az, fp, bi, num_graphs=len(idx), lattice_feats=lat)
                preds.append(p_.cpu().numpy())
        pred = np.concatenate(preds)

        err = pred - y_test
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        fold_scores.append(dict(fold=fold, mae=mae, r2=r2, n=len(y_test)))
        log.info("FOLD %d  MAE=%.4f  R²=%.4f  n=%d", fold, mae, r2, len(y_test))

    maes = [r["mae"] for r in fold_scores]
    r2s = [r["r2"] for r in fold_scores]
    log.info("TASK %s  MAE=%.4f ± %.4f  R²=%.4f ± %.4f",
             args.task, np.mean(maes), np.std(maes), np.mean(r2s), np.std(r2s))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(dict(
        task=args.task,
        per_fold=fold_scores,
        mae_mean=float(np.mean(maes)),
        mae_std=float(np.std(maes)),
        r2_mean=float(np.mean(r2s)),
        config=dict(epochs=args.epochs, lr=args.lr, n_max=args.n_max,
                    pretrained=args.pretrained_encoder),
    ), indent=2))
    log.info("saved to %s", results_path)


if __name__ == "__main__":
    main()
