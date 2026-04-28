"""Build CrystalGraph objects for every record with a CIF.

Output: data/cache/crystals.pkl (list[CrystalGraph | None], aligned with records)
        data/cache/labels.npz   (log_sigma, mask, T_K)

Usage:
    python scripts/02_featurize.py
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.data import build_crystal_graph, read_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="data/processed/records.jsonl")
    p.add_argument("--out-crystals", default="data/cache/crystals.pkl")
    p.add_argument("--out-labels", default="data/cache/labels.npz")
    args = p.parse_args()

    records = read_records(Path(args.records))
    log.info("Loaded %d records", len(records))

    crystals: list = []
    log_sigma = np.full(len(records), np.nan, dtype=np.float32)
    mask = np.zeros(len(records), dtype=np.float32)
    T_K = np.full(len(records), 298.15, dtype=np.float32)
    families = []

    t0 = time.time()
    n_built = 0
    for i, r in enumerate(records):
        if r.log_sigma is not None:
            log_sigma[i] = r.log_sigma
            mask[i] = 1.0
        if r.T_K is not None:
            T_K[i] = r.T_K
        families.append(r.structural_family or "unknown")

        if r.cif:
            cg = build_crystal_graph(r.cif, r.mobile_ion)
            crystals.append(cg)
            if cg is not None:
                n_built += 1
        else:
            crystals.append(None)

        if (i + 1) % 100 == 0:
            log.info("  %d/%d  (%d graphs built, %.1fs)", i + 1, len(records), n_built, time.time() - t0)

    Path(args.out_crystals).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_crystals, "wb") as fh:
        pickle.dump(crystals, fh)
    np.savez(
        args.out_labels,
        log_sigma=log_sigma,
        mask=mask,
        T_K=T_K,
        families=np.array(families, dtype=object),
    )
    log.info("Saved %d graphs to %s", n_built, args.out_crystals)
    log.info("Saved labels to %s", args.out_labels)


if __name__ == "__main__":
    main()
