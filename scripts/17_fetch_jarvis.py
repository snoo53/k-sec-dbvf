"""Fetch JARVIS-DFT 3D crystals + properties via jarvis-tools.

~75k crystals with DFT-computed properties.

Output: data/raw/jarvis_dft.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dft_3d",
                   help="JARVIS dataset name: dft_3d (default), dft_2d, etc.")
    p.add_argument("--out", default="data/raw/jarvis_dft.jsonl.gz")
    p.add_argument("--max-nsites", type=int, default=80)
    args = p.parse_args()

    try:
        from jarvis.db.figshare import data
    except ImportError:
        sys.exit("pip install jarvis-tools")

    log.info("loading JARVIS %s (may take a minute on first run, caches to ~/.jarvis/) ...",
             args.dataset)
    t0 = time.time()
    dset = data(args.dataset)
    log.info("loaded %d entries in %.1fs", len(dset), time.time() - t0)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    with gzip.open(outp, "wt", encoding="utf-8") as fh:
        for entry in dset:
            atoms = entry.get("atoms")
            if atoms is None:
                n_skipped += 1; continue
            elements = atoms.get("elements", [])
            if len(elements) < 1 or len(elements) > args.max_nsites:
                n_skipped += 1; continue
            rec = dict(
                jid=str(entry.get("jid", "")),
                formula=entry.get("formula"),
                nsites=len(elements),
                # JARVIS uses optb88vdw_bandgap for DFT bandgap
                band_gap=entry.get("optb88vdw_bandgap"),
                formation_energy_per_atom=entry.get("formation_energy_peratom"),
                ehull=entry.get("ehull"),
                elements=elements,
                lattice=atoms.get("lattice_mat"),
                coords=atoms.get("coords"),
                cartesian=atoms.get("cartesian", False),
            )
            # Skip if both targets missing
            def _bad(x):
                if x is None:
                    return True
                try:
                    xv = float(x)
                except Exception:
                    return True
                return xv != xv  # NaN
            if _bad(rec["band_gap"]) and _bad(rec["formation_energy_per_atom"]):
                n_skipped += 1; continue
            fh.write(json.dumps(rec) + "\n")
            n_written += 1
            if n_written % 5000 == 0:
                log.info("  %d written  (%d skipped)", n_written, n_skipped)

    log.info("done: %d written, %d skipped → %s", n_written, n_skipped, outp)


if __name__ == "__main__":
    main()
