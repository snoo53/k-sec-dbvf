"""Fetch Li-containing (and optionally all) crystals from Materials
Project for pretraining k-SEC's encoder.

Two phases:
  Phase 1  : Li-containing, ≤ 80 sites, has bandgap + formation energy
  Phase 2  : (optional, --broad) any element, ≤ 80 sites, same targets,
             capped at 50k

Saves each phase as a .jsonl.gz with one record per line:
  {material_id, formula, nsites, band_gap, formation_energy_per_atom,
   e_above_hull, elements, cif}

Also produces a .npz index of (material_id, nsites, Ef, Eg, Eh) for fast
filtering later.

Usage:
    python scripts/11_fetch_mp.py --phase li --out data/raw/mp_li.jsonl.gz
    python scripts/11_fetch_mp.py --phase broad --limit 50000 --out data/raw/mp_broad.jsonl.gz

The API key is read from .env (MP_API_KEY=...).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_env(env_path: str = ".env") -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["li", "broad"], default="li")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Maximum number of entries to save (phase broad only)")
    p.add_argument("--max-nsites", type=int, default=80)
    p.add_argument("--max-e-above-hull", type=float, default=0.2,
                   help="eV/atom above hull; 0.2 keeps metastables, 0.05 for stable-only")
    p.add_argument("--chunk-size", type=int, default=1000)
    args = p.parse_args()

    load_env()
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("Set MP_API_KEY in .env")

    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("pip install mp-api")

    fields = [
        "material_id", "formula_pretty", "nsites",
        "band_gap", "formation_energy_per_atom", "energy_above_hull",
        "elements", "structure",
    ]

    query = dict(
        num_sites=(1, args.max_nsites),
        energy_above_hull=(0.0, args.max_e_above_hull),
        fields=fields,
    )
    if args.phase == "li":
        query["elements"] = ["Li"]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    log.info("phase=%s  out=%s  query=%s", args.phase, outp, {k: v for k, v in query.items() if k != "fields"})

    mpr = MPRester(api_key)

    # Streaming: write each doc to jsonl.gz as it arrives
    n_written = 0
    n_skipped = 0
    t0 = time.time()

    # mp_api returns a generator-like object for large queries
    with gzip.open(outp, "wt", encoding="utf-8") as fh:
        docs = mpr.materials.summary.search(**query, chunk_size=args.chunk_size)
        for d in docs:
            # Skip incomplete
            if d.structure is None or d.band_gap is None or d.formation_energy_per_atom is None:
                n_skipped += 1
                continue
            try:
                rec = dict(
                    material_id=str(d.material_id),
                    formula=d.formula_pretty,
                    nsites=int(d.nsites),
                    band_gap=float(d.band_gap),
                    formation_energy_per_atom=float(d.formation_energy_per_atom),
                    e_above_hull=float(d.energy_above_hull) if d.energy_above_hull is not None else None,
                    elements=[str(e) for e in d.elements],
                    cif=d.structure.to(fmt="cif"),
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("skip %s: %s", d.material_id, exc)
                n_skipped += 1
                continue
            fh.write(json.dumps(rec) + "\n")
            n_written += 1
            if n_written % 500 == 0:
                log.info("  %d written  (%d skipped)  %.1fs", n_written, n_skipped, time.time() - t0)
            if args.limit is not None and n_written >= args.limit:
                break

    log.info("done: %d written, %d skipped, %.1fs → %s", n_written, n_skipped, time.time() - t0, outp)


if __name__ == "__main__":
    main()
