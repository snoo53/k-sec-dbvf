"""Fetch OQMD crystals + formation energies via their public REST API.

OQMD has ~1M DFT calculations. We take a representative subset filtered by
formation energy and formula validity. No API key required.

Rate-limited: ~2 requests/sec to be polite.

Output: data/raw/oqmd.jsonl.gz
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


BASE_URL = "https://oqmd.org/oqmdapi/formationenergy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/oqmd.jsonl.gz")
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--min-formation-energy", type=float, default=-5.0,
                   help="eV/atom lower bound")
    p.add_argument("--max-formation-energy", type=float, default=1.0)
    p.add_argument("--max-nsites", type=int, default=80)
    args = p.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    # OQMD API params: delta_e (formation energy), stability, composition, fields
    params_base = {
        "fields": "name,delta_e,stability,ntypes,volume,band_gap,spacegroup,"
                  "composition_generic,unit_cell,sites",
        "filter": f"delta_e>{args.min_formation_energy} AND delta_e<{args.max_formation_energy}",
        "limit": args.page_size,
        "offset": 0,
    }

    n_written = 0
    n_skipped = 0
    t0 = time.time()
    with gzip.open(outp, "wt", encoding="utf-8") as fh:
        offset = 0
        while n_written < args.limit:
            params = dict(params_base); params["offset"] = offset
            try:
                r = requests.get(BASE_URL, params=params, timeout=60)
                r.raise_for_status()
            except Exception as exc:
                log.warning("request failed at offset=%d: %s", offset, exc)
                time.sleep(5)
                continue
            data = r.json()
            results = data.get("data", [])
            if not results:
                log.info("no more results at offset=%d", offset)
                break
            for entry in results:
                try:
                    n_sites = int(entry.get("ntypes", 0)) * len(entry.get("sites", []) or [])
                    if n_sites < 1 or n_sites > args.max_nsites:
                        n_skipped += 1; continue
                    rec = dict(
                        oqmd_id=entry.get("name"),
                        formula=entry.get("name"),
                        formation_energy_per_atom=entry.get("delta_e"),
                        band_gap=entry.get("band_gap"),
                        stability=entry.get("stability"),
                        nsites=n_sites,
                        spacegroup=entry.get("spacegroup"),
                        unit_cell=entry.get("unit_cell"),
                        sites=entry.get("sites"),
                    )
                    if rec["formation_energy_per_atom"] is None:
                        n_skipped += 1; continue
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1
                except Exception as exc:
                    log.debug("skip: %s", exc)
                    n_skipped += 1
                    continue
                if n_written >= args.limit:
                    break
            offset += len(results)
            if n_written % 2000 < args.page_size:
                log.info("  %d written  (%d skipped)  offset=%d  (%.1fs)",
                         n_written, n_skipped, offset, time.time() - t0)
            time.sleep(0.5)  # be polite

    log.info("done: %d written, %d skipped → %s", n_written, n_skipped, outp)


if __name__ == "__main__":
    main()
