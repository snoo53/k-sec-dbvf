"""Fetch AFLOW crystals + DFT properties via AFLUX API.

AFLOW has ~2.98M DFT-computed entries. Their REST API uses matchbook-style
URL syntax: aflow.org/API/aflux/?KEYWORD,KEYWORD,paging(N,SIZE),format(json)

Response is a JSON dict keyed by "N of TOTAL" with one entry per record.

Output: data/raw/aflow.jsonl.gz (each line one entry)

Usage:
    python scripts/23_fetch_aflow.py --limit 1000000 --page-size 1000
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


BASE = "https://aflow.org/API/aflux/"


def query(fields: list[str], page: int, size: int):
    """Make AFLUX query for one page (no nsites filter; we filter locally)."""
    import requests
    keys = ",".join(fields)
    matchbook = f"{keys},paging({page},{size}),format(json)"
    url = BASE + "?" + matchbook
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/aflow.jsonl.gz")
    p.add_argument("--limit", type=int, default=2000000)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--max-nsites", type=int, default=80)
    p.add_argument("--start-page", type=int, default=1)
    args = p.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    fields = [
        "compound", "Egap", "enthalpy_formation_atom",
        "natoms", "spacegroup_relax", "species", "auid",
    ]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    page = args.start_page
    t0 = time.time()
    consecutive_empty = 0

    with gzip.open(outp, "wt", encoding="utf-8") as fh:
        while n_written < args.limit:
            try:
                data = query(fields, page, args.page_size)
            except Exception as exc:  # noqa: BLE001
                log.warning("page=%d failed: %s", page, exc)
                time.sleep(5)
                page += 1
                consecutive_empty += 1
                if consecutive_empty > 10:
                    log.warning("10 consecutive failures; stopping")
                    break
                continue
            consecutive_empty = 0

            if not data:
                log.info("empty response at page=%d, stopping", page)
                break

            for key, entry in data.items():
                try:
                    nat = entry.get("natoms", 0)
                    if nat is None or nat > args.max_nsites:
                        n_skipped += 1
                        continue
                    rec = dict(
                        auid=entry.get("auid"),
                        compound=entry.get("compound"),
                        nsites=nat,
                        band_gap=entry.get("Egap"),
                        formation_energy_per_atom=entry.get("enthalpy_formation_atom"),
                        spacegroup=entry.get("spacegroup_relax"),
                        species=entry.get("species"),
                    )
                    if rec["formation_energy_per_atom"] is None and rec["band_gap"] is None:
                        n_skipped += 1
                        continue
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1
                except Exception:
                    n_skipped += 1
                    continue
                if n_written >= args.limit:
                    break

            log.info("page=%d  total_so_far=%d  skipped=%d  (%.0fs)",
                     page, n_written, n_skipped, time.time() - t0)
            page += 1
            time.sleep(0.3)

    log.info("done: %d written, %d skipped → %s", n_written, n_skipped, outp)


if __name__ == "__main__":
    main()
