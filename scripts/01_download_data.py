"""Download OBELiX (562 entries + 285 CIFs) and write canonical records.

Usage:
    python scripts/01_download_data.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.data import fetch_obelix, write_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out", default="data/processed/records.jsonl")
    args = p.parse_args()

    records = fetch_obelix(Path(args.cache_dir))
    write_records(records, Path(args.out))
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
