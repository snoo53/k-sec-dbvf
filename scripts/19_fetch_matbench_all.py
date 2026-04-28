"""Download every Matbench dataset via matminer and cache locally.

Produces:
  data/raw/matbench/<task_name>.parquet

These give the encoder multi-target variety beyond Ef+Eg:
  - matbench_mp_e_form     (106k, formation energy)
  - matbench_mp_gap        (106k, bandgap)
  - matbench_log_gvrh      (10k, log10 shear modulus)
  - matbench_log_kvrh      (10k, log10 bulk modulus)
  - matbench_dielectric    (4.7k, refractive index)
  - matbench_phonons       (1.2k, phonon DOS peak)
  - matbench_perovskites   (18k, formation energy ABX3)
  - matbench_jdft2d        (0.6k, 2D exfoliation energy)
  - matbench_expt_gap      (4.6k, experimental bandgap, composition-only)
  - matbench_expt_is_metal (4.9k, experimental is-metal, composition-only)
  - matbench_glass         (5.6k, glass-forming ability, composition-only)
  - matbench_steels        (0.3k, yield strength of steels, composition-only)
  - matbench_mp_is_metal   (106k, is-metal classification)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


TASKS = [
    # (name, structure_required)
    "matbench_dielectric",
    "matbench_log_gvrh",
    "matbench_log_kvrh",
    "matbench_phonons",
    "matbench_perovskites",
    "matbench_jdft2d",
    "matbench_mp_e_form",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    # Composition-only (no CIF) — still useful as Magpie-head pretraining
    "matbench_expt_gap",
    "matbench_expt_is_metal",
    "matbench_glass",
    "matbench_steels",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/matbench")
    p.add_argument("--skip-big", action="store_true",
                   help="Skip the two 106k MP tasks (they duplicate MP data we already have)")
    args = p.parse_args()

    try:
        from matminer.datasets import load_dataset
    except ImportError:
        sys.exit("pip install matminer")

    outp = Path(args.out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    total = 0
    for task in TASKS:
        if args.skip_big and task in {"matbench_mp_e_form", "matbench_mp_gap", "matbench_mp_is_metal"}:
            log.info("skipping big task %s (--skip-big)", task)
            continue
        target = outp / f"{task}.parquet"
        if target.exists():
            log.info("%s already exists, skipping", target.name)
            continue
        t0 = time.time()
        log.info("loading %s ...", task)
        try:
            df = load_dataset(task)
            # Parquet can't serialize pymatgen Structure objects. Convert any
            # Structure column to a CIF string.
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].iloc[0] if len(df) else None
                    if sample is not None and hasattr(sample, "lattice"):
                        log.info("  converting column %s (pymatgen Structure → CIF)", col)
                        df[col] = df[col].apply(lambda s: s.to(fmt="cif") if s is not None else None)
            df.to_parquet(target, engine="pyarrow")
            log.info("  %s  n=%d  cols=%s  (%.1fs)", task, len(df), list(df.columns), time.time() - t0)
            total += len(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s  failed: %s", task, exc)

    log.info("done: %d total rows across %d tasks → %s", total, len(TASKS), outp)


if __name__ == "__main__":
    main()
