"""Unify all pretraining data into one parquet.

Reads:
  - data/raw/mp_full.jsonl.gz      (Materials Project, ≤80 sites)
  - data/raw/jarvis_dft.jsonl.gz   (JARVIS-DFT 3D)
  - data/raw/oqmd.jsonl.gz         (OQMD subset)
  - data/raw/matbench/*.parquet    (various Matbench tasks with structures)

Writes:
  data/cache/unified_pretrain.parquet with columns:
    source, source_id, formula, nsites, formation_energy_per_atom,
    band_gap, structure_json

Filters: n_sites ≤ 80, must have at least one of (Ef, Eg).
Deduplication: by (source, source_id). No cross-source dedup (redundancy
between MP/OQMD/JARVIS is actually useful for pretraining — different
DFT functionals give different numerical labels).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def jsonl_gz_iter(path: Path):
    """Tolerate truncated gzip (files mid-write)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
    except (EOFError, OSError) as exc:
        log.warning("%s: stopped early (%s) — partial file OK", path.name, exc)


def process_mp(path: Path, max_nsites: int):
    for r in jsonl_gz_iter(path):
        if r.get("nsites", 0) > max_nsites:
            continue
        yield dict(
            source="mp", source_id=r["material_id"],
            formula=r.get("formula"),
            nsites=r.get("nsites"),
            formation_energy_per_atom=r.get("formation_energy_per_atom"),
            band_gap=r.get("band_gap"),
            structure_cif=r.get("cif"),
        )


def process_jarvis(path: Path, max_nsites: int):
    try:
        from pymatgen.core import Structure, Lattice
    except ImportError:
        sys.exit("pip install pymatgen")
    for r in jsonl_gz_iter(path):
        if r.get("nsites", 0) > max_nsites:
            continue
        try:
            lat = Lattice(r["lattice"])
            species = r["elements"]
            coords = r["coords"]
            struct = Structure(
                lat, species, coords,
                coords_are_cartesian=bool(r.get("cartesian", False)),
            )
            cif = struct.to(fmt="cif")
        except Exception as exc:
            log.debug("jarvis %s skip: %s", r.get("jid"), exc)
            continue
        yield dict(
            source="jarvis", source_id=r["jid"],
            formula=r.get("formula"),
            nsites=r.get("nsites"),
            formation_energy_per_atom=r.get("formation_energy_per_atom"),
            band_gap=r.get("band_gap"),
            structure_cif=cif,
        )


def process_oqmd(path: Path, max_nsites: int):
    # OQMD sites field is a flat list like "Li @ 0 0 0" strings; skip structure
    # reconstruction — we only keep the numeric props for composition-only pretraining.
    for r in jsonl_gz_iter(path):
        if (r.get("nsites") or 0) > max_nsites:
            continue
        yield dict(
            source="oqmd", source_id=r["oqmd_id"],
            formula=r.get("formula"),
            nsites=r.get("nsites"),
            formation_energy_per_atom=r.get("formation_energy_per_atom"),
            band_gap=r.get("band_gap"),
            structure_cif=None,  # OQMD entries composition-only in our dump
        )


def process_matbench_parquet(path: Path, max_nsites: int):
    import pandas as pd
    df = pd.read_parquet(path)
    struct_col = "structure" if "structure" in df.columns else None
    target_col = [c for c in df.columns if c != struct_col][0]
    task = path.stem
    for i, row in df.iterrows():
        try:
            if struct_col:
                s = row[struct_col]
                if not hasattr(s, "lattice"):
                    # It's already a dict/string, need to parse
                    from pymatgen.core import Structure
                    s = Structure.from_dict(s) if isinstance(s, dict) else Structure.from_str(s, fmt="cif")
                if len(s) > max_nsites:
                    continue
                cif = s.to(fmt="cif")
                nsites = len(s)
            else:
                cif = None; nsites = None
            target = row[target_col]
            # Heuristic: map target to Ef/Eg fields when the task is obviously one of those
            ef = target if "e_form" in task or "perovskites" in task or "jdft2d" in task else None
            eg = target if "gap" in task else None
            yield dict(
                source=f"matbench_{task.replace('matbench_', '')}",
                source_id=f"{task}:{i}",
                formula=row.get("formula", None),
                nsites=nsites,
                formation_energy_per_atom=ef,
                band_gap=eg,
                structure_cif=cif,
            )
        except Exception as exc:
            log.debug("matbench row skip: %s", exc)
            continue


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mp", default="data/raw/mp_full.jsonl.gz")
    p.add_argument("--jarvis", default="data/raw/jarvis_dft.jsonl.gz")
    p.add_argument("--oqmd", default="data/raw/oqmd.jsonl.gz")
    p.add_argument("--matbench-dir", default="data/raw/matbench")
    p.add_argument("--out", default="data/cache/unified_pretrain.parquet")
    p.add_argument("--max-nsites", type=int, default=80)
    args = p.parse_args()

    rows = []
    for label, path, proc in [
        ("mp", Path(args.mp), process_mp),
        ("jarvis", Path(args.jarvis), process_jarvis),
        ("oqmd", Path(args.oqmd), process_oqmd),
    ]:
        if not path.exists():
            log.warning("%s missing, skipping: %s", label, path)
            continue
        n_before = len(rows)
        for rec in proc(path, args.max_nsites):
            # require at least one of Ef/Eg
            if rec["formation_energy_per_atom"] is None and rec["band_gap"] is None:
                continue
            rows.append(rec)
        log.info("%s: +%d rows", label, len(rows) - n_before)

    mb_dir = Path(args.matbench_dir)
    if mb_dir.exists():
        for parquet in sorted(mb_dir.glob("*.parquet")):
            n_before = len(rows)
            for rec in process_matbench_parquet(parquet, args.max_nsites):
                if rec["formation_energy_per_atom"] is None and rec["band_gap"] is None:
                    continue
                rows.append(rec)
            log.info("%s: +%d rows", parquet.stem, len(rows) - n_before)

    log.info("TOTAL: %d unified pretraining rows", len(rows))

    import pandas as pd
    df = pd.DataFrame(rows)

    # Coerce numeric columns; any non-numeric becomes NaN
    for col in ("formation_energy_per_atom", "band_gap", "nsites"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where BOTH targets are now NaN (could have been non-numeric)
    keep = df["formation_energy_per_atom"].notna() | df["band_gap"].notna()
    n_drop = int((~keep).sum())
    if n_drop:
        log.warning("dropped %d rows with non-numeric targets", n_drop)
    df = df[keep].reset_index(drop=True)

    # String columns — cast to str (parquet-safe)
    for col in ("source", "source_id", "formula", "structure_cif"):
        df[col] = df[col].astype("string")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow")
    log.info("saved %d rows to %s (%.1f MB)", len(df), args.out, Path(args.out).stat().st_size / 1e6)
    log.info("by source:\n%s", df["source"].value_counts().to_string())
    log.info("Ef present: %d  Eg present: %d", int(df["formation_energy_per_atom"].notna().sum()),
             int(df["band_gap"].notna().sum()))


if __name__ == "__main__":
    main()
