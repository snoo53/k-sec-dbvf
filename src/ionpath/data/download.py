"""Dataset download.

For k-SEC we need crystals with CIFs and experimental ionic conductivity
at 298 K. The only public source that gives us both is OBELiX (NRC/Mila
2025). Liverpool has more entries but no CIFs. We also ship a small
hand-curated supplement of landmark fast conductors (compositions + σ
only, no CIFs — these go to a composition-only auxiliary set if needed).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import requests

from .schema import CanonicalRecord, FIDELITY_EXPERIMENTAL

log = logging.getLogger(__name__)


OBELIX_CSV_URL = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.csv"
OBELIX_CIFS_URL = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all_cifs.zip"


def _cache_path(cache_dir: Path, url: str, suffix: str = "") -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}{suffix}"


def _get_cached(url: str, cache_dir: Path, suffix: str = "") -> bytes | None:
    path = _cache_path(cache_dir, url, suffix)
    if path.exists():
        return path.read_bytes()
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return r.content
        log.warning("GET %s → status %d", url, r.status_code)
    except Exception as exc:          # noqa: BLE001
        log.warning("GET %s failed: %s", url, exc)
    return None


def fetch_obelix(cache_dir: Path, want_cifs: bool = True) -> list[CanonicalRecord]:
    """Fetch OBELiX CSV + CIFs. Returns records with CIFs attached."""
    import pandas as pd

    raw = _get_cached(OBELIX_CSV_URL, cache_dir, ".csv")
    if raw is None:
        log.error("OBELiX CSV unreachable.")
        return []
    df = pd.read_csv(io.BytesIO(raw))

    cif_map: dict[str, str] = {}
    if want_cifs:
        cif_bytes = _get_cached(OBELIX_CIFS_URL, cache_dir, ".zip")
        if cif_bytes:
            try:
                with zipfile.ZipFile(io.BytesIO(cif_bytes)) as z:
                    for name in z.namelist():
                        if not name.lower().endswith(".cif"):
                            continue
                        stem = Path(name).stem
                        cif_map[stem] = z.read(name).decode("utf-8", errors="replace")
                log.info("Loaded %d OBELiX CIFs", len(cif_map))
            except Exception as exc:          # noqa: BLE001
                log.warning("Failed to extract OBELiX CIF zip: %s", exc)

    records: list[CanonicalRecord] = []
    for _, row in df.iterrows():
        sig = row.get("Ionic conductivity (S cm-1)")
        if pd.isna(sig):
            continue
        try:
            sigv = float(sig)
        except (TypeError, ValueError):
            continue
        if sigv <= 0:
            continue
        comp = row.get("True Composition") or row.get("Reduced Composition")
        if pd.isna(comp) or not comp:
            continue
        rid = str(row.get("ID", "unknown")).strip()
        family = row.get("Family")
        family = str(family).strip().lower() if isinstance(family, str) else None
        sg = row.get("Space group")
        sg = str(sg).strip() if isinstance(sg, str) else None

        records.append(
            CanonicalRecord(
                record_id=f"obelix-{rid}",
                source="obelix-nrc-mila-2025",
                fidelity=FIDELITY_EXPERIMENTAL,
                composition=str(comp),
                mobile_ion="Li",
                structural_family=family,
                space_group=sg,
                cif=cif_map.get(rid),
                T_K=298.15,
                log_sigma=float(np.log10(sigv)),
                doi=str(row.get("DOI")) if not pd.isna(row.get("DOI")) else None,
            )
        )
    log.info("OBELiX: %d records (%d with CIF)",
             len(records), sum(1 for r in records if r.cif))
    return records


def write_records(records: list[CanonicalRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict()) + "\n")


def read_records(path: Path) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            records.append(CanonicalRecord(**json.loads(line)))
    return records
