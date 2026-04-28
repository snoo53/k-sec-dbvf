"""Hargreaves 2023 Liverpool Li-ion conductor dataset loader.

820 experimentally measured entries with composition + temperature +
ionic conductivity. 465 are near room temperature (15-35 C). No CIFs —
composition-only. Useful as an auxiliary dataset to pretrain the tabular
(Magpie) head of hybrid k-SEC.

Cite: Hargreaves et al., npj Computational Materials 9, Article 9 (2023).
License: academic/research use with citation; commercial use requires a
University of Liverpool licence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/raw/LiIonDatabase.csv")


def load_hargreaves(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Return a cleaned DataFrame with columns
       [id, composition, temperature_C, log_sigma, family, chem_family, source].
    """
    df = pd.read_csv(path, skiprows=3, header=0)
    df = df.rename(columns={
        "ID": "id",
        "composition": "composition",
        "source": "source",
        "temperature": "temperature_C",
        "target": "sigma_S_per_cm",
        "log_target": "log_sigma",
        "family": "family",
        "ChemicalFamily": "chem_family",
    })
    df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce")
    df["log_sigma"] = pd.to_numeric(df["log_sigma"], errors="coerce")
    df["sigma_S_per_cm"] = pd.to_numeric(df["sigma_S_per_cm"], errors="coerce")
    df = df.dropna(subset=["composition", "log_sigma", "temperature_C"]).reset_index(drop=True)
    return df


def near_rt(df: pd.DataFrame, low: float = 15.0, high: float = 35.0) -> pd.DataFrame:
    """Filter to room-temperature entries (15-35 °C). 465 rows."""
    return df[(df["temperature_C"] >= low) & (df["temperature_C"] <= high)].reset_index(drop=True)


def dedupe_against_obelix(
    hargreaves: pd.DataFrame,
    obelix_compositions: set[str],
) -> pd.DataFrame:
    """Remove Hargreaves rows whose composition matches any OBELiX composition,
    using pymatgen's reduced-formula normalization. Returns the de-duplicated
    Hargreaves subset."""
    from pymatgen.core import Composition
    obelix_reduced = set()
    for c in obelix_compositions:
        try:
            obelix_reduced.add(Composition(c).reduced_formula)
        except Exception:
            pass
    keep = []
    for comp in hargreaves["composition"].tolist():
        try:
            red = Composition(comp).reduced_formula
        except Exception:
            red = comp
        keep.append(red not in obelix_reduced)
    keep = np.array(keep, dtype=bool)
    return hargreaves.iloc[keep].reset_index(drop=True)
