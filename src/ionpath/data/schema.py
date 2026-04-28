"""Canonical schema used across all downloaded datasets after parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

FIDELITY_EXPERIMENTAL = "experimental"
FIDELITY_AIMD = "aimd"
FIDELITY_DFT = "dft_proxy"


@dataclass
class CanonicalRecord:
    # Identification
    record_id: str
    source: str
    fidelity: str

    # Chemistry
    composition: str
    mobile_ion: str
    structural_family: Optional[str] = None
    space_group: Optional[str] = None

    # Structure (CIF string) — optional when only composition is available.
    cif: Optional[str] = None

    # Transport properties (any may be None)
    T_K: Optional[float] = None
    log_sigma: Optional[float] = None         # log10(σ / S cm^-1)
    E_a_eV: Optional[float] = None
    log_sigma_0: Optional[float] = None       # log10 of Arrhenius prefactor

    # Optional provenance
    doi: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
