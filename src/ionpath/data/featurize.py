"""Featurization for k-SEC.

The only model in this repo is the k-Space Equivariant Convolutional
Network (k-SEC). It operates on per-crystal:

  - atomic numbers   (N,)
  - fractional positions (N, 3)
  - lattice matrix   (3, 3)

We read each CIF with pymatgen and extract these into a minimal
CrystalGraph dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl",
    "Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As",
    "Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In",
    "Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb",
    "Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Th","U",
]
_Z = {el: i + 1 for i, el in enumerate(_ELEMENTS)}


@dataclass
class CrystalGraph:
    """Minimal per-crystal representation for k-SEC."""
    atom_z: np.ndarray          # (N,)  atomic numbers
    frac_pos: np.ndarray        # (N, 3) fractional coords
    cell: np.ndarray            # (3, 3) Å
    composition: str
    mobile_ion: str
    magpie: np.ndarray | None = None  # (F,) Magpie composition features
    lattice_feats: np.ndarray | None = None  # (8,) [a, b, c, alpha, beta, gamma, V, density_atomic]
    geometric: np.ndarray | None = None  # (20,) BV/geometric features from data.geometric
    mace: np.ndarray | None = None       # (4,) MACE-MP-0 features: [E/atom, E/Li, F_rms, valid]


def _sym(site) -> str:
    try:
        return site.specie.symbol
    except AttributeError:
        return max(site.species, key=site.species.get).symbol


def build_crystal_graph(cif: str, mobile_ion: str, with_magpie: bool = True) -> CrystalGraph | None:
    """Parse a CIF into a CrystalGraph. Returns None on failure."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        log.warning("pymatgen not installed — cannot parse CIFs.")
        return None

    try:
        s = Structure.from_str(cif, fmt="cif")
    except Exception as exc:          # noqa: BLE001
        log.debug("CIF parse failed: %s", exc)
        return None

    atom_z = np.array([_Z.get(_sym(site), 0) for site in s], dtype=np.int64)
    frac_pos = np.array([site.frac_coords for site in s], dtype=np.float32) % 1.0
    cell = s.lattice.matrix.astype(np.float32)
    composition = s.composition.reduced_formula

    # Lattice features (8-dim): a, b, c, α, β, γ (radians), volume, atoms/Å³
    try:
        lat = s.lattice
        lattice_feats = np.array([
            float(lat.a), float(lat.b), float(lat.c),
            float(np.deg2rad(lat.alpha)),
            float(np.deg2rad(lat.beta)),
            float(np.deg2rad(lat.gamma)),
            float(lat.volume),
            float(len(s) / max(lat.volume, 1e-6)),
        ], dtype=np.float32)
    except Exception:  # noqa: BLE001
        lattice_feats = np.zeros(8, dtype=np.float32)

    magpie_vec = None
    if with_magpie:
        try:
            from .magpie import featurize_composition
            magpie_vec = featurize_composition(composition)
        except Exception as exc:  # noqa: BLE001
            log.debug("Magpie featurize failed for %s: %s", composition, exc)
            magpie_vec = None

    return CrystalGraph(
        atom_z=atom_z,
        frac_pos=frac_pos,
        cell=cell,
        composition=composition,
        mobile_ion=mobile_ion,
        magpie=magpie_vec,
        lattice_feats=lattice_feats,
    )
