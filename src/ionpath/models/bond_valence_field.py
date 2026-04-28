"""Differentiable Bond-Valence Field (DBVF) — a novel learnable
physics-grounded module for ionic-conductivity prediction.

Background
----------
Brown's bond-valence sum (BVS) rule says that the sum of exponentially-
decaying contributions from neighboring anions reaches the cation's
expected valence at well-fitting sites:

    V = Σ_j exp((r0_ij − d_ij) / b_ij)

Standard practice uses tabulated (r0, b) parameters from the Brown
2002 review. **DBVF treats these as learnable parameters of a neural
module and back-propagates through them**, so the model adapts the
"effective" bond-valence parameters to the OBELiX σ task.

Module
------
- LearnableBVParams: per (cation Z, anion Z) pair, holds (r0_raw, b_raw)
  that are softplus-shifted to give physical (>0) values. Initialized
  from Brown 2002 tables for Li-O, Li-S, Li-F, Li-Cl, Li-Br, Li-I, Li-N
  and zeros for unknown pairs.

- compute_bv_features: given (atom_z, frac_pos, cell, mobile_z), returns
  per-crystal aggregate features of the BV-mismatch surface evaluated
  at every Li site. The aggregates (mean, std, min, max, 25th/75th
  percentile of |V − V_target|) form a 6-d feature vector per crystal.

The full forward is fully differentiable; gradients flow through
exp((r0 - d)/b) into r0 and b.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


# Brown 2002 tabulated values for Li-anion bonds (r0 in Å, b ≈ 0.37–0.40)
# Used only as initialization — the network learns from these.
_LI_BV_INIT = {
    "O":  (1.466, 0.37),
    "S":  (1.85,  0.40),
    "Se": (1.93,  0.40),
    "F":  (1.36,  0.37),
    "Cl": (1.79,  0.40),
    "Br": (1.92,  0.40),
    "I":  (2.07,  0.40),
    "N":  (1.61,  0.37),
}

# Map element symbol → atomic number (from featurize.py order)
_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl",
    "Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As",
    "Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In",
    "Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb",
    "Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Th","U",
]
_Z = {el: i + 1 for i, el in enumerate(_ELEMENTS)}


class LearnableBVParams(nn.Module):
    """Holds learnable (r0, b) parameters for each (mobile_z, anion_z) pair.

    Parameters are stored as raw real numbers and softplus-shifted at use.
    Initial values come from Brown 2002 tabulations where available;
    unknown pairs initialize at (r0 = 1.8, b = 0.4).
    """

    def __init__(self, num_species: int = 100, mobile_z: int = 3):
        super().__init__()
        self.num_species = num_species
        self.mobile_z = mobile_z
        # raw r0 (will softplus → r0 > 0) per anion species
        r0_raw = torch.full((num_species + 1,), self._inv_softplus(1.8))
        b_raw = torch.full((num_species + 1,), self._inv_softplus(0.4))
        # initialize known Li-anion pairs from Brown 2002
        for sym, (r0, b) in _LI_BV_INIT.items():
            j = _Z[sym]
            r0_raw[j] = self._inv_softplus(r0)
            b_raw[j] = self._inv_softplus(b)
        self.r0_raw = nn.Parameter(r0_raw)
        self.b_raw = nn.Parameter(b_raw)

    @staticmethod
    def _inv_softplus(x: float) -> float:
        return math.log(math.expm1(x))

    @property
    def r0(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.r0_raw)

    @property
    def b(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.b_raw)


def compute_bv_features(
    bv: LearnableBVParams,
    atom_z: torch.Tensor,            # (N,) atomic numbers
    frac_pos: torch.Tensor,          # (N, 3) fractional coords
    cell: torch.Tensor,              # (B, 3, 3) per-graph lattice
    batch_idx: torch.Tensor,         # (N,) sample index
    num_graphs: int,
    mobile_z: int = 3,
    cutoff: float = 4.0,
    target_valence: float = 1.0,
) -> torch.Tensor:
    """Per-crystal BV-mismatch aggregate features.

    Returns: (B, 8) — [mean, std, min, max, p25, p50, p75, n_li_sites]
    aggregated across Li sites in each crystal. n_li_sites is normalized
    (atan(n)/π × 2) into [0, 1).
    """
    device = frac_pos.device
    feats = torch.zeros(num_graphs, 8, device=device)

    # Build per-graph cartesian positions
    # frac_pos: (N, 3); cell[batch_idx]: (N, 3, 3) → cartesian = frac @ cell
    cell_per_atom = cell[batch_idx]                         # (N, 3, 3)
    cart = torch.einsum("nj,njk->nk", frac_pos, cell_per_atom)  # (N, 3)

    # 27-image shifts (no extra periodic image search beyond ±1)
    shifts_int = torch.tensor(
        [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=torch.float32, device=device,
    )                                                       # (27, 3)

    for g in range(num_graphs):
        mask_g = (batch_idx == g)
        z_g = atom_z[mask_g]
        cart_g = cart[mask_g]
        cell_g = cell[g]                                    # (3, 3)
        n_atoms = int(mask_g.sum())
        if n_atoms < 2:
            continue
        li_mask = (z_g == mobile_z)
        if not li_mask.any():
            continue
        anion_mask = ~li_mask & (z_g > 0)
        if not anion_mask.any():
            continue

        li_cart = cart_g[li_mask]                           # (n_li, 3)
        anion_cart = cart_g[anion_mask]                     # (n_a, 3)
        anion_z = z_g[anion_mask]                           # (n_a,)

        # Cartesian shifts
        shifts_cart = shifts_int @ cell_g                   # (27, 3)

        # Vectorized per-Li BV sum (single autograd graph, no Python loop)
        # disp: (n_li, n_a, 27, 3)
        disp = (
            anion_cart[None, :, None, :]
            + shifts_cart[None, None, :, :]
            - li_cart[:, None, None, :]
        )
        d = torch.linalg.norm(disp, dim=-1)                  # (n_li, n_a, 27)
        in_range = (d < cutoff).float()
        r0 = bv.r0[anion_z].view(1, -1, 1)                   # broadcast to (1, n_a, 1)
        b_ = bv.b[anion_z].view(1, -1, 1)
        contrib = torch.exp((r0 - d) / b_) * in_range
        v_per_li = contrib.sum(dim=(1, 2))                   # (n_li,)
        mismatch = (v_per_li - target_valence).abs()

        feats[g, 0] = mismatch.mean()
        feats[g, 1] = mismatch.std() if mismatch.numel() > 1 else torch.zeros((), device=device)
        feats[g, 2] = mismatch.min()
        feats[g, 3] = mismatch.max()
        # percentiles
        sorted_m, _ = torch.sort(mismatch)
        n_l = sorted_m.numel()
        feats[g, 4] = sorted_m[n_l // 4]
        feats[g, 5] = sorted_m[n_l // 2]
        feats[g, 6] = sorted_m[(3 * n_l) // 4]
        # n_li normalized
        feats[g, 7] = torch.atan(torch.tensor(float(n_l), device=device)) / (math.pi / 2)

    return feats
