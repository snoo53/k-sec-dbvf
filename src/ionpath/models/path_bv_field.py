"""Site-resolved DBVF with differentiable Li-migration-path integration.

Extends bond_valence_field.LearnableBVParams: instead of evaluating the
BV field only at Li sites (8-d aggregate), also evaluate along straight-
line paths between every Li-Li pair within a cutoff, then take a
differentiable soft-max over the path to extract the migration-saddle
BV mismatch (the actual physics of the migration barrier).

Per-crystal output (12-d):
   - barrier_min     # easiest path's barrier (eV-equivalent BV mismatch)
   - barrier_mean
   - barrier_std
   - barrier_max
   - barrier_p25/p50/p75
   - n_paths         # number of Li-Li pairs within cutoff (atan-normalized)
   - n_low_paths     # fraction of paths with barrier < threshold
   - site_mismatch_mean  # the original site-level mismatch (mean)
   - site_mismatch_min
   - site_mismatch_max

The path-saddle calculation uses a smooth soft-max:
    saddle_barrier_pair ≈ (1/τ) * logsumexp(τ * U(t)) for t in path-points

with τ a learnable inverse temperature so the model can interpolate
between mean (low τ) and hard max (high τ).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .bond_valence_field import LearnableBVParams


PATH_BV_FEATURE_DIM = 12


class LearnablePathBVParams(LearnableBVParams):
    """LearnableBVParams + a learnable inverse-temperature for the
    soft-max path-saddle reduction."""

    def __init__(self, num_species: int = 100, mobile_z: int = 3,
                 init_tau: float = 4.0):
        super().__init__(num_species=num_species, mobile_z=mobile_z)
        self.tau_raw = nn.Parameter(torch.tensor(self._inv_softplus(init_tau)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.tau_raw)


def compute_path_bv_features(
    bv: LearnablePathBVParams,
    atom_z: torch.Tensor,            # (N,) atomic numbers
    frac_pos: torch.Tensor,          # (N, 3) fractional coords
    cell: torch.Tensor,              # (B, 3, 3) per-graph lattice
    batch_idx: torch.Tensor,         # (N,) sample index
    num_graphs: int,
    mobile_z: int = 3,
    cutoff_pair: float = 5.0,        # Li-Li pair cutoff (Å)
    cutoff_anion: float = 4.0,       # anion-distance cutoff for BV sum (Å)
    n_path_points: int = 7,          # discretization (incl. endpoints)
    target_valence: float = 1.0,
    low_barrier_threshold: float = 0.5,  # for "n_low_paths" stat
) -> torch.Tensor:
    """Per-crystal site + path BV features. Returns (B, 12).
    Fully differentiable through the BV r0/b/tau parameters and through
    the soft-max path-saddle.
    """
    device = frac_pos.device
    feats = torch.zeros(num_graphs, PATH_BV_FEATURE_DIM, device=device)

    # Cartesian positions
    cell_per_atom = cell[batch_idx]                                  # (N, 3, 3)
    cart = torch.einsum("nj,njk->nk", frac_pos, cell_per_atom)        # (N, 3)

    # Periodic image shifts (±1)
    shifts_int = torch.tensor(
        [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=torch.float32, device=device,
    )                                                                # (27, 3)

    # Path interpolation parameter (incl. both endpoints)
    t_vec = torch.linspace(0.0, 1.0, n_path_points, device=device)   # (P,)

    tau = bv.tau

    for g in range(num_graphs):
        mask_g = (batch_idx == g)
        z_g = atom_z[mask_g]
        cart_g = cart[mask_g]
        cell_g = cell[g]                                              # (3, 3)
        n_atoms = int(mask_g.sum())
        if n_atoms < 2:
            continue
        li_mask = (z_g == mobile_z)
        if not li_mask.any():
            continue
        anion_mask = ~li_mask & (z_g > 0)
        if not anion_mask.any():
            continue

        li_cart = cart_g[li_mask]                                     # (L, 3)
        anion_cart = cart_g[anion_mask]                               # (A, 3)
        anion_z = z_g[anion_mask]                                     # (A,)
        L = li_cart.shape[0]
        A = anion_cart.shape[0]

        shifts_cart = shifts_int @ cell_g                             # (27, 3)

        # ---- Site-level BV at each Li (vectorized) ----
        # disp_site: (L, A, 27, 3)
        disp_site = (
            anion_cart[None, :, None, :]
            + shifts_cart[None, None, :, :]
            - li_cart[:, None, None, :]
        )
        d_site = torch.linalg.norm(disp_site, dim=-1)                  # (L, A, 27)
        in_range_site = (d_site < cutoff_anion).float()
        r0 = bv.r0[anion_z].view(1, -1, 1)
        b_ = bv.b[anion_z].view(1, -1, 1)
        contrib_site = torch.exp((r0 - d_site) / b_) * in_range_site
        V_site = contrib_site.sum(dim=(1, 2))                          # (L,)
        U_site = (V_site - target_valence).abs()                       # (L,)

        feats[g, 9] = U_site.mean()
        feats[g, 10] = U_site.min()
        feats[g, 11] = U_site.max()

        # ---- Find Li-Li pairs within cutoff (with periodic images) ----
        # diff: (L, L, 27, 3); use upper triangle to avoid double-count
        diff_lili = li_cart[:, None, None, :] - li_cart[None, :, None, :] - shifts_cart[None, None, :, :]
        d_lili = torch.linalg.norm(diff_lili, dim=-1)                  # (L, L, 27)
        # Build a mask: (i, j, k) valid iff i < j (or i == j with k != 13) and d < cutoff_pair
        idx_i, idx_j, idx_k = torch.meshgrid(
            torch.arange(L, device=device),
            torch.arange(L, device=device),
            torch.arange(27, device=device),
            indexing="ij",
        )
        # Self-image case (i == j, k == 13) is the on-site, exclude.
        # Symmetry: we also want each unordered pair only once.
        same_image = (idx_k == 13)
        valid_mask = (
            (d_lili < cutoff_pair)
            & ~(same_image & (idx_i == idx_j))
            & ((idx_i < idx_j) | ((idx_i == idx_j) & ~same_image))
        )
        valid_idx = valid_mask.nonzero(as_tuple=False)                 # (P, 3) → (i, j, k)
        if valid_idx.numel() == 0:
            continue
        P = valid_idx.shape[0]

        # ---- Build path points for each valid pair ----
        # start: li_cart[i]; end: li_cart[j] + shifts_cart[k]
        i_idx, j_idx, k_idx = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        start = li_cart[i_idx]                                          # (P, 3)
        end = li_cart[j_idx] + shifts_cart[k_idx]                       # (P, 3)
        # Path points: (P, n_path_points, 3)
        path = start[:, None, :] + (end - start)[:, None, :] * t_vec[None, :, None]

        # ---- Evaluate BV field at every path point (vectorized) ----
        # path: (P, n_path_points, 3); want d to all anions × shifts
        # disp_path: (P, n_path_points, A, 27, 3)
        disp_path = (
            anion_cart[None, None, :, None, :]
            + shifts_cart[None, None, None, :, :]
            - path[:, :, None, None, :]
        )
        d_path = torch.linalg.norm(disp_path, dim=-1)                   # (P, P_pt, A, 27)
        in_range = (d_path < cutoff_anion).float()
        r0_p = bv.r0[anion_z].view(1, 1, -1, 1)
        b_p = bv.b[anion_z].view(1, 1, -1, 1)
        contrib_path = torch.exp((r0_p - d_path) / b_p) * in_range
        V_path = contrib_path.sum(dim=(2, 3))                            # (P, P_pt)
        U_path = (V_path - target_valence).abs()                         # (P, P_pt)

        # ---- Soft-max over path → barrier per pair ----
        # saddle_barrier ≈ (1/τ) * logsumexp(τ * U_path)
        saddle = (1.0 / tau) * torch.logsumexp(tau * U_path, dim=1)      # (P,)

        feats[g, 0] = saddle.min()
        feats[g, 1] = saddle.mean()
        feats[g, 2] = saddle.std() if saddle.numel() > 1 else torch.zeros((), device=device)
        feats[g, 3] = saddle.max()
        sorted_b, _ = torch.sort(saddle)
        Pn = sorted_b.numel()
        feats[g, 4] = sorted_b[Pn // 4]
        feats[g, 5] = sorted_b[Pn // 2]
        feats[g, 6] = sorted_b[(3 * Pn) // 4]
        feats[g, 7] = torch.atan(torch.tensor(float(P), device=device)) / (math.pi / 2)
        feats[g, 8] = (saddle < low_barrier_threshold).float().mean()

    return feats
