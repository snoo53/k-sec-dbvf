"""Geometric / bond-valence-inspired structural features for ionic
conductivity prediction.

These features explicitly capture Li-site geometry, Li-Li connectivity,
and framework density — the three physics-level descriptors most
directly tied to σ that are NOT captured by k-SEC's Fourier features
or Magpie's composition statistics.

Feature vector is a fixed 20-dim float32 array. Entries with NaN/failed
computation are replaced by zero and masked via the final flag element.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure

log = logging.getLogger(__name__)


GEOMETRIC_FEATURE_DIM = 25


_GEOMETRIC_NAMES = [
    # Li-site coordination (5)
    "li_cn_mean",            # average coordination number over Li sites (CrystalNN)
    "li_cn_std",
    "li_neighbor_d_mean",    # mean Li-X bond distance (Å)
    "li_neighbor_d_min",     # shortest Li-X bond (bottleneck proxy, Å)
    "li_neighbor_d_std",
    # Li-Li connectivity (5)
    "li_li_min",             # minimum Li-Li distance (Å, percolation proxy)
    "li_li_median",          # median Li-Li distance
    "li_li_max",
    "li_li_count_within_4A", # number of Li-Li pairs within 4 Å (hop count)
    "li_volume_per_li",      # cell volume / N_Li (Å³/Li, inverse-density)
    # Framework features (5)
    "framework_density",     # non-Li atoms / cell volume (Å⁻³)
    "frac_li",               # N_Li / N_total
    "frac_anion",            # N of O/S/Cl/Br/I/N/F / N_total (anion content)
    "frac_tm",               # N of transition metals / N_total
    "anion_charge_avg",      # average assumed anion valence proxy
    # Overall geometry (4)
    "cell_volume_per_atom",  # V / N_atoms (Å³/atom, framework compactness)
    "n_atoms",               # N_total
    "n_li",                  # N_Li
    "aspect_ratio",          # max(a,b,c) / min(a,b,c) — anisotropy
    # BV-pathway / percolation (5) — added for Phase A1
    "bottleneck_radius_min", # min radius of Li-Li midpoint to nearest non-Li (Å, channel constriction)
    "bottleneck_radius_mean", # mean across Li-Li midpoints
    "li_percolation_3D",     # 1 if Li-Li graph (cutoff 4 Å) percolates in all 3 dims, 0 otherwise
    "li_percolation_dim",    # estimated channel dimensionality 0..3
    "bv_strain_proxy",       # |sum exp((r0-d)/b) − 1| averaged over Li sites; 0 = Li in well-fitting site
    # Quality flag (1)
    "feat_valid",            # 1.0 if computed successfully, 0.0 if any step failed
]


assert len(_GEOMETRIC_NAMES) == GEOMETRIC_FEATURE_DIM


# Element classes
_ANIONS = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}
_TM = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
       "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
       "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}


def geometric_feature_names() -> list[str]:
    return list(_GEOMETRIC_NAMES)


def _site_symbol(site) -> str:
    try:
        return site.specie.symbol
    except AttributeError:
        return max(site.species, key=site.species.get).symbol


def featurize_structure(structure: "Structure",
                        mobile_ion: str = "Li") -> np.ndarray:
    """Compute the 20-dim geometric/BV-inspired feature vector.

    On any failure, returns zeros with feat_valid=0. Otherwise the last
    element is 1.0 and the rest are the computed values.
    """
    out = np.zeros(GEOMETRIC_FEATURE_DIM, dtype=np.float32)
    try:
        syms = [_site_symbol(s) for s in structure]
        n_total = len(syms)
        if n_total == 0:
            return out

        frac_coords = np.array([s.frac_coords for s in structure], dtype=np.float32) % 1.0
        lattice_M = structure.lattice.matrix.astype(np.float32)
        volume = float(structure.lattice.volume)

        li_idx = [i for i, sym in enumerate(syms) if sym == mobile_ion]
        n_li = len(li_idx)
        n_anion = sum(1 for s in syms if s in _ANIONS)
        n_tm = sum(1 for s in syms if s in _TM)

        # Li-site coordination (use simple distance cutoff to avoid expensive CrystalNN)
        # Find all atoms within 3.5 Å (typical Li-O/S bond = 2.0-3.0 Å) using periodic images
        cn_list, d_means, d_mins, d_stds = [], [], [], []
        if n_li > 0:
            # Compute min-image distances from each Li to all non-Li atoms (with periodic images)
            for i in li_idx:
                li_cart = frac_coords[i] @ lattice_M
                # Gather candidate neighbor sites — only non-Li within 3.5 Å
                dists = []
                for j, sym_j in enumerate(syms):
                    if j == i or sym_j == mobile_ion:
                        continue
                    # brute periodic image search over ±1 shifts
                    nb_cart = frac_coords[j] @ lattice_M
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                shift = np.array([dx, dy, dz], dtype=np.float32) @ lattice_M
                                d = float(np.linalg.norm(li_cart - (nb_cart + shift)))
                                if d < 3.5:
                                    dists.append(d)
                if dists:
                    cn_list.append(len(dists))
                    d_means.append(float(np.mean(dists)))
                    d_mins.append(float(np.min(dists)))
                    d_stds.append(float(np.std(dists)))

        if cn_list:
            out[0] = float(np.mean(cn_list))
            out[1] = float(np.std(cn_list))
            out[2] = float(np.mean(d_means))
            out[3] = float(np.min(d_mins))
            out[4] = float(np.mean(d_stds))

        # Li-Li distances
        if n_li >= 2:
            li_cart = np.array([frac_coords[i] @ lattice_M for i in li_idx], dtype=np.float32)
            dij = []
            # include periodic images of length-1 shift
            for a_i in range(len(li_idx)):
                for b_i in range(a_i + 1, len(li_idx)):
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                shift = np.array([dx, dy, dz], dtype=np.float32) @ lattice_M
                                d = float(np.linalg.norm(li_cart[a_i] - (li_cart[b_i] + shift)))
                                dij.append(d)
            if dij:
                dij_arr = np.array(dij, dtype=np.float32)
                out[5] = float(dij_arr.min())
                out[6] = float(np.median(dij_arr))
                out[7] = float(dij_arr.max())
                out[8] = float(np.sum(dij_arr < 4.0))
        if n_li > 0 and volume > 0:
            out[9] = float(volume / n_li)

        # Framework features
        if volume > 0:
            out[10] = (n_total - n_li) / volume
        out[11] = n_li / max(n_total, 1)
        out[12] = n_anion / max(n_total, 1)
        out[13] = n_tm / max(n_total, 1)
        out[14] = -2.0 if any(s in {"O", "S", "Se"} for s in syms) else (-1.0 if any(s in {"F", "Cl", "Br", "I"} for s in syms) else 0.0)

        # Overall geometry
        if n_total > 0 and volume > 0:
            out[15] = volume / n_total
        out[16] = n_total
        out[17] = n_li
        a = float(structure.lattice.a)
        b = float(structure.lattice.b)
        c = float(structure.lattice.c)
        out[18] = max(a, b, c) / max(min(a, b, c), 1e-6)

        # ----- Phase A1 BV/percolation features -----
        # Bottleneck radius: min distance from each Li-Li midpoint to the
        # nearest non-Li atom (channel constriction).
        if n_li >= 2 and (n_total - n_li) > 0:
            li_cart = np.array([frac_coords[i] @ lattice_M for i in li_idx], dtype=np.float32)
            non_li_cart = np.array([
                frac_coords[i] @ lattice_M for i, sym in enumerate(syms) if sym != mobile_ion
            ], dtype=np.float32)
            shifts = np.array(
                [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                dtype=np.float32,
            ) @ lattice_M  # (27, 3)

            bottleneck_radii = []
            for ai in range(len(li_idx)):
                for bi_ in range(ai + 1, len(li_idx)):
                    midpoint = 0.5 * (li_cart[ai] + li_cart[bi_])
                    # Distance from midpoint to all non-Li atoms with periodic images
                    diffs = non_li_cart[None, :, :] + shifts[:, None, :] - midpoint[None, None, :]
                    d2 = np.sum(diffs * diffs, axis=-1)
                    bottleneck_radii.append(float(np.sqrt(d2.min())))
            if bottleneck_radii:
                out[19] = float(np.min(bottleneck_radii))
                out[20] = float(np.mean(bottleneck_radii))

            # Li percolation: build adjacency graph from Li-Li hops < 4 Å,
            # check connectivity in 3D using lattice translations.
            adj_thresh = 4.0
            try:
                # Connected components via union-find
                parent = list(range(n_li))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]; x = parent[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx != ry: parent[rx] = ry
                connects_in_dim = [False, False, False]  # x, y, z direction crossing
                for ai in range(len(li_idx)):
                    for bi_ in range(ai + 1, len(li_idx)):
                        diffs = li_cart[bi_] + shifts - li_cart[ai]
                        d = np.sqrt(np.sum(diffs * diffs, axis=-1))
                        for k_, dist in enumerate(d):
                            if dist < adj_thresh:
                                union(ai, bi_)
                                shift_vec = shifts[k_]
                                # Identify if connection crosses a periodic image
                                # by checking which dimension dominates the shift
                                if k_ != 13:  # not the (0,0,0) image
                                    sx = int(round(np.dot(shift_vec, np.linalg.inv(lattice_M)[:, 0])))
                                    sy = int(round(np.dot(shift_vec, np.linalg.inv(lattice_M)[:, 1])))
                                    sz = int(round(np.dot(shift_vec, np.linalg.inv(lattice_M)[:, 2])))
                                    if sx != 0: connects_in_dim[0] = True
                                    if sy != 0: connects_in_dim[1] = True
                                    if sz != 0: connects_in_dim[2] = True
                # Count distinct components
                roots = {find(i) for i in range(n_li)}
                largest = max(sum(1 for i in range(n_li) if find(i) == r) for r in roots)
                # Percolating only if largest component spans all 3 dims
                perco_dim = sum(connects_in_dim)
                out[21] = 1.0 if perco_dim == 3 else 0.0
                out[22] = float(perco_dim)
            except Exception:
                pass

            # BV strain proxy: for each Li site, sum exp((r0-d)/b) over
            # nearby anions; deviation from +1 indicates structural strain.
            # Tabulated r0/b values for Li-anion pairs (from Brown 2002):
            r0_b = {"O": (1.466, 0.37), "S": (1.85, 0.40), "Se": (1.93, 0.40),
                    "F": (1.36, 0.37), "Cl": (1.79, 0.40), "Br": (1.92, 0.40),
                    "I": (2.07, 0.40), "N": (1.61, 0.37)}
            li_strains = []
            for i in li_idx:
                li_pos = frac_coords[i] @ lattice_M
                bvs = 0.0
                for j, sym_j in enumerate(syms):
                    if sym_j not in r0_b:
                        continue
                    r0, bb = r0_b[sym_j]
                    nb_pos = frac_coords[j] @ lattice_M
                    diffs = nb_pos + shifts - li_pos
                    d = np.sqrt(np.sum(diffs * diffs, axis=-1))
                    d_min = float(d[d > 1e-3].min()) if (d > 1e-3).any() else 0.0
                    if 0 < d_min < 4.0:
                        bvs += float(np.exp((r0 - d_min) / bb))
                li_strains.append(abs(bvs - 1.0))
            if li_strains:
                out[23] = float(np.mean(li_strains))

        out[24] = 1.0
        return out

    except Exception as exc:  # noqa: BLE001
        log.debug("geometric featurize failed: %s", exc)
        return np.zeros(GEOMETRIC_FEATURE_DIM, dtype=np.float32)
