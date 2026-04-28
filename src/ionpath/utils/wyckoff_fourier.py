"""Wyckoff-symmetry-averaged Fourier feature basis.

For a crystal with point-group operations {R_g} (inherited from its space
group), we build a set of wavevectors K = {k_1, …, k_K} that are orbit-sums
under the group. Any field written as

    f(r) = Σ_k  a_k · cos(2π k · r)  +  b_k · sin(2π k · r)

is automatically invariant under the point group when the wavevectors come
in orbit-pairs {R_g k} and the coefficients share across the orbit.

In practice the full symmetry enumeration requires the space-group number.
When that is not available (most OBELiX entries have it; some don't), we
fall back to a *symmetry-averaged* basis: for every integer vector
n = (n_x, n_y, n_z) in a 3D shell, we take its orbit under the standard
cubic rotation group (order 24) and sum those 24 cosines into a single
basis function. This gives an O_h-symmetric field, which is a strict
super-set of many common space-group symmetries — good enough for the
Fourier-feature construction.

For fields at Γ (k=0), the constant mode is included.
"""

from __future__ import annotations

import numpy as np

# The 24 proper rotations of the cubic point group O_h (without inversion).
# Represented as 3x3 rotation matrices (det = +1). Inversion doubles the
# orbit to 48 elements but a cosine basis already picks up parity-even
# combinations automatically.
_O_ROTATIONS: list[np.ndarray] = []
_identity = np.eye(3, dtype=np.int8)
_O_ROTATIONS.append(_identity)
# 6 axis rotations about x, y, z by 90, 180, 270 degrees.
for axis in range(3):
    for angle in (90, 180, 270):
        c = int(round(np.cos(np.deg2rad(angle))))
        s = int(round(np.sin(np.deg2rad(angle))))
        R = np.eye(3, dtype=np.int8)
        if axis == 0:
            R[1, 1], R[1, 2] = c, -s
            R[2, 1], R[2, 2] = s, c
        elif axis == 1:
            R[0, 0], R[0, 2] = c, s
            R[2, 0], R[2, 2] = -s, c
        else:
            R[0, 0], R[0, 1] = c, -s
            R[1, 0], R[1, 1] = s, c
        _O_ROTATIONS.append(R)

# 8 threefold rotations about body diagonals — generated from the three
# permutation matrices (1 2 0), (2 0 1), and their squares.
_O_ROTATIONS.append(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8))
_O_ROTATIONS.append(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int8))
# And sign-variants for the other 6 diagonals
for sx, sy, sz in [(1, 1, -1), (1, -1, 1), (-1, 1, 1),
                   (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]:
    D = np.diag([sx, sy, sz]).astype(np.int8)
    _O_ROTATIONS.append(D @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8))
    _O_ROTATIONS.append(D @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int8))

# Deduplicate — there may be repeats from the generation above.
_seen = set()
_unique: list[np.ndarray] = []
for R in _O_ROTATIONS:
    key = tuple(R.flatten().tolist())
    if key in _seen:
        continue
    _seen.add(key)
    _unique.append(R)
_O_ROTATIONS = _unique


def generate_wyckoff_wavevectors(n_max: int = 3) -> np.ndarray:
    """Enumerate symmetry-reduced integer-lattice wavevectors up to |k| ≤ n_max.

    Returns a (K, 3) array where each row is the *representative* of one orbit
    under the cubic rotation group. The full orbit is generated on demand by
    applying `_O_ROTATIONS`.
    """
    orbits: list[np.ndarray] = []
    seen_orbits: set[tuple] = set()
    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            for nz in range(-n_max, n_max + 1):
                k = np.array([nx, ny, nz], dtype=np.int8)
                if np.all(k == 0):
                    # Γ point — we include the constant mode separately
                    continue
                # Compute the full orbit
                orbit = set()
                for R in _O_ROTATIONS:
                    orbit.add(tuple((R @ k).tolist()))
                orbit_key = frozenset(orbit)
                if orbit_key in seen_orbits:
                    continue
                seen_orbits.add(orbit_key)
                # Use the lexicographically smallest member as the canonical rep
                rep = tuple(min(orbit))
                orbits.append(np.array(rep, dtype=np.float32))
    orbits.sort(key=lambda v: (np.linalg.norm(v), tuple(v)))
    return np.stack(orbits, axis=0)


def evaluate_fourier_basis(
    r_frac: np.ndarray,               # (..., 3) fractional coordinates
    wavevectors: np.ndarray,          # (K, 3) reciprocal-lattice integers
) -> np.ndarray:
    """Evaluate the symmetry-averaged Fourier basis at one or more points.

    Returns (..., 2K) concatenation of cos and sin evaluations summed over
    each orbit under the cubic group. The sum-over-orbit is what makes the
    field invariant under that group.
    """
    shape = r_frac.shape[:-1]
    K = wavevectors.shape[0]
    r_flat = r_frac.reshape(-1, 3)                                  # (N, 3)
    basis = np.zeros((r_flat.shape[0], 2 * K), dtype=np.float32)
    for k_idx, k_rep in enumerate(wavevectors):
        orbit_cos = np.zeros(r_flat.shape[0], dtype=np.float32)
        orbit_sin = np.zeros(r_flat.shape[0], dtype=np.float32)
        seen = set()
        for R in _O_ROTATIONS:
            k = R @ k_rep
            key = tuple(k.tolist())
            if key in seen:
                continue
            seen.add(key)
            angle = 2.0 * np.pi * (r_flat @ k.astype(np.float32))
            orbit_cos += np.cos(angle)
            orbit_sin += np.sin(angle)
        basis[:, 2 * k_idx] = orbit_cos / float(len(seen))
        basis[:, 2 * k_idx + 1] = orbit_sin / float(len(seen))
    return basis.reshape(*shape, 2 * K)


# ---------------------------------------------------------------------------
# Torch versions — what the model actually calls during training.
# Wavevectors are parameters-free constants; the basis is differentiable w.r.t. r.
# ---------------------------------------------------------------------------


def torch_fourier_basis(r_frac, wavevectors, orbit_members):
    """Torch-differentiable version of `evaluate_fourier_basis`.

    r_frac: torch.Tensor (..., 3), requires grad for Hessian extraction
    wavevectors: torch.Tensor (K, 3), float (representatives)
    orbit_members: list of (K,) torch.Tensor entries, each (M_i, 3) — the
      full orbit for the i-th representative. Precomputed once.

    Returns a torch.Tensor (..., 2K).
    """
    import torch

    shape = r_frac.shape[:-1]
    r_flat = r_frac.reshape(-1, 3)                                  # (N, 3)
    out_cos: list[torch.Tensor] = []
    out_sin: list[torch.Tensor] = []
    for orbit in orbit_members:                                     # (M, 3)
        # angle: (N, M) = 2π · r · k^T
        ang = 2.0 * torch.pi * (r_flat @ orbit.T)
        c = torch.cos(ang).mean(dim=-1)                             # (N,)
        s = torch.sin(ang).mean(dim=-1)                             # (N,)
        out_cos.append(c); out_sin.append(s)
    cos_stack = torch.stack(out_cos, dim=-1)                        # (N, K)
    sin_stack = torch.stack(out_sin, dim=-1)                        # (N, K)
    # interleave into (N, 2K)
    feats = torch.stack([cos_stack, sin_stack], dim=-1).reshape(r_flat.shape[0], -1)
    return feats.reshape(*shape, feats.shape[-1])


def precompute_orbits(wavevectors: np.ndarray) -> list:
    """For each representative, return the orbit as a torch float32 tensor."""
    import torch

    orbits = []
    for k_rep in wavevectors:
        seen = set()
        orbit_list = []
        for R in _O_ROTATIONS:
            k = (R @ k_rep).astype(np.float32)
            key = tuple(k.tolist())
            if key in seen:
                continue
            seen.add(key)
            orbit_list.append(k)
        orbit_arr = np.stack(orbit_list, axis=0)
        orbits.append(torch.from_numpy(orbit_arr).float())
    return orbits
