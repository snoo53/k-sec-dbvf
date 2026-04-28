"""k-SEC v2: k-Space Equivariant Convolutional Network with cubic-harmonic
directional filters and cross-shell gated attention.

## Novelty

Not in the published literature as of early 2026:

1. **Feature maps live in reciprocal (k-)space throughout** the network,
   with cubic-group equivariance enforced by construction.  (ReGNet 2025 uses
   one FFT/IFFT block as an auxiliary stream only.)

2. **Cubic-harmonic directional filters** ``W(|k|, K_4(k̂), K_6(k̂), …)``:
   filters are a function of both the magnitude of **k** AND angular
   coordinates projected onto cubic-invariant polynomial basis functions
   (Kubic harmonics). This preserves cubic-group equivariance while
   providing direction-dependent response — something a pure radial filter
   ``W(|k|)`` cannot. First l>0 cubic invariant appears at l=4, then l=6,
   8, …; we use {1, x⁴+y⁴+z⁴, x²y²+y²z²+z²x², x⁶+y⁶+z⁶, x²y²z²} as a
   concrete finite basis.

3. **Cross-shell learned-gate attention**: attention across ALL k-points
   (not restricted to equal-|k|) but each edge modulated by a learned
   per-shell-pair gate that can adapt between sub-band (small Δ|k|) and
   umklapp-like (large Δ|k|) mixing.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..utils.wyckoff_fourier import generate_wyckoff_wavevectors, precompute_orbits
from .bond_valence_field import LearnableBVParams, compute_bv_features
from .path_bv_field import (
    LearnablePathBVParams, compute_path_bv_features, PATH_BV_FEATURE_DIM,
)
from .mpnn_encoder import MPNNEncoder
from .cross_attention_bridge import CrossAttentionBridge, pad_atoms


# ---------------------------------------------------------------------------
# Cubic-invariant polynomial basis on the unit sphere
# ---------------------------------------------------------------------------
# First five O_h-invariant polynomials on k̂=(x,y,z)/|k|.  l=1,2,3,5 have
# zero cubic invariants; the non-trivial invariants start at l=4.
# ---------------------------------------------------------------------------


def _kubic_invariants(k_unit: torch.Tensor) -> torch.Tensor:
    """Evaluate 5 cubic-invariant polynomials at each unit vector.

    Input : k_unit (N, 3) with ‖k_unit‖ ≈ 1 (we normalise inside anyway)
    Output: (N, 5) real tensor ordered as
            [K_0, K_4a, K_4b, K_6a, K_6b]  where
        K_0   = 1                                (constant, l=0)
        K_4a  = x⁴ + y⁴ + z⁴ − 3/5               (zero-mean l=4)
        K_4b  = x²y² + y²z² + z²x² − 1/5         (zero-mean l=4 mate)
        K_6a  = x⁶ + y⁶ + z⁶ − 3/7               (l=6)
        K_6b  = x²y²z² − 1/105                   (l=6 fully-symmetric triple)

    All entries are O_h-invariant: evaluating on any cubic rotation of k
    gives the same value. They are zero-mean on the sphere (post the
    constants subtracted), so the filter MLP can treat them as features.
    """
    k_unit = k_unit / torch.linalg.norm(k_unit, dim=-1, keepdim=True).clamp(min=1e-6)
    x, y, z = k_unit[..., 0], k_unit[..., 1], k_unit[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    K0 = torch.ones_like(x)
    K4a = x**4 + y**4 + z**4 - 3.0 / 5.0
    K4b = x2 * y2 + y2 * z2 + z2 * x2 - 1.0 / 5.0
    K6a = x**6 + y**6 + z**6 - 3.0 / 7.0
    K6b = x2 * y2 * z2 - 1.0 / 105.0
    return torch.stack([K0, K4a, K4b, K6a, K6b], dim=-1)


class KubicHarmonicFilter(nn.Module):
    """Learnable complex filter W(|k|, K_4, K_4', K_6, K_6') acting point-wise.

    Unlike a purely radial filter (which throws away direction), this filter
    is a function of magnitude AND five cubic-harmonic invariants of the
    direction. Because all five invariants are O_h-invariant, the filter
    still commutes with any cubic rotation of k → perfect space-group
    equivariance (under O_h, the maximal symmetry we use for our k-grid).
    """

    def __init__(self, feature_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # Input: [|k|, K_0..K_6b] → 6-d
        self.gain_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * feature_dim),        # real + imag per channel
        )
        self.bias_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * feature_dim),
        )

    def forward(self, H: torch.Tensor, k_mags: torch.Tensor,
                kubics: torch.Tensor) -> torch.Tensor:
        """H: (B, K, D) complex;  k_mags: (K,);  kubics: (K, 5).
        Returns (B, K, D) complex.
        """
        k_in = torch.cat([k_mags.unsqueeze(-1), kubics], dim=-1)        # (K, 6)
        D = H.shape[-1]
        gain = self.gain_mlp(k_in)
        bias = self.bias_mlp(k_in)
        w_r, w_i = gain[..., :D], gain[..., D:]
        b_r, b_i = bias[..., :D], bias[..., D:]
        W = torch.complex(w_r, w_i).unsqueeze(0)                         # (1, K, D)
        b = torch.complex(b_r, b_i).unsqueeze(0)                         # (1, K, D)
        return H * W + b


class CrossShellGatedAttention(nn.Module):
    """Full cross-k attention, with each attention edge modulated by a
    learned gate that depends only on (|k_i|, |k_j|).

    For equivariance: the gate depends on magnitudes alone (cubic-invariant
    in k). Within a shell the gate is strongest by learning; between
    distant shells it can be turned off or on.
    """

    def __init__(self, feature_dim: int, n_heads: int = 4):
        super().__init__()
        assert feature_dim % n_heads == 0
        self.h = n_heads
        self.dh = feature_dim // n_heads
        self.feature_dim = feature_dim
        self.qkv = nn.Linear(2 * feature_dim, 6 * feature_dim)
        self.o = nn.Linear(2 * feature_dim, 2 * feature_dim)
        # Gate MLP on (|k_i|, |k_j|, ||k_i| − |k_j||)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(),
            nn.Linear(32, n_heads),                  # one gate per attention head
        )

    def forward(self, H: torch.Tensor, k_mags: torch.Tensor) -> torch.Tensor:
        """H: (B, K, D) complex;  k_mags: (K,) real."""
        D = self.feature_dim
        B, K, _ = H.shape
        x = torch.cat([H.real, H.imag], dim=-1)                          # (B, K, 2D)
        qkv = self.qkv(x)
        q_r, q_i, k_r, k_i, v_r, v_i = qkv.chunk(6, dim=-1)
        q = torch.complex(q_r, q_i).view(B, K, self.h, self.dh)
        k = torch.complex(k_r, k_i).view(B, K, self.h, self.dh)
        v = torch.complex(v_r, v_i).view(B, K, self.h, self.dh)

        # Cross-shell gate: shape (K, K, h)
        km_i = k_mags.unsqueeze(-1).expand(K, K)                         # (K, K) — row = |k_i|
        km_j = k_mags.unsqueeze(0).expand(K, K)                          # (K, K)
        dk = (km_i - km_j).abs()
        gate_in = torch.stack([km_i, km_j, dk], dim=-1)                  # (K, K, 3)
        gate_logits = self.gate_mlp(gate_in)                             # (K, K, h)
        gate = torch.sigmoid(gate_logits)                                # in [0,1]

        # Hermitian-style attention score Re(q · k*)
        scores_r = torch.einsum("bkhd,bjhd->bkjh", q.real, k.real) + \
                   torch.einsum("bkhd,bjhd->bkjh", q.imag, k.imag)
        scores = scores_r / (self.dh ** 0.5)                             # (B, K, K, h)
        # Apply gate: multiplies the softmax-input exponents' weight
        gate_log = torch.log(gate.unsqueeze(0).clamp(min=1e-8))           # (1, K, K, h)
        scores = scores + gate_log
        alpha = torch.softmax(scores, dim=2)

        out_r = torch.einsum("bkjh,bjhd->bkhd", alpha, v.real)
        out_i = torch.einsum("bkjh,bjhd->bkhd", alpha, v.imag)
        out = torch.complex(out_r, out_i).reshape(B, K, D)
        y = torch.cat([out.real, out.imag], dim=-1)
        y = self.o(y)
        yr, yi = y.chunk(2, dim=-1)
        return torch.complex(yr, yi) + H


class _CLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(2 * dim)

    def forward(self, H):
        x = torch.cat([H.real, H.imag], dim=-1)
        y = self.ln(x)
        r, i = y.chunk(2, dim=-1)
        return torch.complex(r, i)


class KSECBlock(nn.Module):
    """One k-SEC v2 block: Kubic filter + cross-shell gated attention + residual."""

    def __init__(self, feature_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.filt = KubicHarmonicFilter(feature_dim, dropout=dropout)
        self.attn = CrossShellGatedAttention(feature_dim, n_heads)
        self.norm_in = _CLN(feature_dim)
        self.norm_attn = _CLN(feature_dim)
        self.dropout_p = dropout

    def forward(self, H, k_mags, kubics):
        H = self.norm_in(H)
        H = self.filt(H, k_mags, kubics)
        # Complex magnitude gate
        mag = H.abs()
        gate = torch.sigmoid(mag - 1.0)
        H = H * gate
        H = self.norm_attn(H)
        H = self.attn(H, k_mags)
        return H


class KSECNet(nn.Module):
    """k-SEC v2: full network for per-crystal scalar prediction.

    Parameters
    ----------
    feature_dim   : complex hidden dimension (scalar & direction share)
    num_blocks    : number of k-SEC blocks
    n_heads       : attention heads
    n_max         : k-grid cutoff (|n_i| ≤ n_max for integer wavevectors)
    dropout       : dropout in filter MLP and final readout
    readout_hidden: MLP width for the real-space readout
    """

    def __init__(
        self,
        num_species: int = 100,
        feature_dim: int = 64,
        num_blocks: int = 3,
        n_heads: int = 4,
        n_max: int = 2,
        readout_hidden: int = 192,
        dropout: float = 0.15,
        tabular_dim: int = 0,
        tabular_hidden: int = 96,
        lattice_dim: int = 0,
        lattice_hidden: int = 32,
        geometric_dim: int = 0,
        geometric_hidden: int = 48,
        mace_dim: int = 0,
        mace_hidden: int = 24,
        bv_field: bool = False,
        bv_hidden: int = 32,
        bv_mobile_z: int = 3,
        bv_cutoff: float = 4.0,
        path_bv_field: bool = False,
        path_bv_hidden: int = 48,
        path_bv_pair_cutoff: float = 5.0,
        path_bv_n_points: int = 7,
        # BatteryNet dual-stream extension
        dual_stream: bool = False,
        mpnn_dim: int = 96,
        mpnn_layers: int = 3,
        mpnn_cutoff: float = 5.0,
        bridge_heads: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_species, feature_dim)
        self.feature_dim = feature_dim
        self.tabular_dim = tabular_dim
        self.lattice_dim = lattice_dim
        self.geometric_dim = geometric_dim
        self.mace_dim = mace_dim

        # Build cubic-group-averaged integer wavevector grid.
        wv_np = generate_wyckoff_wavevectors(n_max=n_max)
        orbits = precompute_orbits(wv_np)
        all_k = []
        for orbit in orbits:
            for k_vec in orbit:
                all_k.append(k_vec.numpy())
        all_k = np.array(all_k, dtype=np.float32)
        # prepend Γ
        all_k = np.concatenate([np.zeros((1, 3), dtype=np.float32), all_k], axis=0)
        k_mags = np.linalg.norm(all_k, axis=-1)
        self.register_buffer("k_points", torch.from_numpy(all_k).float(), persistent=False)
        self.register_buffer("k_mags", torch.from_numpy(k_mags).float(), persistent=False)
        # Precompute Kubic invariants once (they depend only on the grid)
        kubics = _kubic_invariants(torch.from_numpy(all_k).float())      # (K, 5)
        # Γ point has no direction — set its Kubic invariants to zeros
        kubics[0] = 0.0
        self.register_buffer("kubics", kubics, persistent=False)

        self.K = all_k.shape[0]
        self.blocks = nn.ModuleList([
            KSECBlock(feature_dim, n_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        if tabular_dim > 0:
            self.tabular_norm = nn.LayerNorm(tabular_dim)
            self.tabular_proj = nn.Sequential(
                nn.Linear(tabular_dim, tabular_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(tabular_hidden, tabular_hidden), nn.SiLU(),
            )
            readout_in = 2 * feature_dim + tabular_hidden
        else:
            self.tabular_norm = None
            self.tabular_proj = None
            readout_in = 2 * feature_dim

        if lattice_dim > 0:
            self.lattice_norm = nn.LayerNorm(lattice_dim)
            self.lattice_proj = nn.Sequential(
                nn.Linear(lattice_dim, lattice_hidden), nn.SiLU(),
                nn.Linear(lattice_hidden, lattice_hidden), nn.SiLU(),
            )
            readout_in += lattice_hidden
        else:
            self.lattice_norm = None
            self.lattice_proj = None

        if geometric_dim > 0:
            self.geometric_norm = nn.LayerNorm(geometric_dim)
            self.geometric_proj = nn.Sequential(
                nn.Linear(geometric_dim, geometric_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(geometric_hidden, geometric_hidden), nn.SiLU(),
            )
            readout_in += geometric_hidden
        else:
            self.geometric_norm = None
            self.geometric_proj = None

        if mace_dim > 0:
            self.mace_norm = nn.LayerNorm(mace_dim)
            self.mace_proj = nn.Sequential(
                nn.Linear(mace_dim, mace_hidden), nn.SiLU(),
                nn.Linear(mace_hidden, mace_hidden), nn.SiLU(),
            )
            readout_in += mace_hidden
        else:
            self.mace_norm = None
            self.mace_proj = None

        # Differentiable Bond-Valence Field (novel learnable physics module)
        self.bv_field = bv_field
        self.bv_mobile_z = bv_mobile_z
        self.bv_cutoff = bv_cutoff
        if bv_field:
            self.bv_params = LearnableBVParams(num_species=num_species, mobile_z=bv_mobile_z)
            self.bv_norm = nn.LayerNorm(8)
            self.bv_proj = nn.Sequential(
                nn.Linear(8, bv_hidden), nn.SiLU(),
                nn.Linear(bv_hidden, bv_hidden), nn.SiLU(),
            )
            readout_in += bv_hidden
        else:
            self.bv_params = None
            self.bv_norm = None
            self.bv_proj = None

        # BatteryNet dual-stream: real-space MPNN + cross-attention bridge
        self.dual_stream = dual_stream
        if dual_stream:
            self.mpnn_encoder = MPNNEncoder(
                num_species=num_species, atom_dim=mpnn_dim,
                n_layers=mpnn_layers, cutoff=mpnn_cutoff,
            )
            # k-SEC produces complex features stored as 2*feature_dim real;
            # the bridge operates on real tensors, so concat real+imag of
            # k-space features → 2*feature_dim then bridge to mpnn_dim.
            self.bridge = CrossAttentionBridge(
                d_k=2 * feature_dim, d_a=mpnn_dim, n_heads=bridge_heads,
            )
            readout_in += mpnn_dim     # mean-pooled atom features
        else:
            self.mpnn_encoder = None
            self.bridge = None

        # Site-resolved DBVF with differentiable path integration
        self.path_bv_field = path_bv_field
        self.path_bv_pair_cutoff = path_bv_pair_cutoff
        self.path_bv_n_points = path_bv_n_points
        if path_bv_field:
            self.path_bv_params = LearnablePathBVParams(num_species=num_species, mobile_z=bv_mobile_z)
            self.path_bv_norm = nn.LayerNorm(PATH_BV_FEATURE_DIM)
            self.path_bv_proj = nn.Sequential(
                nn.Linear(PATH_BV_FEATURE_DIM, path_bv_hidden), nn.SiLU(),
                nn.Linear(path_bv_hidden, path_bv_hidden), nn.SiLU(),
            )
            readout_in += path_bv_hidden
        else:
            self.path_bv_params = None
            self.path_bv_norm = None
            self.path_bv_proj = None

        self.readout = nn.Sequential(
            nn.Linear(readout_in, readout_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(readout_hidden, readout_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(readout_hidden, 1),
        )
        self.log_sigma_shift = nn.Parameter(torch.tensor(-5.0))

    def set_target_shift(self, mean: float):
        with torch.no_grad():
            self.log_sigma_shift.copy_(torch.tensor(float(mean)))

    def forward_structure(
        self,
        atom_z: torch.Tensor,            # (total_N,)
        frac_pos: torch.Tensor,          # (total_N, 3)
        batch_idx: torch.Tensor,         # (total_N,) sample index
        num_graphs: int,
        tabular: torch.Tensor | None = None,       # (B, tabular_dim) optional
        lattice_feats: torch.Tensor | None = None, # (B, lattice_dim) optional
        geometric: torch.Tensor | None = None,     # (B, geometric_dim) optional
        mace: torch.Tensor | None = None,          # (B, mace_dim) optional
        cell: torch.Tensor | None = None,          # (B, 3, 3) lattice matrices, required for bv_field
    ) -> torch.Tensor:
        """Return (B,) predicted log₁₀σ per crystal."""
        z = self.embed(atom_z)                                           # (N, D) real
        z_complex = torch.complex(z, torch.zeros_like(z))

        # Structure factors F_c(k_m) = Σ_j z_{j,c} exp(-2πi k_m · r_j)
        phases = -2.0 * math.pi * (frac_pos @ self.k_points.T)           # (N, K)
        exp_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        contrib = z_complex.unsqueeze(1) * exp_phases.unsqueeze(-1)      # (N, K, D)

        F = torch.zeros(num_graphs, self.K, z.shape[-1],
                        dtype=contrib.dtype, device=contrib.device)
        F.index_add_(0, batch_idx, contrib)

        counts = torch.zeros(num_graphs, device=z.device)
        counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        F = F / counts.clamp(min=1.0).view(-1, 1, 1)

        # k-SEC blocks
        for block in self.blocks:
            F = block(F, self.k_mags, self.kubics)

        # BatteryNet dual-stream: bring in real-space MPNN features,
        # exchange information via cross-attention, then pool.
        if self.dual_stream:
            if cell is None:
                raise ValueError("cell required for dual_stream; pass cell=(B,3,3)")
            # Real-space encoder
            h_atoms, _ = self.mpnn_encoder(atom_z, frac_pos, cell, batch_idx, num_graphs)
            h_a_padded, atom_mask = pad_atoms(h_atoms, batch_idx, num_graphs)
            # k-space stream as real (B, K, 2D)
            h_k_real = torch.cat([F.real, F.imag], dim=-1)
            h_k_real, h_a_padded = self.bridge(h_k_real, h_a_padded, atom_mask)
            # Pool both
            atom_mask_f = atom_mask.float()[..., None]
            atom_mean = (h_a_padded * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp(min=1)
            # k mean across K dim
            h_k_real_mean = h_k_real.mean(dim=1)
            # Combine: use bridge-enriched k-space (B, 2D) and atom mean (B, mpnn_dim)
            h = torch.cat([h_k_real_mean, atom_mean], dim=-1)
        else:
            # Mean-pool over k; separate real/imag halves
            F_re = F.real.mean(dim=1)                                    # (B, D)
            F_im = F.imag.mean(dim=1)
            h = torch.cat([F_re, F_im], dim=-1)                          # (B, 2D)

        if self.tabular_proj is not None:
            if tabular is None:
                raise ValueError("tabular features required but not provided")
            t = self.tabular_norm(tabular)
            t = self.tabular_proj(t)
            h = torch.cat([h, t], dim=-1)

        if self.lattice_proj is not None:
            if lattice_feats is None:
                raise ValueError("lattice features required but not provided")
            l = self.lattice_norm(lattice_feats)
            l = self.lattice_proj(l)
            h = torch.cat([h, l], dim=-1)

        if self.geometric_proj is not None:
            if geometric is None:
                raise ValueError("geometric features required but not provided")
            g = self.geometric_norm(geometric)
            g = self.geometric_proj(g)
            h = torch.cat([h, g], dim=-1)

        if self.mace_proj is not None:
            if mace is None:
                raise ValueError("MACE features required but not provided")
            mc = self.mace_norm(mace)
            mc = self.mace_proj(mc)
            h = torch.cat([h, mc], dim=-1)

        if self.bv_field:
            if cell is None:
                raise ValueError("cell required for bv_field; pass cell=(B,3,3)")
            bv_feats = compute_bv_features(
                self.bv_params, atom_z, frac_pos, cell, batch_idx,
                num_graphs=num_graphs, mobile_z=self.bv_mobile_z, cutoff=self.bv_cutoff,
            )
            bv_feats = self.bv_norm(bv_feats)
            bv_feats = self.bv_proj(bv_feats)
            h = torch.cat([h, bv_feats], dim=-1)

        if self.path_bv_field:
            if cell is None:
                raise ValueError("cell required for path_bv_field; pass cell=(B,3,3)")
            path_feats = compute_path_bv_features(
                self.path_bv_params, atom_z, frac_pos, cell, batch_idx,
                num_graphs=num_graphs, mobile_z=self.bv_mobile_z,
                cutoff_pair=self.path_bv_pair_cutoff, cutoff_anion=self.bv_cutoff,
                n_path_points=self.path_bv_n_points,
            )
            path_feats = self.path_bv_norm(path_feats)
            path_feats = self.path_bv_proj(path_feats)
            h = torch.cat([h, path_feats], dim=-1)

        y = self.readout(h).squeeze(-1)
        return y + self.log_sigma_shift

    def forward_mc_dropout(
        self,
        atom_z: torch.Tensor,
        frac_pos: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
        n_samples: int = 20,
    ):
        """Monte-Carlo-dropout forward. Returns (mean, std) per crystal over
        n_samples stochastic forward passes with dropout active."""
        self.train()                                                      # enable dropout
        preds = []
        for _ in range(n_samples):
            preds.append(self.forward_structure(atom_z, frac_pos, batch_idx, num_graphs))
        self.eval()
        stacked = torch.stack(preds, dim=0)                               # (S, B)
        return stacked.mean(dim=0), stacked.std(dim=0)
