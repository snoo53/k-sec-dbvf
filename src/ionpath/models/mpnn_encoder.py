"""Real-space message-passing encoder for BatteryNet.

A clean CGCNN-style atom-graph MPNN that we own end-to-end:
  - Build a graph from atom positions with a distance cutoff
  - Edge features: gaussian-expanded distance + atomic-pair embedding
  - Message passing: sum-aggregate neighbor messages, GRU-style update
  - Per-atom output → batch via mean pooling

Output: (B, N_max, D) per-atom features (with batch_idx semantics) so
the cross-attention layer can attend over atoms.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class GaussianExpansion(nn.Module):
    """Expand a scalar distance into a smooth basis: exp(-(d - μ)² / 2σ²)."""

    def __init__(self, n_basis: int = 32, d_min: float = 0.5, d_max: float = 8.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_basis)
        self.register_buffer("centers", centers)
        self.sigma = (d_max - d_min) / n_basis

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """d: (E,) → (E, n_basis)."""
        return torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (2 * self.sigma ** 2))


class MPNNLayer(nn.Module):
    """One round of edge → message → aggregate → update."""

    def __init__(self, atom_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * atom_dim + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, atom_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * atom_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, atom_dim),
        )
        self.norm = nn.LayerNorm(atom_dim)

    def forward(self, h_atoms, edge_index, edge_feats):
        """
        h_atoms: (N, D)
        edge_index: (2, E) source, target
        edge_feats: (E, edge_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        h_src = h_atoms[src]
        h_dst = h_atoms[dst]
        msg_input = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        msg = self.msg(msg_input)                                 # (E, D)
        # Sum-aggregate per dst
        agg = torch.zeros_like(h_atoms)
        agg.index_add_(0, dst, msg)
        h_new = self.update(torch.cat([h_atoms, agg], dim=-1))
        return self.norm(h_atoms + h_new)


def build_graph_from_positions(
    atom_z: torch.Tensor,        # (N,)
    cart_pos: torch.Tensor,      # (N, 3)
    batch_idx: torch.Tensor,     # (N,)
    cutoff: float = 5.0,
    max_neighbors: int = 32,
):
    """Build a per-graph radius graph (no periodic images for simplicity;
    OK for OBELiX where we're already in the unit cell with normalized
    coords). Returns edge_index (2, E) and distances (E,).

    Atoms are connected only to atoms in the same graph (same batch_idx).
    """
    N = atom_z.shape[0]
    device = atom_z.device

    # Pairwise distances within same graph
    same_graph = batch_idx[:, None] == batch_idx[None, :]
    diff = cart_pos[:, None, :] - cart_pos[None, :, :]
    d = torch.linalg.norm(diff, dim=-1)
    valid = same_graph & (d > 1e-3) & (d < cutoff)
    src, dst = valid.nonzero(as_tuple=True)
    edge_d = d[src, dst]
    return torch.stack([src, dst], dim=0), edge_d


class MPNNEncoder(nn.Module):
    """CGCNN-style real-space encoder.

    forward(atom_z, frac_pos, cell, batch_idx, num_graphs) → (N, D) per-atom
    features. The downstream cross-attention layer will pool/attend.
    """

    def __init__(
        self,
        num_species: int = 100,
        atom_dim: int = 96,
        n_layers: int = 3,
        edge_basis: int = 32,
        cutoff: float = 5.0,
        hidden: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_species, atom_dim)
        self.gauss = GaussianExpansion(n_basis=edge_basis, d_max=cutoff)
        self.cutoff = cutoff
        self.layers = nn.ModuleList([
            MPNNLayer(atom_dim, edge_basis, hidden) for _ in range(n_layers)
        ])

    def forward(self, atom_z, frac_pos, cell, batch_idx, num_graphs):
        # cart positions
        cell_per_atom = cell[batch_idx]
        cart = torch.einsum("nj,njk->nk", frac_pos, cell_per_atom)

        edge_index, d = build_graph_from_positions(
            atom_z, cart, batch_idx, cutoff=self.cutoff,
        )
        edge_feats = self.gauss(d)
        h = self.embed(atom_z)
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats)
        return h, batch_idx        # (N, D), (N,)
