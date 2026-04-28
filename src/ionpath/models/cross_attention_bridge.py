"""Learnable cross-attention bridge between k-space and real-space streams.

The k-SEC encoder produces (B, K, D_k) reciprocal-space features.
The MPNN encoder produces (N, D_a) per-atom real-space features.

This bridge does:
  1. Pad atoms into (B, N_max, D_a) so cross-attention can be batched
  2. k-space queries attend to atom keys/values → enriched (B, K, D_k)
  3. atom queries attend to k-space keys/values → enriched (B, N_max, D_a)
  4. Pool: mean over k, mean over real → concat → readout-ready (B, 2*D)

This is the novel piece: information flows BOTH ways between the streams.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBridge(nn.Module):
    def __init__(self, d_k: int, d_a: int, n_heads: int = 4, hidden: int = 192):
        super().__init__()
        self.proj_q_k = nn.Linear(d_k, hidden)
        self.proj_kv_a = nn.Linear(d_a, 2 * hidden)
        self.proj_q_a = nn.Linear(d_a, hidden)
        self.proj_kv_k = nn.Linear(d_k, 2 * hidden)
        self.n_heads = n_heads
        self.dh = hidden // n_heads
        self.out_k = nn.Linear(hidden, d_k)
        self.out_a = nn.Linear(hidden, d_a)
        self.norm_k = nn.LayerNorm(d_k)
        self.norm_a = nn.LayerNorm(d_a)

    def _attn(self, q, kv, mask=None):
        """q: (B, Lq, H, dh)  kv: (B, Lk, H, 2*dh)  → (B, Lq, H, dh)."""
        k, v = kv.chunk(2, dim=-1)
        scores = torch.einsum("blhd,bkhd->blkh", q, k) / (q.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :, None], float("-inf"))
        alpha = torch.softmax(scores, dim=2)
        return torch.einsum("blkh,bkhd->blhd", alpha, v)

    def forward(self, h_k, h_a_padded, atom_mask):
        """
        h_k: (B, K, D_k) — k-space stream
        h_a_padded: (B, N_max, D_a) — atom stream
        atom_mask: (B, N_max) bool, True for real atoms (False for padding)
        Returns: (B, K, D_k), (B, N_max, D_a)
        """
        B, K, _ = h_k.shape
        _, N_max, _ = h_a_padded.shape

        # k-space queries → atom keys/values
        q_k = self.proj_q_k(h_k)  # (B, K, H)
        kv_a = self.proj_kv_a(h_a_padded)  # (B, N_max, 2H)
        q_k_h = q_k.view(B, K, self.n_heads, self.dh)
        kv_a_h = kv_a.view(B, N_max, self.n_heads, 2 * self.dh)
        ctx_k = self._attn(q_k_h, kv_a_h, mask=atom_mask)  # (B, K, H, dh)
        ctx_k = ctx_k.reshape(B, K, -1)
        h_k_new = self.norm_k(h_k + self.out_k(ctx_k))

        # atom queries → k-space keys/values
        q_a = self.proj_q_a(h_a_padded)
        kv_k = self.proj_kv_k(h_k)
        q_a_h = q_a.view(B, N_max, self.n_heads, self.dh)
        kv_k_h = kv_k.view(B, K, self.n_heads, 2 * self.dh)
        ctx_a = self._attn(q_a_h, kv_k_h)  # no mask on k-side; all valid
        ctx_a = ctx_a.reshape(B, N_max, -1)
        h_a_new = self.norm_a(h_a_padded + self.out_a(ctx_a))

        return h_k_new, h_a_new


def pad_atoms(h_atoms, batch_idx, num_graphs):
    """Pack (N, D) per-atom into (B, N_max, D) padded with zeros + mask.

    Returns:
        h_padded: (B, N_max, D)
        atom_mask: (B, N_max) bool
    """
    device = h_atoms.device
    D = h_atoms.shape[-1]
    counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    counts.index_add_(0, batch_idx, torch.ones_like(batch_idx))
    N_max = int(counts.max().item())
    h_padded = torch.zeros(num_graphs, N_max, D, device=device, dtype=h_atoms.dtype)
    atom_mask = torch.zeros(num_graphs, N_max, dtype=torch.bool, device=device)
    cursors = torch.zeros(num_graphs, dtype=torch.long, device=device)
    # Fill (sequential — N is small for OBELiX)
    for n_idx in range(h_atoms.shape[0]):
        g = int(batch_idx[n_idx].item())
        c = int(cursors[g].item())
        h_padded[g, c] = h_atoms[n_idx]
        atom_mask[g, c] = True
        cursors[g] = c + 1
    return h_padded, atom_mask
