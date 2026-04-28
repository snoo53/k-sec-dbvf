"""Render the Figure-1 architecture schematic for the JMST manuscript.

Two streams (k-SEC reciprocal-space + DBVF real-space) feed a joint
readout MLP. Designed to reproduce cleanly at 300 dpi in a single column.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"


def box(ax, xy, w, h, text, color, edge="black", fontsize=9, lw=1.2):
    rect = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, src, dst, color="#444444", lw=1.4, mutation=12):
    a = FancyArrowPatch(
        src, dst, arrowstyle="-|>", mutation_scale=mutation,
        color=color, linewidth=lw, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.set_aspect("equal"); ax.axis("off")

    # Colors
    blue_light = "#cfe2f3"
    blue = "#9fc5e8"
    blue_dark = "#1f77b4"
    green_light = "#d9ead3"
    green = "#a3c4a7"
    green_dark = "#2ca02c"
    orange = "#fce5cd"
    gray = "#e0e0e0"

    # Input
    box(ax, (1.5, 4.0), 2.4, 1.6,
        "Crystal\n(L, {Z_j, r_j})\n+ Magpie + lattice +\nWyckoff geometric",
        gray, fontsize=8.5)

    # === k-SEC stream (top) ===
    box(ax, (5.5, 6.4), 2.6, 0.9,
        "Atomic structure factors\n F_c(k) = Σ_j z_{j,c} e^{-2πi k·r_j}",
        blue_light, fontsize=8)
    box(ax, (5.5, 5.3), 2.6, 0.9,
        "Cubic-harmonic filter\n W(|k|, K_0, K_4a, K_4b, K_6a, K_6b)",
        blue, fontsize=8)
    box(ax, (5.5, 4.2), 2.6, 0.9,
        "Cross-shell gated attention\n gate g(|k_i|, |k_j|, Δ|k|)",
        blue, fontsize=8)
    box(ax, (5.5, 3.1), 2.6, 0.7,
        "× 3 blocks → mean pool over k\n→ (B, D=96)",
        blue_dark, edge="black", fontsize=8)

    # === DBVF stream (bottom) ===
    box(ax, (5.5, 1.9), 2.6, 0.9,
        "Per-Li bond-valence sum\n V(r_Li) = Σ exp((r_0 − d)/b)",
        green_light, fontsize=8)
    box(ax, (5.5, 0.8), 2.6, 0.9,
        "Learnable (r_0, b) per anion\n→ pool to 8-d descriptor",
        green_dark, edge="black", fontsize=8)

    # Concatenation node
    box(ax, (9.5, 4.0), 1.8, 1.6,
        "Concatenate\n(B, D=96)\n+ (B, 8) DBVF\n+ (B, 165) tabular",
        orange, fontsize=8.5)

    # Readout MLP
    box(ax, (12.0, 4.0), 1.8, 1.0,
        "3-layer MLP\nDropout 0.15\n→ log_10 σ",
        "#888888", edge="black", fontsize=9)

    # Arrows: input → both streams
    arrow(ax, (2.7, 4.6), (4.2, 6.4))     # to k-SEC top block
    arrow(ax, (2.7, 4.0), (4.2, 1.9))     # to DBVF top block
    arrow(ax, (2.7, 4.0), (8.6, 4.0))     # input -> readout (Magpie etc.)

    # Within k-SEC stream
    arrow(ax, (5.5, 5.95), (5.5, 5.75))
    arrow(ax, (5.5, 4.85), (5.5, 4.65))
    arrow(ax, (5.5, 3.75), (5.5, 3.45))
    # k-SEC → concatenation
    arrow(ax, (6.8, 3.1), (8.6, 3.6))

    # Within DBVF stream
    arrow(ax, (5.5, 1.45), (5.5, 1.25))
    # DBVF → concatenation
    arrow(ax, (6.8, 0.8), (8.6, 4.0 - 0.4))

    # Concatenate → MLP
    arrow(ax, (10.4, 4.0), (11.1, 4.0))

    # Section labels
    ax.text(5.5, 7.2, "k-SEC (reciprocal-space encoder)",
            ha="center", fontsize=10.5, fontweight="bold", color=blue_dark)
    ax.text(5.5, 0.08, "DBVF (differentiable bond-valence)",
            ha="center", fontsize=10.5, fontweight="bold", color=green_dark)

    # Equivariance annotation
    ax.text(5.5, 6.95, "All operations cubic-equivariant by construction",
            ha="center", fontsize=8, style="italic", color="#444444")

    # Title and footer note
    ax.text(7.0, 7.8,
            "Figure 1.  Architecture overview: two-stream k-SEC + DBVF for ionic-conductivity prediction.",
            ha="center", fontsize=10.5, fontweight="bold")

    plt.tight_layout()
    out = FIGS / "fig_1_architecture.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
