"""Generate diagrams for the k-SEC + DBVF technical reference PDF.

Produces three matplotlib block-and-arrow diagrams:
  1. fig_pipeline.png   — full pipeline: Crystal → 6 branches → Readout
  2. fig_ksec_block.png — internals of one KSECBlock
  3. fig_dbvf_flow.png  — DBVF: atom positions + cell → BV mismatch features

Run from the repo root:
    python applications/build_tech_reference_figs.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "applications" / "figs_techref"
OUT.mkdir(parents=True, exist_ok=True)

LIGHT = "#e8eef7"
ACCENT = "#1f4b8e"
WARM = "#c84b1a"
BG = "white"


def _box(ax, x, y, w, h, text, fc=LIGHT, ec=ACCENT, fontsize=9, fontweight="normal"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, x1, y1, x2, y2, text=None, color=ACCENT, ls="-"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12,
        linewidth=1.0, color=color, linestyle=ls,
    )
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text,
                ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.set_axis_off()

    # Crystal input
    _box(ax, 0.2, 3.5, 1.5, 1.0,
         "Crystal\nstructure\n(CIF)", fc="white", ec=WARM, fontweight="bold")

    # Featurization stage (column at x ≈ 2.5)
    feats = [
        ("atom_z\n(N,)", 6.4, "z"),
        ("frac_pos\n(N, 3)", 5.4, "r"),
        ("cell\n(B, 3, 3)", 4.4, "C"),
        ("Magpie\n(B, 132)", 3.3, "M"),
        ("Lattice\n(B, 8)", 2.2, "L"),
        ("Geometric\n(B, 25)", 1.1, "G"),
    ]
    for txt, y, _ in feats:
        _box(ax, 2.2, y, 1.3, 0.7, txt, fontsize=8)
        _arrow(ax, 1.7, 4.0, 2.2, y + 0.35)

    # k-SEC branch
    _box(ax, 4.2, 5.4, 1.7, 1.4,
         "k-SEC\nencoder\n3 × KSECBlock\n(537,804 p)", fc="#fce8d8", ec=WARM)
    # Structure factor pre-step
    _box(ax, 4.2, 6.9, 1.7, 0.5, "F(k) = Σ z·e^(-2πi k·r)", fontsize=8)
    _arrow(ax, 3.5, 6.4, 4.2, 6.6)  # atom_z → kSEC
    _arrow(ax, 3.5, 5.6, 4.2, 6.0)  # frac_pos → kSEC

    # Mean pool box
    _box(ax, 6.2, 5.7, 1.4, 0.8,
         "mean over K\n[Re F̄ ; Im F̄]\n(B, 192)", fontsize=8)
    _arrow(ax, 5.9, 6.1, 6.2, 6.1)

    # DBVF branch
    _box(ax, 4.2, 4.0, 1.7, 1.0,
         "DBVF\nLearnable BV\n(B, 8)", fc="#fce8d8", ec=WARM)
    _arrow(ax, 3.5, 5.6, 4.2, 4.7)
    _arrow(ax, 3.5, 4.7, 4.2, 4.5)
    _arrow(ax, 3.5, 4.0, 4.2, 4.4, ls=":")

    _box(ax, 6.2, 4.0, 1.4, 1.0, "LayerNorm\n+ MLP\n(B, 32)", fontsize=8)
    _arrow(ax, 5.9, 4.5, 6.2, 4.5)

    # Auxiliary projection heads
    aux = [
        ("Magpie head\n(B, 96)", 3.3, "tabular_proj"),
        ("Lattice head\n(B, 32)", 2.2, "lattice_proj"),
        ("Geometric head\n(B, 48)", 1.1, "geometric_proj"),
    ]
    for txt, y, _ in aux:
        _box(ax, 4.2, y, 1.7, 0.7, txt, fontsize=8)
        _arrow(ax, 3.5, y + 0.35, 4.2, y + 0.35)
        _arrow(ax, 5.9, y + 0.35, 8.0, y + 0.35)

    # Concat
    _box(ax, 8.0, 1.0, 0.9, 5.6, "Concat\n(B, 400)",
         fc="#e8f0e8", ec="#3a7a3a", fontweight="bold")
    _arrow(ax, 7.6, 6.1, 8.0, 5.0)
    _arrow(ax, 7.6, 4.5, 8.0, 4.0)

    # Readout MLP
    _box(ax, 9.2, 3.5, 1.6, 1.5,
         "Readout MLP\n400 → 192 → 192 → 1\n(114,241 p)",
         fc="#e8f0e8", ec="#3a7a3a")
    _arrow(ax, 8.9, 4.0, 9.2, 4.2)

    # Output
    _box(ax, 11.0, 3.7, 1.0, 0.9,
         "log₁₀σ\n+ shift", fc="white", ec=WARM, fontweight="bold")
    _arrow(ax, 10.8, 4.2, 11.0, 4.2)

    ax.set_title("Headline forward pass: k-SEC + DBVF + Magpie + Lattice + Geometric  (690,562 trainable params)",
                 fontsize=10, fontweight="bold", pad=8)

    plt.tight_layout()
    out = OUT / "fig_pipeline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"wrote {out}")


def fig_ksec_block():
    fig, ax = plt.subplots(figsize=(7.5, 8.0), dpi=200)
    ax.set_xlim(0, 8); ax.set_ylim(0, 10)
    ax.set_axis_off()

    # Input
    _box(ax, 2.5, 9.0, 3.0, 0.6, "H : (B, K, D) complex", fc="white", ec=WARM)

    # LayerNorm 1
    _box(ax, 2.5, 8.0, 3.0, 0.5, "_CLN  (LayerNorm on [Re ; Im])", fontsize=9)
    _arrow(ax, 4.0, 9.0, 4.0, 8.5)

    # Filter
    _box(ax, 1.0, 6.7, 6.0, 1.0,
         "KubicHarmonicFilter\n  W(|k|, K₀, K₄ₐ, K₄ᵦ, K₆ₐ, K₆ᵦ) ⊙ H + b\n  (cubic-O_h-equivariant by construction)",
         fc="#fce8d8", ec=WARM, fontsize=9, fontweight="bold")
    _arrow(ax, 4.0, 8.0, 4.0, 7.7)

    # Magnitude gate
    _box(ax, 2.5, 5.7, 3.0, 0.5,
         "gate = σ(|H| − 1) ;  H ← H · gate", fontsize=9)
    _arrow(ax, 4.0, 6.7, 4.0, 6.2)

    # LayerNorm 2
    _box(ax, 2.5, 4.7, 3.0, 0.5, "_CLN  (LayerNorm)", fontsize=9)
    _arrow(ax, 4.0, 5.7, 4.0, 5.2)

    # Cross-shell attention
    _box(ax, 1.0, 2.7, 6.0, 1.7,
         "CrossShellGatedAttention\n  Q,K,V = Linear([Re H ; Im H])\n  scores = Re⟨q, k*⟩ / √dₕ + log gate(|kᵢ|, |kⱼ|, ||kᵢ|−|kⱼ||)\n  α = softmax over j ;  out = Linear(α·V)\n  + residual H",
         fc="#fce8d8", ec=WARM, fontsize=9, fontweight="bold")
    _arrow(ax, 4.0, 4.7, 4.0, 4.4)

    # Output
    _box(ax, 2.5, 1.0, 3.0, 0.6, "H' : (B, K, D) complex",
         fc="white", ec=WARM)
    _arrow(ax, 4.0, 2.7, 4.0, 1.6)

    # Param count
    ax.text(7.7, 5.0,
            "Per block:\n179,268 params\n\nFilter: 30,016\nAttention: 148,484\n2 × _CLN: 768",
            fontsize=8, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f0", ec="gray"))

    ax.set_title("KSECBlock internals",
                 fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    out = OUT / "fig_ksec_block.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"wrote {out}")


def fig_dbvf_flow():
    fig, ax = plt.subplots(figsize=(11.0, 5.5), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.set_axis_off()

    # Inputs
    _box(ax, 0.2, 4.5, 1.6, 0.7, "atom_z\n(N,)", fc="white", ec=WARM)
    _box(ax, 0.2, 3.5, 1.6, 0.7, "frac_pos\n(N, 3)", fc="white", ec=WARM)
    _box(ax, 0.2, 2.5, 1.6, 0.7, "cell\n(B, 3, 3)", fc="white", ec=WARM)
    _box(ax, 0.2, 1.0, 1.6, 1.2,
         "Learnable params\n(softplus reparam.)\nr₀ : (101,)\nb  : (101,)",
         fc="#fce8d8", ec=WARM, fontsize=8)

    # Build cartesian + ±1 image shell
    _box(ax, 2.4, 3.3, 2.4, 1.8,
         "1. Cartesian:\n   x = frac_pos · cell\n2. 27 image shifts\n   {-1,0,1}³ · cell",
         fontsize=8)
    _arrow(ax, 1.8, 4.8, 2.4, 4.7)
    _arrow(ax, 1.8, 3.8, 2.4, 4.4)
    _arrow(ax, 1.8, 2.8, 2.4, 4.1)

    # Mask Li sites and anion sites
    _box(ax, 2.4, 1.0, 2.4, 1.8,
         "3. Li sites: z = 3\n4. Anion sites: z ≠ 3, > 0",
         fontsize=8)
    _arrow(ax, 1.8, 4.8, 2.4, 2.3, ls=":")

    # BV sum per Li
    _box(ax, 5.2, 1.5, 3.0, 3.0,
         "5. For each Li site i:\n   d(i, j, k) = ‖xⱼ + Δₖ − xᵢ‖\n   Vᵢ = Σⱼ Σₖ exp((r₀ − d)/b)\n           · 𝟙[d < 4 Å]\n6. Mismatch:\n   Uᵢ = |Vᵢ − 1|",
         fc="#fce8d8", ec=WARM, fontsize=9)
    _arrow(ax, 4.8, 4.2, 5.2, 3.7)
    _arrow(ax, 4.8, 1.9, 5.2, 2.5)
    _arrow(ax, 1.8, 1.6, 5.2, 2.0)  # learnable params direct in

    # Aggregate stats
    _box(ax, 8.6, 1.5, 2.6, 3.0,
         "7. Per-crystal stats:\n   mean(U), std(U)\n   min, max, p25, p50, p75\n   atan(n_Li) / (π/2)\n→ 8 features",
         fontsize=9)
    _arrow(ax, 8.2, 3.0, 8.6, 3.0)

    # Output
    _box(ax, 11.4, 2.7, 0.5, 0.7, "(B, 8)",
         fc="white", ec=WARM, fontsize=8)
    _arrow(ax, 11.2, 3.0, 11.4, 3.0)

    ax.set_title("DBVF: differentiable bond-valence field, atom-level → per-crystal aggregates  (202 learnable params)",
                 fontsize=10, fontweight="bold", pad=8)

    plt.tight_layout()
    out = OUT / "fig_dbvf_flow.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_pipeline()
    fig_ksec_block()
    fig_dbvf_flow()
    print("All diagrams written to", OUT)
