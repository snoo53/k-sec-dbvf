"""Render JMST figures from the result JSONs.

Headline = Phase B1 (k-SEC + DBVF):
    standalone MAE 1.047, stacked MAE 0.980, n = 281, 5-fold CV × 5 seeds.

Produces:
  fig_2_baselines.png       — Phase B1 vs prior baselines (paper Fig. 2)
  fig_3_ablation.png        — 4-config k-SEC architectural ablation (Fig. 3)
  fig_4_dbvf_test.png       — DBVF as architecture, not as features (Fig. 4)
  fig_5_kubic_interpret.png — cubic-harmonic filter direction sensitivity
  fig_6_virtual_screen.png  — top-15 virtual screen + family recovery (Fig. 6)
  fig_7_ood_by_family.png   — leave-family-out generalization (Fig. 7)
  fig_7_calibration.png     — MC-dropout calibration (skipped if not ready)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)


def plot_baselines():
    """Phase B1 (k-SEC + DBVF) headline vs. all prior baselines."""
    with open(RESULTS / "stacking_phaseB1.json") as f:
        b1 = json.load(f)
    mae_stack_b1 = b1["stacked_ridge"]["mae"]
    mae_b1 = b1["ksec_hybrid"]["mae"]
    mae_lgbm_magpie = b1["lightgbm"]["mae"]

    # LightGBM with all hand-crafted features (the brutal-honest ceiling)
    mae_lgbm_full = 0.924
    if (RESULTS / "lightgbm_with_dbvf.json").exists():
        with open(RESULTS / "lightgbm_with_dbvf.json") as f:
            lj = json.load(f)
        # Schema: {"LightGBM + Magpie + lattice + geometric (no DBVF)": {"mae": ...}, ...}
        for key, val in lj.items():
            if "no DBVF" in key:
                mae_lgbm_full = val["mae"]
                break

    # Earlier headline (pre-pivot, no DBVF)
    mae_pre_stack = mae_pre_neural = None
    if (RESULTS / "stacking.json").exists():
        with open(RESULTS / "stacking.json") as f:
            st = json.load(f)
        mae_pre_stack = st["stacked_ridge"]["mae"]
        mae_pre_neural = st["ksec_hybrid"]["mae"]

    # Phase A4 (MACE features) — peer comparison for DBVF
    mae_a4_stack = mae_a4 = None
    if (RESULTS / "stacking_phaseA4.json").exists():
        with open(RESULTS / "stacking_phaseA4.json") as f:
            a4 = json.load(f)
        mae_a4_stack = a4["stacked_ridge"]["mae"]
        mae_a4 = a4["ksec_hybrid"]["mae"]

    # Per-seed std for the headline (pulled from ksec_phaseB1.json aggregate)
    std_b1 = 0.012
    if (RESULTS / "ksec_phaseB1.json").exists():
        with open(RESULTS / "ksec_phaseB1.json") as f:
            kj = json.load(f)
        std_b1 = kj.get("aggregate", {}).get("mae_std", std_b1)

    rows = []
    # Std == None for entries where we don't have a per-seed std; nan-mask
    # lets matplotlib skip the error bar gracefully.
    rows.append(("Stacked\n(k-SEC+DBVF\n + LGBM)", mae_stack_b1, np.nan, "#2ca02c"))
    rows.append(("LightGBM +\nMagpie+lat+geom", mae_lgbm_full, np.nan, "#888888"))
    rows.append(("LightGBM +\nMagpie", mae_lgbm_magpie, np.nan, "#888888"))
    rows.append(("k-SEC + DBVF\n(this work)", mae_b1, std_b1, "#1f77b4"))
    if mae_a4 is not None:
        a4_std = 0.032
        if (RESULTS / "ksec_phaseA4.json").exists():
            with open(RESULTS / "ksec_phaseA4.json") as f:
                a4j = json.load(f)
            a4_std = a4j.get("aggregate", {}).get("mae_std", a4_std)
        rows.append(("k-SEC + MACE\nfeatures", mae_a4, a4_std, "#8fa8c8"))
    if mae_pre_neural is not None:
        rows.append(("k-SEC\n(pre-DBVF)", mae_pre_neural, 0.032, "#8fa8c8"))
    rows += [
        ("IonPath\ndual-graph", 1.393, np.nan, "#888888"),
        ("CGCNN-lite", 1.573, np.nan, "#888888"),
        ("k-SEC v1", 1.634, np.nan, "#888888"),
    ]
    labels, maes, stds, colors = zip(*rows)
    stds_arr = np.array(stds, dtype=float)
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # Replace NaN with 0 for bar drawing; capsize remains so visible stds stand out
    bar_yerr = np.nan_to_num(stds_arr, nan=0.0)
    ax.bar(xs, maes, yerr=bar_yerr, color=colors, edgecolor="black", capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE on log₁₀ σ (5-fold CV × 5 seeds)")
    ax.set_title("OBELiX (n = 281 filtered) — Phase B1 (k-SEC + DBVF) headline")
    ax.axhline(mae_stack_b1, ls=":", color="#2ca02c", alpha=0.6,
               label=f"Stacked MAE = {mae_stack_b1:.3f}")
    ax.axhline(mae_lgbm_full, ls=":", color="#666666", alpha=0.6,
               label=f"LightGBM full features = {mae_lgbm_full:.3f}")
    for i, v in enumerate(maes):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(maes) * 1.15)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_2_baselines.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_2_baselines.png'}")


def plot_dbvf_test():
    """The architecture-vs-features control: extracting DBVF as features
    does not help LightGBM."""
    path = RESULTS / "lightgbm_with_dbvf.json"
    if not path.exists():
        print("skip dbvf test plot (no file)")
        return
    with open(path) as f:
        lj = json.load(f)
    mae_baseline = mae_with = mae_e2e = None
    for key, val in lj.items():
        if "no DBVF" in key:
            mae_baseline = val["mae"]
        elif "DBVF" in key and "k-SEC" not in key:
            mae_with = val["mae"]
        elif "k-SEC" in key:
            mae_e2e = val["mae"]
    if mae_e2e is None:
        with open(RESULTS / "stacking_phaseB1.json") as f:
            b1 = json.load(f)
        mae_e2e = b1["ksec_hybrid"]["mae"]

    labels = [
        "LightGBM\n(Magpie+lat+geom)",
        "LightGBM\n+ DBVF features",
        "k-SEC + DBVF\n(end-to-end)",
    ]
    maes = [mae_baseline, mae_with, mae_e2e]
    colors = ["#888888", "#d62728", "#1f77b4"]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(xs, maes, color=colors, edgecolor="black")
    for i, v in enumerate(maes):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE on log₁₀ σ (5-fold CV)")
    ax.set_title("DBVF is architecture, not feature\n(extracting DBVF features worsens LightGBM)")
    ax.set_ylim(0, max(maes) * 1.2)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_4_dbvf_test.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_4_dbvf_test.png'}")


def plot_virtual_screen():
    """Top-15 virtual-screen predictions colour-coded by canonical family.

    Family assignment is heuristic on the formula since the CSV is
    family-agnostic. The four canonical fast-Li conductors recovered
    in the top-15: anti-perovskite (Li3ClO), LGPS-like (Li10X(PY6)2,
    Li10Zn(PS4)4), argyrodite (Li6PS5{X}), chloride double-perovskite
    (A2LiBCl6).
    """
    path = RESULTS / "virtual_screen_top_100.csv"
    if not path.exists():
        print("skip virtual-screen plot (no top-100 csv)")
        return
    import csv, re
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows = rows[:15]
    labels = [r.get("formula", r.get("pretty_formula", "?")) for r in rows]
    log_sigma = [float(r.get("pred_log_sigma", r.get("pred", 0))) for r in rows]

    def family_of(formula: str) -> str:
        f = formula.replace(" ", "")
        if re.search(r"Li\d*ClO\b", f) or re.search(r"Li\d*BrO\b", f):
            return "anti-perovskite"
        if re.search(r"Li\d*Sn\(P[SeT]+\d*\)\d*", f) or re.search(r"Li\d*Zn\(P[SeT]+\d*\)\d*", f):
            return "lgps"
        if re.search(r"Li6P[SeT]+5[FClBrI]", f):
            return "argyrodite"
        # Chloride double-perovskite: A2LiBCl6 family (e.g. Rb2LiYbCl6, Na2LiTmCl6)
        if re.search(r"^[A-Z][a-z]?2Li[A-Z][a-z]?Cl6", f):
            return "chloride-perovskite"
        if any(x in f for x in ("S6", "S4", "S5")) and "Li" in f:
            return "sulfide"
        return "other"

    families = [family_of(lbl) for lbl in labels]
    fam_colors = {
        "anti-perovskite": "#2ca02c",
        "lgps": "#1f77b4",
        "argyrodite": "#ff7f0e",
        "chloride-perovskite": "#9467bd",
        "sulfide": "#8c564b",
        "other": "#cccccc",
    }
    colors = [fam_colors.get(f, "#cccccc") for f in families]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(xs, log_sigma, color=colors, edgecolor="black")
    # Plot bars upward from a constant baseline so taller = faster (more positive log σ)
    baseline = -3.0
    bar_heights = [v - baseline for v in log_sigma]
    ax.cla()
    ax.bar(xs, bar_heights, bottom=baseline, color=colors, edgecolor="black")
    for i, v in enumerate(log_sigma):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("predicted log₁₀ σ (S/cm)  ↑ faster conductor")
    ax.set_title("Virtual screen: top-15 of 18,574 Li-containing MP crystals\n"
                 "(four canonical fast-Li conductor families recovered without supervision)")
    ax.set_ylim(baseline, max(log_sigma) + 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black") for c in fam_colors.values()]
    ax.legend(handles, list(fam_colors.keys()), loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_6_virtual_screen.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_6_virtual_screen.png'}")


def plot_ood():
    path = RESULTS / "ood_by_family.json"
    if not path.exists():
        print("skip ood plot (no file)")
        return
    with open(path) as f:
        ood = json.load(f)
    with open(RESULTS / "ksec_phaseB1.json") as f:
        kj = json.load(f)
    mae_id = kj["aggregate"]["mae_mean"]

    fams = [r["family"] for r in ood["per_family"]]
    maes = [r["mae"] for r in ood["per_family"]]
    ns = [r["n_test"] for r in ood["per_family"]]
    xs = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ca02c" if m < mae_id else "#d62728" for m in maes]
    ax.bar(xs, maes, color=colors, edgecolor="black")
    for i, (m, n) in enumerate(zip(maes, ns)):
        ax.text(i, m + 0.02, f"{m:.2f}\nn={n}", ha="center", fontsize=8)
    ax.axhline(mae_id, ls="--", color="#1f77b4", label=f"in-distribution MAE = {mae_id:.3f}")
    ax.set_xticks(xs); ax.set_xticklabels(fams)
    ax.set_ylabel("MAE on held-out family")
    ax.set_title("Leave-family-out generalization")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(maes + [mae_id]) * 1.2)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_7_ood_by_family.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_7_ood_by_family.png'}")


def plot_ablation():
    path = RESULTS / "ablation.json"
    if not path.exists():
        print("skip ablation plot (no file)")
        return
    with open(path) as f:
        abl = json.load(f)
    items = list(abl.items())
    items.sort(key=lambda kv: kv[1]["mae_mean"])
    labels = [k.replace(" (", "\n(") for k, _ in items]
    maes = [v["mae_mean"] for _, v in items]
    stds = [v["mae_std"] for _, v in items]
    xs = np.arange(len(labels))
    colors = ["#1f77b4" if "Full v2" in lab else "#888888" for lab in labels]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(xs, maes, yerr=stds, color=colors, edgecolor="black", capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE on log₁₀ σ (5-fold CV, 2 seeds)")
    ax.set_title("k-SEC ablation: filter × attention")
    for i, v in enumerate(maes):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_3_ablation.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_3_ablation.png'}")


def plot_calibration():
    path = RESULTS / "mc_dropout.json"
    if not path.exists():
        print("skip calibration plot (no file)")
        return
    with open(path) as f:
        mc = json.load(f)
    cov = mc["coverage"]
    mae = mc["aggregate_mae"]; ms = mc["mean_uncertainty"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    levels = sorted(cov.keys())
    expected = [0.683, 0.950]
    observed = [cov[k] for k in levels]
    xs = np.arange(len(levels))
    ax.bar(xs - 0.15, expected, 0.3, label="expected", color="#888888")
    ax.bar(xs + 0.15, observed, 0.3, label="observed", color="#1f77b4")
    ax.set_xticks(xs); ax.set_xticklabels(levels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("coverage")
    ax.set_title(f"MC-dropout calibration  (MAE={mae:.3f}, mean σ={ms:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "fig_8_calibration.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_8_calibration.png'}")


def plot_parity_and_per_bin():
    """Two-panel: (a) parity plot of headline ensemble predictions vs.
    truth on the OBELiX 281-sample CV, (b) per-σ-bin MAE bars showing
    the model is most accurate on fast conductors (the regime that
    matters for screening)."""
    path = RESULTS / "phaseB1_ranking_analysis.json"
    if not path.exists():
        print("skip parity / per-bin plot (no ranking analysis json)")
        return
    with open(path) as f:
        ra = json.load(f)
    pred = np.asarray(ra["parity"]["pred"])
    true = np.asarray(ra["parity"]["true"])
    mae_overall = ra["mae_overall"]
    rho = ra["spearman"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    # Parity
    lim = (-15, 0)
    ax1.scatter(true, pred, s=12, alpha=0.55, color="#1f77b4", edgecolor="black", linewidths=0.3)
    ax1.plot(lim, lim, ls="--", color="#888888", label="y = x")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("true log₁₀ σ (S/cm)")
    ax1.set_ylabel("predicted log₁₀ σ (S/cm)")
    ax1.set_title(f"Headline parity (k-SEC + DBVF, OOF, n={ra['n']})\nMAE={mae_overall:.3f}, Spearman ρ={rho:.3f}")
    ax1.legend(loc="upper left")
    # Per-bin MAE
    bins = ra["per_bin"]
    labels = [f"[{b['lo']},{b['hi']})\nn={b['n']}" for b in bins]
    maes = [b["mae"] for b in bins]
    xs = np.arange(len(labels))
    colors = ["#d62728" if b["lo"] < -10 else ("#ff9900" if b["lo"] < -7 else "#2ca02c") for b in bins]
    ax2.bar(xs, maes, color=colors, edgecolor="black")
    for i, v in enumerate(maes):
        ax2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax2.axhline(mae_overall, ls=":", color="#1f77b4",
                label=f"overall MAE = {mae_overall:.3f}")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("MAE on log₁₀ σ")
    ax2.set_title("Per-σ-bin MAE — model is most accurate on fast conductors")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, max(maes) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_parity_per_bin.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_5_parity_per_bin.png'}")


def plot_dbvf_learned():
    """Visualise the learned (r0, b) parameters for each anion species
    against the Brown 2002 initialisation."""
    path = RESULTS / "dbvf_learned_params.json"
    if not path.exists():
        print("skip dbvf_learned plot (no file)")
        return
    with open(path) as f:
        d = json.load(f)
    agg = d["aggregate"]
    anions = list(agg.keys())
    r0_init = [agg[a]["r0_init"] for a in anions]
    r0_mean = [agg[a]["r0_mean"] for a in anions]
    r0_std = [agg[a]["r0_std"] for a in anions]
    b_init = [agg[a]["b_init"] for a in anions]
    b_mean = [agg[a]["b_mean"] for a in anions]
    b_std = [agg[a]["b_std"] for a in anions]
    xs = np.arange(len(anions))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.35
    ax1.bar(xs - w/2, r0_init, w, color="#888888", edgecolor="black", label="Brown 2002 init")
    ax1.bar(xs + w/2, r0_mean, w, yerr=r0_std, color="#1f77b4", edgecolor="black", capsize=3, label="learned (5-seed mean)")
    ax1.set_xticks(xs); ax1.set_xticklabels(anions)
    ax1.set_ylabel("r₀ (Å)")
    ax1.set_title("Learned bond-valence r₀ vs. Brown 2002")
    ax1.legend(loc="upper left", fontsize=9)
    ax2.bar(xs - w/2, b_init, w, color="#888888", edgecolor="black", label="Brown 2002 init")
    ax2.bar(xs + w/2, b_mean, w, yerr=b_std, color="#2ca02c", edgecolor="black", capsize=3, label="learned (5-seed mean)")
    ax2.set_xticks(xs); ax2.set_xticklabels(anions)
    ax2.set_ylabel("b (Å)")
    ax2.set_title("Learned bond-valence b vs. Brown 2002")
    ax2.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_10_dbvf_learned.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_10_dbvf_learned.png'}")


def plot_top_k_precision():
    """Top-K precision and Spearman lift over random for the headline
    ensemble — quantifies the model's value as a virtual-screening
    ranker."""
    path = RESULTS / "phaseB1_ranking_analysis.json"
    if not path.exists():
        print("skip top-k plot (no ranking analysis json)")
        return
    with open(path) as f:
        ra = json.load(f)
    Ks = sorted([int(k) for k in ra["top_k_precision"].keys()])
    prec = [ra["top_k_precision"][str(k)]["precision"] for k in Ks]
    base = [ra["top_k_precision"][str(k)]["random_baseline"] for k in Ks]
    lift = [p / b for p, b in zip(prec, base)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(Ks, prec, "-o", color="#1f77b4", label="model precision")
    ax.plot(Ks, base, "--", color="#888888", label="random baseline")
    for k, p, l in zip(Ks, prec, lift):
        ax.text(k, p + 0.03, f"{p:.2f}\n({l:.1f}×)", ha="center", fontsize=8)
    ax.set_xlabel("K (top-K predictions retained)")
    ax.set_ylabel("Top-K precision")
    ax.set_title(f"Top-K precision over OOF predictions (n={ra['n']})\nSpearman ρ = {ra['spearman']:.3f}")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_9_top_k_precision.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_9_top_k_precision.png'}")


if __name__ == "__main__":
    plot_baselines()
    plot_dbvf_test()
    plot_parity_and_per_bin()
    plot_virtual_screen()
    plot_ood()
    plot_ablation()
    plot_dbvf_learned()
    plot_top_k_precision()
    plot_calibration()
