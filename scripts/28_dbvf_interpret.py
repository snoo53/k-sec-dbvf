"""Extract the learned DBVF (r0, b) bond-valence parameters from each
Phase B1 seed checkpoint and compare them against the Brown 2002
tabulated initialisation. Saves results as JSON for the manuscript SI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ionpath.models.bond_valence_field import _LI_BV_INIT, _Z, LearnableBVParams


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -50, 50)))


def main():
    out: dict[str, object] = {"per_seed": [], "anions": list(_LI_BV_INIT.keys())}
    init_r0 = {sym: r for sym, (r, _) in _LI_BV_INIT.items()}
    init_b = {sym: b for sym, (_, b) in _LI_BV_INIT.items()}
    out["init_r0"] = init_r0
    out["init_b"] = init_b

    for seed in range(5):
        ckpt_path = ROOT / f"results/ksec_phaseB1_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"missing: {ckpt_path}")
            continue
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # The state-dict may be the model itself or a dict with 'state'/'model_state_dict'
        if isinstance(sd, dict):
            if "state" in sd:
                sd = sd["state"]
            elif "model_state_dict" in sd:
                sd = sd["model_state_dict"]
        # Find DBVF param keys
        r0_raw_key = next((k for k in sd if k.endswith("r0_raw")), None)
        b_raw_key = next((k for k in sd if k.endswith("b_raw")), None)
        if r0_raw_key is None or b_raw_key is None:
            print(f"seed {seed}: no DBVF keys (sample keys: {list(sd.keys())[:5]})")
            continue
        r0 = softplus(sd[r0_raw_key].cpu().numpy())
        b = softplus(sd[b_raw_key].cpu().numpy())
        per_anion = {}
        for sym in _LI_BV_INIT:
            j = _Z[sym]
            per_anion[sym] = {
                "r0_learned": float(r0[j]),
                "b_learned": float(b[j]),
                "r0_init": init_r0[sym],
                "b_init": init_b[sym],
                "r0_delta": float(r0[j] - init_r0[sym]),
                "b_delta": float(b[j] - init_b[sym]),
            }
        out["per_seed"].append({"seed": seed, "anions": per_anion})

    if not out["per_seed"]:
        print("no seeds found")
        return

    # Aggregate mean ± std across seeds per anion
    agg = {}
    for sym in _LI_BV_INIT:
        r0s = [s["anions"][sym]["r0_learned"] for s in out["per_seed"]]
        bs = [s["anions"][sym]["b_learned"] for s in out["per_seed"]]
        agg[sym] = {
            "r0_mean": float(np.mean(r0s)), "r0_std": float(np.std(r0s)),
            "b_mean": float(np.mean(bs)), "b_std": float(np.std(bs)),
            "r0_init": init_r0[sym], "b_init": init_b[sym],
            "r0_shift": float(np.mean(r0s) - init_r0[sym]),
            "b_shift": float(np.mean(bs) - init_b[sym]),
        }
    out["aggregate"] = agg

    out_path = ROOT / "results/dbvf_learned_params.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")

    # Pretty print
    print("\nLearned (r0, b) across 5 seeds vs Brown 2002 initialisation:")
    print(f"{'anion':6} {'r0_init':>9} {'r0_learned (mean±std)':>26} {'Δr0':>8}    {'b_init':>8} {'b_learned (mean±std)':>26} {'Δb':>8}")
    for sym in _LI_BV_INIT:
        a = agg[sym]
        print(f"{sym:6} {a['r0_init']:9.3f} {a['r0_mean']:14.3f} ± {a['r0_std']:.3f}      {a['r0_shift']:+.3f}    {a['b_init']:8.3f} {a['b_mean']:14.3f} ± {a['b_std']:.3f}      {a['b_shift']:+.3f}")


if __name__ == "__main__":
    main()
