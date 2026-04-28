"""WP4: Interpret what the cubic-harmonic directional filters learned.

For each KSEC block, the filter is a function W(|k|, K_0, K_4a, K_4b, K_6a, K_6b)
parameterized by a 6→D MLP. We probe what the network has learned by:

1. Computing |W(|k|, ...)|² as a function of (|k|, dominant Kubic axis)
   — does the network amplify direction-dependent components?
2. Comparing the relative magnitude of the directional channels (l=4, l=6)
   to the radial channel (l=0). High direction sensitivity = the filter is
   genuinely using the Kubic-harmonic information.
3. Plotting the gain MLP's response to a sweep through the K-space.

Output: figs/fig_5_kubic_interpret.png + a JSON dump of the channel statistics.

Usage:
    python scripts/22_interpret_kubic.py --checkpoint results/ksec_ckpt_seed0.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ionpath.models import KSECNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-fig", default="figs/fig_5_kubic_interpret.png")
    p.add_argument("--out-json", default="results/kubic_interpret.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = KSECNet(
        feature_dim=cfg.get("feature_dim", 96),
        num_blocks=cfg.get("num_blocks", 3),
        n_max=cfg.get("n_max", 2),
        dropout=0.0,
        tabular_dim=cfg.get("tabular_dim", 0),
        lattice_dim=cfg.get("lattice_dim", 0),
        geometric_dim=cfg.get("geometric_dim", 0),
    ).to(args.device)
    model.load_state_dict(ckpt["state"], strict=False)
    model.eval()

    log.info("model loaded: %d blocks, feature_dim=%d", cfg.get("num_blocks", 3), cfg.get("feature_dim", 96))

    # ----- Probe the Kubic filter response -----
    # Sweep |k| from 0.5 to 5.0 for 6 fixed directions corresponding to the 5 Kubic
    # invariants taking extreme values:
    #   - "radial-only" (Γ-like): K_0=0, K_4a=K_4b=K_6a=K_6b=0  (small |k|)
    #   - "001": cubic-axis maximum K_4a (along x-axis, K_4a = 1 - 3/5 = 2/5)
    #   - "111": body-diagonal, K_4a low, K_6b high (|x|=|y|=|z|)
    #   - "110": face-diagonal
    direction_probes = {
        "100":  np.array([1.0, 0.0, 0.0]),
        "110":  np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "111":  np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        "210":  np.array([2.0, 1.0, 0.0]) / np.sqrt(5),
        "211":  np.array([2.0, 1.0, 1.0]) / np.sqrt(6),
    }

    # Compute Kubic invariants for each probe direction
    from ionpath.models.kspace_conv import _kubic_invariants
    probe_kubics = {}
    for name, vec in direction_probes.items():
        v = torch.from_numpy(vec).float()
        kubic = _kubic_invariants(v.unsqueeze(0))[0].numpy()
        probe_kubics[name] = kubic
        log.info("direction %s  Kubic=[%.3f, %.3f, %.3f, %.3f, %.3f]",
                 name, *kubic.tolist())

    # Sweep |k|
    k_mags = np.linspace(0.5, 5.0, 30)

    # Probe each block's filter
    fig, axes = plt.subplots(1, len(model.blocks), figsize=(5 * len(model.blocks), 4), squeeze=False)
    block_stats = []
    for bi, block in enumerate(model.blocks):
        ax = axes[0, bi]
        responses_per_dir = {}
        for name, kubic in probe_kubics.items():
            magnitudes = []
            for km in k_mags:
                k_in = torch.tensor([[km, *kubic]], dtype=torch.float32, device=args.device)
                with torch.no_grad():
                    gain = block.filt.gain_mlp(k_in)[0].cpu().numpy()
                D = gain.shape[0] // 2
                w_real, w_imag = gain[:D], gain[D:]
                magnitudes.append(float(np.linalg.norm(np.concatenate([w_real, w_imag]))))
            responses_per_dir[name] = magnitudes
            ax.plot(k_mags, magnitudes, label=name)
        ax.set_xlabel("|k|"); ax.set_ylabel("‖W‖₂")
        ax.set_title(f"Block {bi}: filter magnitude vs direction")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Compute stat: spread across directions at fixed |k|=2
        idx_2 = int(np.argmin(np.abs(k_mags - 2.0)))
        responses_at_2 = np.array([responses_per_dir[d][idx_2] for d in direction_probes])
        spread = float(responses_at_2.std() / max(responses_at_2.mean(), 1e-6))
        block_stats.append(dict(
            block=bi, dir_spread_at_k2=spread,
            mean_response_at_k2=float(responses_at_2.mean()),
            response_per_dir_at_k2={d: float(responses_per_dir[d][idx_2]) for d in direction_probes},
        ))
        log.info("Block %d: directional spread at |k|=2 = %.3f (rel std)", bi, spread)

    plt.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=200)
    plt.close(fig)
    log.info("saved %s", args.out_fig)

    # Save JSON
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(dict(
        block_stats=block_stats,
        probe_directions={k: v.tolist() for k, v in direction_probes.items()},
        probe_kubics={k: v.tolist() for k, v in probe_kubics.items()},
    ), indent=2))
    log.info("saved %s", args.out_json)

    # Summary
    avg_spread = np.mean([b["dir_spread_at_k2"] for b in block_stats])
    log.info("==== Summary ====")
    log.info("Average directional spread (rel std at |k|=2): %.3f", avg_spread)
    log.info("Interpretation: higher = filter is MORE direction-sensitive.")
    log.info("If avg_spread > 0.1, the cubic-harmonic features are doing meaningful work.")


if __name__ == "__main__":
    main()
