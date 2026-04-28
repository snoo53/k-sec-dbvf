"""WP-A4: Compute approximate Li migration barriers using MACE-MP-0.

For each OBELiX CIF:
  1. Identify Li sites
  2. Find Li-Li pairs within cutoff (4 Å)
  3. For each pair, build a saddle-point configuration by moving one
     Li to the midpoint (no full NEB — single-point approximation)
  4. Compute MACE energy at original position E_init and at displaced
     position E_saddle. Approximate barrier = E_saddle - E_init.
  5. Average barriers across all Li-Li pairs in the cell → per-crystal
     migration-barrier estimate.

Output: data/cache/mace_barriers.npz with shape (n_obelix,) of per-crystal
        approximate migration barriers (eV).

Usage:
    python scripts/24_mace_barriers.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def cif_to_ase(cif_str: str):
    """Parse a CIF string into an ASE Atoms object via pymatgen."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    s = Structure.from_str(cif_str, fmt="cif")
    return AseAtomsAdaptor.get_atoms(s), s


def li_li_midpoint_displaced_atoms(atoms, ase_module, cutoff: float = 4.0,
                                     max_pairs: int = 5, mobile_z: int = 3,
                                     displacement_frac: float = 0.5):
    """For each near-neighbor Li-Li pair (within cutoff), return a list of
    (original_atoms, displaced_atoms) pairs where one Li is moved a fraction
    `displacement_frac` of the way to the partner Li (in cartesian, with
    min-image convention). 0.5 = midpoint; 0.25 = quarter-way (less strain).
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    li_indices = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == mobile_z]
    if len(li_indices) < 2:
        return []
    pairs = []
    inv_cell = np.linalg.inv(cell.array if hasattr(cell, "array") else np.array(cell))
    for ai, i in enumerate(li_indices):
        for j in li_indices[ai + 1:]:
            dr = positions[j] - positions[i]
            frac = dr @ inv_cell
            frac -= np.round(frac)
            dr_min = frac @ (cell.array if hasattr(cell, "array") else np.array(cell))
            d = float(np.linalg.norm(dr_min))
            if d < cutoff:
                pairs.append((i, j, d, dr_min))
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:max_pairs]

    out = []
    for (i, j, d, dr_min) in pairs:
        a_disp = atoms.copy()
        target = positions[i] + displacement_frac * dr_min
        new_pos = a_disp.get_positions()
        new_pos[i] = target
        a_disp.set_positions(new_pos)
        out.append((atoms.copy(), a_disp, d))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="data/processed/records.jsonl")
    p.add_argument("--out", default="data/cache/mace_barriers.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--max-pairs", type=int, default=5)
    p.add_argument("--displacement-frac", type=float, default=0.25,
                   help="Fraction of pair vector to displace (0.5=midpoint; 0.25=quarter, less strain)")
    p.add_argument("--max-barrier-eV", type=float, default=15.0,
                   help="Reject barriers above this (likely unphysical strain)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of crystals processed (for testing)")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ionpath.data import read_records

    log.info("loading MACE-MP-0 (downloads ~100 MB on first run) ...")
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    log.info("MACE loaded")

    records = read_records(Path(args.records))
    log.info("read %d records", len(records))
    if args.limit:
        records = records[: args.limit]

    # Outputs aligned with full record list (NaN where no CIF or computation failed)
    n = len(records)
    barrier_mean = np.full(n, np.nan, dtype=np.float32)
    barrier_max = np.full(n, np.nan, dtype=np.float32)
    n_pairs_used = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    n_done = 0
    for i, r in enumerate(records):
        if r.cif is None or not r.cif.strip():
            continue
        try:
            atoms, struct = cif_to_ase(r.cif)
        except Exception as exc:  # noqa: BLE001
            log.debug("cif parse failed for %d: %s", i, exc)
            continue
        if "Li" not in atoms.get_chemical_symbols():
            continue

        try:
            from ase.constraints import FixAtoms  # noqa: F401  (placeholder, not used)
            pairs = li_li_midpoint_displaced_atoms(
                atoms, ase_module=None, cutoff=args.cutoff, max_pairs=args.max_pairs,
                displacement_frac=args.displacement_frac,
            )
            if not pairs:
                continue

            barriers = []
            for (a_init, a_disp, d) in pairs:
                a_init.calc = calc
                a_disp.calc = calc
                e_init = float(a_init.get_potential_energy())
                e_disp = float(a_disp.get_potential_energy())
                bar = e_disp - e_init
                # Sensible barrier window (avoid sign-flips and atomic-overlap blowups)
                if 0 < bar < args.max_barrier_eV:
                    barriers.append(bar)

            if barriers:
                barrier_mean[i] = float(np.mean(barriers))
                barrier_max[i] = float(np.max(barriers))
                n_pairs_used[i] = len(barriers)
                n_done += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("mace failed for %d: %s", i, exc)
            continue

        if (i + 1) % 10 == 0:
            log.info("  %d/%d  done=%d  mean_bar=%.2f eV  (%.0fs)",
                     i + 1, len(records), n_done,
                     float(np.nanmean(barrier_mean)) if n_done > 0 else 0.0,
                     time.time() - t0)

    log.info("computed barriers for %d/%d crystals in %.0fs", n_done, len(records), time.time() - t0)
    if n_done > 0:
        log.info("barrier stats: mean=%.3f eV  median=%.3f eV  std=%.3f eV",
                 float(np.nanmean(barrier_mean)), float(np.nanmedian(barrier_mean)),
                 float(np.nanstd(barrier_mean)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             barrier_mean=barrier_mean,
             barrier_max=barrier_max,
             n_pairs_used=n_pairs_used)
    log.info("saved to %s", args.out)


if __name__ == "__main__":
    main()
