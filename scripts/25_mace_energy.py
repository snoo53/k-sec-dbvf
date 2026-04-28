"""WP-A4 simpler: compute MACE-MP-0 total energies + per-atom energies
for every OBELiX CIF. These are DFT-grade auxiliary features.

For each CIF:
  - Total energy (eV)
  - Energy per atom (eV/atom) — this is the real per-cell stability
  - Energy per Li atom (eV/Li) — Li-specific energy proxy
  - Force RMS (eV/Å) — large = far from equilibrium, suggests strain

Output: data/cache/mace_energies.npz aligned with records.

Usage:
    python scripts/25_mace_energy.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="data/processed/records.jsonl")
    p.add_argument("--out", default="data/cache/mace_energies.npz")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ionpath.data import read_records

    log.info("loading MACE-MP-0 ...")
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    log.info("MACE loaded")

    records = read_records(Path(args.records))
    n = len(records)
    log.info("read %d records", n)

    e_total = np.full(n, np.nan, dtype=np.float32)
    e_per_atom = np.full(n, np.nan, dtype=np.float32)
    e_per_li = np.full(n, np.nan, dtype=np.float32)
    force_rms = np.full(n, np.nan, dtype=np.float32)

    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    def to_ordered(struct):
        """Replace each disordered site with its dominant species; ASE
        doesn't support fractional occupancies but we just want one
        approximate energy per crystal."""
        if all(site.is_ordered for site in struct):
            return struct
        new_species = []
        new_coords = []
        for site in struct:
            if site.is_ordered:
                new_species.append(site.specie)
                new_coords.append(site.frac_coords)
            else:
                # dominant species at this site
                dominant = max(site.species, key=site.species.get)
                new_species.append(dominant)
                new_coords.append(site.frac_coords)
        return Structure(struct.lattice, new_species, new_coords)

    t0 = time.time()
    n_done = 0
    n_disordered = 0
    for i, r in enumerate(records):
        if r.cif is None or not r.cif.strip():
            continue
        try:
            s = Structure.from_str(r.cif, fmt="cif")
            if not all(site.is_ordered for site in s):
                s = to_ordered(s)
                n_disordered += 1
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.calc = calc
            E = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            n_at = len(atoms)
            n_li = sum(1 for sym in atoms.get_chemical_symbols() if sym == "Li")
            e_total[i] = E
            e_per_atom[i] = E / max(n_at, 1)
            e_per_li[i] = E / max(n_li, 1) if n_li > 0 else np.nan
            force_rms[i] = float(np.sqrt(np.mean(forces ** 2)))
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("failed %d: %s", i, exc)
            continue

        if (i + 1) % 50 == 0 or n_done % 50 == 1:
            log.info("  %d/%d  done=%d  E_per_atom mean=%.3f eV  (%.0fs)",
                     i + 1, n, n_done,
                     float(np.nanmean(e_per_atom)) if n_done > 0 else 0.0,
                     time.time() - t0)

    log.info("computed energies for %d/%d (%d ordered via dominant-species) in %.0fs",
             n_done, n, n_disordered, time.time() - t0)
    if n_done > 0:
        log.info("E/atom: mean=%.3f std=%.3f  E/Li: mean=%.3f std=%.3f  F_rms: mean=%.3f std=%.3f",
                 float(np.nanmean(e_per_atom)), float(np.nanstd(e_per_atom)),
                 float(np.nanmean(e_per_li)), float(np.nanstd(e_per_li)),
                 float(np.nanmean(force_rms)), float(np.nanstd(force_rms)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             e_total=e_total, e_per_atom=e_per_atom,
             e_per_li=e_per_li, force_rms=force_rms)
    log.info("saved to %s", args.out)


if __name__ == "__main__":
    main()
