"""Smoke tests for k-SEC."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from ionpath.data import CanonicalRecord, CrystalGraph, build_crystal_graph
from ionpath.models import KSECNet
from ionpath.utils import (
    generate_wyckoff_wavevectors,
    precompute_orbits,
    torch_fourier_basis,
)


def test_wyckoff_wavevectors():
    wv = generate_wyckoff_wavevectors(n_max=2)
    assert wv.ndim == 2 and wv.shape[1] == 3
    orbits = precompute_orbits(wv)
    assert len(orbits) == wv.shape[0]
    r = torch.rand(5, 3)
    basis = torch_fourier_basis(r, torch.from_numpy(wv).float(), orbits)
    assert basis.shape == (5, 2 * wv.shape[0])
    assert torch.isfinite(basis).all()
    # Cosine channels are means of cosines → magnitude ≤ 1
    cos_channels = basis[..., 0::2]
    assert (cos_channels.abs() <= 1.001).all()


def test_canonical_record_roundtrip():
    r = CanonicalRecord(
        record_id="test-1", source="unit-test",
        fidelity="experimental", composition="Li7La3Zr2O12",
        mobile_ion="Li", T_K=298.15, log_sigma=-4.0,
    )
    d = r.to_dict()
    r2 = CanonicalRecord(**d)
    assert r2.composition == "Li7La3Zr2O12" and r2.log_sigma == -4.0


def test_build_crystal_graph_from_cif():
    cif = """data_Li2O
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
_cell_length_a 4.619
_cell_length_b 4.619
_cell_length_c 4.619
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Li1 Li 0.25 0.25 0.25 1.0
Li2 Li 0.75 0.75 0.75 1.0
O1 O 0 0 0 1.0
"""
    cg = build_crystal_graph(cif, "Li")
    assert cg is not None
    assert cg.atom_z.shape[0] == cg.frac_pos.shape[0] > 0
    assert cg.cell.shape == (3, 3)
    assert cg.mobile_ion == "Li"


def test_ksec_forward_backward():
    # Small synthetic batch
    atom_z = torch.tensor([3, 3, 8, 3, 8, 3, 8, 8], dtype=torch.long)
    frac_pos = torch.rand(8, 3)
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    model = KSECNet(feature_dim=32, num_blocks=2, n_max=2)
    out = model.forward_structure(atom_z, frac_pos, batch_idx, num_graphs=2)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    loss = ((out - torch.zeros_like(out)) ** 2).mean()
    loss.backward()
    # At least one grad should be non-None
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_ksec_equivariance_under_translation():
    """Translations of all atoms by the same fractional offset should leave the
    structure factor invariant (up to a global phase that averages out)."""
    torch.manual_seed(0)
    atom_z = torch.tensor([3, 3, 8], dtype=torch.long)
    frac_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    batch_idx = torch.tensor([0, 0, 0], dtype=torch.long)

    model = KSECNet(feature_dim=32, num_blocks=1, n_max=1)
    model.eval()

    y1 = model.forward_structure(atom_z, frac_pos, batch_idx, num_graphs=1)
    # Translate all atoms by the same vector
    shift = torch.tensor([0.13, 0.07, 0.22])
    y2 = model.forward_structure(atom_z, (frac_pos + shift) % 1.0, batch_idx, num_graphs=1)
    # Within a few decimal places the real-space translation doesn't change the
    # readout (cos/sin average over orbit cancels the translation phase).
    assert torch.allclose(y1, y2, atol=5e-2), f"translation moved output by {(y1-y2).abs().max()}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
