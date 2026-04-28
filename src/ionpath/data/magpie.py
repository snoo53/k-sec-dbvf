"""Magpie composition featurizer.

Produces the ~132-dim Magpie feature vector (Ward 2016) from a composition
string like "Li7La3Zr2O12". This is the same feature set LightGBM uses to
achieve MAE 1.099 on OBELiX — folding it into the k-SEC readout is the
single intervention most likely to beat the tabular ceiling.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)


_FEATURIZER = None
_FEATURE_DIM = None


def _get_featurizer():
    global _FEATURIZER, _FEATURE_DIM
    if _FEATURIZER is not None:
        return _FEATURIZER
    from matminer.featurizers.composition import ElementProperty

    ep = ElementProperty.from_preset("magpie")
    _FEATURIZER = ep
    _FEATURE_DIM = len(ep.feature_labels())
    return ep


def magpie_feature_dim() -> int:
    if _FEATURE_DIM is None:
        _get_featurizer()
    return int(_FEATURE_DIM)


def featurize_composition(composition: str) -> np.ndarray:
    """Return a (F,) float32 Magpie feature vector for a composition string.

    NaNs are replaced by 0. Unparseable compositions return zeros.
    """
    from pymatgen.core import Composition

    ep = _get_featurizer()
    try:
        comp = Composition(composition)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = np.asarray(ep.featurize(comp), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        log.debug("Magpie featurize failed for %s: %s", composition, exc)
        vec = np.zeros(magpie_feature_dim(), dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def featurize_many(compositions: list[str]) -> np.ndarray:
    """Batch-featurize; returns (N, F)."""
    rows = [featurize_composition(c) for c in compositions]
    return np.stack(rows, axis=0) if rows else np.zeros((0, magpie_feature_dim()), dtype=np.float32)
