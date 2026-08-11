"""The untrained force field against the Q-Chem EDA Pauli component.

The headline check for the whole Pauli path: it exercises the priors, the Angstrom->bohr
conversion, the sqrt(b_i b_j) combination, the damped interaction tensor, the
inter-fragment pair list, and the EDA data loader simultaneously, against a physical target
rather than an internal identity. If any of those is wrong the agreement collapses.

It also records what the learned parts actually have to do. The pyCMM priors get the
*shape* right (correlation 0.99) and the *scale* wrong by ~18%, and a per-pair-type rescale
-- which two learnable per-species log-q deviations can nearly reach -- closes most of that.
So the neural correction is genuinely a delta, not the bulk of the signal.
"""

import numpy as np
import pytest
import torch

from rsfff.ff.units import KJMOL_PER_HARTREE
from rsfff.train.data import load_extxyz

from conftest import DATA_W2, DATA_W3
from test_ff_pauli import make_model, run

N_FRAMES = 200


def predict(dataset, n=N_FRAMES, max_rank=0, **model_kw):
    feat, model = make_model(correction=False, max_rank=max_rank, **model_kw)
    batch = dataset.flat_batch(range(n))
    with torch.no_grad():
        out = run(feat, model, batch)
    return out.energy_ff, batch.eda["mod_pauli"]


@pytest.fixture(scope="module")
def w2():
    return load_extxyz(DATA_W2, dtype=torch.float64)


@pytest.fixture(scope="module")
def w3():
    return load_extxyz(DATA_W3, dtype=torch.float64)


def test_untrained_ff_reproduces_eda_mod_pauli(w2):
    """Priors + formula alone, no ML: the right shape at roughly the right magnitude."""
    pred, label = predict(w2)
    mae = (pred - label).abs().mean().item() * KJMOL_PER_HARTREE
    corr = torch.corrcoef(torch.stack([pred, label]))[0, 1].item()
    assert corr > 0.98, f"correlation {corr:.4f}"
    assert mae < 10.0, f"MAE {mae:.3f} kJ/mol"


def test_the_residual_is_a_scale_error_the_charges_can_absorb(w2):
    """Where the error lives: a per-pair-type rescale, i.e. what learnable q buys.

    The measurement that justifies calling the correction head a delta. Regressing the
    prediction onto the label leaves a residual several times smaller than the raw MAE, so
    most of the untrained error is one multiplicative constant rather than missing physics.
    """
    pred, label = predict(w2)
    p = pred.numpy() * KJMOL_PER_HARTREE
    y = label.numpy() * KJMOL_PER_HARTREE
    slope, intercept = np.polyfit(p, y, 1)
    raw = np.abs(p - y).mean()
    rescaled = np.abs(slope * p + intercept - y).mean()
    assert 0.7 < slope < 0.95, f"slope {slope:.3f}"          # priors over-predict ~18%
    assert rescaled < 0.4 * raw, f"{rescaled:.3f} vs {raw:.3f} kJ/mol"


def test_agreement_holds_for_larger_clusters(w3):
    """Not a two-body accident: the same priors track trimers too."""
    pred, label = predict(w3)
    corr = torch.corrcoef(torch.stack([pred, label]))[0, 1].item()
    assert corr > 0.98, f"correlation {corr:.4f}"


def test_dipoles_do_not_move_the_untrained_prediction(w2):
    """max_rank=1 is a strict superset at init, so the anchor above covers both."""
    charges_only, label = predict(w2, n=50)
    with_dipoles, _ = predict(w2, n=50, max_rank=1)
    torch.testing.assert_close(charges_only, with_dipoles, rtol=1e-14, atol=0)
