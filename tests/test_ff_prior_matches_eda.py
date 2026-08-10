"""The untrained force field against the Q-Chem EDA dispersion component.

This is the headline check for the whole dispersion path: it exercises the priors, the
Angstrom->bohr conversion, the geometric-mean combination, the inter-fragment pair list,
and the EDA data loader simultaneously, against a physical target rather than an internal
identity. If any of those is wrong the agreement collapses immediately.

It also establishes what the neural correction actually has to do: with no ML at all the
backbone is already within a few tenths of a kJ/mol, so the correction is genuinely a
delta and not the bulk of the signal.
"""

import pytest
import torch

from rsfff.ff.units import KJMOL_PER_HARTREE
from rsfff.train.data import load_extxyz

from conftest import DATA_W2, DATA_W3
from test_ff_dispersion import make_model, run

N_FRAMES = 200


def predict(dataset, n=N_FRAMES, **model_kw):
    feat, model = make_model(correction=False, learn_r0=False, **model_kw)
    batch = dataset.flat_batch(range(n))
    with torch.no_grad():
        out = run(feat, model, batch)
    return out.energy_ff, batch.eda["disp"]


@pytest.fixture(scope="module")
def w2():
    return load_extxyz(DATA_W2, dtype=torch.float64)


def test_untrained_ff_reproduces_eda_disp(w2):
    """Priors + formula alone, no switch, no ML: sub-kJ/mol against wB97X-V VV10."""
    pred, label = predict(w2, r0_init=1e-6)
    mae = (pred - label).abs().mean().item() * KJMOL_PER_HARTREE
    corr = torch.corrcoef(torch.stack([pred, label]))[0, 1].item()
    assert mae < 1.0, f"MAE {mae:.3f} kJ/mol"
    assert corr > 0.98, f"correlation {corr:.4f}"


def test_prediction_has_the_right_sign_and_scale(w2):
    """A units or combination-rule error would show up here as an order-of-magnitude miss."""
    pred, label = predict(w2, r0_init=1e-6)
    assert torch.all(pred < 0)
    ratio = pred.mean().item() / label.mean().item()
    assert 0.8 < ratio < 1.2


def test_transfers_to_larger_clusters():
    """Same per-atom priors, bigger clusters: the pair sum must extend without retuning."""
    w3 = load_extxyz(DATA_W3, dtype=torch.float64)
    pred, label = predict(w3, n=100, r0_init=1e-6)
    mae = (pred - label).abs().mean().item() * KJMOL_PER_HARTREE
    corr = torch.corrcoef(torch.stack([pred, label]))[0, 1].item()
    assert mae < 2.0, f"MAE {mae:.3f} kJ/mol"
    assert corr > 0.95, f"correlation {corr:.4f}"


def test_default_r0_leaves_a_workable_residual(w2):
    """At the r0=2.0 default the FF still carries most of the target.

    The remainder is what the pair correction is being asked to learn. If a change to the
    defaults makes this residual dominate, the "delta learning" framing has quietly become
    "the network does everything" -- which is the range-separation ambiguity in
    docs/range_separated_mlip.md §7.
    """
    pred, label = predict(w2, r0_init=2.0)
    residual = (label - pred).abs().mean().item()
    assert residual < 0.5 * label.abs().mean().item()
