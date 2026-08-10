"""The learned pair correction: zero at init, symmetric, and strictly short-ranged.

The compact-support property is the one that matters architecturally -- it is what keeps
the correction from quietly absorbing mid-range energy that belongs to the explicit force
field (``docs/range_separated_mlip.md`` §7, "gauge leakage").
"""

import pytest
import torch

from rsfff.mlip.pair_heads import PairEnergyHead
from rsfff.train.data import Batch, load_extxyz

from conftest import DATA_W2
from test_ff_dispersion import make_model, run

NEIGHBOR_TYPES = [1, 8]


@pytest.fixture(scope="module")
def w2_dataset():
    return load_extxyz(DATA_W2, dtype=torch.float64)


def make_head(*, randomize=True, seed=5, **over):
    kwargs = dict(emb_dim=8, hidden=16, depth=1, n_radial=8, r_on=4.0, r_off=5.0)
    kwargs.update(over)
    head = PairEnergyHead(12, len(NEIGHBOR_TYPES), **kwargs)
    if randomize:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in head.parameters():
                p.add_(0.5 * torch.randn(p.shape, generator=g))
    return head


def toy_inputs(distances, species=(8, 8)):
    """One pair per distance, all placed along x so `r` is the distance."""
    n = 2 * len(distances)
    positions = torch.zeros(n, 3, dtype=torch.float64)
    pair_index = torch.empty(2, len(distances), dtype=torch.long)
    for k, d in enumerate(distances):
        positions[2 * k] = torch.tensor([10.0 * k, 0.0, 0.0])
        positions[2 * k + 1] = torch.tensor([10.0 * k + d, 0.0, 0.0])
        pair_index[:, k] = torch.tensor([2 * k, 2 * k + 1])
    lut = {1: 0, 8: 1}
    species_idx = torch.tensor([lut[s] for s in species] * len(distances))
    g = torch.Generator().manual_seed(11)
    feats = torch.randn(n, 12, generator=g, dtype=torch.float64)
    return feats, species_idx, positions, pair_index


def test_zero_init_gives_pure_force_field(w2_dataset):
    """A fresh model is *exactly* the force field -- not approximately."""
    feat, model = make_model(correction=True)
    out = run(feat, model, w2_dataset.flat_batch([0, 1]))
    assert torch.all(out.e_pair_corr == 0.0)
    assert torch.all(out.energy_corr == 0.0)
    assert torch.equal(out.energy, out.energy_ff)


def test_symmetric_under_pair_swap():
    head = make_head()
    feats, species_idx, positions, pair_index = toy_inputs([2.0, 3.0, 4.5])
    forward = head(feats, species_idx, positions, pair_index)
    swapped = head(feats, species_idx, positions, pair_index.flip(0))
    assert torch.allclose(forward, swapped, atol=1e-14)


def test_compact_support_beyond_r_off():
    """Exactly zero past r_off, and nonzero before it -- with randomized weights."""
    head = make_head()
    feats, species_idx, positions, pair_index = toy_inputs([3.0, 4.9, 5.0, 5.5, 8.0])
    de = head(feats, species_idx, positions, pair_index)
    assert de[0].abs() > 0
    assert de[1].abs() > 0
    assert torch.all(de[2:] == 0.0)      # exact zeros, not merely small


def test_envelope_derivative_is_continuous_at_r_off():
    """C2 smoothstep: the force must not kink where the correction switches off."""
    head = make_head()
    h = 1e-5

    def de_at(d):
        feats, species_idx, positions, pair_index = toy_inputs([d])
        positions = positions.clone().requires_grad_(True)
        value = head(feats, species_idx, positions, pair_index).sum()
        (g,) = torch.autograd.grad(value, positions)
        return value.item(), g[1, 0].item()

    for d in (4.9, 4.99, 5.0, 5.01):
        analytic = de_at(d)[1]
        numeric = (de_at(d + h)[0] - de_at(d - h)[0]) / (2 * h)
        assert analytic == pytest.approx(numeric, abs=1e-9)


def test_species_changes_the_prediction():
    """inv_feats carries the *neighbor* density, so the species embedding is load-bearing.

    Identical distance and identical features, differing only in element: without the
    embedding these two would be indistinguishable.
    """
    head = make_head()
    feats, _, positions, pair_index = toy_inputs([3.0])
    oo = head(feats, torch.tensor([1, 1]), positions, pair_index)
    hh = head(feats, torch.tensor([0, 0]), positions, pair_index)
    assert not torch.allclose(oo, hh)


def test_output_scale_is_respected():
    """energy_scale sets the head's magnitude, so it competes on equal footing with the FF."""
    head = make_head(energy_scale=1e-3)
    feats, species_idx, positions, pair_index = toy_inputs([2.0, 2.5, 3.0, 3.5])
    de = head(feats, species_idx, positions, pair_index).abs().mean().item()
    assert 1e-5 < de < 1e-1


def test_correction_switches_off_when_fragments_separate(w2_dataset):
    """Pull the molecules apart: the learned term vanishes, only the FF tail survives."""
    feat, model = make_model(correction=True, randomize=True)
    batch = w2_dataset.flat_batch([0])
    batch.positions = batch.positions.clone()
    batch.positions[3:] += 20.0
    out = run(feat, model, batch)
    assert out.energy_corr.item() == 0.0


def test_rejects_inverted_envelope():
    with pytest.raises(ValueError, match="r_off > r_on"):
        PairEnergyHead(12, 2, r_on=5.0, r_off=4.0)


def test_contributes_to_forces(w2_dataset):
    """dE must reach the forces -- a correction that only shifts energies is useless."""
    feat, model = make_model(correction=True, randomize=True)
    batch = w2_dataset.flat_batch([0])
    batch.positions = batch.positions.clone().requires_grad_(True)
    out = run(feat, model, batch)
    (g_corr,) = torch.autograd.grad(out.energy_corr.sum(), batch.positions)
    assert g_corr.abs().max() > 0
