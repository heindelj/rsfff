"""Charge-aware featurizer path: equivalence with the species scatter, empty-edge
systems, and gradient parity through the geometry cache."""

import pytest
import torch

from rsfff.features import FlatLambdaSOAPFeaturizer

from conftest import make_batch, water_h3o_batch


def _featurizer(**over):
    kw = dict(
        cutoff=5.0, n_max=3, l_max=2, neighbor_types=[1, 8],
        selected_lambdas=(0, 1, 2), density_channels=4,
    )
    kw.update(over)
    return FlatLambdaSOAPFeaturizer(**kw)


def test_weighted_scatter_reproduces_forward():
    feat = _featurizer()
    batch = water_h3o_batch(jitter=0.02)
    ref = feat(batch)
    cache = feat.precompute_geometry(batch)
    one_hot = torch.nn.functional.one_hot(cache.species_idx, 2).to(batch.positions.dtype)
    got = feat.features_from_weights(cache, one_hot)
    assert torch.equal(ref.inv_feats, got.inv_feats)
    assert torch.equal(ref.equiv_feats, got.equiv_feats)
    assert torch.equal(ref.vec_feats, got.vec_feats)


def test_gradient_parity_through_cache():
    feat = _featurizer()
    batch = water_h3o_batch(jitter=0.02)

    def loss_direct():
        batch.positions.grad = None
        batch.positions.requires_grad_(True)
        out = feat(batch)
        return out.inv_feats.pow(2).sum() + out.equiv_feats.pow(2).sum()

    def loss_cached():
        batch.positions.grad = None
        batch.positions.requires_grad_(True)
        cache = feat.precompute_geometry(batch)
        w = torch.nn.functional.one_hot(cache.species_idx, 2).to(batch.positions.dtype)
        out = feat.features_from_weights(cache, w)
        return out.inv_feats.pow(2).sum() + out.equiv_feats.pow(2).sum()

    (g1,) = torch.autograd.grad(loss_direct(), batch.positions)
    (g2,) = torch.autograd.grad(loss_cached(), batch.positions)
    assert (g1 - g2).abs().max() < 1e-12


def test_single_atom_zero_features():
    feat = _featurizer(charge_channels=3)
    single = make_batch([torch.zeros(1, 3)], [torch.tensor([8])], [0.0])
    cache = feat.precompute_geometry(single)
    assert cache.edge_index.shape == (2, 0)
    out = feat.features_from_weights(cache, torch.ones(1, 3))
    assert out.inv_feats.abs().max() == 0.0
    assert out.equiv_feats.abs().max() == 0.0


def test_charge_channels_guard_forward():
    feat = _featurizer(charge_channels=3)
    with pytest.raises(RuntimeError, match="charge_channels"):
        feat(water_h3o_batch())


def test_charge_channels_rejects_bispectrum():
    import types

    with pytest.raises(NotImplementedError):
        _featurizer(charge_channels=3, bispectrum=types.SimpleNamespace(out_mul=4))


def test_weight_shape_validation():
    feat = _featurizer(charge_channels=3)
    batch = water_h3o_batch()
    cache = feat.precompute_geometry(batch)
    with pytest.raises(ValueError, match="atom_weights"):
        feat.features_from_weights(cache, torch.ones(cache.num_atoms, 5))


def test_invariance_with_arbitrary_weights():
    """inv_feats invariant, |equiv|/|vec| norms invariant under rotation+translation
    for generic (non-species) atom weights."""
    from e3nn import o3

    feat = _featurizer(charge_channels=3)
    batch = water_h3o_batch(jitter=0.02)
    w = torch.randn(batch.positions.shape[0], 3)

    out = feat.features_from_weights(feat.precompute_geometry(batch), w)

    R = o3.rand_matrix(1)[0].to(torch.float64)
    rotated = water_h3o_batch(jitter=0.02)
    rotated.positions = rotated.positions @ R.t() + torch.tensor([1.0, -2.0, 0.5])
    out_r = feat.features_from_weights(feat.precompute_geometry(rotated), w)

    assert (out.inv_feats - out_r.inv_feats).abs().max() < 1e-10
    for name in ("equiv_feats", "vec_feats"):
        n0 = getattr(out, name).pow(2).sum(dim=1)
        n1 = getattr(out_r, name).pow(2).sum(dim=1)
        assert (n0 - n1).abs().max() < 1e-10
