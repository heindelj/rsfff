"""Phase-4 tests: permanent multipole heads and the exact charge projection."""

from __future__ import annotations

import torch
from e3nn import o3

from rsfff.ff.film.permanent import PermanentMultipoleHeads
from rsfff.ff.multipole import irrep2_to_spherical
from rsfff.ff.response import DEFAULT_Q0_PRIOR
from rsfff.train.data import Batch

from film_helpers import make_projector, make_state, water_cluster_batch


def make_heads(latent_dim=8):
    proj = make_projector()
    feat = proj.featurizer
    q0 = torch.tensor([DEFAULT_Q0_PRIOR[1], DEFAULT_Q0_PRIOR[8]])  # species order H, O
    heads = PermanentMultipoleHeads(
        latent_dim,
        feat.feature_dims.get(1),
        feat.feature_dims.get(2),
        feat.n_species,
        q0_prior=q0,
        irrep2_to_spherical=irrep2_to_spherical(feat.backend.irrep6_to_voigt()),
        hidden=24, depth=1, equiv_channels=6,
    )
    return heads, proj


def test_charge_projection_exact():
    """Per-fragment sums equal the formal charges exactly, for random head outputs."""
    heads, proj = make_heads()
    batch = water_cluster_batch(3)
    state = make_state(batch, proj)
    with torch.no_grad():
        for p in heads.parameters():
            p.add_(0.3 * torch.randn(p.shape))

    out = proj(batch, state)
    z = torch.randn(9, 8)
    q, _, _ = heads(z, out.x_in, state)
    sums = torch.zeros(3).index_add_(0, state.fragment_idx, q)
    assert torch.allclose(sums, state.fragment_charge, atol=1e-14)

    # nonzero formal charges too
    state.fragment_charge = torch.tensor([1.0, -1.0, 0.0])
    q, _, _ = heads(z, out.x_in, state)
    sums = torch.zeros(3).index_add_(0, state.fragment_idx, q)
    assert torch.allclose(sums, state.fragment_charge, atol=1e-14)


def test_fresh_heads_at_prior():
    """Zero-init: charges are exactly the projected q0 prior; mu and quad exactly zero."""
    heads, proj = make_heads()
    batch = water_cluster_batch(2)
    state = make_state(batch, proj)
    out = proj(batch, state)
    q, mu, quad_s = heads(torch.randn(6, 8), out.x_in, state)

    expected = torch.tensor([DEFAULT_Q0_PRIOR[8], DEFAULT_Q0_PRIOR[1], DEFAULT_Q0_PRIOR[1]])
    expected = expected - expected.sum() / 3.0        # projected onto Q_f = 0
    assert torch.allclose(q, expected.repeat(2), atol=1e-12)
    assert torch.count_nonzero(mu) == 0
    assert torch.count_nonzero(quad_s) == 0


def test_equivariance():
    """mu rotates as a vector; quad_s rotates under the Wigner D2 in the spherical basis."""
    heads, proj = make_heads()
    with torch.no_grad():
        for p in heads.parameters():
            p.add_(0.1 * torch.randn(p.shape))
    batch = water_cluster_batch(2)
    state = make_state(batch, proj)
    z = torch.randn(6, 8)

    out = proj(batch, state)
    q, mu, quad_s = heads(z, out.x_in, state)

    R = o3.rand_matrix().to(batch.positions.dtype)
    rotated = Batch(**{**batch.__dict__, "positions": batch.positions @ R.T})
    out_r = proj(rotated, state)
    q_r, mu_r, quad_r = heads(z, out_r.x_in, state)

    assert torch.allclose(q_r, q, atol=1e-10)
    assert torch.allclose(mu_r, mu @ R.T, atol=1e-10)
    # quad_s is in the (q20, q21c, q21s, q22c, q22s) spherical convention; check rotation
    # through the Cartesian form, which rotates as R Q R^T.
    from rsfff.ff.multipole import spherical_to_cartesian_quadrupole

    Q = spherical_to_cartesian_quadrupole(quad_s)
    Q_r = spherical_to_cartesian_quadrupole(quad_r)
    assert torch.allclose(Q_r, torch.einsum("ab,nbc,dc->nad", R, Q, R), atol=1e-10)


def test_no_environment_dependence():
    """Moving a neighbor water changes a fragment's permanent multipoles not at all."""
    heads, proj = make_heads()
    with torch.no_grad():
        for p in heads.parameters():
            p.add_(0.1 * torch.randn(p.shape))

    near = water_cluster_batch(2)
    far = Batch(**{**near.__dict__})
    far.positions = near.positions.clone()
    far.positions[3:] += torch.tensor([25.0, 0.0, 0.0])

    state_near = make_state(near, proj)
    state_far = make_state(far, proj)
    z = torch.randn(6, 8)

    q_n, mu_n, _ = heads(z, proj(near, state_near).x_in, state_near)
    q_f, mu_f, _ = heads(z, proj(far, state_far).x_in, state_far)
    # fragment 0's atoms: identical internal features -> identical permanent multipoles
    assert torch.allclose(q_n[:3], q_f[:3], atol=1e-12)
    assert torch.allclose(mu_n[:3], mu_f[:3], atol=1e-12)
