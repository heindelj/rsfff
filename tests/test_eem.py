"""Closed-form EEM layer: constraint, stationarity, variational property, and the
analytic polarizability against autograd / finite differences."""

import copy

import torch

from rsfff.mlip.eem import eem_charges, eem_energy
from rsfff.train.loss import compute_dipole_derivatives, compute_polarizability

from conftest import make_batch, make_model, water_h3o_batch


def _forward(model, batch, *, field=None, grads=False):
    b = copy.copy(batch)
    b.positions = batch.positions.detach().clone().requires_grad_(grads)
    return model(b, field=field), b


def test_charge_conservation_and_single_atom():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    out, _ = _forward(model, batch)
    sums = torch.zeros(2).index_add_(0, batch.batch_idx, out.charges)
    assert (sums - batch.total_charge).abs().max() < 1e-14

    single = make_batch(
        [torch.zeros(1, 3), torch.zeros(1, 3) + 5.0],
        [torch.tensor([8]), torch.tensor([1])],
        [-1.0, 1.0],
    )
    out_s, _ = _forward(model, single)
    assert torch.allclose(out_s.charges, single.total_charge, atol=1e-14)


def test_stationarity_lagrange():
    """At the closed-form solution, dE_EEM/dq_i = -lambda_m uniformly per molecule."""
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    feats = model.featurizer(batch)
    chi, eta, chivec, alpha = model.params(feats)

    q, lam = eem_charges(
        chi, eta, batch.positions, batch.batch_idx, 2, batch.total_charge
    )
    q_leaf = q.detach().requires_grad_(True)
    e = eem_energy(
        q_leaf, chi, eta, chivec, alpha, batch.positions, batch.batch_idx, 2
    )
    (g,) = torch.autograd.grad(e.sum(), q_leaf)
    assert (g + lam[batch.batch_idx]).abs().max() < 1e-12


def test_variational_minimum():
    """Perturbing q along the constraint tangent strictly increases E_EEM."""
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    feats = model.featurizer(batch)
    chi, eta, chivec, alpha = model.params(feats)
    q, _ = eem_charges(chi, eta, batch.positions, batch.batch_idx, 2, batch.total_charge)

    def energy(qq):
        return eem_energy(
            qq, chi, eta, chivec, alpha, batch.positions, batch.batch_idx, 2
        ).sum()

    e0 = energy(q)
    g = torch.Generator().manual_seed(11)
    for _ in range(5):
        d = torch.randn(q.shape, generator=g)
        # project onto the constraint tangent (zero per-molecule sum)
        mean = torch.zeros(2).index_add_(0, batch.batch_idx, d) / torch.bincount(
            batch.batch_idx
        ).to(d.dtype)
        d = d - mean[batch.batch_idx]
        assert energy(q + 0.1 * d) > e0 + 1e-12


def test_analytic_alpha_matches_autograd_and_fd():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)

    # analytic
    out, _ = _forward(model, batch)
    alpha = out.alpha
    assert (alpha - alpha.transpose(-1, -2)).abs().max() < 1e-12
    assert torch.linalg.eigvalsh(alpha).min() > 0.0  # PSD (strictly, via psd_floor)

    # autograd route: d mu / dF at F = 0
    field = torch.zeros(2, 3, requires_grad=True)
    out_f, _ = _forward(model, batch, field=field)
    alpha_ag = compute_polarizability(out_f.dipole, field, create_graph=False)
    assert (alpha - alpha_ag).abs().max() < 1e-10

    # finite-difference route
    h = 1e-4
    for a in range(3):
        fp = torch.zeros(2, 3); fp[:, a] = h
        mu_p = _forward(model, batch, field=fp)[0].dipole
        mu_m = _forward(model, batch, field=-fp)[0].dipole
        fd = (mu_p - mu_m) / (2 * h)                      # (M, 3) d mu_b / dF_a
        assert (alpha[:, :, a] - fd).abs().max() < 1e-8


def test_energy_concave_in_field():
    """The minimized EEM energy is concave in F (the permanent dipole contributes
    at first order, so compare symmetrically): E(F) + E(-F) < 2 E(0), with the
    gap = -F^T alpha F < 0."""
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    e0 = _forward(model, batch)[0].energy
    f = torch.full((2, 3), 0.01)
    ep = _forward(model, batch, field=f)[0].energy
    em = _forward(model, batch, field=-f)[0].energy
    assert (ep + em < 2 * e0 - 1e-12).all()


def test_dmu_dr_translational_sum_rule():
    """sum_i d mu_b / d R_{i,a} = Q delta_ab for the model's dipole."""
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    out, b = _forward(model, batch, grads=True)
    dmu = compute_dipole_derivatives(out.dipole, b.positions, create_graph=False)
    eye = torch.eye(3, dtype=torch.float64)
    for m in range(2):
        s = dmu[batch.batch_idx == m].sum(dim=0)
        assert (s - batch.total_charge[m] * eye).abs().max() < 1e-9, m


def test_training_gradient_matches_fd():
    """dLoss/dtheta spot check against central finite differences (single graph)."""
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)

    def loss_value():
        out, b = _forward(model, batch, grads=True)
        forces = -torch.autograd.grad(
            out.energy.sum(), b.positions, create_graph=True
        )[0]
        return (
            out.energy.sum()
            + forces.pow(2).sum()
            + out.dipole.pow(2).sum()
            + out.alpha.pow(2).sum()
        )

    loss = loss_value()
    model.zero_grad(set_to_none=True)
    loss.backward()

    probes = [
        (model.params.chi0, 0),
        (model.params.eta0_raw, 1),
        (model.params.chivec_head.gate_mlp[-1].weight, 3),
        (model.params.alpha_head.base_mlp[-1].weight, 2),
        (model.energy_head.mlp[-1].weight, 5),
    ]
    h = 1e-5
    for param, flat_idx in probes:
        grad = param.grad.reshape(-1)[flat_idx].item()
        with torch.no_grad():
            param.reshape(-1)[flat_idx] += h
        lp = float(loss_value().detach())
        with torch.no_grad():
            param.reshape(-1)[flat_idx] -= 2 * h
        lm = float(loss_value().detach())
        with torch.no_grad():
            param.reshape(-1)[flat_idx] += h
        fd = (lp - lm) / (2 * h)
        assert abs(grad - fd) / max(abs(fd), 1e-8) < 1e-6, param.shape
