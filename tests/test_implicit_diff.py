"""M0 gatekeeper: derivatives through the charge SCF vs central finite differences.

All in float64 with SCF tol 1e-12. Checks, in order of depth:
  (a) forces = -dE*/dR            (first derivative; envelope theorem)
  (b) d mu / dR                   (first derivative of the attached charges)
  (c) alpha = d mu / dF           (charge response to the applied field)
  (d) d loss / d theta            (training gradient through the attach)
  (e) second-order training gradient of an alpha-type loss: attach_steps=2
"""

import copy

import torch

from rsfff.train.loss import compute_dipole_derivatives, compute_polarizability

from conftest import make_model, water_h3o_batch

TOL = 1e-5  # relative, per the design doc's M0 criterion


def _forward(model, positions_value, batch, field_value=None, *, grads=False):
    b = copy.copy(batch)
    b.positions = positions_value.detach().clone().requires_grad_(grads)
    m = b.n_systems
    field = torch.zeros(m, 3, dtype=torch.float64)
    if field_value is not None:
        field = field + field_value
    field.requires_grad_(grads)
    out = model(b, field=field)
    return out, b, field


def _rel_err(got, want):
    got, want = got.detach(), want.detach()
    denom = want.abs().max().clamp(min=1e-10)
    return float((got - want).abs().max() / denom)


def test_forces_match_fd():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    out, b, _ = _forward(model, batch.positions, batch, grads=True)
    (grad,) = torch.autograd.grad(out.energy.sum(), b.positions)

    h = 1e-4
    probes = [(0, 0), (0, 2), (2, 1), (5, 0)]
    for i, a in probes:
        for sign, store in ((1.0, "p"), (-1.0, "m")):
            pos = batch.positions.clone()
            pos[i, a] += sign * h
            e = _forward(model, pos, batch)[0].energy.sum().item()
            if store == "p":
                ep = e
            else:
                em = e
        fd = (ep - em) / (2 * h)
        assert abs(grad[i, a].item() - fd) / max(abs(fd), 1e-10) < TOL, (i, a)


def test_dmu_dr_matches_fd():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    out, b, _ = _forward(model, batch.positions, batch, grads=True)
    dmu = compute_dipole_derivatives(out.dipole, b.positions, create_graph=False)

    h = 1e-4
    for i, a in [(0, 1), (3, 0), (6, 2)]:
        pos_p = batch.positions.clone(); pos_p[i, a] += h
        pos_m = batch.positions.clone(); pos_m[i, a] -= h
        mu_p = _forward(model, pos_p, batch)[0].dipole
        mu_m = _forward(model, pos_m, batch)[0].dipole
        fd = ((mu_p - mu_m) / (2 * h)).sum(dim=0)  # (3,) d mu_b / dR_{i,a}, summed mols
        assert _rel_err(dmu[i, a, :], fd) < TOL, (i, a)


def test_alpha_matches_fd_field():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    out, _, field = _forward(model, batch.positions, batch, grads=True)
    alpha = compute_polarizability(out.dipole, field, create_graph=False, symmetrize=False)

    h = 1e-3
    for a in range(3):
        f_p = torch.zeros(2, 3, dtype=torch.float64); f_p[:, a] += h
        f_m = -f_p
        mu_p = _forward(model, batch.positions, batch, field_value=f_p)[0].dipole
        mu_m = _forward(model, batch.positions, batch, field_value=f_m)[0].dipole
        fd = (mu_p - mu_m) / (2 * h)              # (M, 3) = d mu_b / dF_a
        assert _rel_err(alpha[:, :, a], fd) < TOL, a
    # symmetry at stationarity (unsymmetrized!) is itself an implicit-diff check
    assert (alpha - alpha.transpose(-1, -2)).abs().max() < 1e-8


def _theta_probes(model):
    """A few scalar parameter slots across the q-coupled modules."""
    return [
        (model.element_energy.chi, 0),
        (model.charge_embedding.q_encoder[-1].weight, 3),
        (model.weight_net.net[-1].weight, 1),
        (model.energy_head.mlp[-1].weight, 2),
    ]


def test_training_gradient_matches_fd():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)

    def loss_value():
        out, b, _ = _forward(model, batch.positions, batch, grads=True)
        forces = -torch.autograd.grad(
            out.energy.sum(), b.positions, create_graph=True
        )[0]
        return out.energy.sum() + forces.pow(2).sum() + out.dipole.pow(2).sum()

    loss = loss_value()
    model.zero_grad(set_to_none=True)
    loss.backward()

    h = 1e-5
    for param, flat_idx in _theta_probes(model):
        grad = param.grad.reshape(-1)[flat_idx].item()
        with torch.no_grad():
            param.reshape(-1)[flat_idx] += h
        lp = float(loss_value())
        with torch.no_grad():
            param.reshape(-1)[flat_idx] -= 2 * h
        lm = float(loss_value())
        with torch.no_grad():
            param.reshape(-1)[flat_idx] += h
        fd = (lp - lm) / (2 * h)
        assert abs(grad - fd) / max(abs(fd), 1e-8) < TOL, param.shape


def test_alpha_loss_gradient_attach_steps():
    """Training gradient of an alpha-type loss (2nd-order in q*): attach_steps=2
    must match finite differences; steps=1 is the cheaper approximation."""
    batch = water_h3o_batch(jitter=0.02)

    def grad_and_fd(steps):
        model = make_model()
        model.attach_steps = steps

        def loss_value():
            out, _, field = _forward(model, batch.positions, batch, grads=True)
            alpha = compute_polarizability(out.dipole, field, create_graph=True)
            return alpha.pow(2).sum()

        loss = loss_value()
        model.zero_grad(set_to_none=True)
        loss.backward()
        param = model.element_energy.eta_raw
        grad = param.grad[1].item()
        h = 1e-5
        with torch.no_grad():
            param[1] += h
        lp = float(loss_value())
        with torch.no_grad():
            param[1] -= 2 * h
        lm = float(loss_value())
        with torch.no_grad():
            param[1] += h
        fd = (lp - lm) / (2 * h)
        return abs(grad - fd) / max(abs(fd), 1e-10)

    err2 = grad_and_fd(2)
    assert err2 < 1e-4, err2
