"""Symmetry checks on model outputs: translation and rotation behavior of
energy, forces, charges, dipole (neutral and charged), and polarizability."""

import copy

import torch

from conftest import make_model, water_h3o_batch


def _run(model, batch, *, grads=False):
    b = copy.copy(batch)
    b.positions = batch.positions.detach().clone().requires_grad_(grads)
    return model(b), b


def test_translation():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    t = torch.tensor([1.3, -0.7, 2.1], dtype=torch.float64)

    out0, b0 = _run(model, batch, grads=True)
    shifted = copy.copy(batch)
    shifted.positions = batch.positions + t
    out1, b1 = _run(model, shifted, grads=True)

    assert (out0.energy - out1.energy).abs().max() < 1e-10
    f0 = -torch.autograd.grad(out0.energy.sum(), b0.positions)[0]
    f1 = -torch.autograd.grad(out1.energy.sum(), b1.positions)[0]
    assert (f0 - f1).abs().max() < 1e-9
    # charges and atomic dipoles are translation-invariant
    assert (out0.charges - out1.charges).abs().max() < 1e-10
    assert (out0.atomic_dipoles - out1.atomic_dipoles).abs().max() < 1e-10
    # mu(r + t) = mu(r) + Q t : invariant for H2O (Q=0), shifts by t for H3O+ (Q=1)
    dmu = out1.dipole - out0.dipole
    assert dmu[0].abs().max() < 1e-10
    assert (dmu[1] - t).abs().max() < 1e-9
    # analytic alpha is translation-invariant (charge-flow term uses r - Rbar)
    assert (out0.alpha - out1.alpha).abs().max() < 1e-9


def test_rotation():
    from e3nn import o3

    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    R = o3.rand_matrix(1)[0].to(torch.float64)

    out0, b0 = _run(model, batch, grads=True)
    rotated = copy.copy(batch)
    rotated.positions = batch.positions @ R.t()
    out1, b1 = _run(model, rotated, grads=True)

    assert (out0.energy - out1.energy).abs().max() < 1e-9
    assert (out0.charges - out1.charges).abs().max() < 1e-9

    f0 = -torch.autograd.grad(out0.energy.sum(), b0.positions)[0]
    f1 = -torch.autograd.grad(out1.energy.sum(), b1.positions)[0]
    assert (f0 @ R.t() - f1).abs().max() < 1e-9

    assert (out0.atomic_dipoles @ R.t() - out1.atomic_dipoles).abs().max() < 1e-9
    assert (out0.dipole @ R.t() - out1.dipole).abs().max() < 1e-9
    assert (R @ out0.alpha @ R.t() - out1.alpha).abs().max() < 1e-8
