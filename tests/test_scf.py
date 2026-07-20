"""Charge SCF forward solver: convergence, exact constraint, init independence."""

import torch

from rsfff.mlip import RaggedIndex

from conftest import make_batch, make_model, water_h3o_batch


def _solve(model, batch, q_init=None):
    cache = model.featurizer.precompute_geometry(batch)
    ragged = RaggedIndex.build(batch.batch_idx, batch.n_systems)
    total_q = batch.total_charge
    field = batch.positions.new_zeros(batch.n_systems, 3)

    def energy_fn(q):
        e, _, _, _, _ = model._evaluate(
            cache, batch.positions, field, batch.n_systems, q
        )
        return e

    if q_init is None:
        q_init = (total_q / ragged.counts.to(batch.positions.dtype))[batch.batch_idx]
    from rsfff.mlip import solve_charges

    return solve_charges(energy_fn, q_init, ragged, max_iter=40, tol=1e-12), ragged


def test_converges_on_real_geometries():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    (q, info), ragged = _solve(model, batch)
    assert info.converged
    assert info.residual < 1e-12
    assert info.n_iter <= 10
    assert info.min_tangent_eig > 0.0  # convex at near-init


def test_charge_conservation_machine_precision():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    (q, info), ragged = _solve(model, batch)
    sums = torch.zeros(batch.n_systems).index_add_(0, batch.batch_idx, q)
    assert (sums - batch.total_charge).abs().max() < 1e-14


def test_solution_independent_of_feasible_init():
    model = make_model()
    batch = water_h3o_batch(jitter=0.02)
    (q_a, _), ragged = _solve(model, batch)
    # a different feasible start: uniform + a tangent perturbation
    g = torch.Generator().manual_seed(7)
    pert = torch.randn(q_a.shape, generator=g)
    from rsfff.mlip.charge_scf import projected_gradient

    pert = projected_gradient(pert, ragged)  # remove per-molecule mean -> tangent
    q0 = (batch.total_charge / ragged.counts.to(torch.float64))[batch.batch_idx]
    (q_b, info_b), _ = _solve(model, batch, q_init=q0 + 0.1 * pert)
    assert info_b.converged
    assert (q_a - q_b).abs().max() < 1e-9


def test_single_atom_charge_pinned():
    model = make_model()
    batch = make_batch(
        [torch.zeros(1, 3), torch.zeros(1, 3) + 5.0],
        [torch.tensor([8]), torch.tensor([1])],
        [-1.0, 1.0],
    )
    (q, info), _ = _solve(model, batch)
    assert info.converged
    assert torch.allclose(q, torch.tensor([-1.0, 1.0]), atol=1e-14)
