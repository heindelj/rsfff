"""The bond-order solve: constraints, competition, derivatives, co-membership."""

import pytest
import torch

from rsfff.ff.pairing.bond_order import (
    comembership_from_bond_order,
    pairing_energy,
    solve_bond_order,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _water_like():
    pair_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    J = torch.tensor([0.35, 0.35, 0.05])
    kappa = torch.full((3,), 0.2)
    valence = torch.tensor([2.0, 1.0, 1.0])
    batch = torch.zeros(3, dtype=torch.long)
    return J, kappa, valence, pair_index, batch


def test_saturated_bonds_and_marginals():
    J, kappa, v, pi, b = _water_like()
    sol = solve_bond_order(J, kappa, v, 0.002, pi, b, 1)
    assert bool(sol.converged.all())
    assert torch.allclose(sol.p[:2], torch.ones(2), atol=1e-9)
    assert float(sol.p[2]) < 1e-9                       # the 1-3 H-H pair does not bond
    marg = sol.u.index_add(0, pi[0], sol.p).index_add(0, pi[1], sol.p)
    assert torch.allclose(marg, v, atol=1e-9)


def test_valence_competition_squeezes_out_the_acceptor():
    # H(1) bonded to O(0) with J = 0.35; an acceptor O(2) at J = 0.12 (> 0.5 kappa)
    pi = torch.tensor([[0, 1], [1, 2]])
    J = torch.tensor([0.35, 0.12])
    kappa = torch.full((2,), 0.2)
    v = torch.tensor([2.0, 1.0, 2.0])
    b = torch.zeros(3, dtype=torch.long)
    sol = solve_bond_order(J, kappa, v, 0.002, pi, b, 1)
    assert float(sol.p[1]) < 1e-2
    # the same acceptor with a free hydrogen pairs partially: p ~ J / kappa
    lone = solve_bond_order(
        J[1:], kappa[1:], torch.tensor([1.0, 2.0]), 0.002,
        torch.tensor([[0], [1]]), b[:2], 1,
    )
    assert abs(float(lone.p[0]) - 0.6) < 0.05


def test_isolated_atoms_have_zero_pairing_energy():
    J = torch.zeros(0)
    kappa = torch.zeros(0)
    v = torch.tensor([2.0, 1.0])
    pi = torch.zeros(2, 0, dtype=torch.long)
    b = torch.zeros(2, dtype=torch.long)
    sol = solve_bond_order(J, kappa, v, 0.002, pi, b, 1)
    assert torch.allclose(sol.u, v)
    e_pair, e_atom = pairing_energy(J, kappa, 0.002, sol.p, sol.u, v)
    assert float(e_atom.abs().max()) == 0.0


def _two_frames(J, kappa, v):
    pi = torch.tensor([[0, 0, 1, 3, 3], [1, 2, 2, 4, 5]])
    b = torch.tensor([0, 0, 0, 1, 1, 1])
    sol = solve_bond_order(J, kappa, v, 0.01, pi, b, 2)
    e_pair, e_atom = pairing_energy(J, kappa, 0.01, sol.p, sol.u, v)
    return e_pair.sum() + e_atom.sum(), sol.p


def test_first_and_second_derivatives_are_exact():
    J = torch.tensor([0.25, 0.3, 0.1, 0.15, 0.22], requires_grad=True)
    kappa = torch.full((5,), 0.2, requires_grad=True)
    v = torch.tensor([2.0, 1.0, 1.0, 2.0, 1.0, 1.0], requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c: _two_frames(a, b, c)[0], (J, kappa, v), eps=1e-6, atol=1e-6)
    assert torch.autograd.gradcheck(lambda a, b, c: _two_frames(a, b, c)[1], (J, kappa, v), eps=1e-6, atol=1e-6)
    assert torch.autograd.gradgradcheck(lambda a, b, c: _two_frames(a, b, c)[0], (J, kappa, v), eps=1e-5, atol=1e-5)


def test_comembership_13_is_product_of_bond_orders():
    pi = torch.tensor([[0, 0, 1], [1, 2, 2]])
    p = torch.tensor([0.9, 0.8, 0.0])
    c = comembership_from_bond_order(p, torch.arange(3), pi, 3)
    assert torch.allclose(c, torch.tensor([0.9, 0.8, 0.72]))
    c0 = comembership_from_bond_order(p, torch.arange(3), pi, 3, include_13=False)
    assert torch.allclose(c0, p)
