"""The explicit bond order on its own: raw order, the two saturation rules, derivatives."""

import pytest
import torch

from rsfff.ff.pairing.electronic_state import capacity_table
from rsfff.ff.tersoff.bond_order import (
    explicit_state,
    raw_bond_order,
    saturate_rebo,
    saturate_waterfill,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
T = 0.002


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _star(J, v_center, v_partner=10.0):
    """Atom 0 with partners 1..n whose own capacities never bind."""
    J = torch.as_tensor(J)
    n = J.shape[0]
    kappa = torch.full((n,), 0.2)
    i = torch.zeros(n, dtype=torch.long)
    j = torch.arange(1, n + 1)
    v = torch.full((n + 1,), v_partner)
    v[0] = v_center
    return J, kappa, i, j, v


def test_raw_order_is_the_clipped_unconstrained_pairing():
    J = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.6])
    kappa = torch.full((5,), 0.2)
    b = raw_bond_order(J, kappa, T)
    assert abs(float(b[2]) - 0.5) < 1e-6           # J = kappa / 2
    assert abs(float(b[1]) - 0.25) < 1e-3
    assert float(b[4]) > 1 - 1e-12 and float(b[0]) < 0.01


def test_waterfill_is_exact_for_one_atom():
    """Two full bonds and a third partner sharing what is left: p = (J - lam) / kappa."""
    J, kappa, i, j, v = _star([0.62, 0.62, 0.487, 0.104, 0.03], 2.965)
    b = raw_bond_order(J, kappa, T)
    p, lam, res = saturate_waterfill(J, kappa, b, v, i, j, 6, T)
    assert abs(float(lam[0]) - (0.487 - 0.965 * 0.2)) < 2e-3
    assert torch.allclose(p[:2], torch.ones(2), atol=1e-6)
    assert abs(float(p[2]) - 0.965) < 3e-3
    assert float(p[3:].max()) < 5e-3                 # the reconcile width
    assert float(res.max()) < 1e-3
    assert float(p.sum()) < 2.965 + 0.01


def test_waterfill_squeezes_the_weak_partner_out():
    """The pairing model's valence competition: a full bond leaves nothing for an acceptor
    whose bare order would be 0.35 (the water dimer at the priors)."""
    J, kappa, i, j, v = _star([0.62, 0.07], 1.0)
    b = raw_bond_order(J, kappa, T)
    assert float(b[1]) > 0.3
    p, lam, _ = saturate_waterfill(J, kappa, b, v, i, j, 3, T)
    assert float(p[0]) > 1 - 1e-6 and float(p[1]) < 5e-3
    p_rebo, _ = saturate_rebo(b, v, i, j, 3)
    assert float(p_rebo[1]) > 0.2                   # linear sharing leaks


def test_waterfill_splits_equal_partners_evenly():
    J, kappa, i, j, v = _star([0.414, 0.414], 1.0)
    b = raw_bond_order(J, kappa, T)
    p, _, _ = saturate_waterfill(J, kappa, b, v, i, j, 3, T)
    assert torch.allclose(p, torch.full((2,), 0.5), atol=3e-3)   # + the reconcile width


def test_rebo_bound_and_exactness_at_capacity():
    J, kappa, i, j, v = _star([0.62, 0.62], 2.0)
    b = raw_bond_order(J, kappa, T)
    p, s = saturate_rebo(b, v, i, j, 3)
    assert float(p.sum()) <= 2.0 + 1e-9 and float(p.min()) > 1 - 0.02
    J, kappa, i, j, v = _star([0.62, 0.62, 0.62], 2.0)
    p, _ = saturate_rebo(raw_bond_order(J, kappa, T), v, i, j, 4)
    assert float(p.sum()) <= 2.0 + 1e-9


def test_capacity_bound_holds_on_a_random_graph():
    g = torch.Generator().manual_seed(3)
    n, m = 12, 40
    pairs = torch.randint(0, n, (2, m), generator=g)
    pairs = pairs[:, pairs[0] != pairs[1]]
    J = 0.7 * torch.rand(pairs.shape[1], generator=g)
    kappa = torch.full((pairs.shape[1],), 0.2)
    tables = capacity_table([8])[torch.zeros(n, dtype=torch.long)]
    q = torch.zeros(n)
    for sat in ("waterfill", "rebo"):
        st = explicit_state(J, kappa, tables, q, pairs, torch.zeros(n, dtype=torch.long), 1, T, saturation=sat)
        coord = torch.zeros(n).index_add_(0, pairs[0], st.p).index_add_(0, pairs[1], st.p)
        slack = 1e-3 + (0.0 if sat == "rebo" else 2e-3 * torch.bincount(pairs.flatten(), minlength=n))
        assert bool((coord <= st.valence + slack).all()), sat
        assert bool((st.p >= 0).all()) and bool((st.p <= 1 + 1e-9).all())


@pytest.mark.parametrize("sat", ["waterfill", "rebo"])
def test_first_and_second_derivatives(sat):
    J, kappa, i, j, v = _star([0.62, 0.55, 0.30, 0.09], 2.0)
    tables = capacity_table([8, 1])[torch.tensor([0, 1, 1, 1, 1])]
    q = torch.zeros(5)
    pairs = torch.stack((i, j))
    batch_idx = torch.zeros(5, dtype=torch.long)
    J = J.clone().requires_grad_(True)

    def energy(J_):
        st = explicit_state(J_, kappa, tables, q, pairs, batch_idx, 1, T, saturation=sat)
        return (-J_ * st.p + 0.5 * kappa * st.p * st.p).sum()

    assert torch.autograd.gradcheck(energy, (J,), eps=1e-6, atol=1e-6)
    assert torch.autograd.gradgradcheck(energy, (J,), eps=1e-5, atol=1e-5)
