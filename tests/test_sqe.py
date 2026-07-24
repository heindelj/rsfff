"""Split-charge equilibration: the identities that must hold exactly, not approximately.

Charge conservation, the two compliance limits, and the analytic charge-flow polarizability are
*structural* properties of the SQE coordinate system (docs/mixture_of_diabatic_embeddings.md
§2.4), not things training is supposed to achieve. If any of these drifts, the model has lost a
guarantee the whole range-separated architecture is built on.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.mlip.eem import eem_charges
from rsfff.mlip.sqe import PairComplianceHead, sqe_solve


@pytest.fixture
def system():
    """One H2O (Q=0, 2 channels) and one H3O+ (Q=+1, 3 channels) in one ragged batch.

    Deliberately different sizes: the padded block is where the numerically delicate cases
    live (a padded channel sits at exactly ``s = 0``).
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469],
         [0.0, 0.0, 0.077], [0.0, 0.921, -0.331], [0.798, -0.461, -0.331],
         [-0.798, -0.461, -0.331]]
    )
    g = torch.Generator().manual_seed(7)
    return dict(
        chi=torch.randn(7, generator=g) * 0.3,
        eta=torch.rand(7, generator=g) + 0.4,
        compliance=torch.rand(5, generator=g) + 0.2,
        q0=torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]),
        positions=positions,
        bond_index=torch.tensor([[0, 0, 3, 3, 3], [1, 2, 4, 5, 6]]),
        batch_idx=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        bond_batch=torch.tensor([0, 0, 1, 1, 1]),
        n_systems=2,
    )


def _total_charge(q, batch_idx, n_systems):
    return torch.zeros(n_systems, dtype=q.dtype).index_add_(0, batch_idx, q)


def test_charge_conservation_is_structural(system):
    """Sum q == sum q0 exactly, for *any* compliance -- it is the coordinate system, not a fit."""
    target = _total_charge(system["q0"], system["batch_idx"], system["n_systems"])
    for scale in (0.0, 1e-6, 1.0, 1e3, 1e8):
        sol = sqe_solve(**{**system, "compliance": system["compliance"] * scale})
        got = _total_charge(sol.charges, system["batch_idx"], system["n_systems"])
        assert torch.allclose(got, target, atol=1e-13), f"scale={scale}"


def test_closed_channels_pin_the_baseline_charges(system):
    """s -> 0: nothing flows, so q is exactly the diabatic baseline q^(0)."""
    sol = sqe_solve(**{**system, "compliance": torch.zeros(5)})
    assert torch.allclose(sol.charges, system["q0"], atol=1e-14)
    assert torch.allclose(sol.transfers, torch.zeros(5), atol=1e-14)
    assert torch.allclose(sol.alpha_flow, torch.zeros(2, 3, 3), atol=1e-14)


def test_soft_channels_reproduce_eem(system):
    """s -> inf on a connected graph: the channel penalty vanishes and SQE becomes EEM."""
    sol = sqe_solve(**{**system, "compliance": torch.full((5,), 1e10)})
    q_eem, _ = eem_charges(
        system["chi"], system["eta"], system["positions"], system["batch_idx"],
        system["n_systems"],
        _total_charge(system["q0"], system["batch_idx"], system["n_systems"]),
        None,
    )
    assert torch.allclose(sol.charges, q_eem, atol=1e-8)


def test_solution_is_stationary_and_energy_is_the_minimum(system):
    """dE/dp == 0 at the returned transfers, and the returned energy is E(p*)."""
    sol = sqe_solve(**system)
    p = sol.transfers.clone().requires_grad_(True)
    n, nb = system["positions"].shape[0], p.shape[0]
    B = torch.zeros(n, nb, dtype=p.dtype)
    B[system["bond_index"][0], torch.arange(nb)] = 1.0
    B[system["bond_index"][1], torch.arange(nb)] = -1.0
    q = system["q0"] + B @ p
    energy = (
        (system["chi"] * q + 0.5 * system["eta"] * q * q).sum()
        + 0.5 * (p * p / system["compliance"]).sum()
    )
    (grad,) = torch.autograd.grad(energy, p)
    assert grad.abs().max() < 1e-12
    assert torch.allclose(sol.energy.sum(), energy.detach(), atol=1e-12)


def test_alpha_flow_matches_autograd_and_is_translation_invariant(system):
    """The analytic charge-flow tensor equals d(sum q r)/dF, and does not move with the origin."""
    sol = sqe_solve(**system)

    field = torch.zeros(2, 3, requires_grad=True)
    sol_f = sqe_solve(**{**system, "field": field})
    mu = torch.zeros(2, 3, dtype=field.dtype).index_add_(
        0, system["batch_idx"], sol_f.charges.unsqueeze(-1) * system["positions"]
    )
    rows = [
        torch.autograd.grad(mu[:, k].sum(), field, retain_graph=True)[0] for k in range(3)
    ]
    assert torch.allclose(torch.stack(rows, dim=1), sol.alpha_flow, atol=1e-12)

    shifted = sqe_solve(
        **{**system, "positions": system["positions"] + torch.tensor([3.0, -1.0, 2.0])}
    )
    assert torch.allclose(shifted.alpha_flow, sol.alpha_flow, atol=1e-12)
    assert torch.linalg.eigvalsh(sol.alpha_flow).min() >= -1e-12  # PSD


def test_padded_channels_do_not_poison_second_derivatives(system):
    """Mixed system sizes put a padded channel at exactly s = 0.

    The solve is rescaled by ``S`` rather than ``S^(1/2)`` precisely so that everything stays
    polynomial in the compliance and differentiable at zero. With a square-root scaling this
    test produces NaN in the double backward -- which is also what force training does.
    """
    compliance = system["compliance"].clone().requires_grad_(True)
    positions = system["positions"].clone().requires_grad_(True)
    sol = sqe_solve(**{**system, "compliance": compliance, "positions": positions})
    (force,) = torch.autograd.grad(sol.energy.sum(), positions, create_graph=True)
    (g_s,) = torch.autograd.grad(force.pow(2).sum(), compliance)
    assert torch.isfinite(force).all()
    assert torch.isfinite(g_s).all()


def test_gradient_is_finite_at_exactly_closed_channels(system):
    """s = 0 is the channel-closure limit Phase 2 lives in; it must be differentiable."""
    compliance = torch.zeros(5, requires_grad=True)
    sol = sqe_solve(**{**system, "compliance": compliance})
    (grad,) = torch.autograd.grad(sol.energy.sum(), compliance)
    assert torch.isfinite(grad).all()


def test_systems_are_independent(system):
    """Perturbing one system leaves the other's charges untouched (block structure)."""
    sol = sqe_solve(**system)
    moved = system["positions"].clone()
    moved[3:] += 0.3
    chi2 = system["chi"].clone()
    chi2[3:] += 1.0
    other = sqe_solve(**{**system, "positions": moved, "chi": chi2})
    assert torch.allclose(other.charges[:3], sol.charges[:3], atol=1e-13)


def test_channelless_systems_coexist_with_bonded_ones(system):
    """A batch may mix lone atoms (no channels) with molecules; each keeps its own charge.

    This is the Phase-2 dissociation layout in miniature -- once a fragment separates, its
    channels close and it must hold exactly integer charge while its neighbors keep flowing.
    """
    positions = torch.cat((system["positions"], torch.tensor([[8.0, 0.0, 0.0]])))
    extended = {
        **system,
        "positions": positions,
        "chi": torch.cat((system["chi"], torch.tensor([0.2]))),
        "eta": torch.cat((system["eta"], torch.tensor([0.7]))),
        "q0": torch.cat((system["q0"], torch.tensor([-1.0]))),
        "batch_idx": torch.cat((system["batch_idx"], torch.tensor([2]))),
        "n_systems": 3,
    }
    sol = sqe_solve(**extended)
    assert sol.charges[-1] == pytest.approx(-1.0, abs=1e-14)  # integer, structurally
    assert sol.alpha_flow[2].abs().max() < 1e-14              # no channels, no charge flow
    reference = sqe_solve(**system)
    assert torch.allclose(sol.charges[:7], reference.charges, atol=1e-13)


def test_compliance_head_is_symmetric_under_atom_swap():
    """A channel is undirected: s_ij must not depend on which atom is listed first."""
    torch.manual_seed(0)
    head = PairComplianceHead(6, hidden=16, depth=1, n_radial=4, cutoff=5.0)
    with torch.no_grad():  # break the zero-init readout so the features actually matter
        head.net[-1].weight.normal_(0.0, 0.5)
    h = torch.randn(4, 6)
    positions = torch.randn(4, 3)
    forward = head(h, positions, torch.tensor([[0, 1], [2, 3]]))
    reversed_ = head(h, positions, torch.tensor([[2, 3], [0, 1]]))
    assert torch.allclose(forward, reversed_, atol=1e-14)
    assert (forward > 0).all()
