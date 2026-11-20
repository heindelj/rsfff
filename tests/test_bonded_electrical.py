"""The split-charge response (rsfff.ff.bonded.response) and the electrical head of BondedModel.

* response: the closed form equals explicit minimization in a field, with the bond electronegativity differences
  (``dchi``) standing in for per-atom ones; charge is conserved; P is positive semidefinite;
* model: charges sum to the molecule's charge; dipole, second moment and polarizability transform as tensors under
  proper and improper rotations (the origin moving with the molecule); the polarizability is positive definite; the
  tied split-charge model ("sqe") has a charge-flow response and "flow" has none; the energy does not see the
  electrical head.
"""
import math

import pytest
import torch

from rsfff.ff.bonded import ResponseParameters, collate, field_energy, response
from rsfff.ff.units import BOHR_ANG

from test_bonded_model import build, molecules

torch.set_default_dtype(torch.float64)


def test_response_closed_form_matches_explicit_minimization_with_dchi():
    torch.manual_seed(1)
    pos = torch.randn(7, 3)
    bonds = torch.tensor([[0, 1], [1, 2], [1, 3], [4, 5], [5, 6]])
    bi, bb = torch.tensor([0, 0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 0, 1, 1])
    A, m2, chi = torch.randn(7, 3, 3), torch.randn(7, 3, 3), torch.randn(7)
    kw = dict(q0=torch.tensor([0, 0, 0, 0, -1 / 3, -1 / 3, -1 / 3.0]), eta=torch.rand(7) + 0.5, compliance=torch.rand(5),
              mu0=torch.randn(7, 3), alpha=A @ A.transpose(1, 2), m2=m2 + m2.transpose(1, 2), c_quad=torch.rand(7))
    p = ResponseParameters(dchi=chi[bonds[:, 1]] - chi[bonds[:, 0]], **kw)
    origin = torch.randn(2, 3)
    sol = response(p, pos, bonds, bi, bb, 2, origin)
    ref = response(ResponseParameters(chi=chi, **kw), pos, bonds, bi, bb, 2, origin)      # per-atom chi: same physics
    for a, b in ((sol.charges, ref.charges), (sol.dipole, ref.dipole), (sol.second_moment, ref.second_moment), (sol.P, ref.P)):
        torch.testing.assert_close(a, b)
    F = torch.zeros(2, 3, requires_grad=True)
    G = torch.zeros(2, 3, 3, requires_grad=True)
    e = field_energy(p, pos, bonds, bi, bb, 2, F, 0.5 * (G + G.transpose(1, 2)), origin)
    assert e.abs().max() < 1e-13
    gF, gG = torch.autograd.grad(e.sum(), (F, G), create_graph=True)
    torch.testing.assert_close(-gF, sol.dipole)
    torch.testing.assert_close(-2 * gG, sol.second_moment)
    al = torch.stack([torch.autograd.grad(-gF[:, k].sum(), F, retain_graph=True)[0] for k in range(3)], 1)
    torch.testing.assert_close(al, sol.polarizability)
    Ft, Gt = 0.1 * torch.randn(2, 3), 0.1 * torch.randn(2, 3, 3)
    Gt = Gt + Gt.transpose(1, 2)
    torch.testing.assert_close(sol.energy(Ft, Gt), field_energy(p, pos, bonds, bi, bb, 2, Ft, Gt, origin))
    assert torch.linalg.eigvalsh(sol.P).min() > -1e-12
    torch.testing.assert_close(torch.zeros(2).index_add(0, bi, sol.charges), torch.tensor([0.0, -1.0]))


def _electrical_model(**over):
    m, b = build(**over)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():                                    # tables away from their neutral defaults
        m.tables.atom.add_(0.3 * torch.randn(m.tables.atom.shape, generator=g))
        m.tables.link.add_(0.3 * torch.randn(m.tables.link.shape, generator=g))
    return m, b


@pytest.mark.parametrize("charges", ["sqe", "flow"])
def test_moments_are_tensors_and_charge_is_conserved(charges):
    m, b = _electrical_model(charges=charges)
    o = m(b)
    q = torch.zeros(b.n_mol).index_add(0, b.mol_idx, o["charges"])
    torch.testing.assert_close(q, b.charge, atol=1e-12, rtol=0)
    assert o["charges"].abs().max() > 1e-3 and o["dipole"].abs().max() > 1e-3
    assert torch.linalg.eigvalsh(o["polarizability"]).min() > 0
    th = 0.8
    rot = torch.tensor([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]])
    rot = rot @ torch.tensor([[math.cos(0.3), -math.sin(0.3), 0.0], [math.sin(0.3), math.cos(0.3), 0.0], [0.0, 0.0, 1.0]])
    for op in (rot, -rot):
        o2 = m(b, b.positions @ op.T)                        # rotation about the coordinate origin
        torch.testing.assert_close(o2["energy"], o["energy"])
        torch.testing.assert_close(o2["charges"], o["charges"])
        torch.testing.assert_close(o2["dipole"], o["dipole"] @ op.T)
        torch.testing.assert_close(o2["second_moment"], op @ o["second_moment"] @ op.T)
        torch.testing.assert_close(o2["polarizability"], op @ o["polarizability"] @ op.T)
    # a translation moves the moments as point charges do: neutral water's dipole is unchanged
    o3 = m(b, b.positions + torch.tensor([0.3, -0.2, 0.5]))
    torch.testing.assert_close(o3["dipole"][0], o["dipole"][0])
    torch.testing.assert_close(o3["polarizability"], o["polarizability"])


def test_tied_charges_respond_and_flows_do_not():
    sqe, b = _electrical_model(polarizability="none", quadrupole_polarizability=False)
    flow, _ = _electrical_model(charges="flow", polarizability="none", quadrupole_polarizability=False)
    a_sqe, a_flow = sqe(b)["polarizability"], flow(b)["polarizability"]
    assert torch.linalg.eigvalsh(a_sqe)[:, -1].min() > 1e-3          # charge flow along the bonds
    assert a_flow.abs().max() == 0
    # the flow polarizability lies along the bond: OH- (molecule 2) has no perpendicular response
    i, j = b.topo.bond_index[:, b.topo.bond_frag == 2][:, 0]
    axis = b.positions[j] - b.positions[i]
    perp = torch.linalg.cross(axis, torch.tensor([1.0, 0.3, 0.2]))
    assert abs(float(perp @ a_sqe[2] @ perp)) < 1e-12 and float(axis @ a_sqe[2] @ axis) > 1e-4
    none, _ = build(charges="none", dipoles=False, quadrupoles=False, polarizability="none", quadrupole_polarizability=False)
    assert not none.has_electrical and "dipole" not in none(b)
    with pytest.raises(ValueError):
        none.electrical(b)


def test_dipole_is_the_field_derivative_and_energy_ignores_the_head():
    m, b = _electrical_model()
    o = m(b)
    sol = o["solution"]
    F = torch.zeros(b.n_mol, 3, requires_grad=True)
    (g,) = torch.autograd.grad(sol.energy(F, None).sum(), F)
    torch.testing.assert_close(-g, o["dipole"])
    assert float(sol.energy().abs().max()) == 0.0                     # zero-field subtracted
    e_only = m(b, electrical=False)
    assert torch.equal(e_only["energy"], o["energy"]) and "dipole" not in e_only
    # point charges and atomic dipoles add up to the molecular dipole about the origin
    p = m.response_parameters(b, *m._encode(b, b.positions, False))
    mu = torch.zeros(b.n_mol, 3).index_add(0, b.mol_idx, sol.charges[:, None] * b.positions / BOHR_ANG + p.mu0)
    torch.testing.assert_close(mu, o["dipole"])


def test_isotropic_polarizability_and_constant_model():
    m, b = _electrical_model(polarizability="isotropic", quadrupoles=False)
    assert m.n2 == 0 and m.gate_readout.irreps.lmax == 1
    p = m.response_parameters(b, *m._encode(b, b.positions, False))
    assert torch.allclose(p.alpha, p.alpha[:, :1, :1] * torch.eye(3)) and p.m2 is None
    const = m(b, constant=True)
    pc = m.response_parameters(b, *m._encode(b, b.positions, True))
    assert pc.mu0.abs().max() == 0                                   # no atomic dipoles from the tables alone
    assert not torch.allclose(const["dipole"], m(b)["dipole"])
    lab = collate(molecules())
    assert lab.dipole is None and not lab.labelled("dipole").any()
