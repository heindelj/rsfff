"""The opposite bond-angle coupling (``film.opposite_coupling``; :mod:`rsfff.ff.film.terms`).

* its instances are the angle pairs' triples: the leg ``apex-s`` and the angle ``x-apex-y``;
* the energy is ``2 rho sqrt(D_s k_theta,xy / 2) y_s dc_xy``, checked against a direct sum for
  H3O+ with the constants of the constant-parameter fit (``scripts/fit_constant_monomers.py``);
* off by default (no parameters, no state-dict change); switched on, a fresh model is the
  model without it, and a trained one is conservative and invariant to relabeling.
"""

from __future__ import annotations

import math

import pytest
import torch

from rsfff.ff.film.bonded import BondedParameters, morse_energy
from rsfff.ff.film.terms import TermParameterHead, TermParameters, term_energies
from rsfff.ff.units import BOHR_ANG

from test_bonding import _topo, acetaldehyde, bonding_model, make_batch

# H3O+ "+bb+ba+aa+bo" of params/bonded_monomers_wb97mv_tzvpd.json (atomic units)
H3O = dict(r_eq=1.8531, d=0.1600, k=0.4757, theta_eq=math.radians(112.4938), k_theta=0.1428,
           bb=0.0072, ba=-0.0783, bo=-0.0411, aa=-0.0660)


def h3o(seed=0):
    torch.manual_seed(seed)
    pos = torch.tensor([[0.0, 0.0, 0.075], [0.941, 0.0, -0.201], [-0.470, 0.815, -0.201],
                        [-0.470, -0.815, -0.201]], dtype=torch.float64)
    return pos + 0.08 * torch.randn(4, 3, dtype=torch.float64), [8, 1, 1, 1], [(0, 1), (0, 2), (0, 3)]


def test_instances_are_the_angle_pair_triples():
    t = _topo([8, 1, 1, 1], [(0, 1), (0, 2), (0, 3)])
    assert t.n_angle_pairs == 3
    for n in range(3):
        apex, s, x, y = (int(v) for v in t.angle_pair_index[:, n])
        assert sorted(t.bond_index[:, t.angle_pair_bond[n]].tolist()) == sorted([apex, s])
        i, a, k = (int(v) for v in t.angle_index[:, t.angle_pair_opposite[n]])
        assert a == apex and sorted([i, k]) == [x, y] and s not in (x, y)
    water = _topo([8, 1, 1], [(0, 1), (0, 2)])
    assert water.n_angle_pairs == 0 and water.angle_pair_bond.numel() == 0


def test_energy_matches_the_direct_sum():
    pos, z, bonds = h3o()
    t = _topo(z, bonds)
    p = H3O
    one_b, one_a, one_p = torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    bp = BondedParameters(r_eq=p["r_eq"] * one_b, d=p["d"] * one_b, k=p["k"] * one_b,
                          cos_theta_eq=math.cos(p["theta_eq"]) * one_a, k_theta=p["k_theta"] * one_a)
    tp = TermParameters(bb_rho=p["bb"] * one_a, ba_rho=p["ba"] * torch.ones(3, 2, dtype=torch.float64),
                        aa_rho=p["aa"] * one_p, bo_rho=p["bo"] * one_p)
    r, cos_t = t.geometry(pos)
    e_b, e_a = bp.energy(r, cos_t, t)
    got = float(e_b.sum() + e_a.sum() + sum(e.sum() for e, _ in term_energies(pos, t, bp, tp)))

    b = (pos[1:] - pos[0]) / BOHR_ANG
    rr = b.norm(dim=-1)
    u = b / rr[:, None]
    y = 1.0 - torch.exp(-math.sqrt(p["k"] / (2 * p["d"])) * (rr - p["r_eq"]))
    ref = float(morse_energy(rr, p["r_eq"] * one_b, p["d"] * one_b, p["k"] * one_b).sum())
    pairs = [(0, 1), (0, 2), (1, 2)]
    dc = {jl: float(u[jl[0]] @ u[jl[1]]) - math.cos(p["theta_eq"]) for jl in pairs}
    s_bt = math.sqrt(p["d"] * p["k_theta"] / 2)
    for (j, l), d in dc.items():
        ref += 0.5 * p["k_theta"] * d * d + 2 * p["bb"] * p["d"] * float(y[j] * y[l])
        ref += 2 * p["ba"] * s_bt * float(y[j] + y[l]) * d + 2 * p["bo"] * s_bt * float(y[3 - j - l]) * d
    for a in range(3):
        for c in range(a + 1, 3):
            ref += 2 * p["aa"] * (p["k_theta"] / 2) * dc[pairs[a]] * dc[pairs[c]]
    assert abs(got - ref) < 1e-12

    # the opposite term alone is not zero here
    only = TermParameters(bo_rho=tp.bo_rho)
    assert abs(float(sum(e.sum() for e, _ in term_energies(pos, t, bp, only)))) > 1e-6


def test_off_by_default_and_fresh_model_unchanged():
    base = bonding_model().double()
    assert base.network.term_head.pair_table.shape[1] == 1
    assert base(make_batch(acetaldehyde())).parameters.terms0.bo_rho is None
    m = bonding_model(opposite_coupling=True).double()
    assert m.network.term_head.pair_table.shape[1] == 2
    out = m(make_batch(acetaldehyde()))
    assert torch.equal(out.parameters.terms0.bo_rho, torch.zeros_like(out.parameters.terms0.bo_rho))
    assert abs(float(out.energy) - float(base(make_batch(acetaldehyde())).energy)) < 1e-12
    with pytest.raises(ValueError):
        TermParameterHead(8, 3, torsions=True, couplings=False, opposite=True)


def test_trained_opposite_coupling_is_conservative_and_invariant():
    m = bonding_model(randomize=True, opposite_coupling=True).double()
    b = make_batch(acetaldehyde())
    out = m(b)
    assert out.parameters.terms0.bo_rho.abs().max() > 0
    bp, tp = out.parameters.bonded0, out.parameters.terms0
    assert float(sum(e.abs().sum() for e, _ in term_energies(
        b.positions, out.topology, bp, TermParameters(bo_rho=tp.bo_rho)))) > 0

    b = make_batch(acetaldehyde())
    b.positions.requires_grad_(True)
    (g,) = torch.autograd.grad(m(b).energy.sum(), b.positions)
    h = 1e-5
    for atom, comp in ((1, 0), (2, 1), (4, 2)):
        e = []
        for sgn in (1, -1):
            bb = make_batch(acetaldehyde())
            with torch.no_grad():
                bb.positions[atom, comp] += sgn * h
            e.append(float(m(bb).energy.sum()))
        fd = (e[0] - e[1]) / (2 * h)
        assert abs(fd - float(g[atom, comp])) < 1e-6 * max(1.0, abs(fd))

    e0 = float(m(make_batch(acetaldehyde())).energy)
    pos, z, bonds = acetaldehyde()
    perm = [1, 0, 2, 3, 6, 4, 5]
    inv = {old: new for new, old in enumerate(perm)}
    e1 = float(m(make_batch((pos[perm], [z[i] for i in perm], [(inv[a], inv[b]) for a, b in bonds]))).energy)
    assert abs(e1 - e0) < 1e-10
