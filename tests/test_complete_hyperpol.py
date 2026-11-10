"""Dipole hyperpolarization (``CompleteModel(hyperpol=True)``): the convention of hyper_dalpha against the free-ion
fit's (a charge q at R on z: alpha_zz += gamma/2 E^2 + b 2q/R^3, alpha_xx += gamma/6 E^2 - b q/R^3), gamma = b = 0
as the model without it, and forces through the one-shot polarizability."""
import dataclasses
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
HYP = {"Na": (8.1, 0.708), "Cl": (5512.8, 542.0)}


@pytest.fixture(scope="module")
def setup():
    tab0, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    tab = dataclasses.replace(tab0, hyper_gamma=torch.tensor([HYP[s][0] for s in tab0.symbols], dtype=torch.float64),
                              hyper_b=torch.tensor([HYP[s][1] for s in tab0.symbols], dtype=torch.float64))
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)

    def make(t=tab, **kw):
        torch.manual_seed(3)
        return CompleteModel(t, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                             pauli_form="shells", env_cap=3.0, xpol=True, **kw)
    return tab0, tab, chan, table, make


def _bt(table, tab, chan, X, Z, m, **kw):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=m.inv_form, **kw)


def test_convention_far_from_a_point_charge(setup):
    """Na+ 12 A from Cl- on z (no penetration, no damping): Cl's correction is the free-ion fit's tensor."""
    tab0, tab, chan, table, make = setup
    m = make(hyperpol=True)
    R = 12.0
    for d in (torch.tensor([0, 0, 1.0]), torch.tensor([1.0, 1.0, 0]) / 2 ** 0.5, torch.tensor([1.0, 0, 0]),
              torch.tensor([0.3, -0.5, 0.8]) / (0.98 ** 0.5)):
        X = torch.stack([torch.zeros(3), R * d]).to(torch.float64)
        bt = _bt(table, tab, chan, X, torch.tensor([17, 11]), m)
        sp = bt["species"]
        dA = m.hyper_dalpha(bt["L"][bt["mask"]], sp)[0]
        Rb = R / K.BOHR_ANG
        g, b = HYP["Cl"]
        dd = d.to(torch.float64)
        ref = g / 6 * Rb ** -4 * (torch.eye(3, dtype=torch.float64) + 2 * dd[:, None] * dd[None]) \
            + b * (3 * dd[:, None] * dd[None] - torch.eye(3, dtype=torch.float64)) / Rb ** 3
        assert torch.allclose(dA, ref, rtol=1e-6, atol=1e-9)
        par = float(dd @ dA @ dd)
        assert abs(par - (g / 2 * Rb ** -4 + 2 * b / Rb ** 3)) < 1e-6 * par


def test_off_and_zero_coefficients_are_the_base_model(setup):
    tab0, tab, chan, table, make = setup
    X = torch.tensor([[0, 0, 0], [0, 0, 2.7], [2.6, 0.3, 0.1], [2.5, 0.2, 2.8]], dtype=torch.float64)
    Z = torch.tensor([11, 17, 17, 11])
    m0, m1, m2 = make(), make(hyperpol=True), make(t=tab0, hyperpol=True)
    assert m1.hyperpol and not m0.hyperpol and not m2.hyperpol            # tab0: no coefficients -> off
    bt = _bt(table, tab, chan, X, Z, m0)
    with torch.no_grad():
        o0, o1 = m0(bt), m1(bt)
    assert float(o0["e_pol"]) != float(o1["e_pol"])
    m1.tab = dataclasses.replace(tab, hyper_gamma=torch.zeros_like(tab.hyper_gamma), hyper_b=torch.zeros_like(tab.hyper_b))
    with torch.no_grad():
        assert abs(float(m1(bt)["e_pol"] - o0["e_pol"])) < 1e-14
    m1.tab = tab


def test_forces(setup):
    tab0, tab, chan, table, make = setup
    m = make(hyperpol=True)
    X0 = torch.tensor([[0, 0, 0], [0, 0, 2.7], [2.6, 0.3, 0.1], [2.5, 0.2, 2.8]], dtype=torch.float64)
    Z = torch.tensor([11, 17, 17, 11])

    def energy(X):
        return m(_bt(table, tab, chan, X, Z, m))["e_pol"].sum()
    X = X0.clone().requires_grad_(True)
    g = torch.autograd.grad(energy(X), X)[0]
    h = 1e-4
    for a, c in ((1, 2), (2, 0), (3, 1)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        with torch.no_grad():
            fd = (energy(Xp) - energy(Xm)) / (2 * h)
        assert abs(float(g[a, c] - fd)) < 1e-7 + 1e-5 * abs(float(fd)), (a, c, float(g[a, c]), float(fd))
