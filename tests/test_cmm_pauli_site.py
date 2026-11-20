"""The site-potential form of the frozen Pauli energy (``CmmConfig.pauli_form = "site"``): the closed-form kernel is
the two-centre integral it stands for; an isolated pair's Pauli energy is <n_i | v_j> + <n_j | v_i> with the table's
shells and site potentials; the environment factor of both ions multiplies the pair; the forces are exact
derivatives; the default form is untouched and a table without site potentials is refused."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K
from rsfff.ff.cmm.model import site_pauli_kernel

SITE = Path(__file__).resolve().parents[1] / "params" / "ions_salts8_site.json"
GOLDEN = Path(__file__).parent / "cmm_golden" / "ions_golden.json"
f64 = torch.float64


def _quadrature(b, beta, R, n=400):
    """int rho_b(r) v_beta(|r - R|) d3r in bipolar coordinates (Gauss-Legendre on [0, R] and [R, R + 80 / b])."""
    x, w = np.polynomial.legendre.leggauss(n)
    cum = lambda s: (1.5 - np.exp(-beta * s) * (1.5 + 0.5 * beta * s)) / beta          # int_0^s t v(t) dt
    tot = 0.0
    for lo, hi in ((0.0, R), (R, R + 80.0 / b)):
        r = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        tot += np.sum(0.5 * (hi - lo) * w * r * b ** 3 / (8 * np.pi) * np.exp(-b * r) * (cum(r + R) - cum(np.abs(r - R))))
    return 2 * np.pi / R * tot


@pytest.mark.parametrize("b, beta, R", [(1.86, 3.1, 4.5), (3.9, 2.0, 5.0), (1.3, 1.3, 3.0), (0.93, 12.0, 2.2), (2.2, 2.3, 6.0)])
def test_site_kernel_is_the_two_centre_integral(b, beta, R):
    t = lambda v: torch.tensor(v, dtype=f64)
    k = float(site_pauli_kernel(t(b), t(beta), t(R)))
    assert k > 0
    assert k == pytest.approx(_quadrature(b, beta, R), rel=1e-8)


@pytest.fixture(scope="module")
def pair():
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    chan = ChannelTable.direct(("Na", "Cl"))
    m = CmmModel(tab, chan, CmmConfig(film=False, pauli_form="site"))
    return tab, m


def _run(m, sym, X):
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m(m.precompute(X, Z, torch.zeros(len(sym), dtype=torch.long)))


def test_pair_energy_is_electrons_in_the_site_potentials(pair):
    tab, m = pair
    R = 2.8
    out = _run(m, ["Na", "Cl"], torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=f64))
    rb = torch.tensor(R / K.BOHR_ANG, dtype=f64)
    a, s = tab.pauli_site_a, tab.pauli_site_s
    ref = 0.0
    for i, j in ((0, 1), (1, 0)):                      # i's shells in j's potential
        for t in range(tab.shell_n.shape[1]):
            for u in range(tab.shell_n.shape[1]):
                if tab.shell_n[i, t] > 0 and tab.shell_n[j, u] > 0:
                    ref += float(tab.shell_n[i, t] * a[j, u] * site_pauli_kernel(tab.shell_b[i, t], s[j] * tab.shell_b[j, u], rb))
    assert float(out["e_pauli"]) == pytest.approx(ref, rel=1e-10)
    assert 20.0 < ref * 2625.4996 < 60.0               # the EDA's mod_pauli of Na-Cl at 2.8 A is about 36 kJ/mol


def test_environment_factors_multiply_the_pair(pair):
    """With random readouts the pair's energy is the bare one times e^{cap(u_i) + cap(u_j)} (the model's own u)."""
    tab, m = pair
    sym, X = ["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.6, 0, 0]], dtype=f64)
    e0 = float(_run(m, sym, X)["e_pauli"])
    torch.manual_seed(1)
    with torch.no_grad():
        m.encoder.readout.weight.copy_(50.0 * torch.randn_like(m.encoder.readout.weight))
    try:
        e1 = float(_run(m, sym, X)["e_pauli"])
        u = m.cap(m._u["pauli"])
    finally:
        with torch.no_grad():
            m.encoder.readout.weight.zero_()
    assert abs(float(u.sum())) > 1e-3 and float(u.abs().max()) < m.config.env_cap
    assert e1 == pytest.approx(e0 * math.exp(float(u.sum())), rel=1e-10)


def test_forces_are_exact_derivatives(pair):
    tab, m = pair
    torch.manual_seed(3)
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
    sym = ["Na", "Cl", "Na", "Cl"]
    X = torch.tensor([[0.0, 0, 0], [2.8, 0, 0], [2.8, 2.9, 0.1], [0, 2.75, 0.2]], dtype=f64)
    E = lambda P: _run(m, sym, P)["e_pauli"].sum()
    pg = X.clone().requires_grad_(True)
    g = torch.autograd.grad(E(pg), pg)[0]
    h = 1e-4
    for a, c in ((0, 0), (1, 1), (3, 2)):
        d = torch.zeros_like(X); d[a, c] = h
        fd = (E(X + d) - E(X - d)) / (2 * h)
        assert float(g[a, c]) == pytest.approx(float(fd), rel=2e-5, abs=1e-9)
    with torch.no_grad():
        m.encoder.readout.weight.zero_()


def test_a_table_without_site_potentials_is_refused():
    tab = IonTable.load(GOLDEN, symbols=("Na", "Cl"))
    with pytest.raises(ValueError, match="pauli_site"):
        CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(film=False, pauli_form="site"))
    with pytest.raises(ValueError, match="pauli_form"):
        CmmConfig(pauli_form="overlap")
