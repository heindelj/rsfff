"""``CmmConfig.xpol_bound``: the tie between the exchange drive and Gram weights that keeps a pair's Pauli wall
(theory/response_functional.tex, the Pauli-wall proposition).

With the bound ("rank" / "scalar") one ion's response removes at most 1 / theta <= 1/2 of a pair's frozen Pauli
energy, so Pauli + exchange polarization >= 0 for every pair, any readouts, any response z; without it
(:meth:`CmmModel.xpol_fraction` as the diagnostic) the fraction is unbounded. The forces stay exact derivatives."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl

FRAMES = [(["Na", "Cl"], [[0.0, 0, 0], [2.8, 0.1, 0]]),
          (["Na", "Cl"], [[0.0, 0, 0], [1.9, 0.0, 0]]),                                     # compressed
          (["Na", "Cl", "Na", "Cl"], [[0.0, 0, 0], [2.8, 0, 0], [2.8, 2.9, 0.1], [0, 2.75, 0.2]]),
          (["Na", "Cl", "Na", "Cl"], [[0.0, 0, 0], [2.0, 0, 0], [2.0, 2.1, 0.1], [0, 1.95, 0.2]]),   # compressed square
          (["Li", "F", "Cl", "Na", "F"], [[0.0, 0, 0], [1.9, 0, 0.1], [-2.6, 0.3, 0], [0.2, 3.1, 0.4],
                                          [-1.0, -2.2, 1.4]])]


def _model(mode, scale=1.5, seed=7):
    torch.manual_seed(seed)
    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    m = CmmModel(tab, chan, CmmConfig(film=False, xpol_bound=mode))
    with torch.no_grad():                       # readouts and constants far from the start: any output must obey
        m.encoder.readout.weight.copy_(scale * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2):
            p.add_(scale * torch.randn_like(p))
    return m


def _batch(m, frames=FRAMES, pos=None):
    X = torch.cat([torch.tensor(x, dtype=torch.float64) for _, x in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(X, Z, fr)


def test_config():
    assert CmmConfig().xpol_bound == "off" and CmmConfig.from_dict({}).xpol_bound == "off"
    assert CmmConfig.from_dict(CmmConfig(xpol_bound="rank").to_dict()).xpol_bound == "rank"
    with pytest.raises(ValueError):
        CmmConfig(xpol_bound="yes")


@pytest.mark.parametrize("mode", ["rank", "scalar"])
def test_fraction_is_bounded(mode):
    """Each side of each pair can remove at most 1 / theta <= 1/2 of the pair's Pauli energy."""
    m = _model(mode)
    bt = _batch(m)
    with torch.no_grad():
        rho = m.xpol_fraction(bt)
        sp = bt["species"]
        i, j = bt["pairs"][:, bt["x_sel"]]
        r = m.xpol_ratio(sp)
    assert float(r.max()) <= 0.5 and float(r.min()) > 0
    cap = torch.cat([r[i], r[j]])
    assert float(rho.max()) > 0.01                       # the pairs do respond
    assert bool((rho <= cap * (1 + 1e-6) + 1e-12).all()), float((rho - cap).max())


def test_scalar_has_one_weight_per_atom():
    m = _model("scalar")
    bt = _batch(m)
    with torch.no_grad():
        m.environment(bt)
        k = m.w1(bt["species"])
    assert torch.equal(k[:, :3], k[:, :1].expand(-1, 3)) and torch.equal(k[:, 3:8], k[:, :1].expand(-1, 5))


@pytest.mark.parametrize("mode", ["rank", "scalar"])
def test_pauli_wall(mode):
    """Pauli + the exchange drive and Gram terms >= sum over pairs of E0 (1 - 1/theta_i - 1/theta_j) >= 0 for random
    responses of any size and for the response that minimizes them."""
    m = _model(mode)
    bt = _batch(m)
    with torch.no_grad():
        sp = bt["species"]
        m.environment(bt)
        m.frozen_terms(bt, sp)
        (i, j), g, Gi, Gj = m.exchange(bt, sp)
        e0 = m._pauli_pairs[1][bt["x_sel"]]
        r = m.xpol_ratio(sp)
        floor = (e0 * (1 - r[i] - r[j])).sum()
        assert float(floor) >= 0
        n, D = sp.shape[0], g.shape[1]
        H = torch.zeros(n, D, D, dtype=g.dtype).index_add(0, i, Gi).index_add(0, j, Gj)

        def energy(z):
            return e0.sum() + (g * z).sum() + 0.5 * torch.einsum("na,nab,nb->", z, H, z)

        zmin = -torch.einsum("nab,nb->na", torch.linalg.pinv(H, hermitian=True), g)
        assert float(energy(zmin)) >= float(floor) * (1 - 1e-6) - 1e-12
        torch.manual_seed(0)
        for s in (1e-3, 1e-1, 1.0, 10.0):
            for _ in range(20):
                assert float(energy(zmin + s * torch.randn(n, D, dtype=g.dtype))) >= float(floor) * (1 - 1e-6) - 1e-12


def test_unbound_model_can_exceed_the_bound():
    """The diagnostic on a model without the bound: a strong drive on a soft Gram block removes more than the pair's
    Pauli energy."""
    m = _model("off", scale=0.0)
    with torch.no_grad():
        m.ln_w1.fill_(1.0)
        m.ln_w2.fill_(-1.5)
        rho = m.xpol_fraction(_batch(m))
    assert float(rho.max()) > 1.0
    with pytest.raises(ValueError):
        m.xpol_ratio(torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("mode", ["rank", "scalar"])
def test_forces_are_exact_derivatives(mode):
    m = _model(mode, scale=0.3)
    frames = [FRAMES[2], FRAMES[4]]
    pos = torch.cat([torch.tensor(x, dtype=torch.float64) for _, x in frames])

    def E(p):
        return m(_batch(m, frames, p))["e_int"].sum()

    pg = pos.clone().requires_grad_(True)
    g = torch.autograd.grad(E(pg), pg)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((0, 0), (3, 1), (6, 2)):
            pp, pm = pos.clone(), pos.clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7
