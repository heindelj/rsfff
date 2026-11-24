"""``pauli_env = "edge"``: f_ij = e^{u_i + u_j}, u_i = sum_k u(s_i, s_k; s_ik) from a shared network (PauliEdge).
Starts at the bare site Pauli; u is bounded, one-signed and monotone in the overlap; the 4-body Pauli is second order
in the edges (it scales as the square of their size, the 3-body linearly); forces."""
from __future__ import annotations

import itertools

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm.model import EDGE_AMAX

from test_cmm_ct_site import SITE

f64 = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]], dtype=f64)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
               ct_gap_converge=40, ct_gap_overlap="site", ct_gap_donor=True, pauli_env="edge")
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**{**cfg, **kw}))


def _pauli(m, idx, X=X0):
    sym = [SYM[k] for k in idx]
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m(m.precompute(X[list(idx)], Z, torch.zeros(len(sym), dtype=torch.long)))["e_pauli"].sum()


def _randomize(m, scale=1.0, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        last = m.pauli_edge.mlp[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g, dtype=f64) * 0.5)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g, dtype=f64) * 0.5)
        last.bias[0] = 0.0
        last.weight[0] *= scale
    return m


def test_start_is_bare():
    e = float(_pauli(_model(), range(4)))
    p = _model(pauli_env="product")
    with torch.no_grad():
        p.encoder.readout.weight.zero_()
    assert e == pytest.approx(float(_pauli(p, range(4))), rel=1e-12)


def test_monotone_bounded_one_signed():
    m = _randomize(_model())
    s = torch.linspace(0, 6, 601, dtype=f64)
    for a, b in itertools.product(range(2), range(2)):
        u = m.pauli_edge(torch.full_like(s, a, dtype=torch.long), torch.full_like(s, b, dtype=torch.long), s).detach()
        assert abs(float(u[0])) < 1e-15
        assert float(u.abs().max()) <= EDGE_AMAX
        d = torch.diff(u) * torch.sign(u[-1])
        assert float(d.min()) >= -1e-15 and float(u.abs()[-1]) > 1e-3


def _increments(m):
    inc = {}
    for r in (2, 3, 4):
        for S in itertools.combinations(range(4), r):
            inc[S] = float(_pauli(m, S)) - sum(inc[T] for k in range(2, r) for T in itertools.combinations(S, k))
    return inc


def test_four_body_is_second_order():
    """Halving the edges' size (their sign output) halves the 3-body and quarters the 4-body (to leading order)."""
    r3, r4 = [], []
    for sc in (0.02, 0.01):
        inc = _increments(_randomize(_model(), scale=sc))
        r3.append(max(abs(v) for S, v in inc.items() if len(S) == 3))
        r4.append(abs(inc[(0, 1, 2, 3)]))
    assert r3[0] / r3[1] == pytest.approx(2.0, rel=0.05)
    assert r4[0] / r4[1] == pytest.approx(4.0, rel=0.1)
    assert r4[1] < 0.05 * r3[1]


def test_forces():
    m = _randomize(_model())
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    fr = torch.zeros(4, dtype=torch.long)
    X = X0.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(m.precompute(X, Z, fr))["e_int"].sum(), X)[0]
    h = 1e-4
    for a_, c_ in ((0, 0), (1, 2), (3, 1)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a_, c_] += h
        Xm[a_, c_] -= h
        num = -(m(m.precompute(Xp, Z, fr))["e_int"].sum() - m(m.precompute(Xm, Z, fr))["e_int"].sum()) / (2 * h)
        assert float(F[a_, c_]) == pytest.approx(float(num), abs=2e-7)
