"""``pauli_env = "linear"``: the site Pauli pair factor (f_i + f_j) / 2 with f_i = 1 + sum_k c[s_i, s_k] sw O_ik / O_REF.
At c = 0 it is the bare site energy; for any c the Pauli energy has no 4-body increment; a pair carries
1 + (c_ab + c_ba) w / (2 O_REF); forces are exact."""
from __future__ import annotations

import itertools

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm.model import PAULI_LIN_CAP, PAULI_LIN_OREF

from test_cmm_ct_site import SITE

f64 = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]], dtype=f64)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
               ct_gap_converge=40, ct_gap_overlap="site", ct_gap_donor=True, pauli_env="linear")
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**{**cfg, **kw}))


def _pauli(m, idx, X=X0):
    sym = [SYM[k] for k in idx]
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m(m.precompute(X[list(idx)], Z, torch.zeros(len(sym), dtype=torch.long)))["e_pauli"].sum()


@pytest.fixture(scope="module")
def model():
    m = _model()
    with torch.no_grad():
        m.pauli_lin_c.copy_(torch.tensor([[0.05, -0.08], [0.12, -0.03]], dtype=f64))
    return m


def test_zero_c_is_the_bare_energy():
    m = _model()
    torch.manual_seed(1)
    with torch.no_grad():                   # the encoder's Pauli readout plays no part
        m.encoder.readout.weight.copy_(50.0 * torch.randn_like(m.encoder.readout.weight))
    e = float(_pauli(m, range(4)))
    p = _model(pauli_env="product")
    with torch.no_grad():
        p.encoder.readout.weight.zero_()
        if p.encoder.readout.bias is not None:
            p.encoder.readout.bias.zero_()
    assert e == pytest.approx(float(_pauli(p, range(4))), rel=1e-12)


def test_no_four_body_term(model):
    m = model
    inc = {}
    for r in (2, 3, 4):
        for S in itertools.combinations(range(4), r):
            inc[S] = float(_pauli(m, S)) - sum(inc[T] for k in range(2, r) for T in itertools.combinations(S, k))
    assert abs(inc[(0, 1, 2, 3)]) < 1e-12 * abs(float(_pauli(m, range(4))))
    assert max(abs(inc[S]) for S in inc if len(S) == 3) > 1e-6        # the 3-body term is there


def test_pair_factor(model):
    m = model
    e = float(_pauli(m, (0, 1)))
    bare = float(_pauli(_model(), (0, 1)))
    bt = m.precompute(X0[:2], torch.tensor([ATOMIC_NUMBER[a] for a in SYM[:2]]), torch.zeros(2, dtype=torch.long))
    w = float(bt["x_w"][0]) / PAULI_LIN_OREF
    c = PAULI_LIN_CAP * torch.tanh(m.pauli_lin_c / PAULI_LIN_CAP)
    s = list(m.tab.symbols)
    a, b = s.index("Na"), s.index("Cl")
    assert e == pytest.approx(bare * (1 + 0.5 * float(c[a, b] + c[b, a]) * w), rel=1e-10)


def test_forces(model):
    m = model
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
