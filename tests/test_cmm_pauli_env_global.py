"""``pauli_env = "global"``: the linear site Pauli factor with one constant on the normalized overlap,
f_i = 1 + c sum_k sw O_ik / (sqrt(O_ii O_kk) S_REF). No 4-body term; the pair factor; the self-overlap; forces."""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K
from rsfff.ff.cmm.model import PAULI_GLOB_CAP, PAULI_GLOB_SREF

from test_cmm_ct_site import SITE

f64 = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]], dtype=f64)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
               ct_gap_converge=40, ct_gap_overlap="site", ct_gap_donor=True, pauli_env="global")
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**{**cfg, **kw}))


def _pauli(m, idx, X=X0):
    sym = [SYM[k] for k in idx]
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m(m.precompute(X[list(idx)], Z, torch.zeros(len(sym), dtype=torch.long)))["e_pauli"].sum()


@pytest.fixture(scope="module")
def model():
    m = _model()
    with torch.no_grad():
        m.pauli_glob_c.fill_(0.17)
    return m


def test_zero_c_is_the_linear_zero(model):
    assert float(_pauli(_model(), range(4))) == pytest.approx(float(_pauli(_model(pauli_env="linear"), range(4))),
                                                              rel=1e-12)


def test_self_overlap_matches_quadrature():
    m = _model()
    tab = m.tab
    k, wk = K.table_ovl_grid(f64, "cpu")
    one = torch.zeros(4, dtype=f64)
    one[0] = 1.0
    for s in range(len(tab.symbols)):
        F = sum(tab.shell_n[s, t] * K.shaped_density_ft(tab.shell_b[s, t].reshape(1), one.reshape(1, 4), k)[0]
                for t in range(tab.shell_n.shape[1]) if tab.shell_n[s, t] > 0)
        num = float((F * F * wk * k * k).sum() / (2 * math.pi ** 2))
        assert float(m.pauli_self_rt[s]) ** 2 == pytest.approx(num, rel=1e-6)


def test_no_four_body_term(model):
    m = model
    inc = {}
    for r in (2, 3, 4):
        for S in itertools.combinations(range(4), r):
            inc[S] = float(_pauli(m, S)) - sum(inc[T] for k in range(2, r) for T in itertools.combinations(S, k))
    assert abs(inc[(0, 1, 2, 3)]) < 1e-12 * abs(float(_pauli(m, range(4))))
    assert max(abs(inc[S]) for S in inc if len(S) == 3) > 1e-6


def test_pair_factor(model):
    m = model
    e = float(_pauli(m, (0, 1)))
    bare = float(_pauli(_model(), (0, 1)))
    bt = m.precompute(X0[:2], torch.tensor([ATOMIC_NUMBER[a] for a in SYM[:2]]), torch.zeros(2, dtype=torch.long))
    s = list(m.tab.symbols)
    a, b = s.index("Na"), s.index("Cl")
    S = float(bt["x_w"][0]) / float(m.pauli_self_rt[a] * m.pauli_self_rt[b]) / PAULI_GLOB_SREF
    c = PAULI_GLOB_CAP * math.tanh(0.17 / PAULI_GLOB_CAP)
    assert e == pytest.approx(bare * (1 + c * S), rel=1e-10)


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
