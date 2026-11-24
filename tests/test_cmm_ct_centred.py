"""``ct_gap_centred``: the gap plus delta_D(s_i) n.grad phi3(X_i) + delta_A(s_j) n.grad phi3(X_j), n the channel's
direction (donor -> acceptor), phi3 the third ions' frozen potential. A dimer's gap is unchanged; the field is the
atom's frozen drive less its partner's pair row; far away it is the point charges' gradient; forces are exact."""
from __future__ import annotations

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm.kernels import BOHR_ANG

from test_cmm_ct_site import SITE

f64 = torch.float64
SYM = ["Na", "Cl", "Na"]
X0 = torch.tensor([[0.0, 0, 0], [2.7, 0.1, 0], [4.9, 1.9, 0.3]], dtype=f64)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
               ct_gap_converge=40, ct_gap_overlap="site", ct_gap_donor=True, ct_cation_donor=False,
               ct_gap_centred=True)
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**{**cfg, **kw}))


def _bt(m, X, sym=SYM):
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m.precompute(X, Z, torch.zeros(len(sym), dtype=torch.long))


def _gaps(m, bt):
    got = {}
    orig = m.gap_channels

    def hook(b, sp):
        g = orig(b, sp)
        got["Delta"] = g["Delta"].detach()
        return g
    m.gap_channels = hook
    try:
        m(bt)
    finally:
        del m.gap_channels
    return got["Delta"]


@pytest.fixture(scope="module")
def model():
    m = _model()
    with torch.no_grad():
        m.ct_centre_raw.add_(torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=f64))
    return m


def test_dimer_gap_unchanged(model):
    bt = _bt(model, X0[:2], SYM[:2])
    assert torch.allclose(_gaps(model, bt), _gaps(_model(ct_gap_centred=False), bt), rtol=0, atol=1e-13)


def test_gap_shift(model):
    m = model
    bt = _bt(m, X0)
    i, j = bt["ch_index"]
    d = m.ct_centre()
    sp = bt["species"]
    want = d[sp[i], 0] * bt["ch_L3"][:, 0] + d[sp[j], 1] * bt["ch_L3"][:, 1]
    assert want.abs().max() > 1e-4
    assert torch.allclose(_gaps(m, bt) - _gaps(_model(ct_gap_centred=False), bt), want.detach(), atol=1e-13)


def test_field_is_the_third_ions(model):
    """ch_L3 at donor and acceptor = the trimer's drive less the dimer's (the pair rows are additive)."""
    m = model
    bt3, bt2 = _bt(m, X0), _bt(m, X0[:2], SYM[:2])
    L3, L2 = bt3["hyp_L"][:, :3], bt2["hyp_L"][:, :3]
    for c, (a, b) in enumerate(bt3["ch_index"].T.tolist()):
        if {a, b} != {0, 1}:
            continue
        n = bt3["ch_rhat"][c]
        assert float(bt3["ch_L3"][c, 0]) == pytest.approx(float(((L3[a] - L2[a]) * n).sum()), abs=1e-12)
        assert float(bt3["ch_L3"][c, 1]) == pytest.approx(float(((L3[b] - L2[b]) * n).sum()), abs=1e-12)


def test_far_field_is_grad_of_point_potential(model):
    """A third ion far off: n.grad phi3 is the point charge's (L = grad phi, so phi3 lower toward the acceptor makes
    the electron cheaper to take from the donor)."""
    m = model
    X = X0.clone()
    X[2] = torch.tensor([-9.0, 4.0, 1.0], dtype=f64)
    bt = _bt(m, X)
    q = float(m.tab.charge[list(m.tab.symbols).index("Na")])
    for c, (a, b) in enumerate(bt["ch_index"].T.tolist()):
        if {a, b} != {0, 1}:
            continue
        n = bt["ch_rhat"][c]
        for slot, at in ((0, a), (1, b)):
            dr = (X[at] - X[2]) / BOHR_ANG
            grad = -q * dr / dr.norm() ** 3
            assert float(bt["ch_L3"][c, slot]) == pytest.approx(float((grad * n).sum()), rel=2e-3)


def test_forces(model):
    m = model
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    fr = torch.zeros(3, dtype=torch.long)
    X = X0.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(m.precompute(X, Z, fr))["e_int"].sum(), X)[0]
    h = 1e-4
    for a_, c_ in ((0, 0), (1, 1), (2, 2)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a_, c_] += h
        Xm[a_, c_] -= h
        num = -(m(m.precompute(Xp, Z, fr))["e_int"].sum() - m(m.precompute(Xm, Z, fr))["e_int"].sum()) / (2 * h)
        assert float(F[a_, c_]) == pytest.approx(float(num), abs=2e-7)


# -- ct_gap_centred = "overlap": s_D = lam t_D, s_A = lam (R - t_D), the other ions' frozen potentials exact -------
import math  # noqa: E402

import numpy as np  # noqa: E402

from rsfff.ff.cmm.model import frozen_ion_potential, overlap_centroid, shaped_ion_potential  # noqa: E402


def _overlap(raw=None, **kw):
    m = _model(ct_gap_centred="overlap", **kw)
    if raw is not None:
        with torch.no_grad():
            m.ct_centre_lam.fill_(raw)
    return m


def test_centroid_brute_force():
    a, b, R = (torch.tensor(v, dtype=f64) for v in ([0.93, 1.32, 0.93], [1.44, 1.6, 0.93], [5.3, 3.6, 6.5]))
    got = overlap_centroid(a, b, R)
    z, x = torch.linspace(-20, 30, 1001, dtype=f64), torch.linspace(0, 20, 401, dtype=f64)
    Zg, Xg = torch.meshgrid(z, x, indexing="ij")
    for k in range(3):
        w = torch.exp(-a[k] * torch.hypot(Zg, Xg) - b[k] * torch.hypot(Zg - R[k], Xg)) * Xg
        assert float(got[k]) == pytest.approx(float((w * Zg).sum() / w.sum()), abs=2e-3)
    assert float(got[2]) == pytest.approx(3.25, abs=1e-12)


def test_overlap_dimer_unchanged():
    bt = _bt(_overlap(2.0), X0[:2], SYM[:2])
    assert torch.allclose(_gaps(_overlap(2.0), bt), _gaps(_model(ct_gap_centred="off"), bt), rtol=0, atol=1e-13)


def test_shaped_potential_quadrature():
    """The ion's frozen potential averaged over a Slater shape, against a quadrature in spherical coordinates about
    the ion (radial trapezoid x Gauss-Legendre in cos theta); finite at r -> 0."""
    q = torch.tensor([-1.0], dtype=f64)
    n, b = torch.tensor([[7.0, 1.5]], dtype=f64), torch.tensor([[1.9, 0.9]], dtype=f64)
    beta = torch.tensor([1.3], dtype=f64)
    sr = torch.linspace(1e-6, 60.0, 120001, dtype=f64)
    u, wu = (torch.tensor(t, dtype=f64) for t in np.polynomial.legendre.leggauss(400))
    phi, _ = frozen_ion_potential(q, n, b, sr)
    for r in (0.7, 2.5, 6.0):
        d = torch.sqrt((sr[:, None] ** 2 + r * r - 2 * r * sr[:, None] * u).clamp(min=0))
        shape = (beta ** 3 / (8 * math.pi) * torch.exp(-beta * d) * wu).sum(1) / 2      # angular mean
        want = float(torch.trapezoid(4 * math.pi * sr ** 2 * phi * shape, sr))
        got = float(shaped_ion_potential(q, n, b, beta, torch.tensor([r], dtype=f64)))
        assert got == pytest.approx(want, abs=1e-6)
    v = shaped_ion_potential(q.expand(2), n.expand(2, 2), b.expand(2, 2), beta.expand(2), torch.tensor([1e-4, 1e-3], dtype=f64))
    assert torch.isfinite(v).all() and float((v[0] - v[1]).abs()) < 1e-2


def test_overlap_shift_direct_sum():
    """The shift against a direct sum over the third ions with the shapes at the moved points (cluster)."""
    m = _overlap(0.7)
    bt = _bt(m, X0)
    sp = bt["species"]
    sh = m.centre_overlap_shift(bt, sp).detach()
    lam = float(torch.sigmoid(m.ct_centre_lam))
    bD, bA = (t.detach() for t in m.ct_exponents())
    Xb = X0 / BOHR_ANG
    tab = m.tab
    for c, (a, b) in enumerate(bt["ch_index"].T.tolist()):
        n = bt["ch_rhat"][c]
        R = float((Xb[b] - Xb[a]).norm())
        tD = float(overlap_centroid(bD[sp[a]], bA[sp[b]], torch.tensor(R, dtype=f64)))
        for slot, at, P, be in ((0, a, Xb[a] + lam * tD * n, bD[sp[a]]), (1, b, Xb[b] - lam * (R - tD) * n, bA[sp[b]])):
            want = 0.0
            for k in range(3):
                if k in (a, b):
                    continue
                q, nk, bk = tab.charge[sp[k]].reshape(1), tab.shell_n[sp[k]][None], tab.shell_b[sp[k]][None]
                bk = torch.where(nk > 0, bk, torch.ones_like(bk))
                V = [shaped_ion_potential(q, nk, bk, be.reshape(1), (Y - Xb[k]).norm().reshape(1)) for Y in (P, Xb[at])]
                want += float(V[0] - V[1])
            assert float(sh[c, slot]) == pytest.approx(want, abs=1e-12)
    assert sh.abs().max() > 1e-3


def test_overlap_first_order_sign():
    """Small lam: the shift is lam t n.grad of the third ions' shaped potential, the sign of ch_L3 (far ion)."""
    m = _overlap(-9.0)
    X = X0.clone()
    X[2] = torch.tensor([-9.0, 4.0, 1.0], dtype=f64)
    bt = _bt(m, X)
    sh = m.centre_overlap_shift(bt, bt["species"]).detach()
    lam = float(torch.sigmoid(m.ct_centre_lam))
    i, j = bt["ch_index"]
    sp = bt["species"]
    R = bt["ch_r"] / BOHR_ANG
    bD, bA = m.ct_exponents()
    tD = overlap_centroid(bD[sp[i]], bA[sp[j]], R).detach()
    want = torch.stack([lam * tD * bt["ch_L3"][:, 0], -lam * (R - tD) * bt["ch_L3"][:, 1]], 1)
    assert torch.allclose(sh, want, rtol=3e-3, atol=1e-14)


def test_overlap_forces():
    m = _overlap(0.4)
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    fr = torch.zeros(3, dtype=torch.long)
    X = X0.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(m.precompute(X, Z, fr))["e_int"].sum(), X)[0]
    h = 1e-4
    for a_, c_ in ((0, 0), (1, 1), (2, 2), (2, 0)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a_, c_] += h
        Xm[a_, c_] -= h
        num = -(m(m.precompute(Xp, Z, fr))["e_int"].sum() - m(m.precompute(Xm, Z, fr))["e_int"].sum()) / (2 * h)
        assert float(F[a_, c_]) == pytest.approx(float(num), abs=2e-7)


def test_overlap_contact_window():
    """The centring is on at contact and off for channels through a bridging ion (rocksalt third neighbours) and for
    like-charge channels."""
    m = _overlap(0.7)
    s = list(m.tab.symbols)
    na, cl = s.index("Na"), s.index("Cl")
    assert float(m.ctc_ron[cl, na]) > 3.2 and float(m.ctc_roff[cl, na]) < 4.7
    X = torch.tensor([[0.0, 0, 0], [2.8, 0, 0], [5.6, 0, 0]], dtype=f64)
    bt = _bt(m, X, ["Cl", "Na", "Cl"])
    sh = m.centre_overlap_shift(bt, bt["species"]).detach()
    for c, (a, b) in enumerate(bt["ch_index"].T.tolist()):
        if {a, b} == {0, 2}:
            assert float(sh[c].abs().max()) == 0.0              # Cl > Cl through the Na
        else:
            assert float(sh[c].abs().max()) > 1e-3
