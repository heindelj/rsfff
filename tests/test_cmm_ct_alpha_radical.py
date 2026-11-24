"""``ct_alpha_radical``: the sites' dipole polarizabilities move with their transferred charge toward the CT
radicals'. "outer": inside the joint minimum (passes to a fixed point); "joint": first order about the ion model's
joint minimum (one factorization, two CT solves). Zero steps reproduce the ion model; E_pol is the ion model's; forces
and the learned lam's gradient are exact; a donor anion's polarizability drops, an acceptor cation's rises (capped);
"joint" is close to "outer" and its autograd polarizability matches a finite field."""
from __future__ import annotations

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

from test_cmm_ct_site import SITE

f64 = torch.float64
SYM = ["Cl", "Na", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.7, 0.1, 0], [-1.2, 2.4, 0.3], [1.6, 2.9, -0.4]], dtype=f64)


def _model(**kw):
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    cfg = dict(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank", ct_model="gap",
               ct_gap_converge=40, ct_gap_overlap="site", ct_gap_donor=True, ct_cation_donor=False,
               hyperpol="saturating", ct_alpha_radical="fixed", ct_alpha_solve="outer")
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")), CmmConfig(**{**cfg, **kw}))


def _run(m, X=X0, **kw):
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    return m(m.precompute(X, Z, torch.zeros(len(SYM), dtype=torch.long)), **kw)


def test_zero_step_is_the_ion_model():
    m = _model()
    with torch.no_grad():
        m.ct_alpha_step.zero_()
    a, b = _run(m), _run(_model(ct_alpha_radical="off"))
    for k in ("e_int", "e_pol", "e_ct"):
        assert torch.allclose(a[k], b[k], rtol=0, atol=1e-13)


def test_steps_and_partition():
    m = _model()
    o, off = _run(m, alpha=True), _run(_model(ct_alpha_radical="off"), alpha=True)
    s = list(m.tab.symbols)
    st = m.ct_alpha_steps()
    assert float(st[s.index("Cl"), 0]) < 0 and float(st[s.index("Na"), 1]) > 0
    assert float(st[s.index("Na"), 1]) <= 0.9 * float(m.hyper_room[s.index("Na")]) + 1e-12
    da = o["ct_dalpha"]
    assert float(da[0]) < 0 and float(da[1]) > 0                       # Cl gives, Na takes
    assert torch.allclose(o["e_pol"], off["e_pol"], rtol=0, atol=1e-14)    # E_pol: the ion model's
    assert torch.allclose(o["alpha_pol"], off["alpha_pol"], rtol=0, atol=1e-12)
    assert float((o["e_int"] - off["e_int"]).abs().max()) > 1e-6
    assert o["ct_alpha_passes"] < m.config.ct_alpha_outer


def test_forces():
    m = _model()
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    fr = torch.zeros(4, dtype=torch.long)
    X = X0.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(m.precompute(X, Z, fr))["e_int"].sum(), X)[0]
    h = 1e-4
    for a_, c_ in ((0, 0), (1, 1), (2, 2), (3, 0)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a_, c_] += h
        Xm[a_, c_] -= h
        num = -(m(m.precompute(Xp, Z, fr))["e_int"].sum() - m(m.precompute(Xm, Z, fr))["e_int"].sum()) / (2 * h)
        assert float(F[a_, c_]) == pytest.approx(float(num), abs=2e-7)


def test_learned_lambda_gradient():
    m = _model(ct_alpha_radical="learned")
    g = torch.autograd.grad(_run(m)["e_int"].sum(), m.ct_alpha_raw)[0]
    s = list(m.tab.symbols)
    for k in ((s.index("Cl"), 0), (s.index("Na"), 1)):
        h = 1e-4
        with torch.no_grad():
            m.ct_alpha_raw[k] += h
        ep = float(_run(m)["e_int"].sum())
        with torch.no_grad():
            m.ct_alpha_raw[k] -= 2 * h
        em = float(_run(m)["e_int"].sum())
        with torch.no_grad():
            m.ct_alpha_raw[k] += h
        assert float(g[k]) == pytest.approx((ep - em) / (2 * h), rel=1e-4, abs=1e-10)


# -- ct_alpha_solve = "joint" ------------------------------------------------------------------------------------


def test_joint_zero_step_and_partition():
    m = _model(ct_alpha_solve="joint")
    off = _run(_model(ct_alpha_radical="off"), alpha=True)
    o = _run(m, alpha=True)
    assert torch.allclose(o["e_pol"], off["e_pol"], rtol=0, atol=1e-14)
    assert torch.allclose(o["alpha_pol"], off["alpha_pol"], rtol=0, atol=1e-12)
    assert float((o["e_int"] - off["e_int"]).abs().max()) > 1e-6
    with torch.no_grad():
        m.ct_alpha_step.zero_()
    z = _run(m)
    assert torch.allclose(z["e_int"], off["e_int"], rtol=0, atol=1e-13)


def test_joint_close_to_outer():
    a, b = _run(_model(ct_alpha_solve="joint")), _run(_model())
    off = _run(_model(ct_alpha_radical="off"))
    d_joint, d_outer = a["e_int"] - off["e_int"], b["e_int"] - off["e_int"]
    assert float((d_joint - d_outer).abs().max()) < 0.05 * float(d_outer.abs().max())


def test_joint_forces():
    m = _model(ct_alpha_solve="joint")
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    fr = torch.zeros(4, dtype=torch.long)
    X = X0.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(m.precompute(X, Z, fr))["e_int"].sum(), X)[0]
    h = 1e-4
    for a_, c_ in ((0, 0), (1, 1), (2, 2), (3, 0)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a_, c_] += h
        Xm[a_, c_] -= h
        num = -(m(m.precompute(Xp, Z, fr))["e_int"].sum() - m(m.precompute(Xm, Z, fr))["e_int"].sum()) / (2 * h)
        assert float(F[a_, c_]) == pytest.approx(float(num), abs=2e-7)


def test_joint_learned_lambda_gradient():
    m = _model(ct_alpha_radical="learned", ct_alpha_solve="joint")
    g = torch.autograd.grad(_run(m)["e_int"].sum(), m.ct_alpha_raw)[0]
    s = list(m.tab.symbols)
    for k in ((s.index("Cl"), 0), (s.index("Na"), 1)):
        h = 1e-4
        with torch.no_grad():
            m.ct_alpha_raw[k] += h
        ep = float(_run(m)["e_int"].sum())
        with torch.no_grad():
            m.ct_alpha_raw[k] -= 2 * h
        em = float(_run(m)["e_int"].sum())
        with torch.no_grad():
            m.ct_alpha_raw[k] += h
        assert float(g[k]) == pytest.approx((ep - em) / (2 * h), rel=1e-4, abs=1e-10)


def test_joint_polarizability_finite_field():
    m = _model(ct_alpha_solve="joint")
    a = _run(m, alpha=True)["alpha"][0].detach()
    off = _run(_model(ct_alpha_radical="off"), alpha=True)["alpha"][0]
    h = 2e-4
    e0 = float(_run(m)["e_int"].sum())
    for x in range(3):
        f = torch.zeros(1, 3, dtype=f64)
        f[0, x] = h
        num = -(float(_run(m, field=f)["e_int"].sum()) + float(_run(m, field=-f)["e_int"].sum()) - 2 * e0) / h ** 2
        assert float(a[x, x]) == pytest.approx(num, abs=2e-3)
    assert float((a - off).abs().max()) > 0.1
    assert torch.allclose(a, a.T, atol=1e-6)
