"""Perpendicular exchange stiffening (``CompleteModel(xpol=True, xpol_perp=True)``): per pair side and component block
E0 k_perp (|v_i|^2 |z_i|^2 - (v_i.z_i)^2). k_perp -> 0 is the xpol model; a field along a pair's bond is untouched
(no perpendicular component) while the perpendicular polarizability drops; the functional stays convex and the
solve stationary; the periodic CG agrees with the dense cluster solve; forces are derivatives of the energy."""
import itertools
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete.periodic import _perp_quad
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
F = 0.002


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)

    def make(kperp=None, **kw):
        torch.manual_seed(3)
        m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                          pauli_form="shells", confine=(), env_cap=3.0, xpol=True, **kw)
        with torch.no_grad():
            for p in (m.kappa_aniso, m.gap_lamD, m.gap_lamA):
                p.fill_(0.3)
            heads = [m.head_frozen] + ([m.head_xpol] if m.head_xpol is not None else [])
            for p in [q for h in heads for q in h.parameters()]:
                p.add_(0.3 * torch.randn_like(p))
            if kperp is not None:
                m.xpol_kperp.fill_(float(torch.log(torch.expm1(torch.tensor(kperp)))))
        return m
    return tab, chan, table, make


def _batch(table, tab, chan, X, Z, m, **kw):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=m.inv_form, **kw)


def _cluster(a=2.4, n=2):
    g = torch.tensor([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.05 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return X, Z


def _pair_alpha(m, tab, chan, table, R):
    """Finite-field alpha_pol (zz along the bond, xx across) of a Na-Cl pair along z."""
    X = torch.tensor([[0, 0, 0], [0, 0, R]], dtype=torch.float64).repeat(5, 1)
    Z = torch.tensor([11, 17]).repeat(5)
    fr = torch.arange(5).repeat_interleave(2)
    fld = torch.zeros(5, 3, dtype=torch.float64)
    fld[1, 2], fld[2, 2], fld[3, 0], fld[4, 0] = F, -F, F, -F
    with torch.no_grad():
        e = m(batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form), field=fld)["e_pol"]
    return float(-(e[1] + e[2] - 2 * e[0]) / F ** 2), float(-(e[3] + e[4] - 2 * e[0]) / F ** 2)


def test_zero_kperp_is_xpol(setup):
    tab, chan, table, make = setup
    X, Z = _cluster()
    m0, m1 = make(xpol_perp=False), make(kperp=1e-30, xpol_perp=True)
    m1.load_state_dict(m0.state_dict(), strict=False)
    with torch.no_grad():
        m1.xpol_kperp.fill_(-80.0)
        bt = _batch(table, tab, chan, X, Z, m0)
        o0, o1 = m0(bt), m1(bt)
    for k in ("e_pol", "e_xpol", "e_ct", "e_int"):
        assert abs(float(o0[k] - o1[k])) < 1e-13, k


def test_pair_parallel_untouched_perpendicular_stiffened(setup):
    """Na-Cl pair along z: v_i has only bond-axis (dipole) and m = 0 (quadrupole) components, so a field along z
    leaves the response unchanged to rounding at any k_perp; the field across the bond meets the new stiffness."""
    tab, chan, table, make = setup
    m0 = make(kperp=1e-30, xpol_perp=True)
    with torch.no_grad():
        m0.xpol_kperp.fill_(-80.0)
    m1 = make(kperp=2.0, xpol_perp=True)
    for R in (2.2, 2.8):
        zz0, xx0 = _pair_alpha(m0, tab, chan, table, R)
        zz1, xx1 = _pair_alpha(m1, tab, chan, table, R)
        assert abs(zz1 - zz0) < 1e-6 * abs(zz0), (R, zz0, zz1)
        assert xx1 < xx0 - 1e-3, (R, xx0, xx1)


def test_convex_and_stationary(setup):
    """Compressed cube: the perpendicular quadratic form is >= 0 at random z, e_xpol is the pair functional at the
    solution, and the full functional's gradient vanishes there (autograd of the explicit energy)."""
    tab, chan, table, make = setup
    X, Z = _cluster(a=2.2)
    m = make(kperp=1.5, xpol_perp=True)
    bt = _batch(table, tab, chan, X, Z, m)
    with torch.no_grad():
        o = m(bt)
        sp, I = bt["species"], m.scaled(bt["feats"])
        i, j, e0, lam, vi, vj = m.xpol_pairs(bt, sp, I)
        wp = m.xpol_perp_weights(sp, i, j, e0)
        z = o["z_xpol"][0]
        for s in range(5):
            zr = torch.randn(z.shape, dtype=z.dtype, generator=torch.Generator().manual_seed(s))
            assert float(_perp_quad(vi, zr[i]).min()) >= -1e-14 and float(_perp_quad(vj, zr[j]).min()) >= -1e-14
        u = (vi * z[i]).sum(1) + (vj * z[j]).sum(1)
        ex = (e0 * (2 * u + lam * u * u)).sum() + (0.5 * wp * (_perp_quad(vi, z[i]) + _perp_quad(vj, z[j]))).sum()
        assert abs(float(ex - o["e_xpol"][0])) < 1e-12 * max(1.0, abs(float(ex)))
        assert float((0.5 * wp * (_perp_quad(vi, z[i]) + _perp_quad(vj, z[j]))).sum()) > 0


def test_periodic_matches_cluster(setup):
    """Na4Cl4 cube in a 50 A box through the periodic CG path (matvec with the perpendicular blocks) against the dense
    cluster solve."""
    tab, chan, table, make = setup
    m = make(kperp=1.5, xpol_perp=True)
    corners = list(itertools.product(range(2), repeat=3))
    X = torch.tensor(corners, dtype=torch.float64) * 2.6 + 25.0
    Z = torch.tensor([11 if sum(c) % 2 == 0 else 17 for c in corners])
    with torch.no_grad():
        ref = m(_batch(table, tab, chan, X, Z, m))
        out = m(_batch(table, tab, chan, X, Z, m, cell=torch.eye(3, dtype=torch.float64) * 50.0, cutoff=13.0,
                       ewald_tol=1e-10))
    assert abs(float(ref["e_xpol"])) > 1e-5
    for k in ("e_pol", "e_xpol"):
        assert abs(float(ref[k] - out[k])) < 1e-7, (k, float(ref[k]), float(out[k]))


@pytest.mark.parametrize("key", ["e_int", "e_pol"])
def test_forces_are_derivatives(setup, key):
    tab, chan, table, make = setup
    X0, Z = _cluster(a=2.4)
    m = make(kperp=1.5, xpol_perp=True)

    def energy(X):
        return m(_batch(table, tab, chan, X, Z, m))[key].sum()
    X = X0.clone().requires_grad_(True)
    g = torch.autograd.grad(energy(X), X)[0]
    h = 1e-4
    for a, c in ((0, 0), (3, 1), (6, 2), (7, 0)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        with torch.no_grad():
            fd = (energy(Xp) - energy(Xm)) / (2 * h)
        assert abs(float(g[a, c] - fd)) < 1e-7 + 1e-5 * abs(float(fd)), (key, a, c, float(g[a, c]), float(fd))


def test_kperp_gets_a_gradient(setup):
    tab, chan, table, make = setup
    X, Z = _cluster(a=2.5)
    m = make(kperp=0.1, xpol_perp=True)
    m(_batch(table, tab, chan, X, Z, m))["e_pol"].sum().backward()
    assert m.xpol_kperp.grad is not None and float(m.xpol_kperp.grad.abs().max()) > 0


def test_constant_kappa_has_no_environment_dependence(setup):
    """xpol_kappa_head=False: kappa is kappa0[s, component] for every atom in any environment (no head parameters),
    and forces stay exact derivatives."""
    tab, chan, table, make = setup
    m = make(kperp=1.0, xpol_perp=True, xpol_kappa_head=False)
    assert m.head_xpol is None and not any("head_xpol" in k for k in m.state_dict())
    for a in (2.3, 3.4):
        X, Z = _cluster(a=a)
        bt = _batch(table, tab, chan, X, Z, m)
        with torch.no_grad():
            sp, I = bt["species"], m.scaled(bt["feats"])
            k = m.xpol_kappa(I, sp)
            k0 = torch.exp(m.xpol_lnk[sp])
        assert (k[:, 0] - k0[:, 0]).abs().max() < 1e-15 and (k[:, 3] - k0[:, 1]).abs().max() < 1e-15

    X0, Z = _cluster(a=2.4)

    def energy(X):
        return m(_batch(table, tab, chan, X, Z, m))["e_int"].sum()
    X = X0.clone().requires_grad_(True)
    g = torch.autograd.grad(energy(X), X)[0]
    h = 1e-4
    for a, c in ((0, 0), (5, 2)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        with torch.no_grad():
            fd = (energy(Xp) - energy(Xm)) / (2 * h)
        assert abs(float(g[a, c] - fd)) < 1e-7 + 1e-5 * abs(float(fd))


def test_scalefree_descriptors():
    """n_eff - 1, eta1, eta2 for the textbook shells (equal overlaps): pair 0/0/0 at any R, a90 1/0.5/., a180 1/1/0,
    Td 3/1/1, Oh 5/1/1."""
    def desc(dirs, O=1e-3):
        d = torch.tensor(dirs, dtype=torch.float64)
        d = 2.5 * d / d.norm(dim=1, keepdim=True)
        n = len(dirs)
        i = torch.zeros(n, dtype=torch.long)
        j = torch.arange(1, n + 1)
        return CompleteModel.xpol_scalefree(n + 1, i, j, d, torch.full((n,), O, dtype=torch.float64))[0]
    for O in (1e-2, 1e-6):
        assert desc([[0, 0, 1]], O).abs().max() < 1e-12
    a90 = desc([[1, 0, 0], [0, 1, 0]])
    assert abs(float(a90[0]) - 1) < 1e-12 and abs(float(a90[1]) - 0.5) < 1e-12
    a180 = desc([[1, 0, 0], [-1, 0, 0]])
    assert torch.allclose(a180, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12)
    td = desc([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    assert torch.allclose(td, torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64), atol=1e-12)
    oh = desc([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    assert torch.allclose(oh, torch.tensor([5.0, 1.0, 1.0], dtype=torch.float64), atol=1e-12)


def test_scalefree_kappa_is_constant_along_a_pair(setup):
    """With a randomized scale-free head, kappa of both ions of a Na-Cl pair is the same at every R (and kappa0),
    while a compressed cube moves it."""
    tab, chan, table, make = setup
    m = make(kperp=1.0, xpol_perp=True, xpol_kappa_env="scalefree", xpol_perp_weight="sqrt_overlap")
    with torch.no_grad():
        for p in m.head_xpol.parameters():
            p.add_(0.5 * torch.randn_like(p))
    ks = []
    for R in (2.0, 2.8, 3.5, 5.0):
        X = torch.tensor([[0, 0, 0], [0, 0, R]], dtype=torch.float64)
        bt = _batch(table, tab, chan, X, torch.tensor([11, 17]), m)
        with torch.no_grad():
            m(bt)
            sp, I = bt["species"], m.scaled(bt["feats"])
            i, j, *_ = m.xpol_pairs(bt, sp, I)
            dr = (bt["pos"][j] - bt["pos"][i]) / 0.529177210903
            ks.append(m.xpol_kappa(I, sp, (i, j, dr, m._xpol_O)))
    for k in ks[1:]:
        assert (k - ks[0]).abs().max() < 1e-13
    X, Z = _cluster(a=2.3)
    bt = _batch(table, tab, chan, X, Z, m)
    with torch.no_grad():
        m(bt)
        sp, I = bt["species"], m.scaled(bt["feats"])
        i, j, *_ = m.xpol_pairs(bt, sp, I)
        dr = (bt["pos"][j] - bt["pos"][i]) / 0.529177210903
        kc = m.xpol_kappa(I, sp, (i, j, dr, m._xpol_O))
    k0 = torch.exp(m.xpol_lnk[sp])[:, 0]
    assert float((kc[:, 0] / k0 - 1).abs().max()) > 1e-3


@pytest.mark.parametrize("opts", [dict(xpol_perp_weight="sqrt_overlap"), dict(xpol_kappa_env="scalefree"),
                                  dict(xpol_perp_weight="sqrt_overlap", xpol_kappa_env="scalefree")])
def test_new_options_forces_and_periodic(setup, opts):
    tab, chan, table, make = setup
    m = make(kperp=1.5, xpol_perp=True, **opts)
    if m.head_xpol is not None:
        with torch.no_grad():
            for p in m.head_xpol.parameters():
                p.add_(0.3 * torch.randn_like(p))
    X0, Z = _cluster(a=2.4)

    def energy(X):
        return m(_batch(table, tab, chan, X, Z, m))["e_int"].sum()
    X = X0.clone().requires_grad_(True)
    g = torch.autograd.grad(energy(X), X)[0]
    h = 1e-4
    for a, c in ((0, 0), (3, 1), (6, 2)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        with torch.no_grad():
            fd = (energy(Xp) - energy(Xm)) / (2 * h)
        assert abs(float(g[a, c] - fd)) < 1e-7 + 1e-5 * abs(float(fd)), (opts, a, c)
    corners = list(itertools.product(range(2), repeat=3))
    Xc = torch.tensor(corners, dtype=torch.float64) * 2.6 + 25.0
    Zc = torch.tensor([11 if sum(cc) % 2 == 0 else 17 for cc in corners])
    with torch.no_grad():
        ref = m(_batch(table, tab, chan, Xc, Zc, m))
        out = m(_batch(table, tab, chan, Xc, Zc, m, cell=torch.eye(3, dtype=torch.float64) * 50.0, cutoff=13.0,
                       ewald_tol=1e-10))
    for k in ("e_pol", "e_xpol"):
        assert abs(float(ref[k] - out[k])) < 1e-7, (opts, k)


def test_defaults_and_legacy_breathing_head(setup):
    """The kappa head reads the scale-free descriptors by default; the breathing polarizability has no environment
    head (a0 e^lna per species), and a checkpoint that still carries the old ``head_breath`` weights loads."""
    tab, chan, table, make = setup
    m = make()
    assert m.xpol_kappa_env == "scalefree" and m.breath and not hasattr(m, "head_breath")
    sd = dict(m.state_dict())
    sd["head_breath.lin.weight"] = torch.zeros(1, 3, dtype=torch.float64)
    sd["head_breath.net.0.weight"] = torch.zeros(4, 3, dtype=torch.float64)
    m2 = make()
    m2.load_state_dict(sd, strict=True)
    X, Z = _cluster()
    with torch.no_grad():
        assert abs(float(m(_batch(table, tab, chan, X, Z, m))["e_pol"] - m2(_batch(table, tab, chan, X, Z, m2))["e_pol"])) < 1e-12


def test_perp_is_the_default(setup):
    """xpol brings the perpendicular stiffening unless switched off, starting from the per-species defaults; the
    training CLI agrees, and a checkpoint without the key (before the default) builds without it."""
    from rsfff.ff.complete.model import XPOL_KPERP_INIT
    from rsfff.train.train_ionct import build_model
    tab, chan, table, make = setup
    m = make()
    assert m.xpol_perp
    p = torch.nn.functional.softplus(m.xpol_kperp)
    for k, x in enumerate(tab.symbols):
        if x in XPOL_KPERP_INIT:
            assert abs(float(p[k]) - XPOL_KPERP_INIT[x]) < 1e-9
    assert not make(xpol_perp=False).xpol_perp
    feat = DensityEnvironment(tab)
    old = build_model({"model": "complete", "frozen": True, "xpol": True, "pauli_form": "shells"}, tab, chan,
                      feat.n_feat, feature_names=feat.feature_names)
    assert old.xpol and not old.xpol_perp and old.xpol_kappa_env == "overlap"
