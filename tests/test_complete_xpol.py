"""Quadratic exchange polarization (``CompleteModel(xpol=True)``): the overlap table against direct shaped
overlaps, kappa = 0 as the model without exchange drives, positivity (H PSD, every pair's E_P >= 0 at lam >= 1),
stationarity of the solve, and forces as derivatives of the reported energies."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete.model import induction_matrix_blocks
from rsfff.ff.complete.tabulate import XpolTable, xpol_overlap_map
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
IU = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)

    def make(**kw):
        torch.manual_seed(3)
        m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                          pauli_form="shells", confine=(), env_cap=3.0, **kw)
        with torch.no_grad():
            for p in (m.kappa_aniso, m.gap_lamD, m.gap_lamA):
                p.fill_(0.3)
            for p in m.head_frozen.parameters():
                p.add_(0.3 * torch.randn_like(p))
            if m.xpol:
                for p in m.head_xpol.parameters():
                    p.add_(0.3 * torch.randn_like(p))
        return m
    return tab, chan, table, make


def _cluster(a=2.4, n=2):
    """A compressed rock-salt n^3 cube (a = 2.4 A against 2.82 at equilibrium), slightly distorted."""
    g = torch.tensor([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.05 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return X, Z


def _batch(table, tab, chan, X, Z, inv_form):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=inv_form)


def test_table_is_the_shaped_overlap(setup):
    """O_ij = <rho_i | rho_j>, e_ij = <rho_i | breath shape of j> and the induced-moment map d_ij (minus the moments'
    derivative operators on <rho_i | response shape of j>) against kernels.shaped_overlap."""
    tab = setup[0]
    xt = XpolTable(tab, h=0.01)
    Xi, Xj = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64), torch.tensor([1.5, -1.2, 2.1], dtype=torch.float64)
    x = torch.randn(8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    poly = K._poly_map(torch.float64, "cpu") @ x
    one = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    for si, sj in ((1, 0), (0, 1), (1, 1)):
        O, d, e = xpol_overlap_map(xt, torch.tensor([si, sj]), torch.tensor([[0], [1]]), ((Xj - Xi) / K.BOHR_ANG)[None])
        def ovl(xi, xj, b, w):             # <rho_i | shape of j>
            r = ((xj - xi).norm() / K.BOHR_ANG)[None]
            return sum(tab.shell_n[si, t] * K.shaped_overlap(tab.shell_b[si, t].reshape(1), one, b[None], w[None], r)[0]
                       for t in range(tab.shell_n.shape[1]) if tab.shell_n[si, t] > 0)
        ref_O = sum(tab.shell_n[sj, t] * ovl(Xi, Xj, tab.shell_b[sj, t], one[0])
                    for t in range(tab.shell_n.shape[1]) if tab.shell_n[sj, t] > 0)
        assert abs(float(O[0] - ref_O)) < 1e-6 * float(ref_O), (si, sj)
        if tab.a0[sj] > 0:
            ref_e = ovl(Xi, Xj, tab.b_breath[sj], tab.w_breath[sj])
            assert abs(float(e[0] - ref_e)) < 1e-6 * abs(float(ref_e)) + 1e-14, (si, sj)
        xd = Xj.clone().requires_grad_(True)
        gd = torch.autograd.grad(ovl(Xi, xd, tab.b_dip[sj], tab.w_dip[sj]), xd)[0] * K.BOHR_ANG
        xq = Xj.clone().requires_grad_(True)
        gq = torch.autograd.grad(ovl(Xi, xq, tab.b_quad[sj], tab.w_quad[sj]), xq, create_graph=True)[0]
        Hq = torch.stack([torch.autograd.grad(gq[a], xq, retain_graph=True)[0] for a in range(3)]) * K.BOHR_ANG ** 2
        ref = -((gd * x[:3]).sum() + sum(poly[4 + c] * Hq[a, b] for c, (a, b) in enumerate(IU)))
        assert abs(float(d[0] @ x - ref)) < 1e-6 * abs(float(ref)), (si, sj, float(d[0] @ x), float(ref))


def test_table_derivatives_are_consistent(setup):
    """The tabulated first derivatives of the scalar rows are the derivatives of the tabulated values."""
    xt = XpolTable(setup[0], h=0.01)
    r = torch.linspace(1.5, 8.0, 40, dtype=torch.float64)
    for si, sj in ((0, 1), (1, 1)):
        s_i, s_j = torch.full_like(r, si, dtype=torch.long), torch.full_like(r, sj, dtype=torch.long)
        for name in ("xp:ovl", "xp:breath", "xp:dip:1"):
            k = xt.index[name]
            h = 1e-4
            fd = (xt(s_i, s_j, r + h)[:, k] - xt(s_i, s_j, r - h)[:, k]) / (2 * h)
            rr = r.clone().requires_grad_(True)
            g = torch.autograd.grad(xt(s_i, s_j, rr)[:, k].sum(), rr)[0]
            assert (g - fd).abs().max() < 1e-6 * fd.abs().max() + 1e-14, name


def test_zero_kappa_is_the_model_without_exchange_drives(setup):
    """kappa = 0: no exchange term, the breathing amplitude is solved with its Coulomb drive alone, so every output
    is the frozen model's without the pseudo drive (xpol=False, breathing Pauli drive lam0 = 0) to rounding."""
    tab, chan, table, make = setup
    X, Z = _cluster()
    m0, m1 = make(), make(xpol=True)
    m1.load_state_dict(m0.state_dict(), strict=False)
    with torch.no_grad():
        m1.xpol_lnk.fill_(-200.0)
    bt = _batch(table, tab, chan, X, Z, m0.inv_form)
    o0, o1 = m0(bt), m1(bt)
    for k in ("e_pol", "e_ct", "e_pauli", "e_disp", "e_int", "x"):
        assert (o0[k] - o1[k]).abs().max() < 1e-12, k
    assert float(o1["e_xpol"].detach().abs().max()) < 1e-15
    assert (o0["e_breath_atom"] - o1["e_breath_atom"]).abs().max() < 1e-12


def test_positivity_and_stationarity(setup):
    """On a compressed cube: every pair's E_ij^0 (1 + 2u + lam u^2) >= 0 at random z, A + H is positive definite,
    the reported e_xpol is E_P(z*) - E_P(0) from the pairs, the solution is stationary and below z = 0."""
    tab, chan, table, make = setup
    X, Z = _cluster(a=2.2)
    m = make(xpol=True, xpol_perp=False)     # the k_par kernel alone (k_perp: test_complete_xperp)
    bt = _batch(table, tab, chan, X, Z, m.inv_form)
    with torch.no_grad():
        o = m(bt)
        sp, I = bt["species"], m.scaled(bt["feats"])
        i, j, e0, lam, vi, vj = m.xpol_pairs(bt, sp, I)
        assert float(lam.min()) >= 1.0
        z = o["z_xpol"][0]
        n, D = z.shape
        for s in range(5):
            zr = 3.0 * torch.randn(n, D, dtype=torch.float64, generator=torch.Generator().manual_seed(s))
            u = (vi * zr[i]).sum(1) + (vj * zr[j]).sum(1)
            assert float((e0 * (1 + 2 * u + lam * u * u)).min()) >= -1e-14
        u = (vi * z[i]).sum(1) + (vj * z[j]).sum(1)
        ex = (e0 * (2 * u + lam * u * u)).sum()
        assert abs(float(ex - o["e_xpol"][0])) < 1e-12 * max(1.0, abs(float(ex)))
        assert abs(float(o["e_xpol"][0])) > 1e-5       # not trivially zero on a compressed cube (either sign:
        # the exchange part is the cost of the overlap the field pushes the electrons into, minus any relief)

    def F(zf):
        zz = zf.reshape(n, D)
        L = bt["L"][0]
        alpha = m.tab.alpha[sp] * m.confinement(I, None, sp)[:, 0]
        A = induction_matrix_blocks(bt["A_off"], m.alpha_inverse(alpha, sp, bt.get("env"))[None],
                                    (m.tab.C[sp] * m.confinement(I, None, sp)[:, 1])[None])[0]
        x = zz[:, :8].reshape(-1)
        e = 0.5 * x @ A @ x + L.reshape(-1) @ x
        if D == 9:
            a_b = m.tab.a0[sp] * torch.exp(m.breath_lna[sp])
            has = a_b > 0
            b = zz[:, 8]
            V0 = bt["feats"][:, m.breath_col]
            e = e + torch.where(has, 0.5 * b * b / a_b.clamp(min=1e-30) - V0 * b, 0.5 * b * b).sum()
        uu = (vi * zz[i]).sum(1) + (vj * zz[j]).sum(1)
        return e + (e0 * (2 * uu + lam * uu * uu)).sum()
    zs = z.reshape(-1).clone().requires_grad_(True)
    g = torch.autograd.grad(F(zs), zs)[0]
    assert float(g.abs().max()) < 1e-9
    Hm = torch.autograd.functional.hessian(F, z.reshape(-1))
    assert float(torch.linalg.eigvalsh(Hm).min()) > 0
    with torch.no_grad():
        assert float(F(z.reshape(-1))) <= float(F(torch.zeros_like(z).reshape(-1)))


@pytest.mark.parametrize("key", ["e_int", "e_pol", "e_ct"])
def test_forces_are_derivatives_of_the_energy(setup, key):
    """Autograd forces through the one linear solve against central differences of the reported energies."""
    tab, chan, table, make = setup
    X0, Z = _cluster(a=2.4)
    m = make(xpol=True)

    def energy(X):
        return m(_batch(table, tab, chan, X, Z, m.inv_form))[key].sum()
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


def test_double_backward(setup):
    """Force training: the parameter gradient of a force loss exists and is finite (double backward through the
    solve and the overlap table)."""
    tab, chan, table, make = setup
    X0, Z = _cluster(a=2.5)
    m = make(xpol=True)
    X = X0.clone().requires_grad_(True)
    e = m(_batch(table, tab, chan, X, Z, m.inv_form))["e_int"].sum()
    f = torch.autograd.grad(e, X, create_graph=True)[0]
    (f * f).sum().backward()
    for name in ("xpol_lnk", "xpol_lam"):
        gp = getattr(m, name).grad
        assert gp is not None and bool(torch.isfinite(gp).all()) and float(gp.abs().max()) > 0, name


def _rocksalt(n, a=5.64, jitter=0.0, seed=0):
    import itertools
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=torch.float64)
    X, Z = [], []
    for c in itertools.product(range(n), repeat=3):
        for b in base:
            o = (torch.tensor(c, dtype=torch.float64) + b) * a
            X += [o, o + torch.tensor([a / 2, 0, 0], dtype=torch.float64)]
            Z += [11, 17]
    X = torch.stack(X) + jitter * torch.randn(len(Z), 3, generator=g, dtype=torch.float64)
    return X, torch.tensor(Z), torch.eye(3, dtype=torch.float64) * a * n


def test_periodic_huge_box_is_the_cluster(setup):
    """A compressed Na4Cl4 cube in a 50 A box (CG with the Ewald frames, the exchange pairs from the periodic pair
    list) is the cluster's dense solve, to the octupole-image residue (as in test_complete_periodic)."""
    import itertools
    tab, chan, table, make = setup
    m = make(xpol=True)
    corners = list(itertools.product(range(2), repeat=3))
    X = torch.tensor(corners, dtype=torch.float64) * 2.6 + 25.0
    Z = torch.tensor([11 if sum(c) % 2 == 0 else 17 for c in corners])
    fr = torch.zeros(8, dtype=torch.long)
    with torch.no_grad():
        ref = m(batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form))
        out = m(batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form,
                                   cell=torch.eye(3, dtype=torch.float64) * 50.0, cutoff=13.0, ewald_tol=1e-10))
    assert abs(float(ref["e_xpol"])) > 1e-4
    for k in ("e_pauli", "e_pol", "e_xpol", "e_ct", "e_int"):
        assert abs(float(ref[k] - out[k])) < 1e-7, (k, float(ref[k]), float(out[k]))


@pytest.mark.parametrize("env,h", [("overlap", 1e-4), ("scalefree", 2e-5)])
def test_periodic_forces_and_stress(setup, env, h):
    """A small triclinic cell (several images per pair, self images): forces and the strain derivative of the
    periodic energy against central differences. The randomised kappa head on the (unscaled) scale-free descriptors
    curves the energy more, so its central differences take a smaller step (the error falls as h^2)."""
    tab, chan, table, make = setup
    m = make(xpol=True, xpol_kappa_env=env)
    X, Z, cell = _rocksalt(1, a=5.5, jitter=0.2, seed=3)
    cell = cell + torch.tensor([[0, 0, 0], [0.4, 0, 0], [0.2, -0.3, 0]], dtype=torch.float64)
    fr = torch.zeros(len(Z), dtype=torch.long)

    def E(pos, eps):
        F = torch.eye(3, dtype=torch.float64) + eps
        return m(batched_precompute(pos @ F, Z, fr, tab, table, chan, inv_form=m.inv_form, cell=cell @ F,
                                    cutoff=9.5))["e_int"].sum()
    pos = X.clone().requires_grad_(True)
    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    g, ge = torch.autograd.grad(E(pos, eps), (pos, eps))
    z3 = torch.zeros(3, 3, dtype=torch.float64)
    with torch.no_grad():
        for a, b in ((0, 0), (3, 2), (6, 1)):
            pp, pm = X.clone(), X.clone()
            pp[a, b] += h
            pm[a, b] -= h
            fd = (E(pp, z3) - E(pm, z3)) / (2 * h)
            assert abs(float(fd - g[a, b])) < 1e-7, (a, b, float(fd), float(g[a, b]))
        for a, b in ((0, 0), (1, 2)):
            ep, em = z3.clone(), z3.clone()
            ep[a, b], em[a, b] = h, -h
            fd = (E(X, ep) - E(X, em)) / (2 * h)
            assert abs(float(fd - ge[a, b])) < 1e-7 + 2e-7 * abs(float(fd)), (a, b, float(fd), float(ge[a, b]))


@pytest.mark.parametrize("argv,want", [
    (["--model", "complete", "--frozen"], "xpol"),
    (["--model", "complete", "--frozen", "--pseudo"], "pseudo"),
    (["--model", "complete"], "none"),
    (["--model", "ionct", "--pseudo"], "pseudo"),
    (["--model", "complete", "--frozen", "--exchange", "none"], "none"),
])
def test_exchange_default(argv, want):
    """train_ionct: xpol is the default exchange drive of the complete model with the frozen terms; --pseudo keeps
    the old drive; saved args carry consistent pseudo / xpol flags for build_model."""
    import argparse
    from rsfff.train.train_ionct import resolve_exchange
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ionct")
    for f in ("--frozen", "--pseudo", "--xpol"):
        ap.add_argument(f, action="store_true")
    ap.add_argument("--exchange", default="auto")
    a = ap.parse_args(argv)
    resolve_exchange(a)
    assert a.exchange == want and a.pseudo == (want == "pseudo") and a.xpol == (want == "xpol")
