"""Two-state channel CT (``rsfff.ff.complete.twostate``, ``CompleteModel(ct_model="twostate")``).

Reference systems: H2+ (one channel with a degenerate gap: exactly half an electron at zero field, the linear-response
and exact two-state field responses, no self-interaction), a random small system against a direct minimisation of
the full energy, and NaCl (the gap of the ion pair, linear response vs the converged solve on a cluster, forces as
energy derivatives, finite parameter gradients)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize, minimize_scalar

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete import twostate as TS
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
f64 = torch.float64


# ---- the per-channel functional ---------------------------------------------------------------------------------

def test_isolated_minimum_closed_forms():
    """min F = e, argmin = q0, F''(q0) = k for weak, intermediate and degenerate gaps."""
    for e, D in ((-0.004, 0.13), (-0.05, 0.08), (-0.2, 0.0), (-1e-6, 0.3)):
        e_, D_ = torch.tensor(e, dtype=f64), torch.tensor(D, dtype=f64)
        W2, q0, kq0, kinv = TS.isolated(e_, D_)
        W = float(W2.sqrt())
        r = minimize_scalar(lambda q: D * q - 2 * W * np.sqrt(q * (1 - q)), bounds=(1e-15, 0.5),
                            method="bounded", options={"xatol": 1e-14})
        assert r.fun == pytest.approx(e, rel=1e-9)
        assert r.x == pytest.approx(float(q0), rel=1e-5, abs=1e-12)
        k = float(TS.d2F(q0, D_, W2.sqrt()))
        assert 1.0 / k == pytest.approx(float(kinv), rel=1e-9)
        assert float(kq0) == pytest.approx(k * float(q0), rel=1e-9)


# ---- H2+ --------------------------------------------------------------------------------------------------------

def _two_site(J=0.6, Hij=0.5, phi=(0.0, 0.0)):
    """Two slots, one channel 0 -> 1, no induced moments (one dummy site variable with zero coupling)."""
    Ha = torch.tensor([[[J, Hij], [Hij, J]]], dtype=f64)
    Ca = torch.zeros(1, 1, 2, dtype=f64)
    ch = (torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    return Ca, Ha, Ca.clone(), torch.zeros(1, 1, dtype=f64), torch.tensor([phi], dtype=f64), ch


def test_h2plus_half_electron_without_self_interaction():
    """Degenerate gap (I = A, the proton's potential on the hole cancelled by the electron-hole attraction): q = 1/2
    and E = e = -W exactly, in linear response and converged, for any slot Coulomb matrix -- the channel's own charge
    does not interact with itself (with it, the classical q (1 - q) H_ij would break the symmetry)."""
    W = 0.2175                                    # H2+ at R = 2 bohr: half the 1sg / 1su splitting
    e, D = torch.tensor([-W], dtype=f64), torch.tensor([0.0], dtype=f64)
    for J, Hij in ((0.6, 0.5), (1.0, 0.1), (0.3, 0.29)):
        for n in (1, 30):
            E, e_ch, q, z, Q = TS.solve(*_two_site(J, Hij), e, D, n_outer=n)
            assert float(q[0]) == pytest.approx(0.5, abs=1e-12)
            assert float(E[0]) == pytest.approx(-W, rel=1e-12)
            assert torch.allclose(Q[0], torch.tensor([0.5, -0.5], dtype=f64))


def test_h2plus_field_linear_response_vs_iterative():
    """A potential difference dpsi across the bond: one outer step is the linear response (q = 1/2 - dpsi / 4W,
    E = -W + dpsi / 2 - dpsi^2 / 8W, alpha_par = R^2 / 4W); converged outer steps give the exact two-state result,
    q = (1 - dpsi / s) / 2, E = (dpsi - s) / 2, s = sqrt(dpsi^2 + 4 W^2), inside (0, 1) at any field."""
    W = 0.2175
    e, D = torch.tensor([-W], dtype=f64), torch.tensor([0.0], dtype=f64)
    for x in (0.1, 0.5, 1.0, 2.0, 4.0, -3.0):
        dpsi = 2 * W * x
        phi = (0.5 * dpsi, -0.5 * dpsi)
        # h = 0.02 << k = 4W: the proximal guard is ~1e-11, linear response is the plain expansion
        E1, _, q1, _, _ = TS.solve(*_two_site(0.3, 0.29, phi), e, D, n_outer=1)
        assert float(q1[0]) == pytest.approx(0.5 - dpsi / (4 * W), rel=1e-9, abs=1e-10)
        assert float(E1[0]) == pytest.approx(-W + 0.5 * dpsi - dpsi ** 2 / (8 * W), rel=1e-9)
        s = np.sqrt(dpsi ** 2 + 4 * W * W)
        # converged: the exact two-state result for any slot Coulomb matrix, also h = 1.8 > F'' (strong guard)
        for J, Hij in ((0.3, 0.29), (0.6, 0.5), (1.0, 0.1)):
            info = {}
            En, _, qn, _, _ = TS.solve(*_two_site(J, Hij, phi), e, D, n_outer=80, tol=1e-13, info=info)
            assert info["step"] < 1e-12
            assert float(qn[0]) == pytest.approx(0.5 * (1 - dpsi / s), rel=1e-9)
            assert float(En[0]) == pytest.approx(0.5 * (dpsi - s), rel=1e-10)
            assert 0.0 < float(qn[0]) < 1.0


# ---- a random coupled system against a direct minimisation ------------------------------------------------------

def test_newton_converges_to_the_direct_minimum():
    """Three atoms, four channels (two sharing a donor), induced moments coupled to the charges, a field: the
    converged outer steps give the minimum of the full energy (direct L-BFGS-B over z and q in (0, 1)); one step is
    the minimum of its linear-response quadratic."""
    g = torch.Generator().manual_seed(5)
    N, D = 3, 2
    ND = N * D
    Rm = torch.randn(ND, ND, generator=g, dtype=f64)
    A = Rm @ Rm.T / ND + 2.0 * torch.eye(ND, dtype=f64)
    L = 0.05 * torch.randn(ND, generator=g, dtype=f64)
    Ca = 0.05 * torch.randn(ND, N, generator=g, dtype=f64)
    Hm = torch.tensor([[0.6, 0.2, 0.15], [0.2, 0.4, 0.18], [0.15, 0.18, 0.5]], dtype=f64)
    phi = torch.tensor([0.01, -0.02, 0.015], dtype=f64)
    don, acc = torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 1])
    e = torch.tensor([-0.02, -0.01, -0.03, -0.005], dtype=f64)
    Dl = torch.tensor([0.10, 0.15, 0.06, 0.2], dtype=f64)
    G = torch.zeros(N, 4, dtype=f64)
    G[don, torch.arange(4)] += 1
    G[acc, torch.arange(4)] -= 1
    h = torch.diag(G.T @ Hm @ G)
    W2 = e * (e - Dl)

    def energy(v):
        v = torch.tensor(v, dtype=f64, requires_grad=True)
        z, q = v[:ND], v[ND:]
        Q = G @ q
        Fq = Dl * q - 2 * W2.sqrt() * torch.sqrt(q * (1 - q))
        E = 0.5 * z @ A @ z + L @ z + (Fq - 0.5 * h * q * q).sum() + Q @ (Ca.T @ z + phi) + 0.5 * Q @ Hm @ Q
        gv, = torch.autograd.grad(E, v)
        return float(E.detach()), gv.numpy()

    z_pol = -torch.linalg.solve(A, L)
    E_pol = float(0.5 * z_pol @ A @ z_pol + L @ z_pol)
    q0 = TS.isolated(e, Dl)[1]
    x0 = np.concatenate([z_pol.numpy(), q0.numpy()])
    bounds = [(None, None)] * ND + [(1e-12, 1 - 1e-12)] * 4
    r = minimize(energy, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 20000})
    W = torch.linalg.solve(A, Ca)
    ch = (torch.zeros(4, dtype=torch.long), don, acc)
    info = {}
    E, e_ch, q, z, Q = TS.solve(Ca[None], Hm[None], W[None], z_pol[None], phi[None], ch, e, Dl, n_outer=50,
                                tol=1e-13, info=info)
    assert float(E[0]) == pytest.approx(r.fun - E_pol, rel=1e-7)
    assert torch.allclose(q, torch.as_tensor(r.x[ND:]), atol=1e-6)
    assert float(e_ch.sum()) == pytest.approx(float(E[0]), rel=1e-12)
    assert info["outer"] < 50
    # linear response: the minimum of the quadratic about q0 (k from the closed form) -- solve it densely
    k = 1.0 / TS.isolated(e, Dl)[3]
    rho = h * torch.exp(-k / (2 * h))                  # the proximal guard (twostate.expansion)
    Hq = torch.diag(k - h + rho) + G.T @ Hm @ G
    Mfull = torch.zeros(ND + 4, ND + 4, dtype=f64)
    Mfull[:ND, :ND], Mfull[:ND, ND:], Mfull[ND:, :ND], Mfull[ND:, ND:] = A, Ca @ G, G.T @ Ca.T, Hq
    rhs = torch.cat([-L, (k + rho) * q0 - G.T @ phi])
    sol = torch.linalg.solve(Mfull, rhs)
    zz, qq = sol[:ND], sol[ND:]
    E_lr = float(0.5 * zz @ A @ zz + L @ zz + (e + 0.5 * (k + rho) * (qq - q0) ** 2 - 0.5 * h * qq * qq).sum()
                 + (G @ qq) @ (Ca.T @ zz + phi) + 0.5 * (G @ qq) @ Hm @ (G @ qq)) - E_pol
    E1, _, q1, z1, _ = TS.solve(Ca[None], Hm[None], W[None], z_pol[None], phi[None], ch, e, Dl, n_outer=1)
    assert float(E1[0]) == pytest.approx(E_lr, rel=1e-10)
    assert torch.allclose(q1, qq, atol=1e-12)
    assert torch.allclose(z1[0], zz, atol=1e-12)


def test_switched_off_channel_is_inert():
    """e_c = 0 (the kernel's taper end): no charge, no energy, finite everything, in both modes."""
    e, D = torch.tensor([0.0], dtype=f64), torch.tensor([0.05], dtype=f64)
    for n in (1, 5):
        E, e_ch, q, z, Q = TS.solve(*_two_site(phi=(0.01, -0.01)), e, TS.floor_gap(D), n_outer=n)
        assert float(q[0]) == 0.0 and float(E[0]) == 0.0 and torch.isfinite(Q).all()


# ---- NaCl in the complete model ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)
    cache = {}

    def make(**kw):
        torch.manual_seed(3)
        m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                          pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="twostate", **kw)
        with torch.no_grad():                 # a Na -> Cl direction much weaker than Cl -> Na, as trained
            m.ct_lnq0[:, :] -= 1.0
            m.ct_lnq0[0, 0] -= 2.0
            for p in m.head_ctq.parameters():
                p.add_(0.2 * torch.randn_like(p))
        if "table" in cache:
            m._flux_table = cache["table"]
        return m

    def keep(m):
        cache.setdefault("table", m._flux_table)
    return tab, chan, table, feat, make, keep


def _batch(table, tab, chan, X, Z, m):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=m.inv_form)


def _cluster(a=2.75):
    g = torch.tensor([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=f64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.08 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(1), dtype=f64)
    return X, Z


def _gaps(m, bt):
    sp = bt["species"]
    I = m.scaled(bt["feats"])
    *_, e_c, Dl, Draw = m.twostate_operator(bt, sp, m.ct_charges_from(I, sp), 9 if m.breath else 8)
    return bt["ch_index"], sp, e_c, Draw


def test_nacl_pair_gap(setup):
    """Na+ Cl-: Delta(Cl- -> Na+) = I(Cl-) - A(Na+) + 1/R at long range (the ion pair's +2/R potential difference
    minus the electron-hole attraction), Delta(Na+ -> Cl-) = I(Na+) - A(Cl-) - 3/R; at 2.8 A ~ 0.13 Eh (3.6 eV)."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    keep(m)
    I0, A0 = m.ts_I0, m.ts_A0
    assert float(A0[1]) == pytest.approx(float(tab.I_donor[1] - m.flux_jself[1]))    # Cl-: unbound, I - J_self
    for R in (2.8, 4.0, 6.0):
        X = torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=f64)
        bt = _batch(table, tab, chan, X, torch.tensor([11, 17]), m)
        (i, j), sp, e_c, Draw = _gaps(m, bt)
        rb = R / K.BOHR_ANG
        for c in range(i.shape[0]):
            if int(sp[i[c]]) == 1:              # Cl -> Na
                ref = float(I0[1] - A0[0]) + 1 / rb
            else:
                ref = float(I0[0] - A0[1]) - 3 / rb
            tol = 2e-3 if R >= 4.0 else 0.03
            assert float(Draw[c]) == pytest.approx(ref, abs=tol), (R, c)
            if int(sp[i[c]]) == 1 and R == 2.8:
                assert 0.10 < float(Draw[c]) < 0.16


def test_nacl_cluster_linear_response_vs_converged(setup):
    """The 8-ion cluster is weak coupling (Delta >> W): the converged solve is within a few percent of linear
    response in E_CT and the charges; populations in (0, 1/2); the outer loop converges."""
    tab, chan, table, feat, make, keep = setup
    m1, mn = make(), make(ct_outer=40, ct_outer_tol=1e-12)
    X, Z = _cluster()
    bt = _batch(table, tab, chan, X, Z, m1)
    with torch.no_grad():
        o1 = m1(bt)
        keep(m1)
        mn._flux_table = m1._flux_table
        on = mn(bt)
    assert mn._ts_info["step"] < 1e-12 and mn._ts_info["outer"] < 40
    e1, en = float(o1["e_ct"][0]), float(on["e_ct"][0])
    assert en < 0 and e1 < 0
    assert abs(e1 - en) < 0.05 * abs(en)
    q = on["p"]
    assert float(q.min()) >= 0 and float(q.max()) < 0.5
    assert float((o1["p"] - q).abs().max()) < 0.05 * float(q.abs().max())


@pytest.mark.parametrize("n_outer", [1, 30])
def test_forces_are_energy_derivatives(setup, n_outer):
    """Cluster: autograd forces of the total interaction energy against central differences, in linear response
    and with the outer loop converged."""
    tab, chan, table, feat, make, keep = setup
    m = make(ct_outer=n_outer, ct_outer_tol=1e-13)
    X, Z = _cluster()
    X = X.clone().requires_grad_(True)
    bt = _batch(table, tab, chan, X, Z, m)
    E = m(bt)["e_int"].sum()
    keep(m)
    Fa, = torch.autograd.grad(E, X)
    h = 1e-5
    for a, k in ((0, 0), (3, 1), (6, 2)):
        Xp, Xm = X.detach().clone(), X.detach().clone()
        Xp[a, k] += h
        Xm[a, k] -= h
        with torch.no_grad():
            ep = m(_batch(table, tab, chan, Xp, Z, m))["e_int"].sum()
            em = m(_batch(table, tab, chan, Xm, Z, m))["e_int"].sum()
        assert float(Fa[a, k]) == pytest.approx(float((ep - em) / (2 * h)), rel=2e-5, abs=1e-8)


def test_gap_shifts_frozen_by_default(setup):
    """The gap uses the table I, A unless ct_gap_learn: the shifts are frozen (no gradient, not trained)."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    keep(m)
    assert not m.ts_dI.requires_grad and not m.ts_dA.requires_grad
    assert float(m.ts_dI.abs().sum() + m.ts_dA.abs().sum()) == 0.0
    assert all(p is not m.ts_dI and p is not m.ts_dA for p in m.parameters() if p.requires_grad)


def test_parameter_gradients_finite(setup):
    """A training step's gradient is finite (channels at the taper end included) and, with ct_gap_learn, reaches
    the gap shifts."""
    tab, chan, table, feat, make, keep = setup
    m = make(ct_gap_learn=True)
    X, Z = _cluster(a=3.3)
    bt = _batch(table, tab, chan, X, Z, m)
    o = m(bt, field=torch.tensor([[0.002, 0.0, -0.001]], dtype=f64))
    keep(m)
    (o["e_ct"].sum() + o["e_pol"].sum()).backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    assert float(m.ts_dI.grad.abs().sum()) > 0 and float(m.ts_dA.grad.abs().sum()) > 0
