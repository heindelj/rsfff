"""Charge-flux CT (``CompleteModel(ct_model="flux")``): the pair's zero-field energy is the direct-CT kernel's,
the compliance vanishes with it, the parallel CT polarizability of a pair, the joint minimum (stationarity in z and q),
the dimer's zero perpendicular CT response, and forces as derivatives of the energy."""
from pathlib import Path

import pytest
import torch

from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"


@pytest.fixture(scope="module")
def setup():
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)
    cache = {}

    def make(**kw):
        torch.manual_seed(3)
        m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                          pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, ct_model="flux", **kw)
        with torch.no_grad():                 # a Na -> Cl direction much weaker than Cl -> Na, as trained
            m.ct_lnq0[0, 0] -= 2.0
            for p in m.head_ctq.parameters():
                p.add_(0.2 * torch.randn_like(p))
        if "table" in cache:                  # the flux table is the slow part (~20 s): build it once
            m._flux_table = cache["table"]
        return m

    def keep(m):
        cache.setdefault("table", m._flux_table)
    return tab, chan, table, feat, make, keep


def _batch(table, tab, chan, X, Z, m, **kw):
    return batched_precompute(X, Z, torch.zeros(X.shape[0], dtype=torch.long), tab, table, chan, inv_form=m.inv_form, **kw)


def _cluster(a=2.75):
    g = torch.tensor([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.08 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    return X, Z


def _pieces(m, bt, field=None):
    """The flux block and the xpol solution of a batch, as the forward builds them."""
    sp = bt["species"]
    I = m.scaled(bt["feats"])
    B, nmax = bt["B"], bt["nmax"]
    m.frozen_terms(I, bt, sp)
    c = m.confinement(I, None, sp)
    alpha = m.tab.alpha[sp] * c[:, 0]
    Cq = torch.ones(B, nmax, dtype=I.dtype).masked_scatter(bt["mask"], m.tab.C[sp] * c[:, 1])
    ainv = m.alpha_inverse(alpha, sp, bt.get("env"))
    ainv_pad = torch.eye(3, dtype=I.dtype).expand(B, nmax, 3, 3).clone()
    ainv_pad = ainv_pad.masked_scatter(bt["mask"][..., None, None].expand(B, nmax, 3, 3), ainv)
    flux = m.flux_operator(bt, sp, m.ct_charges_from(I, sp), 9 if m.breath else 8)
    z, *_, W = m.xpol_solve(bt, sp, I, bt["L"], bt["A_off"], ainv_pad, Cq, rhs_extra=flux[0])
    return flux, z.reshape(B, -1), W


def test_pair_energy_is_the_kernel_and_compliance_vanishes(setup):
    """Na+ Cl-: without the induced-moment coupling the two channels' zero-field minimum is e_c + e_c' exactly
    (the drive's pair factor), at every distance; the charges vanish with e_c at long range."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    for R in (2.3, 2.8, 3.5, 5.0, 7.0):
        X = torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=torch.float64)
        bt = _batch(table, tab, chan, X, torch.tensor([11, 17]), m)
        with torch.no_grad():
            (Cz, Hq, s, V, G, ch, e_c), z, W = _pieces(m, bt)
            keep(m)
            e, e_ch, q, *_ = m.flux_solve(bt, torch.zeros_like(Cz), Hq, s, V, G, ch, z, torch.zeros_like(W),
                                          torch.zeros(2, dtype=torch.float64))
        assert e_c.shape[0] == 2
        assert abs(float(e[0] / e_c.sum()) - 1) < 1e-9
        assert float(e_ch.sum() - e[0]) == pytest.approx(0.0, abs=1e-15)
        if R >= 7.0:
            assert float(q.abs().max()) < 1e-4


def test_lone_channel_parallel_polarizability(setup):
    """With the reverse channel off and no induced-moment coupling, alpha_CT,par = 2 |e_c| R^2 / V_c^2 = R^2 / (b + h)."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    R = 3.0
    X = torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=torch.float64)
    bt = _batch(table, tab, chan, X, torch.tensor([11, 17]), m)
    with torch.no_grad():
        (Cz, Hq, s, V, G, ch, e_c), z, W = _pieces(m, bt)
        keep(m)
        i, j = bt["ch_index"]
        c = int(((i == 1) & (j == 0)).nonzero()[0])                 # Cl -> Na
        cf, slot = ch
        s1 = s.clone()
        s1[cf[1 - c], slot[1 - c]] = 0.0                            # the reverse channel off
        h = Hq[cf[c], slot[c], slot[c]]
        V1 = V.clone()
        ae = -e_c[c]
        v0 = m.flux_lnv.exp()
        V0 = v0[bt["species"][i[c]]] + v0[bt["species"][j[c]]]
        V1[cf[c], slot[c]] = torch.sqrt(V0 ** 2 + 2 * ae * h)      # lone-channel drive (t = 1)
        Rb = R / K.BOHR_ANG

        def energy(F):
            phi = -F * bt["pos"][:, 0] / K.BOHR_ANG
            return float(m.flux_solve(bt, torch.zeros_like(Cz), Hq, s1, V1, G, ch, z, torch.zeros_like(W), phi)[0][0])
        d = 1e-4
        alpha = -(energy(d) - 2 * energy(0.0) + energy(-d)) / d ** 2
        assert energy(0.0) == pytest.approx(float(e_c[c]), rel=1e-10)
    assert alpha == pytest.approx(float(2 * ae * Rb ** 2 / V1[cf[c], slot[c]] ** 2), rel=1e-5)
    assert alpha == pytest.approx(float(Rb ** 2 / (V0 ** 2 / (2 * ae) + h)), rel=1e-5)


def test_joint_minimum(setup):
    """Cluster: the charges and the joint moments are stationary points of the full quadratic -- z* solves the xpol
    system with the charges' drive added, and every channel's gradient in q vanishes."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    X, Z = _cluster()
    bt = _batch(table, tab, chan, X, Z, m)
    with torch.no_grad():
        (Cz, Hq, s, V, G, ch, e_c), z, W = _pieces(m, bt)
        keep(m)
        n = X.shape[0]
        e, e_ch, q, zj, Q = m.flux_solve(bt, Cz, Hq, s, V, G, ch, z, W, torch.zeros(n, dtype=torch.float64))
        cf, slot = ch
        qp = torch.zeros_like(s).index_put((cf, slot), q)
        # z: A z* = -(b_z + C q*)  <=>  z* = z_pol - W q*
        assert torch.allclose(zj, z - (W @ qp[..., None])[..., 0], atol=1e-14)
        # q: (diag(b) + Hq) q + C^T z* - V = 0 on every channel (b_c = 1 / s_c^2)
        b = torch.where(s > 0, 1 / s.clamp(min=1e-300) ** 2, torch.zeros_like(s))
        g = b * qp + (Hq @ qp[..., None])[..., 0] + torch.einsum("bnc,bn->bc", Cz, zj) - V
        assert float((g * (s > 0)).abs().max()) < 1e-10 * float(V.abs().max())
        assert torch.allclose(Q.sum(1), torch.zeros(1, dtype=torch.float64), atol=1e-14)       # charge conserved
        assert float(e) < 0
        o = m(bt)
    assert float(o["e_ct"]) == pytest.approx(float(e), rel=1e-10)
    assert torch.isfinite(o["e_int"]).all()


def test_dimer_perpendicular_response_is_zero(setup):
    """A dimer's CT has no response to a field across the bond (symmetry); along it, it does."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    X = torch.tensor([[0.0, 0, 0], [2.9, 0, 0]], dtype=torch.float64)
    bt = _batch(table, tab, chan, X, torch.tensor([11, 17]), m)

    def e_ct(F):
        with torch.no_grad():
            return float(m(bt, field=torch.tensor([F], dtype=torch.float64))["e_ct"][0])
    d = 2e-3
    e0 = e_ct([0.0, 0, 0])
    keep(m)
    a_perp = -(e_ct([0, d, 0]) - 2 * e0 + e_ct([0, -d, 0])) / d ** 2
    a_par = -(e_ct([d, 0, 0]) - 2 * e0 + e_ct([-d, 0, 0])) / d ** 2
    assert abs(a_perp) < 1e-6 * abs(a_par)
    assert a_par > 1.0


def test_forces_are_energy_derivatives(setup):
    """Cluster: autograd forces of the total interaction energy against central differences."""
    tab, chan, table, feat, make, keep = setup
    m = make()
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


def test_parameter_gradients_finite(setup):
    """A training step's gradient is finite (channels at the taper end included)."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    X, Z = _cluster(a=3.3)
    bt = _batch(table, tab, chan, X, Z, m)
    o = m(bt)
    keep(m)
    (o["e_ct"].sum() + o["e_pol"].sum()).backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    assert float(m.flux_lnv.grad.abs().sum()) > 0
    assert float(m.flux_eta.grad.abs().sum()) > 0
