"""Exclusion-aware CT (``CmmConfig(ct_model="gap", ct_gap_converge=N, ct_exclusion=True)``): for the channels of one
site alone the energy is the lowest eigenvalue of the arrowhead matrix of its configurations (sharing an acceptor
or a donor); the converged charges are a minimum of the functional written out here, and no site gives or takes
more than one electron; channels that share no site are the independent two-state channels; forces, the
polarizability and their parameter gradients are exact; the periodic solve is the cluster's; the option needs the
converged gap solve."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
RING = (["Cl", "Na", "Cl", "Na", "Cl", "Li"],
        torch.tensor([[2.9, 0.1, 0.0], [1.4, 2.5, 0.1], [-1.5, 2.4, -0.1], [-2.8, 0.0, 0.2], [-1.4, -2.5, 0.0],
                      [1.5, -2.2, -0.2]], dtype=f64))
PAIR = (["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0.1, 0]], dtype=f64))


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, q_boost=0.3, gap_shift=0.19, w0=0.0, excl=True, tol=1e-11):
    """Soft, resonant channels by default (every gap lowered by ``gap_shift`` Hartree)."""
    torch.manual_seed(5)
    m = CmmModel(*tables, CmmConfig(film=False, ct_model="gap", ct_gap_overlap="probe", ct_gap_w0=w0,
                                    ct_gap_converge=40, ct_gap_tol=tol, ct_exclusion=excl))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw, m.ct_ln_kov):
            p.add_(0.1 * torch.randn_like(p))
        m.ct_ln_q0.add_(q_boost)
        m.ct_gap_I.sub_(gap_shift)
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def _lone(m, bt):
    """The batch and model with the channels' electrostatics off (no slot Coulomb or hardness, no moments)."""
    with torch.no_grad():
        m.ct_eta_raw.fill_(-80.0)
        m.ct_jself.zero_()
    return {**bt, "Ca": torch.zeros_like(bt["Ca"]), "Jq": torch.zeros_like(bt["Jq"]), "ch_J": torch.zeros_like(bt["ch_J"])}


@pytest.mark.parametrize("sym,keep", [(["Cl", "Na", "Cl", "F"], "acceptor"), (["Na", "Cl", "Na", "Li"], "donor")])
def test_one_site_is_the_arrowhead_ground_state(tables, sym, keep):
    """Three halides around one Na+ (the channels into it share the acceptor) and three cations around one Cl- (the
    channels out of it share the donor), every other channel switched off, no electrostatics: E_CT is the lowest
    eigenvalue of [[0, -W_c], [-W_c, diag(Delta_c)]] and the charges are its eigenvector's weights."""
    m = _model(tables, gap_shift=0.12)
    tab = m.tab
    X = torch.tensor([[2.7, 0.0, 0.1], [0.0, 0.0, 0.0], [-1.2, 2.5, 0.0], [-0.9, -1.7, 1.6]], dtype=f64)
    with torch.no_grad():
        anion = tab.charge < 0
        m.ct_ln_q0[anion, 1] = -80.0                       # anions accept nothing
        m.ct_ln_q0[~anion, 0] = -80.0                      # cations donate nothing
    bt = _lone(m, _batch(m, [(sym, X)]))
    with torch.no_grad():
        o = m(bt)
    g = m._gap
    i, j = bt["ch_index"]
    live = g["W2"] > 1e-24
    assert int(live.sum()) == 3 and bool(m._gap_info["converged"].all())
    centre = 1
    assert bool(((j if keep == "acceptor" else i)[live] == centre).all())
    W, D = g["W2"][live].sqrt(), g["Delta0"][live]
    A = torch.zeros(4, 4, dtype=f64)
    A[0, 1:] = A[1:, 0] = -W
    A[1:, 1:] = torch.diag(D)
    ev, vec = torch.linalg.eigh(A)
    assert float(o["e_ct"][0]) == pytest.approx(float(ev[0]), rel=1e-10)
    assert torch.allclose(o["q_ch"][live], vec[1:, 0] ** 2, rtol=1e-7, atol=1e-12)
    # independent channels would give more: sum of the two-state energies
    assert float(ev[0]) > float(o["e_kernel"].sum()) + 1e-6


@pytest.mark.parametrize("w0", [0.0, 0.005])
def test_converged_is_a_minimum_of_the_exclusion_functional(tables, w0):
    """E(q) = sum_c [Delta_c q_c - 2 W_b sqrt(q_c (lam_c (1 - b_c) - q_c)) - 1/2 h_c q_c^2] + 1/2 Q^T H_Q Q + Q^T C^T z
    at the returned moments: zero gradient in q, positive definite Hessian of the part without z (moments decoupled),
    its value E_CT; no site gives or takes more than one electron."""
    m = _model(tables, w0=w0)
    bt = _batch(m, [RING])
    bt0 = {**bt, "Ca": torch.zeros_like(bt["Ca"])}
    with torch.no_grad():
        o = m(bt0)
    assert bool(m._gap_info["converged"].all()) and 0 < o["ct_gap_iters"] < 30
    g, sp = m._gap, bt["species"]
    n = sp.shape[0]
    i, j = bt["ch_index"]
    C = i.shape[0]
    ar = torch.arange(C)
    PD = torch.zeros(n, C, dtype=f64).index_put((i, ar), torch.ones(C, dtype=f64))
    PA = torch.zeros(n, C, dtype=f64).index_put((j, ar), torch.ones(C, dtype=f64))
    G = PD - PA
    Hq = (bt["Jq"][0, :n, :n] + torch.diag(torch.nn.functional.softplus(m.ct_eta_raw[sp]))).detach()
    W2, D0, h = g["W2"].detach(), g["Delta0"].detach(), g["h"].detach()
    Wb2 = W2 + w0 ** 2
    lam = W2 / Wb2 if w0 > 0 else torch.ones_like(W2)

    def E(q):
        b = PD.T @ (PD @ q) + PA.T @ (PA @ q) - 2 * q           # the site-mates' charge
        Q = G @ q
        return (D0 * q - 2 * Wb2.sqrt() * torch.sqrt(q * (lam * (1 - b) - q)) - 0.5 * h * q * q).sum() + 0.5 * Q @ Hq @ Q

    q = o["q_ch"].detach().clone().requires_grad_(True)
    assert float(torch.autograd.grad(E(q), q)[0].abs().max()) < 1e-9
    assert float(torch.linalg.eigvalsh(torch.autograd.functional.hessian(E, q.detach())).min()) > 0
    assert float(o["e_ct"][0]) == pytest.approx(float(E(q.detach())), rel=1e-10)
    qd = q.detach()
    assert float((PD @ qd).max()) < 1 and float((PA @ qd).max()) < 1 and float(qd.min()) > 0
    assert float((PD @ qd).max()) > 0.03                                           # and the sites are busy
    # the independent channels' minimum is lower, and its charges differ
    m0 = _model(tables, w0=w0, excl=False)
    with torch.no_grad():
        o0 = m0(bt0)
    assert float(o0["e_ct"]) < float(o["e_ct"]) - 1e-4


def test_channels_without_site_mates_are_the_two_state_channels(tables):
    """A pair: Cl -> Na and Na -> Cl share no donor and no acceptor."""
    m, m0 = _model(tables), _model(tables, excl=False)
    bt = _batch(m, [PAIR])
    with torch.no_grad():
        o, o0 = m(bt), m0(bt)
    for k in ("e_ct", "e_pol", "e_int"):
        assert torch.allclose(o[k], o0[k], rtol=0, atol=1e-11), k
    assert torch.allclose(o["q_ch"], o0["q_ch"], rtol=0, atol=1e-10)


def test_forces_and_alpha(tables):
    m = _model(tables)
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(_batch(m, [RING, PAIR], torch.cat([pos, PAIR[1]])))["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, k in ((0, 0), (3, 1), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[a, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)]))["e_int"]))
            assert float(F[a, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=1e-6, abs=1e-9)
        bt = _batch(m, [RING])
        o, r0 = m(bt, alpha=True), m(bt)
        an = torch.diagonal(o["alpha"], dim1=1, dim2=2)[0]
        err = []
        for hf in (1e-3, 2.5e-4):
            fd = []
            for a in range(3):
                E = {}
                for sg in (1, -1):
                    Fd = torch.zeros(1, 3, dtype=f64)
                    Fd[:, a] = sg * hf
                    r = m(bt, field=Fd)
                    E[sg] = r["e_pol"] + r["e_ct"]
                fd.append(-(E[1] + E[-1] - 2 * (r0["e_pol"] + r0["e_ct"])) / hf ** 2)
            err.append(float((torch.cat(fd) - an).abs().max()))
        assert err[1] < 0.01 and err[1] < err[0] / 8


def test_parameter_gradients_of_forces_and_alpha(tables):
    m = _model(tables, tol=1e-12)
    sym, X = RING
    bt = _batch(m, [RING])

    def tr_alpha():
        return torch.diagonal(m(bt, alpha=True)["alpha"], dim1=1, dim2=2).sum()

    def f2(create_graph):
        pos = X.clone().requires_grad_(True)
        F = torch.autograd.grad(m(_batch(m, [RING], pos))["e_int"].sum(), pos, create_graph=create_graph)[0]
        return (F ** 2).sum()

    for fn, kw in ((tr_alpha, {}), (f2, {"create_graph": True})):
        for p in (m.ct_ln_q0, m.ct_eta_raw):
            gr = torch.autograd.grad(fn(**kw), p)[0]
            k = int(gr.abs().flatten().argmax())
            eps, v = 1e-5, []
            for sg in (1, -1):
                with torch.no_grad():
                    p.view(-1)[k] += sg * eps
                v.append(float(fn(**({"create_graph": False} if kw else {}))))
                with torch.no_grad():
                    p.view(-1)[k] -= sg * eps
            assert float(gr.view(-1)[k]) == pytest.approx((v[0] - v[1]) / (2 * eps), rel=1e-6)


def test_periodic(tables):
    """A cluster in cubes of 30 and 40 A extrapolates (1/L^5) to the cluster; forces under PBC; the warm start."""
    m = _model(tables, gap_shift=0.12, tol=1e-10)
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym, X = ["Na", "Cl", "Li", "F"] * 2, torch.cat([half, -half])
    with torch.no_grad():
        c = m(_batch(m, [(sym, X)]))
        p = {L: m(_batch(m, [(sym, X + L / 2)], cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False))
             for L in (30., 40.)}
        c0 = _model(tables, gap_shift=0.12, excl=False)(_batch(m, [(sym, X)]))
    assert float(c["ct_gap_resid"].max()) < 1e-9 and float(p[40.]["ct_gap_resid"].max()) < 1e-9
    assert float((c["e_ct"] - c0["e_ct"]).abs()) > 1e-5                         # exclusion matters in this cluster
    w = 40.0 ** 5 / (40.0 ** 5 - 30.0 ** 5)
    for k in ("e_pol", "e_ct", "e_int"):
        assert float((p[30.][k] + w * (p[40.][k] - p[30.][k]) - c[k]).abs()) < 3e-7, k
    g = torch.Generator().manual_seed(1)
    a = 5.6
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=f64) * a
    Xc = torch.cat([base, base + torch.tensor([a / 2, 0, 0], dtype=f64)]) + 0.15 * torch.randn(8, 3, generator=g, dtype=f64)
    symc, cell = ["Na"] * 4 + ["Cl"] * 4, torch.eye(3, dtype=f64) * a
    pos = Xc.clone().requires_grad_(True)
    o = m(_batch(m, [(symc, Xc)], pos, cell=cell))
    assert o["ct_gap_iters"] > 0 and float(o["ct_gap_resid"].max()) < 1e-9
    F = -torch.autograd.grad(o["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for at, k in ((0, 0), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = Xc.clone()
                Xp[at, k] += sg * h
                e.append(float(m(_batch(m, [(symc, Xp)], cell=cell))["e_int"]))
            assert float(F[at, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-6, abs=1e-9)
        bt = _batch(m, [(symc, Xc)], cell=cell)
        o2 = m({**bt, "guess": o["guess"]})
        assert o2["ct_gap_iters"] == 0 and float(o2["e_int"]) == pytest.approx(float(o["e_int"]), abs=1e-9)
        assert o2["cg_iters"] < o["cg_iters"]
        ot = m({**bt, "total_only": True})
        assert float(ot["e_int"]) == pytest.approx(float(o["e_int"]), abs=1e-9)


def test_config():
    assert not CmmConfig().ct_exclusion
    with pytest.raises(ValueError):
        CmmConfig(ct_model="gap", ct_exclusion=True)                 # no one-step form
    with pytest.raises(ValueError):
        CmmConfig(ct_exclusion=True, ct_gap_converge=20)             # the flux model has no gaps
    c = CmmConfig.from_dict(CmmConfig(ct_model="gap", ct_gap_converge=20, ct_exclusion=True).to_dict())
    assert c.ct_exclusion
