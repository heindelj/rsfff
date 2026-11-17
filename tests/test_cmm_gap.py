"""The gap CT model (``CmmConfig(ct_model="gap")``): the two-state closed forms; the gap's electrostatics are the
frontier-averaged frozen potentials; the joint solve is the minimum of the energy written out in the module
docstring (dense channel-space reference), also when some channels are too soft to eliminate; a channel that
couples to nothing is exactly its two-state energy and charge; forces and the polarizability are exact; the periodic
solve is the cluster's; an isolated ion has no CT; the flux model is untouched by the option."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
RING = (["Cl", "Na", "Cl", "Na", "Cl", "Li"],
        torch.tensor([[2.9, 0.1, 0.0], [1.4, 2.5, 0.1], [-1.5, 2.4, -0.1], [-2.8, 0.0, 0.2], [-1.4, -2.5, 0.0],
                      [1.5, -2.2, -0.2]], dtype=f64))
PAIR = (["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0.1, 0]], dtype=f64))


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, q_boost=0.0, gap_shift=0.0, overlap="probe", w0=0.0):
    torch.manual_seed(5)
    m = CmmModel(*tables, CmmConfig(film=False, ct_model="gap", ct_gap_overlap=overlap, ct_gap_w0=w0))
    with torch.no_grad():                       # an environment that does something: random readouts and constants
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw, m.ct_ln_kov):
            p.add_(0.1 * torch.randn_like(p))
        m.ct_ln_q0.add_(q_boost)
        m.ct_gap_I.sub_(gap_shift)              # every gap lowered: soft and resonant channels
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def test_two_state_closed_forms(tables):
    """a, q* against the 2x2 eigenproblem [[0, -W], [-W, Delta]]; q* = -d(-a)/dDelta ... = da/d(-Delta); the
    stiffness den / (2 W^2) + h is 1 / (dq*/dDelta) in size."""
    m = _model(tables)
    bt = _batch(m, [RING])
    sp = bt["species"]
    m.environment(bt)
    m._q = m.ct_charges(sp)
    with torch.no_grad():
        g = m.gap_channels(bt, sp)
    W, D = g["W2"].sqrt(), g["Delta"]
    for c in range(0, W.shape[0], 5):
        H = torch.tensor([[0.0, -float(W[c])], [-float(W[c]), float(D[c])]], dtype=f64)
        e, v = torch.linalg.eigh(H)
        assert float(-g["a"][c]) == pytest.approx(float(e[0]), rel=1e-9, abs=1e-14)
        assert float(g["qs"][c]) == pytest.approx(float(v[1, 0] ** 2), rel=1e-7, abs=1e-14)
    # Hellmann-Feynman and the tangent stiffness by finite differences in the gap
    d = 1e-5
    def aq(Dl, W2):
        rt = torch.sqrt(Dl * Dl + 4 * W2)
        return 0.5 * (rt - Dl), 0.5 * (1 - Dl / rt)
    ap, qp = aq(D + d, g["W2"])
    am, qm = aq(D - d, g["W2"])
    strong = g["qs"] > 1e-6
    assert torch.allclose(((am - ap) / (2 * d))[strong], g["qs"][strong], rtol=1e-5)
    k = g["den"] / (2 * g["W2"]) + g["h"]
    assert torch.allclose((-(qp - qm) / (2 * d) * k)[strong], torch.ones(int(strong.sum()), dtype=f64), rtol=1e-4)


def test_gap_electrostatics_are_the_frontier_potentials(tables):
    """bt["phi_front"] (table rows + point charges) against the closed-form frontier-averaged potentials."""
    m = _model(tables)
    bt = _batch(m, [RING, PAIR])
    for col, role in enumerate(("donor", "acceptor")):
        ref = K.frontier_potential_matrix(bt["pos"], bt["species"], bt["frame"], m.tab, role).sum(1)
        assert torch.allclose(bt["phi_front"][:, col], ref, rtol=1e-6, atol=1e-8), role


@pytest.mark.parametrize("gap_shift", [0.0, 0.19])
def test_joint_solution_is_the_dense_minimum(tables, gap_shift):
    """With the moments decoupled (Ca = 0) the channel charges are the stationary point of
    sum_c [-a + 1/2 k (q - q*)^2] + 1/2 Q H_Q Q - 1/2 sum_c h_c q_c^2; with every gap lowered by 0.19 Hartree some
    channels have k - h below the elimination threshold (here negative) and go through the explicit block."""
    from rsfff.ff.cmm.model import GAP_B_EPS
    m = _model(tables, 0.0, gap_shift)
    bt = _batch(m, [RING])
    with torch.no_grad():
        o = m({**bt, "Ca": torch.zeros_like(bt["Ca"])})
        sp = bt["species"]
        n = sp.shape[0]
        g = m._gap
        i, j = bt["ch_index"]
        C = i.shape[0]
        ar = torch.arange(C)
        G = torch.zeros(n, C, dtype=f64).index_put((i, ar), torch.ones(C, dtype=f64)).index_put((j, ar), -torch.ones(C, dtype=f64))
        Hq = bt["Jq"][0, :n, :n] + torch.diag(torch.nn.functional.softplus(m.ct_eta_raw[sp]))
        k = g["den"] / (2 * g["W2"]) + g["h"]
        b = k - g["h"]
        n_bad = int((b < GAP_B_EPS).sum())
        assert (n_bad > 0) == (gap_shift > 0)
        Hc = torch.diag(b) + G.T @ Hq @ G
        assert torch.allclose(torch.diagonal(G.T @ Hq @ G), g["h"], rtol=1e-10)          # h is the channel's own share
        q = torch.linalg.solve(Hc, k * g["qs"])
        E = (-g["a"] + 0.5 * k * (q - g["qs"]) ** 2).sum() + 0.5 * (G @ q) @ Hq @ (G @ q) - 0.5 * (g["h"] * q * q).sum()
        assert torch.allclose(o["q_ch"], q, rtol=1e-8, atol=1e-12)
        assert float(o["e_ct"][0]) == pytest.approx(float(E), rel=1e-9)
        assert float(o["e_ch"].sum()) == pytest.approx(float(o["e_ct"][0]), rel=1e-12)
        assert torch.allclose(o["dq"], G @ q, atol=1e-12)


def test_stationary_with_the_moments(tables):
    """grad_q of the joint energy vanishes at the returned (z, q): k (q - q*) - h q + G^T (H_Q Q + C^T z) = 0."""
    m = _model(tables, 0.3)
    bt = _batch(m, [RING])
    with torch.no_grad():
        o = m(bt)
        sp = bt["species"]
        n = sp.shape[0]
        g = m._gap
        i, j = bt["ch_index"]
        Hq = bt["Jq"][0, :n, :n] + torch.diag(torch.nn.functional.softplus(m.ct_eta_raw[sp]))
        k = g["den"] / (2 * g["W2"]) + g["h"]
        q, Q = o["q_ch"], o["dq"]
        zf = torch.cat([o["x"], torch.zeros(1, bt["nmax"], bt["D"] - 8, dtype=f64)], -1).reshape(1, -1)
        psi = Hq @ Q + torch.einsum("bnk,bn->bk", bt["Ca"], zf)[0, :n]
        grad = k * (q - g["qs"]) - g["h"] * q + psi[i] - psi[j]
        assert float(grad.abs().max()) < 1e-9 * float((k * g["qs"]).abs().max())


def test_a_channel_that_couples_to_nothing_is_its_two_state_answer(tables):
    """H_Q = 0 and Ca = 0: every channel is alone, E_CT = -sum a and q = q*."""
    m = _model(tables)
    bt = _batch(m, [RING])
    with torch.no_grad():
        m.ct_eta_raw.fill_(-80.0)
        m.ct_jself.zero_()
        o = m({**bt, "Ca": torch.zeros_like(bt["Ca"]), "Jq": torch.zeros_like(bt["Jq"]), "ch_J": torch.zeros_like(bt["ch_J"])})
    assert float(o["e_ct"][0]) == pytest.approx(float(o["e_kernel"].sum()), rel=1e-10)
    assert torch.allclose(o["q_ch"], o["q_star"], rtol=1e-9, atol=1e-14)


def test_forces_and_alpha(tables):
    m = _model(tables, 0.3)
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    out = m(_batch(m, [RING], pos))
    F = -torch.autograd.grad(out["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, k in ((0, 0), (3, 1), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[a, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)]))["e_int"]))
            assert float(F[a, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-6, abs=1e-9)
        bt = _batch(m, [RING])
        o = m(bt, alpha=True)
        hf, E = 1e-3, {}
        for a in range(3):
            for sg in (1, -1):
                Fd = torch.zeros(1, 3, dtype=f64)
                Fd[:, a] = sg * hf
                r = m(bt, field=Fd)
                E[a, sg] = r["e_pol"] + r["e_ct"]
        r0 = m(bt)
        fd = torch.stack([-(E[a, 1] + E[a, -1] - 2 * (r0["e_pol"] + r0["e_ct"])) / hf ** 2 for a in range(3)], 1)
        assert torch.allclose(torch.diagonal(o["alpha"], dim1=1, dim2=2), fd, rtol=1e-5, atol=1e-5)
        assert bool((torch.diagonal(o["alpha"] - o["alpha_pol"], dim1=1, dim2=2) > -1e-10).all())


def test_periodic_solve_is_the_cluster(tables):
    """An inversion-symmetric neutral cluster in cubes of 30 and 40 A extrapolates (1/L^5) to the cluster."""
    m = _model(tables, 0.3)
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym = ["Na", "Cl", "Li", "F"] * 2
    X = torch.cat([half, -half])
    with torch.no_grad():
        c = m(_batch(m, [(sym, X)]))
        p = {L: m(_batch(m, [(sym, X + L / 2)], cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False))
             for L in (30., 40.)}
    w = 40.0 ** 5 / (40.0 ** 5 - 30.0 ** 5)
    for k in ("e_pol", "e_ct", "e_int"):
        ext = p[30.][k] + w * (p[40.][k] - p[30.][k])
        assert float((ext - c[k]).abs()) < 3e-7, k
    assert float(c["e_ct"].abs()) > 1e-4


def test_periodic_forces(tables):
    m = _model(tables, 0.3)
    g = torch.Generator().manual_seed(1)
    a = 5.6
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=f64) * a
    X = torch.cat([base, base + torch.tensor([a / 2, 0, 0], dtype=f64)]) + 0.15 * torch.randn(8, 3, generator=g, dtype=f64)
    sym, cell = ["Na"] * 4 + ["Cl"] * 4, torch.eye(3, dtype=f64) * a
    pos = X.clone().requires_grad_(True)
    out = m(_batch(m, [(sym, X)], pos, cell=cell))
    F = -torch.autograd.grad(out["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for at, k in ((0, 0), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[at, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)], cell=cell))["e_int"]))
            assert float(F[at, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=5e-5, abs=1e-8)


def test_isolated_ion_and_one_sided_charges(tables):
    m = _model(tables)
    far = (["Na", "Cl"], torch.tensor([[0.0, 0, 0], [60.0, 0, 0]], dtype=f64))
    with torch.no_grad():
        o = m(_batch(m, [far]))
        r = m(_batch(m, [RING]))
    assert float(o["e_ct"].abs()) == 0.0
    q0 = m.ct_ln_q0.exp()[_batch(m, [RING])["species"]]
    assert bool((r["q_ct"] <= q0 * (1 + 1e-12)).all())


def test_flux_model_has_no_gap_parameters(tables):
    m = CmmModel(*tables, CmmConfig(film=False))
    assert not m.gap and not hasattr(m, "ct_ln_kov")
    assert "ct_ln_kov" in CmmModel(*tables, CmmConfig(film=False, ct_model="gap")).state_dict()
    with pytest.raises(ValueError):
        CmmConfig(ct_model="two-state")
    with pytest.raises(ValueError):
        CmmConfig(ct_model="gap", ct_gap_w0=-1e-3)
    # an anion's acceptor level is I - 2 J_self; a cation's is the table's
    m = CmmModel(*tables, CmmConfig(film=False, ct_model="gap"))
    tab = tables[0]
    an = torch.isnan(tab.A_acceptor)
    assert bool(an.any()) and torch.allclose(m.ct_gap_A[an], (tab.I_donor - 2 * m.ct_jself)[an].to(f64))
    assert torch.allclose(m.ct_gap_A[~an], tab.A_acceptor[~an].to(f64))


def test_broadened_crossing(tables):
    """``ct_gap_w0``: a = lam [sqrt(Delta^2/4 + W_b^2) - Delta/2], W_b^2 = W^2 + W0^2, lam = W^2 / W_b^2. Still
    q* = -da/dDelta and k = 1 / |dq*/dDelta|; W^2 / Delta at a large gap; transfer <= lam and response
    <= W^2 / (4 W_b^3) with every gap inverted; the joint solve is the stationary point and forces are exact."""
    w0 = 0.02
    m = _model(tables, 0.3, 0.6, w0=w0)                                  # 0.6 Hartree off every gap: X -> M inverted
    bt = _batch(m, [RING])
    with torch.no_grad():
        o = m(bt)
        g = m._gap
        W2, D = g["W2"], g["Delta"]
        assert bool((D < -0.1).any()) and bool((D > 0).any())
        Wb2 = W2 + w0 * w0
        lam = W2 / Wb2
        def aq(Dl):
            rt = torch.sqrt(Dl * Dl + 4 * Wb2)
            return lam * 0.5 * (rt - Dl), lam * 0.5 * (1 - Dl / rt)
        a, q = aq(D)
        assert torch.allclose(g["a"], a, rtol=1e-10) and torch.allclose(g["qs"], q, rtol=1e-10)
        d = 1e-5
        (ap, qp), (am, qm) = aq(D + d), aq(D - d)
        assert torch.allclose((am - ap) / (2 * d), g["qs"], rtol=1e-6)
        k = g["den"] / (2 * W2) + g["h"]
        assert torch.allclose(-(qp - qm) / (2 * d) * k, torch.ones_like(k), rtol=1e-5)
        assert bool((g["qs"] <= lam).all()) and bool((1 / k <= W2 / (4 * Wb2 ** 1.5) * (1 + 1e-12)).all())
        inv = D < -0.1                                                    # the exact form moves > 0.9 electron there
        assert float((0.5 * (1 - D / torch.sqrt(D * D + 4 * W2)))[inv].min()) > 0.9 and bool((g["qs"][inv] < lam[inv]).all())
        # stationary in q with the moments (as test_stationary_with_the_moments)
        sp = bt["species"]
        n = sp.shape[0]
        i, j = bt["ch_index"]
        Hq = bt["Jq"][0, :n, :n] + torch.diag(torch.nn.functional.softplus(m.ct_eta_raw[sp]))
        zf = torch.cat([o["x"], torch.zeros(1, bt["nmax"], bt["D"] - 8, dtype=f64)], -1).reshape(1, -1)
        psi = Hq @ o["dq"] + torch.einsum("bnk,bn->bk", bt["Ca"], zf)[0, :n]
        grad = k * (o["q_ch"] - g["qs"]) - g["h"] * o["q_ch"] + psi[i] - psi[j]
        assert float(grad.abs().max()) < 1e-9 * float((k * g["qs"]).abs().max())
        # a large gap: the same W^2 / Delta as the exact form, to relative order W_b^2 / Delta^2
        m.ct_gap_I.add_(0.6 + 3.0)
        g = m.gap_channels(bt, sp)
        assert torch.allclose(g["a"], g["W2"] / g["Delta"], rtol=3 * float(((g["W2"] + w0 * w0) / g["Delta"] ** 2).max()))
        m.ct_gap_I.sub_(3.0)
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(_batch(m, [RING], pos))["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a_, c in ((0, 0), (3, 2)):
            e = []
            for s_ in (1, -1):
                Y = X.clone()
                Y[a_, c] += s_ * h
                e.append(float(m(_batch(m, [RING], Y))["e_int"]))
            assert float(F[a_, c]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-5, abs=1e-9)


def test_pauli_anchored_overlap(tables):
    """``ct_gap_overlap="pauli"``: the acceptor shift is c sum_k (E0_jk / O_jk) S_A(j <- k) from the exchange pairs;
    forces are exact; the periodic solve is the cluster's."""
    m = _model(tables, 0.3, overlap="pauli")
    sym, X = RING
    bt = _batch(m, [RING])
    with torch.no_grad():
        o = m(bt)
        sp = bt["species"]
        shift = m.acceptor_shift(bt, sp)
        xs = bt["x_sel"]
        pi, pj = bt["pairs"][:, xs]
        e0, Px = m._pauli_pairs[1][xs], xs.shape[0]
        ref = torch.zeros(len(sym), dtype=f64)
        for p in range(Px):
            ref[pi[p]] += e0[p] * bt["x_swO"][p] * bt["x_Sa"][p]
            ref[pj[p]] += e0[p] * bt["x_swO"][p] * bt["x_Sa"][Px + p]
        assert torch.allclose(shift, m.overlap_strength()[sp] * ref, rtol=1e-12)
        # the strength starts at 1.5 and stays within a factor 5 of 1 wherever the raw parameter goes
        m0 = CmmModel(*tables, CmmConfig(film=False, ct_model="gap"))
        assert torch.allclose(m0.overlap_strength(), torch.full_like(m0.ct_ln_kov, 1.5), rtol=1e-12)
        for raw, lim in ((40.0, 5.0), (-40.0, 0.2)):
            m0.ct_ln_kov.fill_(raw)
            assert torch.allclose(m0.overlap_strength(), torch.full_like(m0.ct_ln_kov, lim), rtol=1e-9)
        mp = CmmModel(*tables, CmmConfig(film=False, ct_model="gap", ct_gap_overlap="probe"))
        assert torch.allclose(mp.overlap_strength(), torch.full_like(mp.ct_ln_kov, 10.0), rtol=1e-12)
        assert float(shift.min()) > 0 and float(o["e_ct"]) < 0
        # the per-pair probe overlaps add up to the atoms' totals where every neighbour is an exchange pair
        tot = torch.zeros(len(sym), dtype=f64).index_add(0, pi, bt["x_Sa"][:Px]).index_add(0, pj, bt["x_Sa"][Px:])
        assert torch.allclose(tot, bt["S0_atom"][:, 1], rtol=1e-6, atol=1e-9)
    pos = X.clone().requires_grad_(True)
    out = m(_batch(m, [RING], pos))
    F = -torch.autograd.grad(out["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, k in ((0, 0), (3, 1)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[a, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)]))["e_int"]))
            assert float(F[a, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-6, abs=1e-9)
        half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
        sym2 = ["Na", "Cl", "Li", "F"] * 2
        X2 = torch.cat([half, -half])
        c = m(_batch(m, [(sym2, X2)]))
        p = {L: m(_batch(m, [(sym2, X2 + L / 2)], cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False)) for L in (30., 40.)}
        w = 40.0 ** 5 / (40.0 ** 5 - 30.0 ** 5)
        assert float((p[30.]["e_ct"] + w * (p[40.]["e_ct"] - p[30.]["e_ct"]) - c["e_ct"]).abs()) < 3e-7
