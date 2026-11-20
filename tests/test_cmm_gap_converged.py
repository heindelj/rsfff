"""The self-consistent gap solve (``CmmConfig(ct_model="gap", ct_gap_converge=N)``): the converged charges are a
minimum of the full two-state functional written out here (not of its expansion about a predicted point) and satisfy
the gap equation with the coupled moments; the result does not depend on the start; forces, the polarizability and
their parameter gradients are exact; the periodic solve is the cluster's, with exact forces and a warm start; the
charges stay two-state charges where the one-step expansion leaves [0, 1]; off by default."""
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


def _model(tables, q_boost=0.3, gap_shift=0.19, w0=0.0, predict=1, converge=40, tol=1e-11):
    """Soft, resonant channels by default (every gap lowered by ``gap_shift`` Hartree): charges up to ~0.7."""
    torch.manual_seed(5)
    m = CmmModel(*tables, CmmConfig(film=False, ct_model="gap", ct_gap_overlap="probe", ct_gap_w0=w0,
                                    ct_gap_predict=predict, ct_gap_converge=converge, ct_gap_tol=tol))
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


@pytest.mark.parametrize("w0", [0.0, 0.005])
def test_converged_is_a_minimum_of_the_two_state_functional(tables, w0):
    """Moments decoupled (Ca = 0): E(q) = sum_c [Delta_c q_c - 2 W_b sqrt(q_c (lam - q_c)) - 1/2 h_c q_c^2]
    + 1/2 (G q)^T H_Q (G q) with the frozen gaps Delta. At the returned charges its gradient vanishes, its Hessian is
    positive definite, and its value is E_CT."""
    m = _model(tables, w0=w0)
    bt = _batch(m, [RING])
    with torch.no_grad():
        o = m({**bt, "Ca": torch.zeros_like(bt["Ca"])})
    assert bool(m._gap_info["converged"].all()) and 0 < o["ct_gap_iters"] < 25
    g, sp = m._gap, bt["species"]
    n = sp.shape[0]
    i, j = bt["ch_index"]
    C = i.shape[0]
    ar = torch.arange(C)
    G = torch.zeros(n, C, dtype=f64).index_put((i, ar), torch.ones(C, dtype=f64)).index_put((j, ar), -torch.ones(C, dtype=f64))
    Hq = (bt["Jq"][0, :n, :n] + torch.diag(torch.nn.functional.softplus(m.ct_eta_raw[sp]))).detach()
    W2, D0, h = g["W2"].detach(), g["Delta0"].detach(), g["h"].detach()
    Wb2 = W2 + w0 ** 2
    lam = W2 / Wb2

    def E(q):
        Q = G @ q
        return (D0 * q - 2 * Wb2.sqrt() * torch.sqrt(q * (lam - q)) - 0.5 * h * q * q).sum() + 0.5 * Q @ Hq @ Q

    q = o["q_ch"].detach().clone().requires_grad_(True)
    assert float(q.detach().min()) > 0 and bool((q < lam).all()) and float(q.detach().max()) > 0.05
    grad = torch.autograd.grad(E(q), q)[0]
    assert float(grad.abs().max()) < 1e-9
    assert float(torch.linalg.eigvalsh(torch.autograd.functional.hessian(E, q.detach())).min()) > 0
    assert float(o["e_ct"][0]) == pytest.approx(float(E(q)), rel=1e-10)
    assert torch.allclose(o["q_ch"], o["q_star"], rtol=0, atol=1e-10)           # nothing left of the expansion


def test_gap_equation_with_the_moments(tables):
    """delta_c = (psi_i - psi_j) - h_c q_c with psi the potential of the *joint* solution's charges and moments."""
    m = _model(tables)
    bt = _batch(m, [RING, PAIR])
    with torch.no_grad():
        o = m(bt)
    assert bool(m._gap_info["converged"].all())
    g, sp, fr = m._gap, bt["species"], bt["frame"]
    i, j = bt["ch_index"]
    loc = torch.cat([torch.arange(6), torch.arange(2)])
    eta = torch.nn.functional.softplus(m.ct_eta_raw[sp]).detach()
    Qp = torch.zeros(2, bt["nmax"], dtype=f64).index_put((fr, loc), o["dq"])
    z = torch.cat([o["x"], torch.zeros(2, bt["nmax"], bt["D"] - 8, dtype=f64)], -1)
    psi = ((bt["Jq"] @ Qp[..., None])[..., 0] + torch.einsum("bnk,bn->bk", bt["Ca"], z.reshape(2, -1)))[fr, loc] + eta * o["dq"]
    assert torch.allclose(g["delta"], psi[i] - psi[j] - g["h"] * o["q_ch"], rtol=0, atol=1e-9)
    assert torch.allclose(g["Delta"], g["Delta0"] + g["delta"]) and float(g["delta"].abs().max()) > 1e-3
    assert float((o["q_ch"] - o["q_star"]).abs().max()) < 1e-10


def test_independent_of_the_start(tables):
    """From the frozen gaps (predict 0) and from the predictor (1, 2): the same minimum."""
    bt, e = None, []
    for predict in (0, 1, 2):
        m = _model(tables, predict=predict)
        bt = bt or _batch(m, [RING, PAIR])
        with torch.no_grad():
            o = m(bt)
        assert bool(m._gap_info["converged"].all())
        e.append(o["e_int"])
    assert torch.allclose(e[0], e[1], rtol=0, atol=1e-11) and torch.allclose(e[0], e[2], rtol=0, atol=1e-11)
    m1 = _model(tables, converge=0)
    with torch.no_grad():
        o1 = m1(bt)
    assert "ct_gap_iters" not in o1 and m1._gap_info is None
    assert float((o1["e_ct"] - o["e_ct"]).abs().max()) > 1e-3                     # the one-step energy is another one


def test_forces_and_alpha(tables):
    m = _model(tables)
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(_batch(m, [RING], pos))["e_int"].sum(), pos)[0]
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
        for hf in (1e-3, 2.5e-4):           # the energy is not quadratic in the field: the difference converges as hf^2
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
    """What the force and polarizability losses differentiate: d|F|^2 / dp and d tr(alpha) / dp by autograd against
    finite differences in a parameter (third derivatives of the energy: the converged point's own dependence on the
    parameters is in the graph)."""
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
    m = _model(tables, gap_shift=0.12)
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym, X = ["Na", "Cl", "Li", "F"] * 2, torch.cat([half, -half])
    with torch.no_grad():
        c = m(_batch(m, [(sym, X)]))
        p = {L: m(_batch(m, [(sym, X + L / 2)], cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False))
             for L in (30., 40.)}
    assert float(c["ct_gap_resid"].max()) < 1e-10 and float(p[40.]["ct_gap_resid"].max()) < 1e-10
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
    assert o["ct_gap_iters"] > 0 and float(o["ct_gap_resid"].max()) < 1e-10
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
        # the same cell again from the previous solution: converged at once, the same energy
        bt = _batch(m, [(symc, Xc)], cell=cell)
        o2 = m({**bt, "guess": o["guess"]})
        assert o2["ct_gap_iters"] == 0 and float(o2["e_int"]) == pytest.approx(float(o["e_int"]), abs=1e-10)
        assert o2["cg_iters"] < o["cg_iters"]


def test_charges_stay_two_state_where_the_expansion_does_not(tables):
    """Gaps closed well below the couplings: the one-step expansion gives charges outside [0, 1] and a positive
    E_CT; the converged charges are two-state charges and E_CT is below the sum of the lone channels' energies'
    floor -sum W."""
    kw = dict(q_boost=0.6, gap_shift=0.3, w0=0.005)
    m1, m = _model(tables, converge=0, **kw), _model(tables, converge=60, tol=1e-9, **kw)
    bt = _batch(m, [RING])
    with torch.no_grad():
        o1, o = m1(bt), m(bt)
    assert float(o1["e_ct"]) > 0 and float(o1["q_ch"].max()) > 0.9
    assert bool(m._gap_info["converged"].all())
    lam = m._gap["W2"] / (m._gap["W2"] + 0.005 ** 2)
    assert float(o["q_ch"].min()) >= 0 and bool((o["q_ch"] <= lam + 1e-12).all())
    assert float(o["e_ct"]) < 0


def test_config():
    assert CmmConfig().ct_gap_converge == 0
    with pytest.raises(ValueError):
        CmmConfig(ct_gap_converge=-1)
    c = CmmConfig.from_dict(CmmConfig(ct_model="gap", ct_gap_converge=30, ct_gap_tol=1e-8).to_dict())
    assert c.ct_gap_converge == 30 and c.ct_gap_tol == 1e-8
