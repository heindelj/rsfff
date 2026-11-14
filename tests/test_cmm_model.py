"""CmmModel: the precompute agrees with the exact per-frame quantities (induction operator, probe environment) and
treats pairs beyond the table as bare multipoles; the forces are exact derivatives; an isolated ion has no energy and
no environment; a lone ion pair's CT is exactly its kernel energy; the joint induction + charge-flux solution is
stationary; the exact polarizability is the finite-field one; the Cholesky solve is the LU one. (The reproduction of
the trained complete model is tests/test_cmm_golden.py.)"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, DensityEnvironment, IonTable
from rsfff.ff.cmm import kernels as K

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(5)
    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    m = CmmModel(tab, chan, CmmConfig(film=False))
    table = m.table()
    with torch.no_grad():                       # an environment that does something: random readouts and constants
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw):
            p.add_(0.1 * torch.randn_like(p))
    frames = [(["Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0.1, 0]], dtype=torch.float64)),
              (["Na", "Cl", "Na", "Cl"], torch.tensor([[0.0, 0, 0], [2.8, 0, 0], [2.8, 2.9, 0.1], [0, 2.75, 0.2]],
                                                       dtype=torch.float64)),
              (["Li", "F", "Cl", "Na", "F"], torch.tensor([[0.0, 0, 0], [1.9, 0, 0.1], [-2.6, 0.3, 0], [0.2, 3.1, 0.4],
                                                           [-1.0, -2.2, 1.4]], dtype=torch.float64))]
    return tab, chan, table, m, frames


def _batch(frames, tab, table, chan, pos=None):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return _MODEL[0].precompute(pos, Z, fr)


_MODEL = []


@pytest.fixture(autouse=True)
def _model(setup):
    _MODEL[:] = [setup[3]]


def test_precompute_matches_the_exact_quantities(setup):
    """The drive, the operator's couplings and the probes' environment against the closed-form / quadrature
    per-frame evaluation (``kernels.induction_operator``, ``DensityEnvironment``): the table's interpolation error."""
    tab, chan, table, m, frames = setup
    feat = DensityEnvironment(tab)
    for sym, X in frames:
        bt = _batch([(sym, X)], tab, table, chan)
        n, D = len(sym), bt["D"]
        Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
        s, b = tab.species_index(Z), torch.zeros(n, dtype=torch.long)
        L, A_off = K.induction_operator(X, s, b, tab)
        A = bt["A"][0].reshape(n, D, n, D)[:, :8, :, :8]
        for k in range(n):
            A[k, :, k, :] = 0.0
        assert torch.allclose(A.reshape(8 * n, 8 * n), A_off, atol=1e-7)
        assert torch.allclose(bt["Lz"][0, :, 3:8], L[:, 3:8], atol=1e-7)       # the dipole rows carry the offset h
        env = feat(positions=X, atomic_numbers=Z, batch_idx=b).blocks
        gv = bt["gated"].reshape(n, 3, 5)
        for q, p in enumerate(("donor", "acceptor", "breath")):
            has = tab.a0[s] > 0 if p == "breath" else torch.ones(n, dtype=torch.bool)
            assert torch.allclose(gv[:, q, 0], env[p]["S0"] * has, atol=1e-7), p
            assert torch.allclose(gv[:, q, 1], env[p]["V0"] * has, atol=1e-7), p
        if D > 8:
            assert torch.allclose(bt["Lz"][0, :, 8], -env["breath"]["V0"] * (tab.a0[s] > 0), atol=1e-7)


def test_far_pairs_are_bare_multipoles(setup):
    """A frame with a pair beyond the table's reach: the same energies as a table that reaches it."""
    tab, chan, table, m, _ = setup
    X = torch.tensor([[0.0, 0, 0], [2.7, 0, 0], [13.5, 0.4, 0], [16.2, 0.4, 0.3]], dtype=torch.float64)
    fr = [(["Na", "Cl", "Na", "Cl"], X)]
    with torch.no_grad():
        a = m(_batch(fr, tab, table, chan))
        r_max = table.r_max
        table.r_max = 100.0                    # every pair "inside": the table returns zeros past its grid anyway
        try:
            b = m(_batch(fr, tab, table, chan))
        finally:
            table.r_max = r_max
    for k in ("e_pol", "e_ct", "e_int"):
        assert torch.allclose(a[k], b[k], rtol=1e-12, atol=1e-15), k


def test_forces_are_exact_derivatives(setup):
    tab, chan, table, m, frames = setup
    pos = torch.cat([X for _, X in frames])

    def E(p):
        return m(_batch(frames, tab, table, chan, p))["e_int"].sum()

    pg = pos.clone().requires_grad_(True)
    g = torch.autograd.grad(E(pg), pg)[0]
    h = 1e-4
    with torch.no_grad():
        for a, b in ((0, 0), (3, 1), (6, 2), (9, 0)):
            pp, pm = pos.clone(), pos.clone()
            pp[a, b] += h
            pm[a, b] -= h
            assert abs((E(pp) - E(pm)) / (2 * h) - g[a, b]) < 1e-7


def test_isolated_ions(setup):
    tab, chan, table, m, _ = setup
    frames = [([x], torch.zeros(1, 3, dtype=torch.float64)) for x in ("Li", "Na", "F", "Cl")]
    with torch.no_grad():
        out = m(_batch(frames, tab, table, chan))
    assert torch.allclose(out["e_int"], torch.zeros(4, dtype=torch.float64), atol=1e-14)
    assert all(float(v.abs().max()) == 0.0 for v in m._u.values())


def test_lone_pair_ct_is_the_kernel(setup):
    """A Na-Cl pair at zero field: without the charges' coupling to the induced moments, E_CT = e_c + e_c' of its two
    channels exactly (the drives' calibration); with it, the moments' response to the moved charge lowers the CT
    (part of the pair's CT is then booked as induction-coupled relaxation, E_CT < e_c + e_c' in size)."""
    tab, chan, table, m, frames = setup
    bt = _batch(frames[:1], tab, table, chan)
    with torch.no_grad():
        full = m(bt)
    with torch.no_grad():
        bare = m({**bt, "Ca": torch.zeros_like(bt["Ca"])})
    assert bare["e_kernel"].shape[0] == 2
    assert float(bare["e_ct"][0]) == pytest.approx(float(bare["e_kernel"].sum()), rel=1e-10)
    assert float(bare["e_kernel"].sum()) < float(full["e_ct"][0]) < 0


def test_joint_solution_is_stationary(setup):
    """E(z, q) of the module docstring is stationary at the returned joint moments and channel charges."""
    tab, chan, table, m, frames = setup
    bt = _batch(frames[1:2], tab, table, chan)
    with torch.no_grad():
        out = m(bt)
    sp, B, nmax = bt["species"], bt["B"], bt["nmax"]
    i, j = bt["ch_index"]
    q = out["q_ch"]
    # the charge pattern and the energy as an explicit function of (z, q) around the solution
    D = 8 + m.n_b
    fr, loc = m._local(bt)
    Ca = bt["Ca"]
    Hq = bt["Jq"].index_put((fr, loc, loc), torch.nn.functional.softplus(m.ct_eta_raw[sp]), accumulate=True)
    li, lj = loc[i], loc[j]
    cf = bt["ch_frame"]
    h = Hq[cf, li, li] + Hq[cf, lj, lj] - Hq[cf, li, lj] - Hq[cf, lj, li]
    e_c, V, s = m.channel_drives(bt, sp, m._q, h)
    G = torch.zeros(B, nmax, q.shape[0], dtype=q.dtype).index_put((cf, li, torch.arange(q.shape[0])), torch.ones_like(q))
    G = G.index_put((cf, lj, torch.arange(q.shape[0])), -torch.ones_like(q), accumulate=True)
    # stationarity in q: b q - V + G^T (H_Q Q + C^T z) = 0, with Q = G q and z the joint moments (all of D)
    z = out["x"]
    zf = torch.cat([z, torch.zeros(B, nmax, D - 8, dtype=z.dtype)], -1) if D > 8 else z
    # the breathing amplitudes are not returned in x: take them from the joint solve (z_pol - W Q) instead
    Q = (G @ q[None, :, None])[..., 0]
    grad_q = q / (s * s) - V + (G.transpose(1, 2) @ (Hq @ Q[..., None]))[0, :, 0]
    grad_q = grad_q + (G.transpose(1, 2) @ torch.einsum("bnk,bn->bk", Ca, zf.reshape(B, -1))[..., None])[0, :, 0]
    if D > 8:          # the breathing slots carry no charge coupling (Ca has zero rows there), so x suffices
        pass
    assert float(grad_q.detach().abs().max()) < 1e-9 * float(V.detach().abs().max())
    assert torch.allclose(out["dq"], Q[0, :4], atol=1e-14)


def test_alpha_is_the_finite_field_curvature(setup):
    """The exact polarizabilities (extra right-hand sides) against -d2E/dF2 by central differences: the energy is
    quadratic in the field, so the stencil is exact up to rounding."""
    tab, chan, table, m, frames = setup
    bt = _batch(frames, tab, table, chan)
    B = bt["B"]
    with torch.no_grad():
        out = m(bt, alpha=True)
        h = 1e-3
        E = {}
        for a in range(3):
            for sg in (1, -1):
                F = torch.zeros(B, 3, dtype=torch.float64)
                F[:, a] = sg * h
                o = m(bt, field=F)
                E[a, sg] = (o["e_pol"], o["e_pol"] + o["e_ct"])
        o0 = m(bt)
        e0 = (o0["e_pol"], o0["e_pol"] + o0["e_ct"])
    for k, key in enumerate(("alpha_pol", "alpha")):
        fd = torch.stack([-(E[a, 1][k] + E[a, -1][k] - 2 * e0[k]) / h ** 2 for a in range(3)], 1)
        an = torch.diagonal(out[key], dim1=1, dim2=2)
        assert torch.allclose(an, fd, rtol=1e-5, atol=1e-5), key
        assert torch.allclose(out[key], out[key].transpose(1, 2), atol=1e-10), key
    assert bool((torch.diagonal(out["alpha"] - out["alpha_pol"], dim1=1, dim2=2) > -1e-10).all())    # more freedom


def test_cholesky_is_the_lu_solve(setup, monkeypatch):
    tab, chan, table, m, frames = setup
    bt = _batch(frames, tab, table, chan)
    with torch.no_grad():
        a = m(bt)
        monkeypatch.setattr(torch.linalg, "cholesky_ex", lambda M: (M, torch.ones(M.shape[0], dtype=torch.int32)))
        b = m(bt)
    for k in ("e_pol", "e_xpol", "e_ct", "x", "dq"):
        assert torch.allclose(a[k], b[k], rtol=1e-10, atol=1e-13), k
