"""The saturating hyperpolarization (``CmmConfig(hyperpol="saturating")``): the dipole shape's self-energy u1 is the
coalescence limit of the model's own dipole-dipole block; the bounded field energy is the polynomial one at small
field and gradient, its dipole and polarizability are its derivatives, and every eigenvalue of alpha(E; T) stays in
(alpha / 2, ceiling) for any field and gradient; an ion without room responds linearly; in the model the on-site blocks obey
the ceiling, forces are exact (cluster and cell), the polarizability is the energy's second field derivative and the
ceiling fraction is learned; the Legendre model is untouched by the option."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K
from rsfff.ff.cmm.precompute import dipole_self_coulomb, hyperpolarization, saturating_hyperpolarization

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
RING = (["Cl", "Na", "Cl", "Na", "Cl", "Li"],
        torch.tensor([[2.9, 0.1, 0.0], [1.4, 2.5, 0.1], [-1.5, 2.4, -0.1], [-2.8, 0.0, 0.2], [-1.4, -2.5, 0.0],
                      [1.5, -2.2, -0.2]], dtype=f64))


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, **kw):
    torch.manual_seed(4)
    m = CmmModel(*tables, CmmConfig(film=False, hyperpol="saturating", **kw))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        m.hyper_raw.add_(0.3 * torch.randn_like(m.hyper_raw))
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def _drive(E, T):
    """The permanent drive (n, 8) whose field is E (n, 3) and whose traceless gradient tensor is T (n, 3, 3)."""
    h3 = 0.5 * 3.0 ** 0.5
    q20 = T[:, 2, 2]
    q22c = (T[:, 0, 0] - T[:, 1, 1]) / (2 * h3)
    return torch.cat([-E, torch.stack([q20, T[:, 0, 2] / h3, T[:, 1, 2] / h3, q22c, T[:, 0, 1] / h3], 1)], 1)


def _rand_T(n, g):
    X = torch.randn(n, 3, 3, generator=g, dtype=f64)
    T = 0.5 * (X + X.transpose(1, 2))
    return T - torch.eye(3, dtype=f64) * (torch.diagonal(T, dim1=1, dim2=2).sum(-1) / 3)[:, None, None]


def test_u1_is_the_coalescence_limit_of_the_dipole_block(tables):
    tab = tables[0]
    u1 = dipole_self_coulomb(tab)
    for k in range(len(tab.symbols)):
        pos = torch.tensor([[0.0, 0, 0], [0.004, 0, 0]], dtype=f64)
        _, A = K.induction_operator(pos, torch.tensor([k, k]), torch.zeros(2, dtype=torch.long), tab)
        blk = A.reshape(2, 8, 2, 8)[0, :3, 1, :3]
        assert float(blk[1, 1]) == pytest.approx(float(u1[k]), rel=2e-3) and float(blk[0, 0]) == pytest.approx(float(u1[k]), rel=2e-2)
        far = torch.tensor([[0.0, 0, 0], [1.5, 0, 0]], dtype=f64)
        _, A = K.induction_operator(far, torch.tensor([k, k]), torch.zeros(2, dtype=torch.long), tab)
        assert float(torch.linalg.eigvalsh(A.reshape(2, 8, 2, 8)[0, :3, 1, :3]).abs().max()) < float(u1[k])


def test_small_field_is_the_polynomial_form():
    a, kap, gam, b = (torch.tensor([v], dtype=f64) for v in (31.3, 9.0, 5512.0, 542.0))
    g = torch.Generator().manual_seed(0)
    E, T = 1e-3 * torch.randn(1, 3, generator=g, dtype=f64), 1e-4 * _rand_T(1, g)
    Lp = _drive(E, T)

    class Tab:
        alpha, hyper_gamma, hyper_b = a, gam, b
    ai, h, c = saturating_hyperpolarization(Lp, a, kap, gam, b)
    ai0, h0, c0 = hyperpolarization(Lp, torch.tensor([0]), Tab)
    al, al0 = torch.linalg.inv(ai), torch.linalg.inv(ai0)
    assert torch.allclose(al, al0, rtol=0, atol=0.02 * float((al0 - a * torch.eye(3, dtype=f64)).abs().max()))
    assert float((al0 - a * torch.eye(3, dtype=f64)).abs().max()) > 0.05              # the terms are there
    assert torch.allclose(h, h0, rtol=0, atol=0.05 * float(h0.abs().max())) and float(h0.abs().max()) > 0
    assert float(c) == pytest.approx(float(c0), rel=0.05) and float(c0) != 0
    # zero field and gradient: the free ion
    ai, h, c = saturating_hyperpolarization(torch.zeros(1, 8, dtype=f64), a, kap, gam, b)
    assert torch.allclose(ai, torch.eye(3, dtype=f64) / a) and float(h.abs().max()) == 0 and float(c) == 0


def test_dipole_and_polarizability_are_the_derivatives_and_are_bounded():
    """mu_p = -dW/dE, alpha = d mu_p / dE (so h and c are the Legendre transform's), and alpha / 2 < alpha(E; T) < ceiling
    for fields and gradients of any size."""
    g = torch.Generator().manual_seed(1)
    n = 4000
    a = torch.full((n,), 31.3, dtype=f64)
    kap = torch.full((n,), 9.0, dtype=f64)
    gam, b = torch.full((n,), 5512.0, dtype=f64), torch.full((n,), 542.0, dtype=f64)
    E = torch.randn(n, 3, generator=g, dtype=f64) * 10 ** (torch.rand(n, 1, generator=g, dtype=f64) * 5 - 4)
    T = _rand_T(n, g) * 10 ** (torch.rand(n, 1, 1, generator=g, dtype=f64) * 5 - 4)
    ai, h, c = saturating_hyperpolarization(_drive(E, T), a, kap, gam, b)
    ev = torch.linalg.eigvalsh(torch.linalg.inv(ai))
    assert float(ev.min()) > 0.5 * 31.3 - 1e-9 and float(ev.max()) < 31.3 + 9.0 + 1e-9
    assert float(ev.max()) > 40.0 and float(ev.min()) < 0.52 * 31.3                  # both bounds are approached
    # derivatives, at moderate fields: c = W + 1/2 mu alpha^-1 mu, h = alpha^-1 mu - E  =>  W, mu from them
    k = 50
    Ek = E[:k].clone()
    Ek = (Ek / Ek.norm(dim=1, keepdim=True) * torch.linspace(0.005, 0.3, k, dtype=f64)[:, None]).requires_grad_(True)
    Tk = _rand_T(k, g) * 0.004

    def parts(Ev):
        ai_, h_, c_ = saturating_hyperpolarization(_drive(Ev, Tk), a[:k], kap[:k], gam[:k], b[:k])
        mu_ = torch.linalg.solve(ai_, (h_ + Ev)[..., None])[..., 0]
        return c_ - 0.5 * (mu_ * (h_ + Ev)).sum(-1), mu_, torch.linalg.inv(ai_)

    W, mu, al = parts(Ek)
    assert torch.allclose(-torch.autograd.grad(W.sum(), Ek, retain_graph=True)[0], mu, rtol=1e-8, atol=1e-10)
    J = torch.stack([torch.autograd.grad(mu[:, x].sum(), Ek, retain_graph=True)[0] for x in range(3)], 1)
    assert torch.allclose(J, al, rtol=1e-7, atol=1e-8)
    # no room, or no gamma: linear
    for kw in (dict(kappa=torch.zeros(k, dtype=f64), gamma=gam[:k]), dict(kappa=kap[:k], gamma=torch.zeros(k, dtype=f64))):
        ai_, h_, c_ = saturating_hyperpolarization(_drive(Ek.detach(), Tk), a[:k], kw["kappa"], kw["gamma"], b[:k])
        assert torch.allclose(ai_, torch.eye(3, dtype=f64) / 31.3) and float(h_.abs().max()) < 1e-14 and float(c_.abs().max()) < 1e-14


@pytest.mark.parametrize("ceiling", ["global", "species"])
def test_model_blocks_obey_the_ceiling(tables, ceiling):
    m = _model(tables, hyper_ceiling=ceiling)
    assert m.hyper_raw.shape == ((4,) if ceiling == "species" else (1,))
    bt = _batch(m, [RING])
    sp = bt["species"]
    with torch.no_grad():
        hb = m.hyperpolarized(bt, sp)
        o = m(bt)
    D = bt["D"]
    on = hb["A"][0].reshape(bt["nmax"], D, bt["nmax"], D)
    blocks = torch.stack([on[k, :3, k, :3] for k in range(6)])
    ev = torch.linalg.eigvalsh(torch.linalg.inv(blocks))
    cap = (m.hyper_alpha + m.hyper_fraction() * m.hyper_room)[sp]
    assert bool((ev.min(1).values >= 0.5 * m.hyper_alpha[sp] - 1e-10).all()) and bool((ev.max(1).values <= cap + 1e-10).all())
    assert bool((cap <= 1 / m.hyper_u1[sp] + 1e-12).all())                      # inside the Gram certificate
    assert float((ev.max(1).values / m.hyper_alpha[sp]).max()) > 1.001          # and it does something here
    assert bool(torch.isfinite(o["e_int"]).all())
    # an ion with no room (alpha u1 >= 1) keeps its free-ion block
    with torch.no_grad():
        m.hyper_room[sp[0]] = 0.0
        hb = m.hyperpolarized(bt, sp)
    assert torch.allclose(hb["A"][0].reshape(bt["nmax"], D, bt["nmax"], D)[0, :3, 0, :3],
                          torch.eye(3, dtype=f64) / m.hyper_alpha[sp[0]], rtol=0, atol=1e-14)


def test_forces_alpha_and_the_learned_ceiling(tables):
    m = _model(tables)
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    out = m(_batch(m, [RING], pos))
    F = -torch.autograd.grad(out["e_int"].sum(), pos, retain_graph=True)[0]
    assert float(torch.autograd.grad(out["e_int"].sum(), m.hyper_raw)[0].abs().max()) > 0
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


def test_periodic(tables):
    m = _model(tables)
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym, X = ["Na", "Cl", "Li", "F"] * 2, torch.cat([half, -half])
    with torch.no_grad():
        c = m(_batch(m, [(sym, X)]))
        p = {L: m(_batch(m, [(sym, X + L / 2)], cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False))
             for L in (30., 40.)}
    w = 40.0 ** 5 / (40.0 ** 5 - 30.0 ** 5)
    for k in ("e_pol", "e_ct", "e_int"):
        assert float((p[30.][k] + w * (p[40.][k] - p[30.][k]) - c[k]).abs()) < 3e-7, k
    g = torch.Generator().manual_seed(1)
    a = 5.6
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=f64) * a
    Xc = torch.cat([base, base + torch.tensor([a / 2, 0, 0], dtype=f64)]) + 0.15 * torch.randn(8, 3, generator=g, dtype=f64)
    symc, cell = ["Na"] * 4 + ["Cl"] * 4, torch.eye(3, dtype=f64) * a
    pos = Xc.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(_batch(m, [(symc, Xc)], pos, cell=cell))["e_int"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for at, k in ((0, 0), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = Xc.clone()
                Xp[at, k] += sg * h
                e.append(float(m(_batch(m, [(symc, Xp)], cell=cell))["e_int"]))
            assert float(F[at, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=5e-5, abs=1e-8)


def test_legendre_model_is_unchanged(tables):
    m = CmmModel(*tables, CmmConfig(film=False))
    assert not m.hyper_sat and "hyper_raw" not in m.state_dict()
    with pytest.raises(ValueError):
        CmmConfig(hyperpol="tanh")
    with pytest.raises(ValueError):
        CmmConfig(hyper_ceiling="pair")
