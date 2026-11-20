"""No kinks from cutoffs (opt-in, the defaults are the legacy forms): the step of the overlap windows
(``CmmConfig.env_switch``: "cubic" C^1, "quintic" C^2, "smooth" C^infinity), the exchange pairs' overlap window
(``env_overlap``) and the one-sided CT charges without the clip at the free-ion value (``ct_q_headroom``)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm.channels import SWITCHES, overlap_switch, switch_step

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, **kw):
    torch.manual_seed(3)
    m = CmmModel(*tables, CmmConfig(film=False, ct_model="gap", ct_gap_overlap="pauli", ct_gap_w0=0.005,
                                    ct_gap_converge=40, ct_gap_tol=1e-12, **kw))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.5 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw, m.ct_ln_kov):
            p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def _scan(m, a, b, R):
    n = R.shape[0]
    pos = torch.zeros(n, 2, 3, dtype=f64)
    pos[:, 1, 2] = R
    Z = torch.tensor([ATOMIC_NUMBER[a], ATOMIC_NUMBER[b]] * n)
    return m.precompute(pos.reshape(-1, 3), Z, torch.arange(n).repeat_interleave(2))


def _jump(y, order, w=12, skip=3):
    """The largest increment of the order-th difference of y relative to the median increment around it (w points on
    either side, the nearest ``skip`` left out): ~1 for a smooth function, large where its order-th derivative
    jumps."""
    d = torch.diff(y, n=order + 1).abs()
    win = d.unfold(0, 2 * w + 1, 1)                                   # (n - 2 w, 2 w + 1), centre at column w
    near = torch.cat([win[:, :w - skip], win[:, w + skip + 1:]], 1)
    return float((win[:, w] / (near.median(1).values + 1e-300 + 1e-12 * d.max())).max())


# -- the step ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", SWITCHES)
def test_step_values_and_monotone(kind):
    t = torch.linspace(-0.5, 1.5, 4001, dtype=f64)
    s = switch_step(t, kind)
    assert bool((s[t <= 0] == 0).all()) and bool((s[t >= 1] == 1).all())
    assert bool((torch.diff(s) >= 0).all()) and abs(float(switch_step(torch.tensor(0.5, dtype=f64), kind)) - 0.5) < 1e-15
    u = torch.linspace(0.01, 0.99, 99, dtype=f64)
    assert torch.allclose(switch_step(u, kind) + switch_step(1 - u, kind), torch.ones_like(u), atol=1e-14)


def test_step_derivatives_at_the_ends():
    """cubic: s'' = 6 at t -> 0+ (a jump from 0); quintic: s'' -> 0, s''' = 60; smooth: every derivative -> 0."""
    def ders(kind, t0, n=4):
        t = torch.tensor(t0, dtype=f64, requires_grad=True)
        y, out = switch_step(t, kind), []
        for _ in range(n):
            y, = torch.autograd.grad(y, t, create_graph=True)
            out.append(float(y))
        return out
    c, q, s = ders("cubic", 1e-9, 2), ders("quintic", 1e-9, 3), ders("smooth", 2e-3)
    assert abs(c[1] - 6) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2] - 60) < 1e-5
    assert max(abs(v) for v in s) < 1e-150
    assert max(abs(v) for v in ders("smooth", 1 - 2e-3)) < 1e-150
    for kind in SWITCHES:                                   # finite gradients at and beyond the ends
        t = torch.tensor([-1.0, 0.0, 1e-300, 5e-4, 0.3, 1 - 5e-4, 1.0, 2.0], dtype=f64, requires_grad=True)
        g, = torch.autograd.grad(switch_step(t, kind).sum(), t, create_graph=True)
        g2, = torch.autograd.grad(g.sum(), t)
        assert bool(torch.isfinite(g).all()) and bool(torch.isfinite(g2).all())
        assert float(g[0]) == float(g[1]) == float(g[-1]) == 0.0


def test_overlap_switch_window():
    O = torch.tensor([-1e-9, 0.0, 1e-8, 1e-7, 1e-5, 1e-4, 1e-3], dtype=f64)
    for kind in SWITCHES:
        s = overlap_switch(O, 1e-7, 1e-4, kind)
        assert s[:4].tolist() == [0, 0, 0, 0] and s[5:].tolist() == [1, 1] and 0 < float(s[4]) < 1
    assert torch.equal(overlap_switch(O), overlap_switch(O, 1e-7, 1e-6, "cubic"))          # the default is the legacy


def test_config_validation(tables):
    for kw in (dict(env_switch="linear"), dict(ct_q_headroom=1.5), dict(ct_q_headroom=-0.1),
               dict(env_overlap=(1e-8, 1e-6)), dict(env_overlap=(1e-5, 1e-6))):
        with pytest.raises(ValueError):
            CmmConfig(**kw)
    c = CmmConfig()
    assert (c.env_switch, c.env_overlap, c.ct_q_headroom) == ("cubic", (1e-7, 1e-6), 0.0)
    d = CmmConfig(env_switch="smooth", env_overlap=[1e-7, 1e-4], ct_q_headroom=0.2).to_dict()
    assert CmmConfig.from_dict(d).env_overlap == (1e-7, 1e-4) and CmmConfig.from_dict(d).env_switch == "smooth"


# -- the windows in the precompute -----------------------------------------------------------------------------------
def test_gated_invariants_have_no_curvature_jump(tables):
    """Along a Na-Cl scan through the probe gates: the cubic step leaves jumps in the second derivative of the gated
    invariants (and so of every readout), the quintic only in the third, the smooth step in none."""
    R = torch.arange(3.0, 7.0, 0.004, dtype=f64)
    G = {}
    for kind in SWITCHES:
        m = _model(tables, env_switch=kind)
        bt = _scan(m, "Na", "Cl", R)
        assert bt["switch"] == kind
        G[kind] = bt["gated"].reshape(R.shape[0], 2, -1)
    h = float(R[1] - R[0])

    def rel(y, k):          # the largest change of the k-th derivative between neighbouring points over its size
        D = torch.diff(y, n=k) / h ** k
        return float(torch.diff(D).abs().max() / D.abs().max())
    for c in (2, 12):                                               # |V1g|^2 of the anion's donor and breath probes
        j = {k: (rel(v[:, 1, c], 2), rel(v[:, 1, c], 3)) for k, v in G.items()}
        assert j["cubic"][0] > 0.1 and max(j["quintic"][0], j["smooth"][0]) < 0.03, j
        assert j["quintic"][1] > 0.15 and j["smooth"][1] < 0.1, j
    far = R > 6.9                                                  # the same values where every window is closed/open
    assert torch.allclose(G["cubic"][far][:, :, [0, 5, 10]], G["smooth"][far][:, :, [0, 5, 10]], rtol=0, atol=1e-14)


def test_batch_and_model_must_agree(tables):
    m, m2 = _model(tables), _model(tables, env_switch="smooth", env_overlap=(1e-7, 1e-4))
    R = torch.tensor([2.6, 3.4], dtype=f64)
    bt, bt2 = _scan(m, "Na", "Cl", R), _scan(m2, "Na", "Cl", R)
    assert (bt["switch"], bt["overlap"]) == ("cubic", (1e-7, 1e-6)) and bt2["overlap"] == (1e-7, 1e-4)
    with torch.no_grad():
        m(bt), m2(bt2)
        with pytest.raises(ValueError, match="switch"):
            m(bt2)
        with pytest.raises(ValueError, match="overlap"):
            _model(tables, env_overlap=(1e-7, 1e-4))(bt)


# -- the one-sided CT charges ----------------------------------------------------------------------------------------
def test_headroom_map(tables):
    h, m = 0.2, _model(tables, ct_q_headroom=0.2)
    c = m.config.ct_q_cap
    u = torch.linspace(-40, 40, 8001, dtype=f64, requires_grad=True)
    sp = torch.zeros(u.shape[0], dtype=torch.long)
    m._u = {"qD": u, "qA": u.detach().flip(0)}
    f = torch.log(m.ct_charges(sp)[:, 0]) - m.ct_ln_q0[0, 0].detach()
    assert abs(float(f[4000])) < 1e-14                                  # the free ion keeps its charge
    assert float(f.max()) <= h + 1e-12 and abs(float(f[-1]) - h) < 1e-6    # the ceiling e^h q0
    assert float(f.min()) >= h - c - 1e-12 and abs(float(f[0]) - (h - c)) < 1e-6
    g, = torch.autograd.grad(f.sum(), u, create_graph=True)
    assert bool((torch.diff(f.detach()) >= 0).all())
    assert abs(float(g[4000]) - 0.5 * (1 - (h / c) ** 2)) < 1e-12        # a gradient at the start of training
    g2, = torch.autograd.grad(g.sum(), u)
    assert _jump(f.detach()[3900:4100], 1) < 3 and float(g2.abs().max()) < 2.0     # no kink at u = 0
    # the legacy clip: f = min(c tanh(u / c), 0), slope 1 at 0- and 0 at 0+
    m0 = _model(tables)
    m0._u = {"qD": u.detach(), "qA": u.detach()}
    f0 = torch.log(m0.ct_charges(sp)[:, 0]) - m0.ct_ln_q0[0, 0].detach()
    assert torch.allclose(f0, (c * torch.tanh(u.detach() / c)).clamp(max=0.0), atol=1e-14)


def _prescribed(m, r0, slope=1.5):
    """Replace the CT readouts of ``m`` by u = slope (r - r0), r the distance to the partner in a dimer frame: a
    sign change at a known distance, differentiable in the positions."""
    env = m.environment

    def environment(bt):
        u = dict(env(bt))
        X = bt["pos"].reshape(-1, 2, 3)
        r = (X[:, 1] - X[:, 0]).norm(dim=1).repeat_interleave(2)
        u["qD"] = u["qA"] = slope * (r - r0)
        m._u = u
        return u
    m.environment = environment
    return m


def test_no_force_jump_where_a_readout_changes_sign(tables):
    """Na-Cl scans on which the CT readouts cross zero at 3.1 A: with the clip E_CT has a kink there (the force
    jumps); with the headroom it has none, and the force is the derivative of the energy across the crossing."""
    R = torch.arange(2.6, 3.6, 0.004, dtype=f64)
    e = {}
    for h in (0.0, 0.2):
        m = _prescribed(_model(tables, ct_q_headroom=h, env_switch="smooth"), 3.1)
        with torch.no_grad():
            e[h] = m(_scan(m, "Na", "Cl", R))["e_ct"]
    assert _jump(e[0.0], 1) > 30 and _jump(e[0.2], 1) < 3, (_jump(e[0.0], 1), _jump(e[0.2], 1))
    k = int((R - 3.1).abs().argmin())
    d2 = torch.diff(e[0.0], n=2)
    assert int(d2.abs()[k - 20:k + 20].argmax()) + k - 20 in (k - 2, k - 1, k)       # the kink is at the crossing
    m = _prescribed(_model(tables, ct_q_headroom=0.2, env_switch="smooth", env_overlap=(1e-7, 1e-4)), 3.1)
    d = 1e-4
    Rs = torch.tensor([3.1 - d, 3.1, 3.1 + d], dtype=f64, requires_grad=True)
    ei = m(_scan(m, "Na", "Cl", Rs))["e_int"]
    g, = torch.autograd.grad(ei[1], Rs)
    fd = float((ei[2] - ei[0]) / (2 * d))
    assert abs(float(g[1]) - fd) < 1e-6 * max(1e-3, abs(fd))
