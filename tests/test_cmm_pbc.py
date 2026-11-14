"""cmm under PBC: a cluster in a large box is the cluster; forces and stress are exact derivatives of the periodic
energy; the energy does not depend on the pair cutoff (beyond the short-range reach) or on a translation; a warm
start changes nothing but the iteration count."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
TERMS = ("e_elst", "e_pauli", "e_disp", "e_pol", "e_xpol", "e_ct", "e_int")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(5)
    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    m = CmmModel(tab, chan, CmmConfig(film=False))
    with torch.no_grad():                       # an environment that does something: random readouts and constants
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.ct_ln_q0, m.ct_acc_raw):
            p.add_(0.1 * torch.randn_like(p))
    m.eval()
    return m


def _Z(sym):
    return torch.tensor([ATOMIC_NUMBER[a] for a in sym])


def _run(m, X, sym, cell=None, **kw):
    fr = torch.zeros(len(sym), dtype=torch.long)
    return m(m.precompute(X, _Z(sym), fr, cell=cell, **kw))


def _rocksalt(a=5.5, jitter=0.15, seed=1):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=f64) * a
    X = torch.cat([base, base + torch.tensor([a / 2, 0, 0], dtype=f64)])
    X = X + jitter * torch.randn(X.shape, generator=g, dtype=f64)
    return X, ["Na"] * 4 + ["Cl"] * 4, torch.eye(3, dtype=f64) * a


def test_cluster_in_a_large_box_is_the_cluster(model):
    """An inversion-symmetric neutral cluster (no dipole: its images interact from the quadrupoles on) centred in
    cubes of 30 and 40 A: every term differs from the cluster's by the images' 1/L^5 interaction, and the
    extrapolation to L -> infinity is the cluster."""
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym = ["Na", "Cl", "Li", "F"] * 2
    X = torch.cat([half, -half])
    with torch.no_grad():
        c = _run(model, X, sym)
        p = {L: _run(model, X + L / 2, sym, cell=torch.eye(3, dtype=f64) * L, dispersion_tail=False) for L in (30., 40.)}
    assert float(c["e_ct"].abs()) > 1e-4 and float(c["e_xpol"].abs()) > 1e-5      # the terms are doing something
    w = 40.0 ** 5 / (40.0 ** 5 - 30.0 ** 5)
    for k in TERMS:
        d30, d40 = float(p[30.][k] - c[k]), float(p[40.][k] - c[k])
        assert abs(d40) < 1e-5, (k, d40)
        assert abs(w * d40 + (1 - w) * d30) < 1e-7, (k, d30, d40)       # Richardson: the 1/L^5 term removed
    dq = (p[40.]["dq"] - c["dq"].reshape(-1)).abs().max()
    assert float(dq) < 1e-4 * float(c["dq"].abs().max()), float(dq)


def test_periodic_forces_and_stress_are_exact(model):
    X, sym, cell = _rocksalt()
    Xg = X.clone().requires_grad_(True)
    eps = torch.zeros(3, 3, dtype=f64, requires_grad=True)
    F = torch.eye(3, dtype=f64) + eps
    E = _run(model, Xg @ F, sym, cell=cell @ F)["e_int"].sum()
    gx, ge = torch.autograd.grad(E, (Xg, eps))

    def energy(x, e):
        Fe = torch.eye(3, dtype=f64) + e
        with torch.no_grad():
            return float(_run(model, x @ Fe, sym, cell=cell @ Fe)["e_int"].sum())

    h = 1e-4
    for a, b in ((0, 0), (5, 2), (7, 1)):
        xp, xm = X.clone(), X.clone()
        xp[a, b] += h
        xm[a, b] -= h
        fd = (energy(xp, torch.zeros(3, 3, dtype=f64)) - energy(xm, torch.zeros(3, 3, dtype=f64))) / (2 * h)
        assert abs(fd - float(gx[a, b])) < 2e-7 * max(1.0, abs(fd)), (a, b, fd, float(gx[a, b]))
    for a, b in ((0, 0), (1, 2), (2, 2)):
        ep, em = torch.zeros(3, 3, dtype=f64), torch.zeros(3, 3, dtype=f64)
        ep[a, b], em[a, b] = h, -h
        fd = (energy(X, ep) - energy(X, em)) / (2 * h)
        assert abs(fd - float(ge[a, b])) < 2e-6 * max(1.0, abs(fd)), (a, b, fd, float(ge[a, b]))


def test_cutoff_and_translation_independence(model):
    """Past the default cutoff only the table's far complements (relative size < 1e-6, times r^-(2n+1)) and the shell
    penetration tails change: ~1e-6 Hartree for this dense cell."""
    X, sym, cell = _rocksalt()
    from rsfff.ff.cmm.precompute import default_cutoff
    rc = default_cutoff(model.table(), model.chan, model.config.env_gate)
    with torch.no_grad():
        a = _run(model, X, sym, cell=cell)
        b = _run(model, X, sym, cell=cell, cutoff=rc + 1.5)
        c = _run(model, X + torch.tensor([0.37, -1.2, 2.9], dtype=f64), sym, cell=cell)
    for k in ("e_elst", "e_pauli", "e_pol", "e_xpol", "e_ct"):
        assert abs(float(a[k] - b[k])) < 5e-6, (k, float(a[k] - b[k]))
    for k in TERMS:
        assert abs(float(a[k] - c[k])) < 1e-9 * max(1.0, abs(float(a[k]))), k


def test_warm_start(model):
    X, sym, cell = _rocksalt()
    fr = torch.zeros(len(sym), dtype=torch.long)
    with torch.no_grad():
        bt = model.precompute(X, _Z(sym), fr, cell=cell)
        a = model(bt)
        bt2 = model.precompute(X + 0.002, _Z(sym), fr, cell=cell)
        cold = model(dict(bt2))
        bt2["guess"] = a["guess"]
        warm = model(bt2)
    for k in TERMS:
        assert abs(float(cold[k] - warm[k])) < 1e-10, k
    assert warm["cg_iters"] < cold["cg_iters"]


def test_supercell_is_extensive(model):
    """A 2 x 2 x 2 supercell of a (distorted) cell is the same crystal: eight times every energy term (the images,
    self-image channels and Ewald sums counted right)."""
    X, sym, cell = _rocksalt(jitter=0.2)
    sh = torch.tensor([[a, b, c] for a in range(2) for b in range(2) for c in range(2)], dtype=f64) @ cell
    X2 = (X[None] + sh[:, None]).reshape(-1, 3)
    with torch.no_grad():
        e1 = _run(model, X, sym, cell=cell)
        e2 = _run(model, X2, sym * 8, cell=2 * cell)
    for k in TERMS:
        assert abs(float(e2[k] - 8 * e1[k])) < 1e-8 * max(1.0, abs(float(e2[k]))), (k, float(e2[k]), 8 * float(e1[k]))
