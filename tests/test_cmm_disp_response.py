"""The response dispersion (``CmmConfig(disp_model="response")``): the table's Casimir-Polder coefficients; the
confinement scale factor; the free-ion limit is the bare C6 / C8 sum; Pauli confinement can only shrink the
coefficients; forces are exact (cluster and cell); a cluster in a large cell is the cluster; the uniform tail; a
checkpoint carries its table; the emitted dispersion is untouched by the option."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K
from rsfff.ff.cmm.disp_table import build_disp_table, load_disp_table, pair_scale
from rsfff.ff.cmm.io import load_cmm, resolve_params, save_cmm
from rsfff.ff.cmm.periodic import dispersion_tail_integral

PARAMS = Path(__file__).parent / "cmm_golden" / "ions_golden.json"      # Li, Na, F, Cl
f64 = torch.float64
RING = (["Cl", "Na", "Cl", "Na", "Cl", "Li"],
        torch.tensor([[2.9, 0.1, 0.0], [1.4, 2.5, 0.1], [-1.5, 2.4, -0.1], [-2.8, 0.0, 0.2], [-1.4, -2.5, 0.0],
                      [1.5, -2.2, -0.2]], dtype=f64))


@pytest.fixture(scope="module")
def tables():
    return IonTable.load(PARAMS), ChannelTable.load(PARAMS)


def _model(tables, confine="pauli", **kw):
    torch.manual_seed(3)
    m = CmmModel(*tables, CmmConfig(film=False, disp_model="response", disp_confine=confine, **kw))
    with torch.no_grad():
        m.encoder.readout.weight.copy_(0.3 * torch.randn_like(m.encoder.readout.weight))
        for p in (m.ln_w1, m.ln_w2, m.disp_ln_bscale):
            p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def _batch(m, frames, pos=None, **kw):
    pos = torch.cat([X for _, X in frames]) if pos is None else pos
    Z = torch.tensor([ATOMIC_NUMBER[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return m.precompute(pos, Z, fr, **kw)


def test_table_is_casimir_polder():
    """One-pole spectra: C6 = 3/2 a a' w w' / (w + w'), C8dq = 15/4 a1 a2' w1 w2' / (w1 + w2'), and the one-pole
    frequencies are the poles."""
    sp = {"A": {"W": [[0.5], [0.3]], "f": [[10 * 0.25], [50 * 0.09]]},          # alpha1 10, alpha2 50
          "B": {"W": [[2.0], [1.5]], "f": [[1 * 4.0], [0.0]]}}                  # alpha1 1, no quadrupole response
    t = build_disp_table(sp)
    assert t["alpha1"] == pytest.approx([10, 1]) and t["alpha2"] == pytest.approx([50, 0])
    assert t["w1"] == pytest.approx([0.5, 2.0]) and t["w2"] == pytest.approx([0.3, 2.0])
    assert t["C6"][0][1] == pytest.approx(1.5 * 10 * 1 * 0.5 * 2.0 / 2.5) and t["C6"][0][1] == pytest.approx(t["C6"][1][0])
    assert t["C6"][0][0] == pytest.approx(0.75 * 100 * 0.5)
    assert t["C8dq"][1][0] == pytest.approx(3.75 * 1 * 50 * 2.0 * 0.3 / 2.3) and t["C8dq"][0][1] == 0.0


def test_pair_scale():
    r = torch.tensor([0.3, 0.8, 1.0], dtype=f64)
    w = torch.tensor(0.4, dtype=f64)
    assert torch.allclose(pair_scale(r, r, w, 3 * w), r ** 1.5)                 # like confinement: any frequencies
    one = torch.ones_like(r)
    assert torch.allclose(pair_scale(one, one, w, 2 * w), one)
    # the one-pole coefficient with alpha -> r alpha, w -> w / sqrt(r)
    ra, rb, wa, wb = torch.tensor(0.6, dtype=f64), torch.tensor(0.9, dtype=f64), 0.4, 1.3
    lon = lambda a, b, x, y: a * b * x * y / (x + y)                             # noqa: E731
    assert float(pair_scale(ra, rb, wa, wb)) == pytest.approx(
        float(lon(ra, rb, wa / ra.sqrt(), wb / rb.sqrt()) / lon(1.0, 1.0, wa, wb)))
    assert float(pair_scale(torch.tensor(0.0, dtype=f64), rb, wa, wb)) == 0.0


def test_free_ion_limit_is_the_bare_sum(tables):
    """``disp_confine="none"``, a distant pair: -C6 / r^6 - C8 / r^8 with the table's pair coefficients."""
    m = _model(tables, "none")
    t = load_disp_table(resolve_params(m.config.disp_table), ["Na", "Cl"])
    X = torch.tensor([[0.0, 0, 0], [14.0, 0, 0]], dtype=f64)
    with torch.no_grad():
        e = float(m(_batch(m, [(["Na", "Cl"], X)]))["e_disp"])
    r = 14.0 / K.BOHR_ANG
    c6, c8 = float(t["c6"][0, 1]), float(t["c8"][0, 1] + t["c8"][1, 0])
    assert e == pytest.approx(-c6 / r ** 6 - c8 / r ** 8, rel=1e-6)
    assert float(t["c6"][1, 1]) == pytest.approx(287.2, rel=0.01)                # the free chloride's C6
    assert c6 < float((t["c6"][0, 0] * t["c6"][1, 1]).sqrt())                     # an unlike pair: below the geometric mean


@pytest.mark.parametrize("bound", ["off", "rank"])
def test_pauli_confinement_only_shrinks(tables, bound):
    m, mf = _model(tables, "pauli", xpol_bound=bound), _model(tables, "none", xpol_bound=bound)
    bt = _batch(m, [RING])
    with torch.no_grad():
        o, of = m(bt), mf(bt)
        rho = m.confinement(bt["species"], m._gram)
    assert bool((rho <= 1 + 1e-12).all()) and bool((rho > 0).all()) and float(rho.min()) < 0.99
    c6, c8 = m._disp_pairs
    c6f, c8f = mf._disp_pairs
    assert bool((c6 <= c6f * (1 + 1e-12)).all()) and bool((c8 <= c8f * (1 + 1e-12)).all())
    assert float(of["e_disp"]) < float(o["e_disp"]) < 0
    for k in ("e_pauli", "e_pol", "e_ct", "e_elst"):                             # nothing else changes
        assert torch.allclose(o[k], of[k], rtol=0, atol=1e-14)


@pytest.mark.parametrize("confine", ["pauli", "onsite"])
def test_forces(tables, confine):
    m = _model(tables, confine, xpol_bound="rank")
    sym, X = RING
    pos = X.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(_batch(m, [RING], pos))["e_disp"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for a, k in ((0, 0), (3, 1), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[a, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)]))["e_disp"]))
            assert float(F[a, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-6, abs=1e-10)


def test_parameter_gradients_reach_the_exchange_weights(tables):
    """The confinement is the exchange Gram blocks': the dispersion energy has gradients in W1, theta and the range,
    and none in the emitted model's C6."""
    m = _model(tables, "pauli", xpol_bound="rank")
    m(_batch(m, [RING]))["e_disp"].sum().backward()
    for p in (m.ln_w1, m.ln_w2, m.disp_ln_bscale, m.pauli_ln_w):
        assert p.grad is not None and float(p.grad.abs().max()) > 0
    assert m.disp_ln_c6.grad is None


def test_periodic(tables):
    """A cluster in a large cell is the cluster (no tail); forces under PBC; the tail of a uniform like-species gas."""
    m = _model(tables, "pauli", xpol_bound="rank")
    half = torch.tensor([[1.4, 0.2, 0.1], [0.3, 1.5, -0.2], [-0.2, 0.4, 2.3], [2.0, -1.6, 0.9]], dtype=f64)
    sym2, X2 = ["Na", "Cl", "Li", "F"] * 2, torch.cat([half, -half])
    with torch.no_grad():
        c = m(_batch(m, [(sym2, X2)]))
        p = m(_batch(m, [(sym2, X2 + 20.0)], cell=torch.eye(3, dtype=f64) * 40.0, cutoff=19.0, dispersion_tail=False))
    assert float(p["e_disp"]) == pytest.approx(float(c["e_disp"]), rel=1e-9)
    g = torch.Generator().manual_seed(1)
    a = 5.6
    base = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=f64) * a
    X = torch.cat([base, base + torch.tensor([a / 2, 0, 0], dtype=f64)]) + 0.15 * torch.randn(8, 3, generator=g, dtype=f64)
    sym, cell = ["Na"] * 4 + ["Cl"] * 4, torch.eye(3, dtype=f64) * a
    pos = X.clone().requires_grad_(True)
    F = -torch.autograd.grad(m(_batch(m, [(sym, X)], pos, cell=cell))["e_disp"].sum(), pos)[0]
    h = 1e-4
    with torch.no_grad():
        for at, k in ((0, 0), (5, 2)):
            e = []
            for sg in (1, -1):
                Xp = X.clone()
                Xp[at, k] += sg * h
                e.append(float(m(_batch(m, [(sym, Xp)], cell=cell))["e_disp"]))
            assert float(F[at, k]) == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=2e-6, abs=1e-10)
        # the tail: free coefficients, one species -> -2 pi / V N^2 (C6 I6 + C8 I8)
        mf = _model(tables, "none")
        Xc = torch.rand(8, 3, generator=g, dtype=f64) * 11.0
        cl = ["Cl"] * 8
        kw = dict(cell=torch.eye(3, dtype=f64) * 11.0)
        bt = _batch(mf, [(cl, Xc)], dispersion_tail=True, **kw)
        e_t = float(mf(bt)["e_disp"]) - float(mf(_batch(mf, [(cl, Xc)], dispersion_tail=False, **kw))["e_disp"])
        k_cl = mf.tab.symbols.index("Cl")
        c6, c8 = float(mf.disp_c6[k_cl, k_cl]), 2 * float(mf.disp_c8[k_cl, k_cl])
        rc, sw = bt["pbc"]["cutoff"], bt["pbc"]["switch_width"]
        I6, I8 = dispersion_tail_integral(rc, sw), dispersion_tail_integral(rc, sw, power=6)
        V = (11.0 / K.BOHR_ANG) ** 3
        assert e_t == pytest.approx(-2 * math.pi / V * 64 * (c6 * I6 + c8 * I8), rel=1e-10)
        assert I8 < I6 / (rc / K.BOHR_ANG) ** 2 * 1.5


def test_checkpoint_roundtrip(tables, tmp_path):
    m = _model(tables, "pauli")
    with torch.no_grad():
        m.disp_c6.mul_(1.25)                    # a table that is not the file's: the checkpoint's own must come back
    bt = _batch(m, [RING])
    with torch.no_grad():
        e = m(bt)["e_disp"]
    save_cmm(tmp_path / "m.pt", m, params=PARAMS)
    m2 = load_cmm(tmp_path / "m.pt")
    assert m2.config.disp_model == "response" and torch.equal(m2.disp_c6, m.disp_c6)
    with torch.no_grad():
        assert torch.allclose(m2(_batch(m2, [RING]))["e_disp"], e, rtol=1e-13)


def test_emitted_model_is_unchanged(tables):
    m = CmmModel(*tables, CmmConfig(film=False))
    assert not m.disp_response and "disp_c6" not in m.state_dict() and "disp_ln_bscale" not in m.state_dict()
    with pytest.raises(ValueError):
        CmmConfig(disp_model="c6")
    with pytest.raises(KeyError):
        load_disp_table(resolve_params("disp_wb97mv_tzvpd.json"), ["Na", "Xe"])
    assert json.load(open(resolve_params("disp_wb97mv_tzvpd.json")))["symbols"] == ["Li", "Na", "K", "Rb", "Cs", "F", "Cl", "Br", "I"]
