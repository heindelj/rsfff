"""rsfff.ff.ionct.ct: CT channels between ions (pair functions, arrowhead roots, environment shifts)."""
import json
from pathlib import Path

import pytest
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.units import BOHR_ANG

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"
HAS_CHANNELS = PARAMS.exists() and "channels" in json.load(open(PARAMS))
pytestmark = pytest.mark.skipif(not HAS_CHANNELS, reason="params with a 'channels' section not built")
DT = torch.float64


@pytest.fixture(scope="module")
def tabs():
    return IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)


def test_pair_function_interpolates_and_extrapolates_linearly():
    R = torch.linspace(1.0, 4.0, 13, dtype=DT)
    y = torch.exp(-R) + 0.1 * R ** 2
    f = CT.PairFunction(R, y)
    assert torch.allclose(f(R), y, atol=1e-12)
    r = torch.tensor([0.5, 0.9, 4.2, 5.0], dtype=DT, requires_grad=True)
    v = f(r)
    g, = torch.autograd.grad(v.sum(), r)
    assert torch.allclose(g[:2], f.slope_lo.expand(2)) and torch.allclose(g[2:], f.slope_hi.expand(2))
    # C1 at the ends
    eps = 1e-7
    for x0 in (1.0, 4.0):
        a = f(torch.tensor([x0 - eps], dtype=DT))
        b = f(torch.tensor([x0 + eps], dtype=DT))
        assert abs((b - a).item()) < 1e-6


def test_arrowhead_root_is_the_lowest_eigenvalue():
    torch.manual_seed(1)
    n_ch = 5
    W = torch.rand(n_ch, dtype=DT) * 0.01
    D = 0.1 + torch.rand(n_ch, dtype=DT)
    donor = torch.zeros(n_ch, dtype=torch.long)
    eps, Df = CT.arrowhead_roots(W, D, donor, 1)
    assert torch.allclose(Df, D, atol=1e-12)            # the gap floor (0.02 Eh) leaves gaps above 0.1 alone
    H = torch.zeros(n_ch + 1, n_ch + 1, dtype=DT)
    H[0, 1:] = H[1:, 0] = W.sqrt()
    H[1:, 1:] = torch.diag(Df)
    assert eps[0].item() == pytest.approx(torch.linalg.eigvalsh(H)[0].item(), abs=1e-12)


@pytest.mark.parametrize("name,R", [("Cl>Na", 2.4), ("Cl>Na", 2.8), ("Cl>Cl", 3.8)])
def test_isolated_pair_reproduces_its_calibration(tabs, name, R):
    tab, chan = tabs
    t = json.load(open(PARAMS))["channels"]["types"][name]
    k = min(range(len(t["R"])), key=lambda m: abs(t["R"][m] - R))
    R = t["R"][k]
    D, A = name.split(">")
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]], dtype=DT)
    s = tab.species_index(torch.tensor([{"Na": 11, "Cl": 17}[D], {"Na": 11, "Cl": 17}[A]]))
    b = torch.zeros(2, dtype=torch.long)
    c = torch.full((2,), t["confinement"][k], dtype=DT)
    x = K.induction(X, s, b, tab, alpha_scale=c, c_scale=c).x
    res = CT.ct_energy(X, s, b, tab, chan, x_ind=x)
    m = [n == name and int(res.index[0, q]) == 0 for q, n in enumerate(res.type)].index(True)
    assert abs(res.delta[m].item()) < 1e-9                      # no environment shift, W = W_pair
    W = torch.exp(torch.tensor(t["lnW"][k]))
    Dl = t["g"] + t["dphi"][k] + t["dov"][k]
    eps_ref = 0.5 * (Dl - (Dl * Dl + 4 * W) ** 0.5)
    assert res.W[m].item() == pytest.approx(W.item(), rel=1e-10)
    assert res.Delta[m].item() == pytest.approx(Dl, abs=1e-9)
    assert res.eps[0].item() == pytest.approx(eps_ref.item(), rel=1e-8)


def test_isolated_ion_has_no_ct(tabs):
    tab, chan = tabs
    res = CT.ct_energy(torch.zeros(1, 3, dtype=DT), torch.tensor([1]), torch.zeros(1, dtype=torch.long), tab, chan)
    assert res.energy.item() == 0.0 and res.index.shape[1] == 0


def test_uniform_field_shifts_every_gap_by_F_dot_r(tabs):
    tab, chan = tabs
    X = torch.tensor([[0.0, 0.0, 0.0], [2.6, 0.4, 0.2], [0.3, 2.7, -0.1]], dtype=DT)
    s = torch.tensor([0, 1, 1])
    b = torch.zeros(3, dtype=torch.long)
    F = torch.tensor([[0.002, -0.001, 0.003]], dtype=DT)
    i0, _, _, D0, _ = CT.ct_channels(X, s, b, tab, chan)
    i1, _, _, D1, _ = CT.ct_channels(X, s, b, tab, chan, field=F)
    shift = ((X[i0[1]] - X[i0[0]]) / BOHR_ANG) @ F[0] * chan.s
    assert torch.allclose(D1 - D0, shift, atol=1e-12)


def test_ct_forces_and_double_backward(tabs):
    tab, chan = tabs
    X0 = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.3, 0.1], [0.2, 2.6, -0.3], [2.7, 2.4, 0.4]], dtype=DT)
    s = torch.tensor([0, 1, 1, 0])
    b = torch.zeros(4, dtype=torch.long)
    F = torch.tensor([[0.0, 0.003, -0.002]], dtype=DT)
    ov = torch.tensor(1.3, dtype=DT)

    def energy(X):
        x = K.induction(X, s, b, tab, field=F).x
        return CT.ct_energy(X, s, b, tab, chan, x_ind=x, field=F, ov_scale=ov).energy.sum()

    X = X0.clone().requires_grad_(True)
    g, = torch.autograd.grad(energy(X), X, create_graph=True)
    h = 1e-5
    for a in range(4):
        for c in range(3):
            Xp, Xm = X0.clone(), X0.clone()
            Xp[a, c] += h
            Xm[a, c] -= h
            fd = (energy(Xp) - energy(Xm)).item() / (2 * h)
            assert g[a, c].item() == pytest.approx(fd, rel=1e-5, abs=1e-10)
    (g ** 2).sum().backward()
    assert torch.isfinite(X.grad).all()


def test_ct_polarizability_of_the_pair_is_positive(tabs):
    tab, chan = tabs
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.8]], dtype=DT)
    s = torch.tensor([0, 1])
    b = torch.zeros(2, dtype=torch.long)

    def e(Fz):
        F = torch.tensor([[0.0, 0.0, Fz]], dtype=DT)
        x = K.induction(X, s, b, tab, field=F).x
        return CT.ct_energy(X, s, b, tab, chan, x_ind=x, field=F).energy.item()

    h = 2e-3
    a_par = -(e(h) + e(-h) - 2 * e(0.0)) / h ** 2
    assert 0.0 < a_par < 100.0


def test_additive_overlap_gap_shifts_the_gaps_and_lowers_ct(tabs):
    tab, chan = tabs
    X = torch.tensor([[0.0, 0.0, 0.0], [2.6, 0.4, 0.2], [0.3, 2.7, -0.1]], dtype=DT)
    s = torch.tensor([0, 1, 1])
    b = torch.zeros(3, dtype=torch.long)
    idx, _, W0, D0, _ = CT.ct_channels(X, s, b, tab, chan)
    add = torch.linspace(0.0, 0.3, idx.shape[1], dtype=DT)
    _, _, W1, D1, _ = CT.ct_channels(X, s, b, tab, chan, ov_add=add)
    assert torch.allclose(D1 - D0, add) and torch.allclose(W1, W0)
    e0 = CT.ct_energy(X, s, b, tab, chan).energy
    e1 = CT.ct_energy(X, s, b, tab, chan, ov_add=add).energy
    assert (e1 > e0).all()


def test_smooth_wkb_is_two_sided_near_zero_and_bounded():
    x = torch.linspace(-50, 50, 2001, dtype=DT)
    y = CT.wkb_response(x, "smooth:4,1")
    assert y.min() >= -4.0 and y.max() <= 1.0
    small = x.abs() < 0.1
    assert ((y - x)[small].abs() <= x[small].abs() ** 3 / 3 + 1e-12).all()      # x - a tanh(x/a) ~ x^3 / 3a^2
    g = torch.autograd.functional.jacobian(lambda t: CT.wkb_response(t, "smooth").sum(), torch.tensor([0.0], dtype=DT))
    assert g.item() == pytest.approx(1.0)


@pytest.mark.parametrize("pair", [("Cl", "Cl"), ("Na", "Cl")])
def test_smooth_wkb_ct_polarizability_is_analytic_at_symmetric_and_reference_geometries(tabs, pair):
    """The one-sided form has a cusp at delta = 0 (symmetric pairs, the calibration geometry); the smooth form
    gives the two-sided curvature there, independent of the finite-difference step."""
    tab, chan = tabs
    R = 3.2 if pair[0] == pair[1] else 2.8
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]], dtype=DT)
    s = tab.species_index(torch.tensor([{"Na": 11, "Cl": 17}[a] for a in pair]))
    b = torch.zeros(2, dtype=torch.long)

    def alpha(wkb, h):
        def e(Fz):
            F = torch.tensor([[0.0, 0.0, Fz]], dtype=DT)
            x = K.induction(X, s, b, tab, field=F).x
            return CT.ct_energy(X, s, b, tab, chan, x_ind=x, field=F, wkb=wkb).energy.item()
        return -(e(h) + e(-h) - 2 * e(0.0)) / h ** 2

    a1, a2 = alpha("smooth", 1e-3), alpha("smooth", 2e-3)
    assert a1 > 0 and a1 == pytest.approx(a2, rel=0.05)           # quartic terms only
    assert a1 == pytest.approx(alpha("two_sided", 1e-3), rel=0.02)


def test_kappa_floor_tames_the_anion_pair_field_response(tabs):
    """Next to another anion the donor level of Cl- is pushed toward the continuum (I_donor + phi_D -> 0), kappa
    collapses and the WKB field response of W grows like (R / kappa)^2; the floor keeps kappa at least the free
    ion's. Away from that regime (Na-Cl, phi_D > 0) it changes nothing beyond the floor's width."""
    tab, chan = tabs

    def alpha(sym, R, wkb):
        X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]], dtype=DT)
        s = tab.species_index(torch.tensor([{"Na": 11, "Cl": 17}[a] for a in sym]))
        b = torch.zeros(2, dtype=torch.long)

        def e(Fz):
            F = torch.tensor([[0.0, 0.0, Fz]], dtype=DT)
            x = K.induction(X, s, b, tab, field=F).x
            return CT.ct_energy(X, s, b, tab, chan, x_ind=x, field=F, wkb=wkb).energy.item()
        h = 1e-3
        return -(e(h) + e(-h) - 2 * e(0.0)) / h ** 2

    assert 0 < alpha(("Cl", "Cl"), 4.5, "smooth+floor") < 0.5 * alpha(("Cl", "Cl"), 4.5, "smooth")
    assert alpha(("Na", "Cl"), 2.8, "smooth+floor") == pytest.approx(alpha(("Na", "Cl"), 2.8, "smooth"), rel=0.05)
