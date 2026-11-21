"""Exchange polarization on the site potential (``CmmConfig.xpol_form = "site"``): the drive and Gram maps are the
integrals <dn_i | v_j> and int dphi_i dphi_i v_j of the response components; with them the pair's site-potential
energy to second order is a complete square, so each ion's response removes at most 1 / theta of its own half of the
pair's Pauli energy; forces are exact derivatives; the config is validated."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, CmmConfig, CmmModel, IonTable
from rsfff.ff.cmm import kernels as K
from rsfff.ff.cmm.gram import _angular_polys, _shell_density, response_radials

SITE = Path(__file__).resolve().parents[1] / "params" / "ions_salts8_site.json"
f64 = torch.float64
SYM = ["Na", "Cl", "Na", "Cl"]
X0 = torch.tensor([[0.0, 0, 0], [2.6, 0, 0], [2.7, 2.5, 0.2], [0.1, 2.6, -0.3]], dtype=f64)


@pytest.fixture(scope="module")
def model():
    tab = IonTable.load(SITE, symbols=("Na", "Cl"))
    torch.manual_seed(0)
    return CmmModel(tab, ChannelTable.direct(("Na", "Cl")),
                    CmmConfig(film=False, pauli_form="site", xpol_form="site", xpol_bound="rank"))


def _batch(m, X=X0, sym=SYM):
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in sym])
    return m.precompute(X, Z, torch.zeros(len(sym), dtype=torch.long))


def test_maps_are_the_integrals_of_the_response_in_the_site_potential(model):
    """Na+ (response) next to Cl- (potential): <dn_a | y_t> and int dn_a dn_b y_t / (4 rho) by direct quadrature."""
    m = model
    tab = m.tab
    R = torch.tensor([0.3, -0.5, 0.8], dtype=f64)
    R = R / R.norm() * 2.8
    bt = _batch(m, torch.stack([torch.zeros(3, dtype=f64), R]), ["Cl", "Na"])
    si, sj = [int(v) for v in bt["species"]]                         # i = Cl: its response in v_Na and the reverse
    xg, wg = np.polynomial.legendre.leggauss(64)
    nph = 96
    rr = torch.arange(0.005, 30, 0.01, dtype=f64)
    x, ph = torch.tensor(xg), torch.arange(nph, dtype=f64) * (2 * math.pi / nph)
    st = torch.sqrt(1 - x * x)
    rh = torch.stack([(st[:, None] * torch.cos(ph)[None]).reshape(-1), (st[:, None] * torch.sin(ph)[None]).reshape(-1),
                      x[:, None].expand(-1, nph).reshape(-1)], 1)
    wa = (torch.tensor(wg)[:, None] * (2 * math.pi / nph)).expand(-1, nph).reshape(-1)
    pa = _angular_polys(rh, 1)
    w = 0.01 * rr * rr
    D = bt["D"]
    for (a, b, rows_g, rows_G, sign) in ((sj, si, bt["xs_g"][1:], bt["xs_Gj"], -1.0),):      # Na's response in v_Cl
        u = response_radials(tab, a, rr)
        U = torch.stack([u["d"]] * 3 + [u["q"]] * 5 + [u["b"]], 1)
        rho = _shell_density(tab, a, rr)
        for t in range(tab.shell_n.shape[1]):
            if float(tab.shell_n[b, t]) <= 0:
                continue
            beta = float(tab.pauli_site_s[b] * tab.shell_b[b, t])
            s = (rr[:, None, None] * rh[None] - (sign * R / K.BOHR_ANG)[None, None]).norm(dim=-1)
            yv = torch.exp(-beta * s) * (1 / s + beta / 2)
            g_ref = ((w[:, None] * U) * ((yv * wa) @ pa)).sum(0)[:D]
            ang = torch.einsum("rg,ga,gb->rab", yv * wa, pa, pa)
            G_ref = torch.einsum("r,ra,rb,rab->ab", w / (4 * rho), U, U, ang)[:D, :D]
            assert torch.allclose(rows_g[0, t], g_ref, rtol=2e-3, atol=1e-3 * float(g_ref.abs().max()))
            assert torch.allclose(rows_G[0, t], G_ref, rtol=2e-3, atol=1e-3 * float(G_ref.abs().max()))


def test_complete_square_and_bound(model):
    m = model
    bt = _batch(m)
    sp = bt["species"]
    rho, r = m.xpol_fraction(bt), m.xpol_ratio(sp)
    i, j = bt["pairs"][:, bt["x_sel"]]
    assert float(r.max()) <= 1.0
    assert bool((rho <= torch.cat([r[i], r[j]]) * (1 + 1e-6)).all())
    assert float(rho.max()) > 0.1                                    # the exchange terms are on
    # unit weights, theta -> 1: half + g z + z G z >= 0 for every z
    st = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        m.ln_w1.zero_()
        m.ln_w2.fill_(-40.0)
    try:
        m.environment(bt)
        m.frozen_terms(bt, sp)
        (i, j), g, Gi, Gj = m.exchange(bt, sp)
        P = i.shape[0]
        ci, cj = m._site_amplitudes(sp, i, j)
        gen = torch.Generator().manual_seed(3)
        for half, c, gr, G in ((m._pauli_half[0][bt["x_sel"]], ci, bt["xs_g"][:P], Gi),
                               (m._pauli_half[1][bt["x_sel"]], cj, bt["xs_g"][P:], Gj)):
            gp = torch.einsum("pt,ptd->pd", c, gr)
            for scale in (0.01, 0.1, 1.0, 10.0, 100.0):
                z = scale * torch.randn(200, P, gp.shape[1], generator=gen, dtype=f64)
                e = half + (gp * z).sum(-1) + 0.5 * torch.einsum("npa,pab,npb->np", z, G, z)
                assert bool((e >= -1e-3 * half - 1e-12).all())
    finally:
        m.load_state_dict(st)


def test_forces_are_exact(model):
    m = model
    Z = torch.tensor([ATOMIC_NUMBER[a] for a in SYM])
    fr = torch.zeros(4, dtype=torch.long)
    X = X0.clone().requires_grad_(True)
    out = m(m.precompute(X, Z, fr))
    assert float(out["e_xpol"].abs().sum()) > 1e-4
    F = -torch.autograd.grad(out["e_int"].sum(), X)[0]
    h = 1e-4
    for a, c in ((0, 0), (1, 2), (3, 1)):
        Xp, Xm = X0.clone(), X0.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        num = -(m(m.precompute(Xp, Z, fr))["e_int"].sum() - m(m.precompute(Xm, Z, fr))["e_int"].sum()) / (2 * h)
        assert float(F[a, c]) == pytest.approx(float(num), abs=2e-7)


def test_site_scale_is_fixed_and_config_checked(model):
    assert not model.pauli_ln_bscale.requires_grad
    with pytest.raises(ValueError, match="pauli_form"):
        CmmConfig(xpol_form="site")
    with pytest.raises(ValueError, match="xpol_form"):
        CmmConfig(pauli_form="site", xpol_form="gram")
