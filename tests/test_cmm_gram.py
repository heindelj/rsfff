"""Parameter-free exchange polarization (gram.py, xpol_form="gram") and the consistent hyperpolarization
(hyperpol_form="legendre"); theory/parameter_free_xpol.tex."""
import math

import numpy as np
import pytest
import torch

from rsfff.ff.cmm.gram import GramTable, _angular_polys, _shell_density, gram_angular_coef, gram_blocks, response_radials
from rsfff.ff.cmm.params import IonTable

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def tab():
    return IonTable.load(symbols=("Na", "Cl"))


@pytest.fixture(scope="module")
def gtab(tab):
    return GramTable(tab, r_max=6.0, h=0.02)


def _brute(tab, si, sj, dr, nr=900, h=0.012, nx=48, nph=48):
    r = torch.arange(nr) * h + h / 2
    xg, wg = np.polynomial.legendre.leggauss(nx)
    x, wx = torch.tensor(xg), torch.tensor(wg)
    ph = torch.arange(nph) * 2 * math.pi / nph
    st = torch.sqrt(1 - x * x)
    rh = torch.stack([(st[:, None] * torch.cos(ph)).reshape(-1), (st[:, None] * torch.sin(ph)).reshape(-1),
                      x[:, None].expand(-1, nph).reshape(-1)], 1)
    wa = (wx[:, None] * 2 * math.pi / nph).expand(-1, nph).reshape(-1)
    p = _angular_polys(rh)
    u = response_radials(tab, si, r)
    rho_i = _shell_density(tab, si, r)
    U = torch.stack([u["d"]] * 3 + [u["q"]] * 5 + [u["b"]], 1)
    G = torch.zeros(9, 9)
    for k in range(0, nr, 50):
        rr = r[k:k + 50]
        rhoj = _shell_density(tab, sj, (rr[:, None, None] * rh[None] - dr).norm(dim=-1))
        f = torch.einsum("ca,ap,aq->cpq", rhoj * wa, p, p)
        G += (f * (U[k:k + 50, :, None] * U[k:k + 50, None, :]) * (rr * rr * h / (4 * rho_i[k:k + 50]))[:, None, None]).sum(0)
    return G


def test_angular_fit_exact():
    C = gram_angular_coef()
    assert C.shape == (10, 9, 9, 35)
    assert torch.allclose(C, C.transpose(1, 2))


def test_gram_matches_quadrature_on_axis(gtab, tab):
    """Partner on z (the brute-force grid resolves it): table + angular polynomials = direct 3D quadrature."""
    for si, sj, R in ((1, 0, 2.8), (1, 1, 3.4)):
        dr = torch.tensor([0.0, 0.0, R / 0.529177210903])
        G = gram_blocks(gtab, torch.tensor([si, sj]), (torch.tensor([0]), torch.tensor([1])), dr[None])[0]
        Gb = _brute(tab, si, sj, dr)
        assert torch.allclose(G, Gb, rtol=2e-2, atol=2e-3 * float(Gb.abs().max()))


def test_gram_rotates_and_is_psd(gtab):
    """Gt(Q R) = D(Q) Gt(R) D(Q)^T on the dipole block, and every block PSD."""
    s = torch.tensor([1, 0])
    dp = (torch.tensor([0]), torch.tensor([1]))
    dz = torch.tensor([[0.0, 0.0, 5.3]])
    Q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(3)))[0]
    G0 = gram_blocks(gtab, s, dp, dz)[0]
    G1 = gram_blocks(gtab, s, dp, dz @ Q.T)[0]
    assert torch.allclose(Q @ G0[:3, :3] @ Q.T, G1[:3, :3], atol=1e-10 * float(G0.abs().max()) + 1e-14)
    assert torch.linalg.eigvalsh(G1).min() > -1e-10 * float(G1.abs().max())
    # cylindrical symmetry on z: pi components equal, sigma-pi uncoupled
    assert torch.isclose(G0[0, 0], G0[1, 1]) and abs(float(G0[0, 2])) < 1e-12




