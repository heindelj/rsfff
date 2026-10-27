"""C0 environment features (complete model plan): projections of the environment onto the atom's own densities."""
import math

import numpy as np
import pytest
import torch

from rsfff.ff.complete import DensityEnvironment, env_invariants, shaped_overlap_derivs
from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import kernels as K

PARAMS = "params/ions_wb97mv_tzvpd_pf.json"


@pytest.fixture(scope="module")
def env():
    return DensityEnvironment(IonTable.load(PARAMS))


def test_isolated_is_zero(env):
    f = env(torch.zeros(1, 3), torch.tensor([17]), torch.tensor([0]))
    assert f.inv_feats.shape == (1, env.n_feat) and float(f.inv_feats.abs().max()) == 0.0


def test_invariance_and_equivariance(env):
    pos = torch.tensor([[0., 0., 0.], [2.8, 0., 0.], [0., 3.5, 0.5]], dtype=torch.float64)
    Z, b = torch.tensor([11, 17, 17]), torch.zeros(3, dtype=torch.long)
    f0 = env(pos, Z, b)
    g = torch.Generator().manual_seed(0)
    Rm = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=g))[0]
    if torch.det(Rm) < 0:
        Rm[:, 0] *= -1
    f1 = env(pos @ Rm.T, Z, b)
    assert torch.allclose(f0.inv_feats, f1.inv_feats, atol=1e-12)
    for p in env.probes:
        for name in ("V1", "S1"):
            assert torch.allclose(f0.blocks[p][name] @ Rm.T, f1.blocks[p][name], atol=1e-12)
        assert torch.allclose(Rm @ f0.blocks[p]["S2"] @ Rm.T, f1.blocks[p]["S2"], atol=1e-12)


def test_overlap_derivatives():
    bD = torch.tensor([1.3578], dtype=torch.float64)
    w = torch.tensor([[1., 0, 0, 0]], dtype=torch.float64)
    bA = torch.tensor([3.82], dtype=torch.float64)
    R = torch.tensor([5.0], dtype=torch.float64)
    h = 1e-4
    S, S1, S2 = shaped_overlap_derivs(bD, w, bA, w, R)
    f = lambda x: K.shaped_overlap(bD, w, bA, w, x)
    assert abs(float(S - f(R))) < 1e-15
    assert abs(float(S1) - float((f(R + h) - f(R - h)) / (2 * h))) < 1e-9
    assert abs(float(S2) - float((f(R + h) - 2 * f(R) + f(R - h)) / h ** 2)) < 1e-8
    # against a direct two-dimensional quadrature of the two 1s densities
    si = pytest.importorskip("scipy.integrate")
    b1, b2, RR = 1.3578, 3.82, 5.0
    rho = lambda bb, r: bb ** 3 / (8 * math.pi) * np.exp(-bb * r)
    val = si.dblquad(lambda r, th: rho(b1, r) * rho(b2, np.sqrt(r * r + RR * RR - 2 * r * RR * np.cos(th)))
                     * 2 * math.pi * r * r * np.sin(th), 0, math.pi, 0, 40)[0]
    assert abs(float(S) - val) < 1e-9


def test_coulomb_blocks_are_fields(env):
    """Far apart: the penetration potential vanishes and |V1| is the point-charge field q / r^2."""
    f = env(torch.tensor([[0., 0., 0.], [10., 0., 0.]], dtype=torch.float64), torch.tensor([11, 17]), torch.tensor([0, 0]))
    r = 10.0 / 0.529177210903
    assert float(f.blocks["shell"]["V0"].abs().max()) < 1e-8
    assert torch.allclose(f.blocks["shell"]["V1"].norm(dim=-1), torch.full((2,), 1 / r ** 2, dtype=torch.float64), rtol=1e-3)


def _octahedron(d=2.8, shift=0.0):
    """Na+ at the centre of a Cl- octahedron (O_h: every V^(1), S^(1), V^(2), S^(2) of the centre vanishes),
    displaced by ``shift`` (Angstrom) along a general direction."""
    u = torch.tensor([0.6, -0.3, 0.74], dtype=torch.float64)
    cl = d * torch.cat([torch.eye(3), -torch.eye(3)]).double()
    pos = torch.cat([shift * u[None] / u.norm(), cl])
    return pos, torch.tensor([11] + [17] * 6), torch.zeros(7, dtype=torch.long)


def test_sq_form_is_squared_norms():
    tab = IonTable.load(PARAMS)
    en, es = DensityEnvironment(tab, inv_form="norm"), DensityEnvironment(tab, inv_form="sq")
    pos = torch.tensor([[0., 0., 0.], [2.8, 0., 0.], [0., 3.5, 0.5]], dtype=torch.float64)
    Z, b = torch.tensor([11, 17, 17]), torch.zeros(3, dtype=torch.long)
    fn, fs = en(pos, Z, b), es(pos, Z, b)
    for k, (a, c) in enumerate(zip(en.feature_names, es.feature_names)):
        if c.endswith("^2"):
            assert c == a + "^2"
            assert torch.allclose(fs.inv_feats[:, k], fn.inv_feats[:, k] ** 2, rtol=1e-12, atol=1e-300)
        else:
            assert a == c and torch.equal(fs.inv_feats[:, k], fn.inv_feats[:, k])
    # the cached equivariant blocks reproduce either form
    blocks = {f"{p}:{q}": v for p, blk in fs.blocks.items() for q, v in blk.items()}
    assert torch.allclose(env_invariants(blocks, es.probes, "sq"), fs.inv_feats, rtol=1e-14, atol=0)
    assert torch.allclose(env_invariants(blocks, en.probes, "norm"), fn.inv_feats, rtol=1e-14, atol=0)


def test_sq_form_is_smooth_at_a_lattice_site():
    """Through the centre of an octahedron the magnitudes have a cone (one-sided slopes of opposite sign); the
    squared norms are smooth: their slope is continuous (and zero) there."""
    tab = IonTable.load(PARAMS)
    h = 1e-4
    for form, smooth in (("norm", False), ("sq", True)):
        env = DensityEnvironment(tab, inv_form=form)
        f = {t: env(*_octahedron(shift=t)).inv_feats[0] for t in (-h, 0.0, h)}
        assert float(f[0.0][[1, 2, 4, 5]].abs().max()) < 1e-12        # shell probe: |V1|, |V2|, |S1|, |S2| vanish
        kink = ((f[h] - f[0.0]) / h - (f[0.0] - f[-h]) / h).abs()       # jump of the one-sided slopes
        scale = env(*_octahedron(shift=0.2)).inv_feats[0].abs() / 0.2 + 1e-12
        rel = float((kink / scale).max())
        if smooth:
            assert rel < 1e-2, rel
        else:
            assert rel > 0.5, rel
