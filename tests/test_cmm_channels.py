"""m33 environment channels (``channels.py``) and the shared environment (``heads.SharedEnvironment``,
``CompleteModel(env_channels=True)``): the three-body identity against a brute-force triple loop, dimer and isolated
limits, invariances, continuity as a neighbour leaves, readout masks, the pinned a0, and forces as derivatives of
the energy through the channels."""
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import legval

from rsfff.ff.cmm import DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.cmm.channels import (CHANNEL_NAMES, THREE_BODY_COLS, TWO_BODY_COLS, channels_from_pairs,
                                        pair_electrostatics, real_harmonics)
from rsfff.ff.cmm import params as CT
from rsfff.ff.cmm.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"
f64 = torch.float64


def test_harmonics_addition_theorem():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(7, 3, generator=g, dtype=f64)
    b = torch.randn(7, 3, generator=g, dtype=f64)
    a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    for L in range(1, 5):
        lhs = (real_harmonics(a, L) * real_harmonics(b, L)).sum(1).numpy()
        np.testing.assert_allclose(lhs, legval((a * b).sum(1).numpy(), [0] * L + [1]), atol=1e-13)


def test_channels_match_the_triple_loop():
    g = torch.Generator().manual_seed(1)
    dr = torch.randn(5, 3, generator=g, dtype=f64)
    w = torch.rand(5, generator=g, dtype=f64)
    s = torch.randn(5, generator=g, dtype=f64)
    X = channels_from_pairs(1, torch.zeros(5, dtype=torch.long), dr, w, s)[0].numpy()
    rh = (dr / dr.norm(dim=1, keepdim=True)).numpy()
    wn, sn = w.numpy(), s.numpy()
    ref = [wn.sum(), sn.sum()]
    for A, B in ((wn, wn), (wn, sn), (sn, sn)):
        for L in range(5):
            t = sum(0.5 * (A[j] * B[k] + A[k] * B[j]) * (1.0 if L == 0 else 1 - legval(rh[j] @ rh[k], [0] * L + [1]))
                    for j in range(5) for k in range(j + 1, 5))
            ref.append(t / wn.sum())
    np.testing.assert_allclose(X, ref, rtol=1e-12, atol=1e-15)
    assert len(ref) == len(CHANNEL_NAMES)


def test_single_neighbour_has_no_three_body_and_no_pairs_gives_zero():
    dr = torch.tensor([[0.0, 0.0, 2.8]], dtype=f64)
    X = channels_from_pairs(2, torch.tensor([0]), dr, torch.tensor([3e-3], dtype=f64), torch.tensor([-1e-3], dtype=f64))
    assert float(X[0, list(THREE_BODY_COLS)].abs().max()) < 1e-18
    assert float(X[0, 0]) == pytest.approx(3e-3) and float(X[0, 1]) == pytest.approx(-1e-3)
    assert float(X[1].abs().max()) == 0.0                           # atom 1 is the centre of no pair




def _cluster(a=2.8, seed=1):
    g = torch.tensor([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=f64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.1 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(seed), dtype=f64)
    return X, Z






















# --- vectorized channels against the per-L reference loop ----------------------------------------------------------

@pytest.mark.parametrize("lmax", [1, 2, 4, 6])
def test_stacked_harmonics_are_e3nn_blocks(lmax):
    from rsfff.ff.cmm.channels import stacked_harmonics
    g = torch.Generator().manual_seed(lmax)
    v = torch.randn(300, 3, generator=g, dtype=f64)
    v = v / v.norm(dim=1, keepdim=True)
    v[:6] = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], dtype=f64)
    Y = stacked_harmonics(v, lmax)
    ref = torch.cat([real_harmonics(v, L) for L in range(1, lmax + 1)], 1)
    assert Y.shape == ref.shape
    torch.testing.assert_close(Y, ref, atol=1e-13, rtol=0)


def _random_pairs(n=9, P=60, seed=0, dtype=f64):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n - 2, (P,), generator=g)                      # the last two atoms have no pairs
    dr = torch.randn(P, 3, generator=g, dtype=dtype) * 3
    dr[:3] = torch.tensor([[0, 0, 2.0], [0, 0, -2.5], [1.5, 0, 0]], dtype=dtype)   # axis-aligned bonds
    w = torch.rand(P, generator=g, dtype=dtype) * 1e-2
    s = (torch.rand(P, generator=g, dtype=dtype) - 0.5) * 1e-2
    return n, i, dr, w, s


@pytest.mark.parametrize("lmax", [0, 1, 4])
@pytest.mark.parametrize("dtype,tol", [(f64, 1e-15), (torch.float32, 1e-7)])
def test_vectorized_channels_match_reference(lmax, dtype, tol):
    from rsfff.ff.cmm.channels import _channels_from_pairs_ref
    n, i, dr, w, s = _random_pairs(dtype=dtype)
    X = channels_from_pairs(n, i, dr, w, s, lmax)
    R = _channels_from_pairs_ref(n, i, dr, w, s, lmax)
    assert X.shape == R.shape == (n, 2 + 3 * (lmax + 1)) and X.dtype == dtype
    assert float(X[-2:].abs().max()) == 0.0
    torch.testing.assert_close(X, R, atol=tol * float(R.abs().max()), rtol=0)


def test_vectorized_channels_gradients_match_reference():
    from rsfff.ff.cmm.channels import _channels_from_pairs_ref
    n, i, dr, w, s = _random_pairs(seed=3)
    out = torch.randn(n, 2 + 3 * 5, generator=torch.Generator().manual_seed(1), dtype=f64)
    grads = []
    for f in (channels_from_pairs, _channels_from_pairs_ref):
        args = [t.clone().requires_grad_(True) for t in (dr, w, s)]
        grads.append(torch.autograd.grad((f(n, i, *args) * out).sum(), args))
    for a, b in zip(*grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-12 * float(b.abs().max()), rtol=0)


def test_vectorized_channels_empty_pair_list():
    e = torch.zeros(0, dtype=f64)
    X = channels_from_pairs(3, torch.zeros(0, dtype=torch.long), torch.zeros(0, 3, dtype=f64), e, e)
    assert X.shape == (3, 17) and float(X.abs().max()) == 0.0
