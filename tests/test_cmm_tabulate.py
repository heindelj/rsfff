"""The radial-table machinery: the vectorized lookup and damped-tensor basis against their transcriptions, and the
frozen electrostatics. (The precompute against the exact per-frame quantities: tests/test_cmm_model.py.)"""
from pathlib import Path

import pytest
import torch

from rsfff.ff.cmm import params as CT
from rsfff.ff.cmm.params import IonTable
from rsfff.ff.cmm import RadialTable, permanent_electrostatics

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2p.json"


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(3)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = RadialTable(tab, h=0.05)            # coarse grid keeps the test fast; the training default is 0.01 A
    feat = None
    frames = []
    for n in (2, 4, 6):
        sym = (["Na", "Cl"] * 3)[:n]
        while True:
            X = torch.rand(n, 3, dtype=torch.float64) * 5.0
            if (torch.cdist(X, X) + torch.eye(n) * 10).min() > 2.0:
                break
        frames.append((sym, X))
    return tab, chan, table, feat, frames


def _flat(frames):
    pos = torch.cat([X for _, X in frames])
    Z = torch.tensor([{"Na": 11, "Cl": 17}[a] for sym, _ in frames for a in sym])
    fr = torch.cat([torch.full((len(sym),), k) for k, (sym, _) in enumerate(frames)])
    return pos, Z, fr


def test_fused_basis_matches_transcription():
    from rsfff.ff.cmm.tabulate import _basis_tensors_fused, _basis_tensors_ref
    dr = (torch.randn(200, 3, dtype=torch.float64) * 4).requires_grad_(True)
    a, b = _basis_tensors_ref(dr), _basis_tensors_fused(dr)
    with torch.no_grad():
        assert torch.allclose(b, a, rtol=1e-12, atol=1e-14 * float(a.abs().max()))
    w = torch.randn_like(a)
    ga = torch.autograd.grad((a * w).sum(), dr, create_graph=True)[0]
    gb = torch.autograd.grad((b * w).sum(), dr, create_graph=True)[0]
    assert torch.allclose(gb, ga, rtol=1e-12, atol=1e-14 * float(ga.abs().max()))
    v = torch.randn_like(ga)        # the double backward force training takes
    ha, hb = (torch.autograd.grad((g * v).sum(), dr)[0] for g in (ga, gb))
    assert torch.allclose(hb, ha, rtol=1e-11, atol=1e-13 * float(ha.abs().max()))


def test_electrostatics_matches_pairwise(setup):
    from rsfff.ff.cmm import kernels as K
    tab, chan, table, feat, frames = setup
    pos, Z, fr = _flat(frames)
    s = tab.species_index(Z)
    pairs = K.all_pairs(fr)
    ref = torch.zeros(len(frames), dtype=torch.float64).index_add(0, fr[pairs[0]],
                                                                  K.permanent_pair_energy(pos, s, pairs, tab))
    assert torch.allclose(permanent_electrostatics(pos, s, fr, tab), ref, rtol=1e-12, atol=1e-12)


def test_electrostatics_pair_limit(setup):
    tab, chan, table, feat, frames = setup
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]], dtype=torch.float64)
    s = tab.species_index(torch.tensor([11, 17]))
    e = permanent_electrostatics(X, s, torch.zeros(2, dtype=torch.long), tab)[0]
    assert abs(e + 1.0 / (20.0 / 0.529177210903)) < 1e-9          # two point charges at 20 A


def _table_points(table, n=400, seed=5):
    g = torch.Generator().manual_seed(seed)
    S = len(table.tab.symbols)
    r = torch.rand(n, generator=g, dtype=torch.float64) * (table.r_max + 1.0 - 0.1) + 0.1    # below r_min to past r_max
    nodes = table.r_min + table.h * torch.arange(0, table.G, max(table.G // 40, 1), dtype=torch.float64)
    r = torch.cat([r, nodes, torch.tensor([table.r_min, table.r_max], dtype=torch.float64)])   # t = 0 exactly
    si = torch.randint(0, S, (r.shape[0],), generator=g)
    sj = torch.randint(0, S, (r.shape[0],), generator=g)
    return si, sj, r


def test_vectorized_lookup_matches_transcription(setup):
    _, _, table, _, _ = setup
    si, sj, r = _table_points(table)
    a, b = table._call_ref(si, sj, r), table(si, sj, r)
    assert b.shape == a.shape and b.dtype == a.dtype
    assert torch.allclose(b, a, rtol=1e-12, atol=1e-13 * float(a.abs().max()))


def test_vectorized_lookup_derivatives_match_transcription(setup):
    """First and second derivatives in r (forces, and their gradient in the force loss)."""
    _, _, table, _, _ = setup
    si, sj, r0 = _table_points(table)
    outs = []
    for fn in (table._call_ref, table.__call__):
        r = r0.clone().requires_grad_(True)
        y = fn(si, sj, r)
        w = torch.linspace(-1.0, 1.0, y.numel(), dtype=y.dtype).reshape(y.shape)
        g = torch.autograd.grad((y * w).sum(), r, create_graph=True)[0]
        h = torch.autograd.grad((g * torch.cos(r0)).sum(), r)[0]
        outs.append((g.detach(), h))
    # the second derivative carries the summation-order rounding times 1/h^2 (400 on this grid): ~2e-12 of its scale
    for (x, y), scale in zip(zip(*outs), (1e-12, 1e-11)):
        assert torch.isfinite(y).all()
        assert torch.allclose(y, x, rtol=1e-11, atol=scale * float(x.abs().max()))


def test_vectorized_lookup_is_a_handful_of_ops(setup):
    from torch.overrides import TorchFunctionMode

    class Count(TorchFunctionMode):
        n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            Count.n += 1
            return func(*args, **(kwargs or {}))

    _, _, table, _, _ = setup
    si, sj, r = _table_points(table)
    table(si, sj, r)
    with Count():
        table(si, sj, r)
    assert Count.n < 40, Count.n
