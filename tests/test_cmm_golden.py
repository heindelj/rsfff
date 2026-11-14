"""The cmm model reproduces the complete model as trained in the complete.* runs (tag complete-explore-final).

The references are written by tests/cmm_golden/make_golden.py from the legacy model with randomized parameters:
golden_nofilm.pt (no species conditioning) and, where present, golden_film.pt (FiLM; needs the real ``bb``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent / "cmm_golden"
PARAMS = HERE / "ions_golden.json"
F_SET = 2e-3


def polar_fields(B):
    f = torch.zeros(7, 3, dtype=torch.float64)
    for k in range(3):
        f[1 + 2 * k, k], f[2 + 2 * k, k] = F_SET, -F_SET
    return f.repeat(B, 1)


def _case(name):
    p = HERE / name
    if not p.exists():
        pytest.skip(f"no {name} (tests/cmm_golden/make_golden.py writes it)")
    return torch.load(p, weights_only=False)


def _run(ck):
    from rsfff.ff.cmm import load_legacy
    from rsfff.ff.complete import batched_precompute
    from rsfff.ff.ionct import ct as CT
    from rsfff.ff.ionct.params import ATOMIC_NUMBER, IonTable
    from rsfff.train.train_ionct import radial_table

    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    table = radial_table(tab, PARAMS, os.environ.get("RSFFF_TABLE_CACHE"))
    m = load_legacy(ck, tab, chan)
    m.eval()
    fr = ck["frames"]
    pos = torch.tensor([x for _, _, X in fr for x in X], dtype=torch.float64, requires_grad=True)
    Z = torch.tensor([ATOMIC_NUMBER[x] for _, s, _ in fr for x in s])
    idx = torch.cat([torch.full((len(s),), k, dtype=torch.long) for k, (_, s, _) in enumerate(fr)])
    bt = batched_precompute(pos, Z, idx, tab, table, chan, inv_form="sq")
    out = m(bt)
    F = -torch.autograd.grad(out["e_int"].sum(), pos)[0]
    u = {k: v.detach().clone() for k, v in m._u.items()}
    # finite-field alpha, the trainer's stencil
    rep = [f for f in fr for _ in range(7)]
    posr = torch.tensor([x for _, _, X in rep for x in X], dtype=torch.float64)
    Zr = torch.tensor([ATOMIC_NUMBER[x] for _, s, _ in rep for x in s])
    idr = torch.cat([torch.full((len(s),), k, dtype=torch.long) for k, (_, s, _) in enumerate(rep)])
    with torch.no_grad():
        op = m(batched_precompute(posr, Zr, idr, tab, table, chan, inv_form="sq"), field=polar_fields(len(fr)))
    e = torch.stack([op["e_pol"], op["e_ct"]], -1).reshape(len(fr), 7, 2)
    alpha = -(e[:, 1::2] + e[:, 2::2] - 2 * e[:, :1]) / F_SET ** 2
    return u, out, F, alpha


@pytest.mark.parametrize("name", ["golden_nofilm.pt", "golden_film.pt"])
def test_cmm_reproduces_complete(name):
    ck = _case(name)
    if ck["film"]:
        pytest.importorskip("bb.nn.conditioning")
    u, out, F, alpha = _run(ck)
    ref = ck["ref"]
    for k in ("e_elst", "e_pauli", "e_disp", "e_pol", "e_xpol", "e_ct", "e_int"):
        torch.testing.assert_close(out[k].detach(), ref[k], rtol=1e-9, atol=1e-12, msg=k)
    torch.testing.assert_close(F, ref["forces"], rtol=1e-8, atol=1e-11)
    torch.testing.assert_close(out["x"].detach(), ref["x"], rtol=1e-8, atol=1e-11)
    torch.testing.assert_close(out["dq"].detach(), ref["dq"], rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(out["q_ct"].detach(), ref["q_ct"], rtol=1e-10, atol=1e-12)
    legacy = {"kappa_dip": "w1_dip", "kappa_quad": "w1_quad", "kappa_breath": "w1_breath", "gram": "w2"}
    for k, v in ref["u"].items():
        torch.testing.assert_close(u[legacy.get(k, k)], v, rtol=1e-10, atol=1e-12, msg=k)
    torch.testing.assert_close(alpha, ref["alpha"], rtol=1e-6, atol=1e-6)
