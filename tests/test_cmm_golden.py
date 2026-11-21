"""The cmm model reproduces the complete model as trained in the complete.* runs (tag complete-explore-final).

The references are written by tests/cmm_golden/make_golden.py from the legacy model with randomized parameters:
golden_nofilm.pt (no species conditioning) and, where present, golden_film.pt (FiLM; needs the real ``bb``).

Tolerances: rounding, except downstream of the gated probe invariants. Their gate overlap now comes from the one
table on the 0.01 A grid instead of the 0.02 A probe table the references were made with, which moves those inputs by
their interpolation error (~4e-9 relative; every other input agrees with the old code to rounding) and the latents,
moments and charges by ~1e-10. The polarizability is now the model's exact linear response; the references are the
trainer's 7-point finite-field stencil.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent / "cmm_golden"
PARAMS = HERE / "ions_golden.json"


def _case(name):
    p = HERE / name
    if not p.exists():
        pytest.skip(f"no {name} (tests/cmm_golden/make_golden.py writes it)")
    return torch.load(p, weights_only=False)


def _run(ck):
    from rsfff.ff.cmm import ATOMIC_NUMBER, ChannelTable, IonTable, load_legacy

    tab, chan = IonTable.load(PARAMS), ChannelTable.load(PARAMS)
    m = load_legacy(ck, tab, chan)
    m.eval()
    fr = ck["frames"]
    pos = torch.tensor([x for _, _, X in fr for x in X], dtype=torch.float64, requires_grad=True)
    Z = torch.tensor([ATOMIC_NUMBER[x] for _, s, _ in fr for x in s])
    idx = torch.cat([torch.full((len(s),), k, dtype=torch.long) for k, (_, s, _) in enumerate(fr)])
    bt = m.precompute(pos, Z, idx)
    out = m(bt, alpha=True)
    F = -torch.autograd.grad(out["e_int"].sum(), pos)[0]
    u = {k: v.detach().clone() for k, v in m._u.items()}
    ap = torch.diagonal(out["alpha_pol"].detach(), dim1=1, dim2=2)
    alpha = torch.stack([ap, torch.diagonal(out["alpha"].detach(), dim1=1, dim2=2) - ap], -1)
    return u, out, F, alpha


@pytest.fixture
def legacy_ovl_grid(monkeypatch):
    """The references were made with the tabulated overlaps on the runtime k-grid (kmax 40, 1000 points); the tables
    now use a finer one (rsfff.ff.cmm.kernels.TABLE_OVL_KMAX), which moves the energies by ~1e-7 Hartree."""
    from rsfff.ff.cmm import kernels as K
    monkeypatch.setattr(K, "TABLE_OVL_KMAX", 40.0)
    monkeypatch.setattr(K, "TABLE_OVL_N", 1000)
    monkeypatch.setattr(K, "_TABLE_OVL_K", None)
    yield
    K._TABLE_OVL_K = None


@pytest.mark.parametrize("name", ["golden_nofilm.pt", "golden_film.pt"])
def test_cmm_reproduces_complete(name, legacy_ovl_grid):
    ck = _case(name)
    if ck["film"]:
        try:
            from bb.nn.conditioning import ConditionedTrunk
            ConditionedTrunk(2, 2, 1, d_c=2, mode="film")
        except Exception:  # noqa: BLE001
            pytest.skip("needs bb (FiLM trunk)")
    u, out, F, alpha = _run(ck)
    ref = ck["ref"]
    for k in ("e_elst", "e_pauli", "e_disp", "e_pol", "e_xpol", "e_ct", "e_int"):
        torch.testing.assert_close(out[k].detach(), ref[k], rtol=1e-9, atol=1e-12, msg=k)
    torch.testing.assert_close(F, ref["forces"], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(out["x"].detach(), ref["x"], rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(out["dq"].detach(), ref["dq"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(out["q_ct"].detach(), ref["q_ct"], rtol=1e-8, atol=1e-9)
    legacy = {"kappa_dip": "w1_dip", "kappa_quad": "w1_quad", "kappa_breath": "w1_breath", "gram": "w2"}
    for k, v in ref["u"].items():
        torch.testing.assert_close(u[legacy.get(k, k)], v, rtol=1e-8, atol=1e-9, msg=k)
    torch.testing.assert_close(alpha, ref["alpha"], rtol=1e-6, atol=1e-6)
