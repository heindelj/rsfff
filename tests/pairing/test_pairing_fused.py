"""The fused Triton inner solves agree with the torch reference loops.

Without a GPU the kernels run under the Triton interpreter (``TRITON_INTERPRET=1``, which
Triton reads at import, hence the subprocess); skipped when Triton is not importable.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="triton not installed")

BODY = r"""
import os, torch
torch.set_default_dtype(torch.float64)
from rsfff.ff.pairing import bond_order as bo, electronic_state as es, fused
from rsfff.ff.pairing.electronic_state import capacity_table
assert fused.fused_enabled(torch.zeros(1, device=DEVICE)), "fused path not active"

def both(fn, *args, **kwargs):
    os.environ["RSFFF_PAIRING_FUSED"] = "0"; ref = fn(*args, **kwargs)
    os.environ["RSFFF_PAIRING_FUSED"] = "1"; out = fn(*args, **kwargs)
    return ref, out

a = torch.linspace(-0.5, 1.0, 61, device=DEVICE); kappa = torch.full_like(a, 0.2)
ref, out = both(bo._solve_pair_logit, a, kappa, torch.tensor(0.002, device=DEVICE))
assert torch.allclose(out, ref, atol=1e-12, rtol=0.0), float((out - ref).abs().max())

tables = capacity_table([1, 8]).repeat(3, 1).to(DEVICE)
n0, _, aa, bb, shell = tables.unbind(-1)
chi = torch.tensor([0.248, 0.281] * 3, device=DEVICE); eta = torch.tensor([0.492, 0.452] * 3, device=DEVICE)
g = torch.Generator().manual_seed(0)
worst = 0.0
for _ in range(50):
    lam = (torch.rand(6, generator=g) * 0.6 - 0.05).to(DEVICE)
    mu = (torch.rand(1, generator=g) * 0.6).expand(6).to(DEVICE)
    n_init = n0 + (torch.randn(6, generator=g) * 0.5).to(DEVICE)
    (n_ref, dl_ref, dm_ref), (n_out, dl_out, dm_out) = both(
        es._formal_count, lam, mu, chi, eta, n0, aa, bb, shell, torch.tensor(0.002, device=DEVICE), n_init=n_init)
    worst = max(worst, float((n_out - n_ref).abs().max()))
    assert torch.allclose(dl_out, dl_ref, rtol=1e-8, atol=1e-12)
    assert torch.allclose(dm_out, dm_ref, rtol=1e-8, atol=1e-12)
assert worst < 1e-12, worst
print("ok", worst)
"""


def test_fused_inner_solves_match_reference():
    env = dict(os.environ)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        env["TRITON_INTERPRET"] = "1"
    env["RSFFF_PAIRING_FUSED"] = "1"
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", f"DEVICE = {device!r}\n" + BODY],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "ok" in proc.stdout
