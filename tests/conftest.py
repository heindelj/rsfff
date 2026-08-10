"""Shared fixtures for the MLIP+EEM model tests.

Everything runs on CPU in float64 so analytic identities (constraint, stationarity,
alpha cross-checks) hold to near machine precision.
"""

from __future__ import annotations

import os
import types

# macOS: avoid the duplicate-libomp abort (torch + conda MKL); must precede torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# macOS: torch and pyscf each pull in an OpenMP runtime; multi-threaded pyscf DFT
# then segfaults in the grid loop. Pinning one thread before torch loads (so the
# env var is read at runtime init) avoids the crash for the qcgen tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from rsfff.mlip import build_eem_model
from rsfff.train.data import Batch

DATA_H2O = "data/labels/h2o.extxyz"
DATA_H3O = "data/labels/h3o+.extxyz"

# Q-Chem ALMO-EDA water clusters (scripts/parse_qchem_eda.py): no forces, a per-atom
# fragment_idx column, and eda_* component labels on the header line.
DATA_W2 = "data/eda_data/w2_wb97xv_qzvppd.xyz"
DATA_W3 = "data/eda_data/w3_wb97xv_qzvppd.xyz"


@pytest.fixture(autouse=True)
def _float64_cpu():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def features_cfg(**over):
    base = dict(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
        backend="e3nn", density_channels=4,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def mlip_cfg(**over):
    base = dict(emb_dim=8, hidden=24, depth=1)
    base.update(over)
    return types.SimpleNamespace(**base)


def eem_cfg(**over):
    base = dict(
        emb_dim=8, hidden=24, depth=1, equiv_channels=6,
        eta_init=0.5, eta_floor=0.05, psd_floor=1e-4,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def make_model(features=None, mlip=None, eem=None, *, randomize=True, seed=1):
    """Small MLIPEEMModel over H/O with (optionally) randomized readouts.

    Zero-init readouts leave chi environment-independent and chivec = 0; the
    perturbation activates every parameter pathway (chi/eta environment
    dependence, chivec, alpha anisotropy) so tests exercise real coupling.
    """
    model = build_eem_model(
        features or features_cfg(),
        mlip or mlip_cfg(),
        eem or eem_cfg(),
        [1, 8],
        torch.tensor([-0.5013, -75.0093]),
    )
    if randomize:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def make_batch(positions, numbers, charges) -> Batch:
    """Assemble a Batch from per-molecule (positions, atomic numbers, total charge)."""
    counts = torch.tensor([p.shape[0] for p in positions])
    return Batch(
        positions=torch.cat(positions).to(torch.get_default_dtype()),
        atomic_numbers=torch.cat(numbers).long(),
        batch_idx=torch.repeat_interleave(torch.arange(len(positions)), counts),
        n_systems=len(positions),
        energy=torch.zeros(len(positions)),
        forces=torch.zeros(int(counts.sum()), 3),
        total_charge=torch.tensor([float(c) for c in charges]),
    )


def water_h3o_batch(jitter: float = 0.0, seed: int = 3) -> Batch:
    """One H2O (Q=0) and one H3O+ (Q=+1), optionally jittered."""
    h2o = torch.tensor(
        [[0.0, 0.0, 0.1174], [0.0, 0.7572, -0.4696], [0.0, -0.7572, -0.4696]]
    )
    h3o = torch.tensor(
        [[0.0, 0.0, 0.0774], [0.0, 0.9210, -0.3306],
         [0.7976, -0.4605, -0.3306], [-0.7976, -0.4605, -0.3306]]
    )
    if jitter:
        g = torch.Generator().manual_seed(seed)
        h2o = h2o + jitter * torch.randn(h2o.shape, generator=g)
        h3o = h3o + jitter * torch.randn(h3o.shape, generator=g)
    return make_batch(
        [h2o, h3o],
        [torch.tensor([8, 1, 1]), torch.tensor([8, 1, 1, 1])],
        [0.0, 1.0],
    )
