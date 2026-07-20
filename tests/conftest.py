"""Shared fixtures for the charge-aware model tests.

Everything runs on CPU in float64: the implicit-differentiation gradient checks
need ~1e-12 SCF residuals, and the batched ``linalg.solve``/``eigvalsh`` used by
the KKT solver have CPU/CUDA kernels only.
"""

from __future__ import annotations

import types

import pytest
import torch

from rsfff.mlip import build_charge_model
from rsfff.train.data import Batch

DATA_H2O = "data/labels/h2o.extxyz"
DATA_H3O = "data/labels/h3o+.extxyz"


@pytest.fixture(autouse=True)
def _float64_cpu():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def features_cfg(**over):
    base = dict(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 2),
        backend="e3nn", density_channels=4,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def charge_cfg(**over):
    base = dict(
        emb_dim=8, q_hidden=8, weight_hidden=16, charge_channels=3,
        hidden=24, depth=1, alpha_hidden=24, alpha_depth=1, alpha_equiv_channels=6,
        alpha_positive_isotropic=True, dipole_head=False, eta_init=0.5,
        scf_max_iter=40, scf_tol=1e-12, scf_damping=0.0,
        attach_steps=1, hessian_in_graph=True,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def make_model(features=None, charge=None, *, randomize=True, seed=1):
    """Small ChargeAwareModel over H/O with (optionally) randomized readouts.

    The zero-initialized readouts make E independent of q beyond the hardness
    parabola; the perturbation makes every q-pathway (embedding -> heads,
    embedding -> density weights) active so the tests exercise real coupling.
    """
    model = build_charge_model(
        features or features_cfg(),
        charge or charge_cfg(),
        [1, 8],
        torch.tensor([-0.5013, -75.0093]),
    )
    if randomize:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for module in (
                model.energy_head, model.charge_embedding, model.weight_net,
                model.alpha_head, model.element_energy,
            ):
                for p in module.parameters():
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
