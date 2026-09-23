"""Checks of the uncertainty-driven sampler against the real committee (CPU, ~2 min).

    RSFFF_COMMITTEE=committees/film_committee_100k python -m pytest tests -q
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsfff_al.committee import Committee, Topology  # noqa: E402
from rsfff_al.dynamics import DynamicsConfig, flat_bottom_wall, run_replicas  # noqa: E402
from rsfff_al.select import choose, pool_indices, scores  # noqa: E402

COMMITTEE = os.environ.get("RSFFF_COMMITTEE", str(ROOT / "committees" / "film_committee_100k"))
TRIMER = np.array([
    [0.000, 0.000, 0.000], [0.957, 0.000, 0.000], [-0.240, 0.927, 0.000],
    [2.900, 0.100, 0.000], [3.300, 0.950, 0.200], [3.250, -0.600, 0.550],
    [1.300, 2.600, 0.300], [0.700, 1.850, 0.200], [2.150, 2.200, 0.100],
])


@pytest.fixture(scope="module")
def committee():
    if not Path(COMMITTEE).exists():
        pytest.skip(f"no committee at {COMMITTEE}")
    return Committee.load(COMMITTEE, device="cpu")


def test_grad_sigma_matches_finite_difference(committee):
    top = Topology.water(3)
    x = torch.tensor(np.stack([TRIMER, TRIMER + 0.05 * np.random.default_rng(0).normal(size=TRIMER.shape)]))
    g = committee.evaluate(x, top).grad_sigma()
    h = 1e-4
    for atom, axis in [(0, 0), (4, 1), (8, 2)]:
        xp, xm = x.clone(), x.clone()
        xp[:, atom, axis] += h
        xm[:, atom, axis] -= h
        fd = (committee.evaluate(xp, top).sigma_energy
              - committee.evaluate(xm, top).sigma_energy) / (2 * h)
        assert torch.allclose(g[:, atom, axis], fd, rtol=1e-4, atol=1e-8)


def test_replicas_are_independent(committee):
    """A replica's energy does not depend on what else is in the batch."""
    top = Topology.water(3)
    one = committee.evaluate(torch.tensor(TRIMER[None]), top)
    two = committee.evaluate(torch.tensor(np.stack([TRIMER, TRIMER + 0.3])), top)
    assert torch.allclose(one.energy[:, 0], two.energy[:, 0], atol=1e-10)
    assert torch.allclose(one.forces[:, 0], two.forces[:, 0], atol=1e-9)


def test_wall_has_no_net_force():
    top = Topology.water(3)
    x = torch.tensor(TRIMER[None] * 2.0)
    e, f = flat_bottom_wall(x, top, torch.tensor([1.0]), 0.5, 1.2)
    assert float(e[0]) > 0
    assert torch.allclose(f.sum(1), torch.zeros(1, 3, dtype=f.dtype), atol=1e-12)


def test_short_biased_run_and_rewind(committee):
    """A few hundred steps: frames recorded, a forced guard failure rewinds instead of dying."""
    cfg = DynamicsConfig(time_ps=0.02, warmup_fs=10.0, stride=5, max_oh=1.0, max_restarts=1,
                         rewind_fs=5.0, bias_mode="matched")
    out = run_replicas(committee, Topology.water(3), np.stack([TRIMER, TRIMER]), cfg,
                       log=lambda s: None)
    # max_oh=1.0 A is tighter than thermal O-H motion, so some replica must have failed
    assert any(r["failures"] for r in out["replicas"])
    assert all(r["status"] in ("complete", "retired") for r in out["replicas"])
    assert np.all(np.isfinite(out["sigma_energy"]))


def test_selection_respects_budget_and_eda():
    rng = np.random.default_rng(1)
    rep = np.repeat(np.arange(4), 50)
    t = np.tile(np.arange(50) * 10.0, 4)
    se, sf = rng.random(200), rng.random(200)
    sc = scores(se, sf)
    rows = pool_indices(rep, t, sc, window_fs=100.0)
    assert len(rows) == 4 * 5
    chosen, eda = choose(sc[rows], rep[rows], 8, n_eda=3)
    assert len(chosen) == 8 and len(eda) == 3 and set(eda) <= set(chosen)
    assert max(np.bincount(rep[rows][chosen])) <= 4
