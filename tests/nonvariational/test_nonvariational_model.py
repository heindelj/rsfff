"""``NonvariationalModel`` (``film.model: nonvariational``) as a whole.

* the film model is untouched: no new parameters, same energies, when the flag is off;
* with the learned widths and quadrupole response off and many iterations, the model is the
  film model at the same parameters (the loop converges to the same functional's minimum);
* an isolated fragment has zero induction and ``b_ind == b`` exactly, at every ``K``;
* in a cluster the induced widths only broaden (``b_ind <= b``);
* finite-difference forces through the loop, and a force loss is twice differentiable;
* energy invariant and induced dipoles equivariant under rotation;
* ``diagnose`` reports a non-negative energy gap to the converged solve.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

import pytest
import torch

_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "film"))
from film_helpers import water_cluster_batch  # noqa: E402
from test_exclusions import excl_model  # noqa: E402

from rsfff.ff.film import FilmModel  # noqa: E402
from rsfff.ff.nonvariational import NonvariationalModel, diagnose  # noqa: E402
from rsfff.train.data import Batch  # noqa: E402


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def nv_model(seed=0, *, randomize=True, **over):
    kw = dict(model="nonvariational", n_iter=3)
    kw.update(over)
    return excl_model(seed, randomize=randomize, **kw)


def _with_positions(batch, pos):
    return Batch(**{**batch.__dict__, "positions": pos})


# --- the film is untouched ---------------------------------------------------------------------

def test_film_model_has_no_new_parameters():
    film = excl_model()
    assert type(film) is FilmModel
    names = [n for n, _ in film.named_parameters()]
    assert not any("width" in n or "cquad" in n or "iterate_weights" in n for n in names)
    nv = nv_model(randomize=False)
    assert isinstance(nv, NonvariationalModel)
    extra = set(n for n, _ in nv.named_parameters()) - set(names)
    assert {"iterate_weights_raw"} <= extra
    assert any("width_mlp" in n for n in extra) and any("cquad" in n for n in extra)


def test_converged_loop_without_the_learned_parts_is_the_film_model():
    """Same seed, same randomisation, flags off: identical parameters. 60 iterations of the
    plain series then reproduce the CG minimum of the same functional."""
    film = excl_model(3, randomize=True, cg_rtol=1e-13, cg_atol=1e-15, cg_maxiter=400)
    nv = nv_model(3, n_iter=60, iterate_weights=False, induced_width=False,
                  induced_quadrupoles=False)
    assert [n for n, _ in film.named_parameters()] == [n for n, _ in nv.named_parameters()]
    for (_, a), (_, b) in zip(film.named_parameters(), nv.named_parameters()):
        assert torch.equal(a, b)
    batch = water_cluster_batch(3)
    o_film, o_nv = film(batch), nv(batch)
    assert torch.allclose(o_film.energy, o_nv.energy, atol=1e-9)
    assert torch.allclose(o_film.interaction["induction"], o_nv.interaction["induction"], atol=1e-9)
    assert torch.allclose(o_film.level_ind.charges, o_nv.level_ind.charges, atol=1e-8)
    assert o_nv.solver["ind"][0] == 60 and bool(o_nv.solver["ind"][1].all())


# --- vacuum limit and the widths ----------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 3])
def test_isolated_fragment_has_zero_induction_and_physical_widths(k):
    model = nv_model(n_iter=k)
    out = model(water_cluster_batch(1))
    assert torch.allclose(out.interaction["induction"], torch.zeros(1), atol=1e-14)
    resp = out.parameters.response
    assert torch.equal(resp.b_ind, resp.b)
    assert torch.equal(resp.s_ind, torch.zeros_like(resp.s_ind))


def test_cluster_widths_only_broaden_and_monomer_polarizability_is_k_independent():
    model = nv_model(iterate_weights=False)
    outs = []
    for k in (1, 4):
        model.n_iter = k
        outs.append(model(water_cluster_batch(3), with_polarizability=True))
    resp = outs[0].parameters.response
    assert bool((resp.b_ind <= resp.b).all()) and bool((resp.s_ind > 0).any())
    assert torch.allclose(outs[0].polarizability, outs[1].polarizability)
    # induced quadrupoles are on: the coupled level's quadrupoles move off the permanent ones
    assert not torch.allclose(outs[0].level_ind.quad_s, outs[0].parameters.quad_perm)


def test_free_width_mode_can_narrow():
    model = nv_model(induced_width_mode="free")
    resp = model(water_cluster_batch(3)).parameters.response
    assert bool((resp.s_ind != 0).any())
    assert torch.equal(model(water_cluster_batch(1)).parameters.response.s_ind,
                       torch.zeros(3))


# --- derivatives ------------------------------------------------------------------------------

def test_finite_difference_forces():
    model = nv_model()
    batch = water_cluster_batch(2)
    pos = batch.positions.clone().requires_grad_(True)
    out = model(_with_positions(batch, pos))
    grad = torch.autograd.grad(out.energy.sum(), pos)[0]
    eps = 1e-5
    for a, x in [(0, 0), (1, 2), (2, 1), (3, 1), (5, 0)]:
        plus, minus = batch.positions.clone(), batch.positions.clone()
        plus[a, x] += eps
        minus[a, x] -= eps
        e_p = model(_with_positions(batch, plus)).energy.sum()
        e_m = model(_with_positions(batch, minus)).energy.sum()
        fd = (e_p - e_m) / (2 * eps)
        assert torch.allclose(grad[a, x], fd, atol=1e-7), (a, x, grad[a, x].item(), fd.item())


def test_force_loss_is_differentiable_in_the_parameters():
    model = nv_model()
    model.train()
    batch = water_cluster_batch(2)
    pos = batch.positions.clone().requires_grad_(True)
    out = model(_with_positions(batch, pos))
    forces = -torch.autograd.grad(out.energy.sum(), pos, create_graph=True)[0]
    loss = forces.pow(2).sum() + out.energy.sum()
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                allow_unused=True)
    named = [n for n, p in model.named_parameters() if p.requires_grad]
    got = {n: g for n, g in zip(named, grads) if g is not None}
    assert all(torch.isfinite(g).all() for g in got.values())
    assert "iterate_weights_raw" in got
    assert any("width_mlp" in n for n in got) and any("cquad" in n for n in got)


# --- symmetry -----------------------------------------------------------------------------------

def test_rotation_invariance_and_equivariance():
    model = nv_model()
    batch = water_cluster_batch(3)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    o1 = model(batch)
    o2 = model(_with_positions(batch, batch.positions @ q.T))
    assert torch.allclose(o1.energy, o2.energy, atol=1e-9)
    assert torch.allclose(o1.level_ind.charges, o2.level_ind.charges, atol=1e-9)
    assert torch.allclose(o1.level_ind.mu @ q.T, o2.level_ind.mu, atol=1e-9)


# --- diagnostics -------------------------------------------------------------------------------

def test_diagnose_reports_a_nonnegative_gap_that_shrinks_with_k():
    gaps = []
    for k in (0, 2, 5):
        model = nv_model(2, n_iter=k, induced_width=False)
        d = diagnose(model, water_cluster_batch(3), rtol=1e-13, atol=1e-15, maxiter=400)
        assert float(d["energy_gap"]) >= -1e-13
        assert torch.isfinite(d["residual"]).all()
        gaps.append(float(d["energy_gap"]))
    assert gaps[0] > gaps[1] > gaps[2]
