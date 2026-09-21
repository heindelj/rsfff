"""``rsfff.ff.backend``: the torch path and the torchff path are the same function.

On CPU (here) the torchff path runs ``torchff.ffterms``' pure-torch references, so this pins
that those references *are* rsfff's formulas; on a CUDA machine with the compiled extension
the same tests exercise the kernels, including double backward. Both are run through the
whole film model, not just the leaves, so the wiring in ``FilmModel.forward`` is covered.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from rsfff.ff import backend
from rsfff.ff.film.bonded import BondedTopology
from rsfff.train.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "film"))
from film_helpers import water_cluster_batch  # noqa: E402
from test_model import small_model  # noqa: E402

needs_torchff = pytest.mark.skipif(not backend.HAVE_TORCHFF, reason="torchff not installed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def both_backends():
    yield
    backend.set_backend("auto")


def _to(batch: Batch, device):
    from dataclasses import fields, replace
    kw = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            kw[f.name] = v.to(device)
    return replace(batch, **kw)


def _run(model, batch, name: str, *, second_order: bool):
    backend.set_backend(name)
    batch = _to(batch, DEVICE)
    batch.positions = batch.positions.detach().clone().requires_grad_(True)
    out = model(batch)
    (forces,) = torch.autograd.grad(out.energy.sum(), batch.positions, create_graph=second_order)
    res = dict(
        energy=out.energy.detach(),
        forces=forces.detach(),
        bonded=out.energy_bonded.detach(),
        **{f"e_{k}": v.detach() for k, v in out.interaction.items()},
    )
    if second_order:
        # the force-loss gradient into every parameter: this is the double backward that a
        # first-order-only kernel would silently get wrong
        loss = forces.pow(2).sum() + out.energy.pow(2).sum()
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
        res["param_grad"] = torch.cat([g.reshape(-1) for g in grads if g is not None]).detach()
    return res


@needs_torchff
@pytest.mark.parametrize("n_waters", [1, 3])
def test_film_model_agrees_across_backends(both_backends, n_waters):
    model = small_model(seed=3, randomize=True).to(DEVICE)
    batch = water_cluster_batch(n_waters, jitter=0.08, seed=11)
    a = _run(model, batch, "torch", second_order=True)
    b = _run(model, batch, "torchff", second_order=True)
    for key in a:
        assert torch.allclose(a[key], b[key], rtol=1e-9, atol=1e-11), key


@needs_torchff
def test_leaf_dispersion_matches(both_backends):
    torch.manual_seed(0)
    pos = (torch.rand(9, 3, dtype=torch.float64, device=DEVICE) * 4).requires_grad_(True)
    ii, jj = torch.triu_indices(9, 9, 1)
    pair_index = torch.stack([ii, jj]).to(DEVICE)
    c6 = (torch.rand(pair_index.shape[1], dtype=torch.float64, device=DEVICE) * 30 + 1).requires_grad_(True)
    b = (torch.rand(pair_index.shape[1], dtype=torch.float64, device=DEVICE) + 1.0).requires_grad_(True)
    backend.set_backend("torch")
    e_t = backend.tt_dispersion(pos, pair_index, c6, b)
    backend.set_backend("torchff")
    e_f = backend.tt_dispersion(pos, pair_index, c6, b)
    assert torch.allclose(e_t, e_f, rtol=1e-12, atol=1e-15)
    g_t = torch.autograd.grad(e_t.sum(), (pos, c6, b))
    g_f = torch.autograd.grad(e_f.sum(), (pos, c6, b))
    for x, y in zip(g_t, g_f):
        assert torch.allclose(x, y, rtol=1e-11, atol=1e-14)


@needs_torchff
def test_leaf_bonded_matches(both_backends):
    from rsfff.ff.film.state import StateDescriptor
    from film_helpers import make_projector
    batch = _to(water_cluster_batch(2, jitter=0.1, seed=5), DEVICE)
    projector = make_projector().to(DEVICE)
    species = projector.species_index(batch.atomic_numbers)
    state = StateDescriptor.from_batch(batch, species, projector.featurizer.n_species)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    nb, na = topo.bond_index.shape[1], topo.angle_index.shape[1]
    from rsfff.ff.film.bonded import BondedParameters
    params = BondedParameters(
        r_eq=torch.full((nb,), 1.81, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        d=torch.full((nb,), 0.2, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        k=torch.full((nb,), 0.54, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        cos_theta_eq=torch.full((na,), -0.25, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        k_theta=torch.full((na,), 0.17, dtype=torch.float64, device=DEVICE).requires_grad_(True),
    )
    pos = batch.positions.detach().clone().requires_grad_(True)
    backend.set_backend("torch")
    eb_t, ea_t = backend.bonded_energy(pos, topo, params)
    backend.set_backend("torchff")
    eb_f, ea_f = backend.bonded_energy(pos, topo, params)
    assert torch.allclose(eb_t, eb_f, rtol=1e-12, atol=1e-15)
    assert torch.allclose(ea_t, ea_f, rtol=1e-12, atol=1e-15)
    leaves = (pos, params.r_eq, params.d, params.k, params.cos_theta_eq, params.k_theta)
    g_t = torch.autograd.grad((eb_t.sum() + ea_t.sum()), leaves)
    g_f = torch.autograd.grad((eb_f.sum() + ea_f.sum()), leaves)
    for x, y in zip(g_t, g_f):
        assert torch.allclose(x, y, rtol=1e-11, atol=1e-14)


def test_backend_selection():
    backend.set_backend("torch")
    assert backend.active_backend(torch.zeros(1)) == "torch"
    with pytest.raises(ValueError):
        backend.set_backend("nope")
    backend.set_backend("auto")
    assert backend.active_backend(torch.zeros(1)) == "torch"  # CPU tensor -> torch path
