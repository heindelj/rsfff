"""The film model with hard 1-2/1-3 exclusions (``film.nonbonded: exclusions``).

What has to hold for the non-reactive model:

* **exclusion list** -- read off the covalent graph: every 1-2 and 1-3 pair (and 1-4 with
  ``exclude_through=4``), never a distance;
* **no intramolecular nonbonded energy** -- for water every intra pair is excluded, so the
  fragment energy is exactly ``sum E0 + Morse + angle`` and a monomer stretch scan is the
  bonded curve itself (no Fermi switch to put a kink in it);
* **no range heads** -- no ``r0``/``alpha`` parameters, and the inter gate is the taper alone;
* the film invariants carry over: vertex, spectator isolation, accounting, forces.
"""

from __future__ import annotations

import types

import pytest
import torch

from rsfff.ff.film.bonded import BondedTopology
from rsfff.train.build_film import build_film_model
from rsfff.train.data import Batch

from film_helpers import water_cluster_batch


def excl_model(seed: int = 0, *, randomize: bool = False, **film_over):
    torch.manual_seed(seed)
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
        backend="e3nn", density_channels=4,
    )
    film = types.SimpleNamespace(
        hidden=32, block_dim=24, head_hidden=24, head_depth=1,
        equiv_channels=6, bonded_hidden=24, nonbonded="exclusions", **film_over,
    )
    model = build_film_model(features, film, [1, 8], torch.tensor([-0.5013, -75.0093]))
    if randomize:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def _chain_topology(n: int) -> BondedTopology:
    """A linear chain 0-1-2-...-(n-1): only ``bond_index`` matters for the exclusions."""
    bonds = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
    z = torch.zeros(0, dtype=torch.long)
    return BondedTopology(bonds, z, z.double(), torch.zeros(3, 0, dtype=torch.long), z,
                          z.double())


def _pairs(t: torch.Tensor) -> set[tuple[int, int]]:
    return {tuple(p) for p in t.T.tolist()}


# --- the exclusion list ----------------------------------------------------------------------

def test_chain_exclusions():
    topo = _chain_topology(5)
    one_two = {(0, 1), (1, 2), (2, 3), (3, 4)}
    one_three = {(0, 2), (1, 3), (2, 4)}
    one_four = {(0, 3), (1, 4)}
    assert _pairs(topo.exclusions(5, 3)) == one_two | one_three
    assert _pairs(topo.exclusions(5, 4)) == one_two | one_three | one_four
    assert _pairs(topo.exclusions(5, 2)) == one_two


def test_ring_exclusions_have_no_self_pairs():
    """A 3-ring: walks come back to the start; the self pairs must be dropped."""
    bonds = torch.tensor([[0, 1, 0], [1, 2, 2]])
    z = torch.zeros(0, dtype=torch.long)
    topo = BondedTopology(bonds, z, z.double(), torch.zeros(3, 0, dtype=torch.long), z,
                          z.double())
    assert _pairs(topo.exclusions(3, 4)) == {(0, 1), (0, 2), (1, 2)}


def test_water_exclusions_are_every_intra_pair():
    model = excl_model()
    batch = water_cluster_batch(3)
    out = model(batch)
    # O-H, O-H (1-2) and H-H (1-3) in each water: nothing intramolecular survives
    assert not bool(out.is_intra.any())
    assert out.pair_frag.eq(-1).all()
    assert torch.count_nonzero(out.energy_intra) == 0
    # ... while every intermolecular pair within the cutoff is still there
    frag = batch.fragment_idx
    i, j = torch.triu_indices(9, 9, offset=1)
    inter = frag[i] != frag[j]
    assert out.pair_index.shape[1] == int(inter.sum())


# --- the model ------------------------------------------------------------------------------

def test_no_range_heads():
    model = excl_model()
    assert model.range_heads is None
    assert not any("range_heads" in k for k in model.state_dict())
    out = model(water_cluster_batch(2))
    assert out.r0 == {} and out.alpha == {} and out.r0_pair == {}
    # the inter gate is the taper alone: exactly 1 inside cutoff - taper_width
    for name, spec in model.classical.items():
        inside = out.r < spec.cutoff - spec.taper_width
        assert torch.equal(out.gate[name][inside], torch.ones_like(out.r[inside])), name


def test_fragment_energy_is_bonded_only():
    model = excl_model(randomize=True)
    out = model(water_cluster_batch(3))
    assert torch.allclose(
        out.fragment_energy, out.energy_ref + out.energy_bonded, atol=0, rtol=0
    )


def test_monomer_stretch_is_the_morse_curve():
    """An O-H stretch of a lone water is the bonded energy and nothing else: smooth all the
    way out (the range-separated model's intra switch is what put the kink here)."""
    model = excl_model(randomize=True)
    batch = water_cluster_batch(1, jitter=0.0)
    oh = batch.positions[1] - batch.positions[0]
    unit = oh / oh.norm()
    energies, bonded = [], []
    rs = torch.linspace(0.75, 3.0, 181)
    for r in rs:
        pos = batch.positions.clone()
        pos[1] = pos[0] + r * unit
        out = model(Batch(**{**batch.__dict__, "positions": pos}))
        energies.append(out.energy)
        bonded.append(out.energy_ref + out.energy_bonded)
    e = torch.cat(energies)
    assert torch.allclose(e, torch.cat(bonded), atol=0, rtol=0)
    assert torch.isfinite(e).all()


def test_vertex_and_spectator():
    model = excl_model(randomize=True)
    lone = model(water_cluster_batch(1))
    assert lone.interaction["induction"].item() == 0.0
    assert torch.allclose(lone.energy, lone.fragment_energy, atol=0, rtol=0)

    alone = water_cluster_batch(1)
    far = water_cluster_batch(2)
    far.positions = torch.cat(
        (alone.positions, alone.positions + torch.tensor([50.0, 0.0, 0.0]))
    )
    out_far = model(far)
    assert torch.allclose(out_far.fragment_energy[0], lone.fragment_energy[0], atol=1e-14,
                          rtol=0)
    assert abs(out_far.interaction["induction"].item()) < 1e-14


def test_accounting():
    model = excl_model(randomize=True)
    batch = water_cluster_batch(3)
    out = model(batch)
    total = out.fragment_energy.sum() + sum(v.sum() for v in out.interaction.values())
    assert torch.allclose(total, out.energy.sum(), atol=1e-12)


@pytest.mark.parametrize("induction", [False, True])
def test_finite_difference_forces(induction):
    over = dict(cg_rtol=1e-13, cg_atol=1e-15, cg_maxiter=400) if induction else {}
    model = excl_model(randomize=True, **over)
    model.induction = induction
    batch = water_cluster_batch(2)
    pos = batch.positions.clone().requires_grad_(True)
    out = model(Batch(**{**batch.__dict__, "positions": pos}))
    grad = torch.autograd.grad(out.energy.sum(), pos)[0]
    eps = 1e-5
    for a, x in [(0, 0), (1, 2), (2, 1), (3, 1), (5, 0)]:
        plus, minus = batch.positions.clone(), batch.positions.clone()
        plus[a, x] += eps
        minus[a, x] -= eps
        e_p = model(Batch(**{**batch.__dict__, "positions": plus})).energy.sum()
        e_m = model(Batch(**{**batch.__dict__, "positions": minus})).energy.sum()
        fd = (e_p - e_m) / (2 * eps)
        assert torch.allclose(grad[a, x], fd, atol=1e-6), (a, x, grad[a, x].item(), fd.item())


def test_short_contact_is_smooth():
    """Push one water's H onto the other's O: no switch turns on anywhere along the way,
    so the dimer energy has no kink (second difference stays O(h^2))."""
    model = excl_model(randomize=True)
    batch = water_cluster_batch(2, jitter=0.0)
    o_b = batch.positions[3]
    h_a = batch.positions[1]
    direction = (o_b - h_a) / (o_b - h_a).norm()
    e = []
    shifts = torch.linspace(0.0, 1.2, 241)
    for s in shifts:
        pos = batch.positions.clone()
        pos[3:] = pos[3:] - s * direction
        e.append(model(Batch(**{**batch.__dict__, "positions": pos})).energy)
    e = torch.cat(e)
    d2 = e[2:] - 2 * e[1:-1] + e[:-2]
    # no isolated spike: the largest curvature is comparable to its neighbors'
    ratio = d2.abs().max() / d2.abs()[1:-1].median().clamp(min=1e-300)
    assert torch.isfinite(e).all()
    assert ratio < 1e3


def test_training_penalties_skip_range_terms():
    from rsfff.train.config import Config
    from rsfff.train.train_film import FilmStreams

    model = excl_model(randomize=True)
    cfg = Config()
    cfg.film.nonbonded = "exclusions"
    streams = FilmStreams(model, "cpu", fragment_dataset=[], fragment_batch_size=4)
    batch = water_cluster_batch(2)
    out = model(batch)
    extra = streams.penalties(out, batch, cfg)
    assert "r0" not in extra and "r0_spread" not in extra
    metrics = streams.diagnostics(out, batch, None)
    assert not any(k.startswith("r0_") for k in metrics)


def test_range_separated_default_unchanged():
    """Nothing about the default model changed: it still has its range heads."""
    torch.manual_seed(0)
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2),
        backend="e3nn", density_channels=4,
    )
    film = types.SimpleNamespace(hidden=32, block_dim=24, head_hidden=24, head_depth=1,
                                 equiv_channels=6, bonded_hidden=24)
    model = build_film_model(features, film, [1, 8], torch.tensor([-0.5013, -75.0093]))
    assert model.nonbonded == "range_separated" and model.range_heads is not None
    out = model(water_cluster_batch(2))
    assert bool(out.is_intra.any())
