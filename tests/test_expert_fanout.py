"""Several experts in one batch: the gather/scatter, and what it must not change.

Every ion-cluster frame mixes an H3O+ or OH- with one or more waters, so the forward has to
evaluate different weights on different atoms and stitch the pieces back together. Nothing
about that is visible in the output -- a wrong gather produces a number, not an error -- so
the checks here are all *comparisons against a computation that does not fan out*.

Two of them carry the weight:

``test_fanout_matches_the_single_group_path``
    with every expert holding identical weights, the fanned-out answer must equal the
    single-group one. That isolates exactly the plumbing: same weights, same data, same
    arithmetic, only the routing differs.
``test_one_body_sector_is_still_sealed``
    the property the whole model exists for -- ``fragment_energy`` reads no environment --
    has to survive the refactor, and it is checked on a *mixed* batch where the fan-out is
    actually running.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.ff.expert import ExpertGroup
from rsfff.ff.expert_model import _stitch
from rsfff.mlip.heads import env_parameters
from rsfff.mlip.reference_states import AtomicStateReference
from rsfff.train.build_expert import build_expert_model
from rsfff.train.data import load_extxyz, load_reference_energies

from test_expert_model import NEIGHBOR_TYPES, _config

#: A three-fragment cation and a three-fragment anion, at two different decompositions each.
#: `fragmentation=1` and `=2` are the strained assignments, which is where the atom order in
#: the file is interleaved and the loader's re-sort is doing work.
CASES = [
    ("w1_h3o+", 0), ("w1_h3o+", 1),
    ("w2_h3o+", 1), ("w1_oh-", 0), ("w2_oh-", 2),
]

COMPOSITIONS = ("H2O", "H3O", "HO")


@pytest.fixture(scope="module")
def build():
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    states = AtomicStateReference.from_json(
        "data/atomic_reference_states_wb97mv_tzvpd.json", NEIGHBOR_TYPES,
        dtype=torch.float64,
    )

    def make(*, induction=True, applicability=True, identical=False, seed=0):
        cfg = _config(environment=True, induction=induction)
        cfg.expert.compositions = COMPOSITIONS
        cfg.expert.applicability = applicability
        torch.manual_seed(seed)
        model = build_expert_model(cfg, NEIGHBOR_TYPES, e0, states).double()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.normal_(0.0, 0.05)
            for _n, p in env_parameters(model):
                p.normal_(0.0, 0.05)
            if identical:
                # Every expert the same, so the only difference between two runs is routing.
                reference = model.experts["H2O"].state_dict()
                for key in COMPOSITIONS[1:]:
                    model.experts[key].load_state_dict(reference)
        return model

    return make


def ion_batch(stem, k, n=3):
    ds = load_extxyz(
        f"data/wb97mv_tzvpd/{stem}_wb97mv_tzvpd.xyz", dtype=torch.float64, fragmentation=k
    )
    return ds.flat_batch(range(min(n, len(ds))))


def group_keys(model, batch):
    feats = model.featurizer(batch, batch.fragment_idx)
    return [
        g.key
        for g in model.experts.groups(
            feats.species_idx, batch.fragment_idx, int(batch.n_fragments)
        )
    ]


# ---------------------------------------------------------------------------------------
# the plumbing
# ---------------------------------------------------------------------------------------

def test_the_batches_really_are_multi_expert(build):
    """Otherwise every comparison below is measuring the single-group fast path twice."""
    model = build()
    for stem, k in CASES:
        keys = group_keys(model, ion_batch(stem, k))
        assert len(set(keys)) > 1, (stem, k, keys)
        assert "H2O" in keys


@pytest.mark.parametrize("stem,k", CASES)
def test_fanout_matches_the_single_group_path(build, stem, k):
    """With identical experts, routing must not change the answer.

    Not bit-identical: concatenating per-expert pieces and gathering them back reorders the
    reductions, so float64 round-off of order 1e-16 Hartree is expected and anything larger
    is a real mis-stitch.
    """
    model = build(identical=True)
    batch = ion_batch(stem, k)

    fanned = model(batch)
    original = model.experts.groups
    model.experts.groups = lambda *_a, **_k: [
        ExpertGroup("H2O", model.experts["H2O"], None, None)
    ]
    try:
        single = model(batch)
    finally:
        model.experts.groups = original

    def same(a, b, name):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-14, rtol=0.0), (
            f"{name}: max|diff| {float((a - b).detach().abs().max()):.3g}"
        )

    for name in ("energy", "fragment_energy", "energy_internal", "energy_bond",
                 "energy_intra", "energy_ref"):
        same(getattr(fanned, name), getattr(single, name), name)
    same(fanned.response.charges, single.response.charges, "charges")
    for channel in fanned.interaction:
        same(fanned.interaction[channel], single.interaction[channel], channel)


@pytest.mark.parametrize("stem,k", CASES)
def test_the_accounting_identity_survives_the_fanout(build, stem, k):
    model = build()
    batch = ion_batch(stem, k)
    out = model(batch)
    total = out.fragment_energy.new_zeros(batch.n_systems).index_add_(
        0, batch.fragment_to_batch, out.fragment_energy
    )
    for value in out.interaction.values():
        total = total + value
    assert torch.allclose(out.energy, total, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("stem,k", CASES)
def test_charge_is_conserved_per_fragment_across_experts(build, stem, k):
    """The SQE solve runs once for the whole batch; its blocks must still close per fragment."""
    model = build()
    batch = ion_batch(stem, k)
    q = model(batch).response.charges
    per_fragment = q.new_zeros(int(batch.n_fragments)).index_add_(0, batch.fragment_idx, q)
    assert torch.allclose(
        per_fragment.detach(), batch.fragment_charge.to(per_fragment.dtype),
        atol=1e-12, rtol=0.0,
    )


def test_one_body_sector_is_still_sealed(build):
    """``fragment_energy`` reads no environment -- checked where the fan-out is live."""
    model = build()
    batch = ion_batch("w2_h3o+", 1)
    before = model(batch)
    with torch.no_grad():
        for _n, p in env_parameters(model):
            p.normal_(0.0, 0.5)
    after = model(batch)
    assert torch.equal(before.fragment_energy, after.fragment_energy)
    # And the interaction did move, or the test proved nothing about a sealed sector.
    assert not torch.equal(before.interaction["disp"], after.interaction["disp"])


def test_gradients_reach_every_expert(build):
    """A stitch that dropped a group would still produce an energy, just a wrong one."""
    model = build()
    batch = ion_batch("w2_h3o+", 1)
    model(batch).energy.sum().backward()
    for key in ("H2O", "H3O"):
        grads = [
            p.grad for p in model.experts[key].parameters()
            if p.grad is not None and float(p.grad.abs().max()) > 0
        ]
        assert grads, f"no gradient reached the {key} expert"


# ---------------------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------------------

def test_alpha_is_shared_between_experts():
    """One number per channel for the whole model -- a pair spanning two experts needs that.

    Shared by object identity rather than by value, so training moves one tensor and no copy
    can drift or be decayed away while unused.
    """
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    cfg = _config(environment=True)
    cfg.expert.compositions = COMPOSITIONS
    model = build_expert_model(cfg, NEIGHBOR_TYPES, e0, None).double()
    for channel in model.experts["H2O"].range_heads.channel_names:
        first = model.experts["H2O"].range_heads.alpha_raw[channel]
        for key in COMPOSITIONS[1:]:
            assert model.experts[key].range_heads.alpha_raw[channel] is first
    # r0, by contrast, is per element per expert and must NOT be shared.
    assert (
        model.experts["H2O"].range_heads.d_log_r0["elst"]
        is not model.experts["HO"].range_heads.d_log_r0["elst"]
    )


def test_stitch_puts_rows_back_where_they_came_from():
    torch.manual_seed(0)
    values = torch.randn(7, 3)
    a = torch.tensor([0, 3, 4])
    b = torch.tensor([1, 2, 5, 6])
    assert torch.equal(_stitch([(a, values[a]), (b, values[b])], 7), values)


def test_stitch_handles_nested_structures_and_none():
    a, b = torch.tensor([0, 2]), torch.tensor([1])
    parts = [
        (a, ({"x": torch.tensor([1.0, 3.0])}, None)),
        (b, ({"x": torch.tensor([2.0])}, None)),
    ]
    got = _stitch(parts, 3)
    assert got[1] is None
    assert torch.equal(got[0]["x"], torch.tensor([1.0, 2.0, 3.0]))


def test_stitch_refuses_a_partition_with_holes():
    a = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="do not partition"):
        _stitch([(a, torch.zeros(2))], 3)


def test_an_unknown_composition_still_raises(build):
    """The fan-out must not have turned a missing expert into a silent fallback."""
    model = build()
    batch = ion_batch("w1_h3o+", 0)
    del model.experts.experts["H3O"]
    with pytest.raises(KeyError, match="no expert for fragment composition"):
        model(batch)


def test_experts_must_agree_on_the_multipole_form():
    """One solve serves the batch, so the experts cannot disagree about what it is solving."""
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    from rsfff.ff.expert_model import FragmentExpertModel

    cfg = _config(environment=True)
    cfg.expert.compositions = ("H2O", "HO")
    model = build_expert_model(cfg, NEIGHBOR_TYPES, e0, None).double()
    model.experts["HO"].response.direct_multipoles = not model._direct_multipoles
    with pytest.raises(ValueError, match="disagree about `direct_multipoles`"):
        FragmentExpertModel(
            model.featurizer, model.experts, e0, max_rank=cfg.expert.max_rank
        )


def test_applicability_is_emitted_per_fragment_by_the_owning_expert(build):
    model = build(applicability=True)
    batch = ion_batch("w2_oh-", 2)
    out = model(batch)
    assert out.applicability.shape == (int(batch.n_fragments),)
    # Waking only the hydroxide expert's head must move only hydroxide rows.
    feats = model.featurizer(batch, batch.fragment_idx)
    groups = {
        g.key: g for g in model.experts.groups(
            feats.species_idx, batch.fragment_idx, int(batch.n_fragments)
        )
    }
    before = out.applicability.detach().clone()
    with torch.no_grad():
        model.experts["HO"].applicability.net[-1].bias.add_(1.0)
    after = model(batch).applicability.detach()
    moved = (after - before).abs() > 1e-12
    assert torch.equal(
        moved.nonzero().squeeze(-1).sort().values, groups["HO"].fragment_index.sort().values
    )
