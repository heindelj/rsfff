"""Many-body expansion of the dispersion energy.

The load-bearing test is :func:`test_additive_model_has_no_many_body`: a pair sum with
per-species coefficients is *exactly* additive, so its 3-body and higher terms must vanish
to machine precision. If the MBE code has an indexing or sign error, that test fails --
and only then is nonzero many-body content from the environment-aware model meaningful.
"""

import pytest
import torch

from rsfff.ff.many_body import mbe_dataset, mbe_decompose, subset_batch
from rsfff.train.data import load_extxyz

from conftest import DATA_W3
from test_ff_dispersion import make_featurizer, make_model

from rsfff.ff.dispersion import DispersionModel


@pytest.fixture(scope="module")
def w3():
    return load_extxyz(DATA_W3, dtype=torch.float64)


def wrap(feat, disp, **kw):
    return DispersionModel(feat, disp, **kw)


def additive_model():
    """Per-species C6/b, no environment MLP, no correction: strictly pairwise."""
    feat, disp = make_model(correction=False, r0_init=1.5, learn_r0=False)
    disp.params.c6_mlp = None
    disp.params.b_mlp = None
    return wrap(feat, disp)


def environment_model(seed=3):
    """Environment-dependent C6 plus a pair correction -- both sources of non-additivity."""
    feat, disp = make_model(correction=True, r0_init=1.5, learn_r0=False, randomize=True,
                            seed=seed)
    return wrap(feat, disp)


def frame(dataset, i=0):
    b = dataset.flat_batch([i])
    return b.positions, b.atomic_numbers, b.fragment_idx


# ---------------------------------------------------------------------------
# subset batching
# ---------------------------------------------------------------------------

def test_subset_batch_layout(w3):
    positions, numbers, fragment_idx = frame(w3)
    batch = subset_batch(positions, numbers, fragment_idx, [(0, 1), (2,), (0, 1, 2)])
    assert batch.n_systems == 3
    assert batch.batch_idx.tolist() == [0] * 6 + [1] * 3 + [2] * 9
    # Fragment ids are renumbered per subset and offset batch-globally, as flat_batch
    # does: 2 + 1 + 3 = 6 distinct fragments across the three systems.
    assert batch.fragment_idx.tolist() == (
        [0, 0, 0, 1, 1, 1] + [2, 2, 2] + [3, 3, 3, 4, 4, 4, 5, 5, 5]
    )
    assert batch.n_fragments == 6
    assert torch.all(batch.fragment_idx.diff() >= 0)
    # the (0,1,2) system must reproduce the original geometry exactly
    assert torch.equal(batch.positions[9:], positions)


def test_subset_batch_selects_the_right_atoms(w3):
    positions, numbers, fragment_idx = frame(w3)
    batch = subset_batch(positions, numbers, fragment_idx, [(1, 2)])
    keep = (fragment_idx != 0)
    assert torch.equal(batch.positions, positions[keep])
    assert torch.equal(batch.atomic_numbers, numbers[keep])


def test_subset_batch_carries_the_fragment_state(w3):
    """Charge and spin follow their fragment into every subset that contains it.

    Dropping them is invisible on neutral closed-shell water -- ``FragmentStateEmbedding`` is
    identically zero there -- and wrong the moment a cluster carries an ion: every fragment
    would read as neutral, so the decomposition would describe a different system than the
    one asked about, silently and with no shape error to catch it.
    """
    positions, numbers, fragment_idx = frame(w3)
    charge = torch.tensor([1.0, 0.0, -1.0], dtype=positions.dtype)
    two_s = torch.tensor([0.0, 2.0, 0.0], dtype=positions.dtype)
    subsets = [(0, 1), (1, 2), (0, 1, 2)]
    batch = subset_batch(
        positions, numbers, fragment_idx, subsets,
        fragment_charge=charge, fragment_two_s=two_s,
    )

    want_c = torch.tensor([1.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0], dtype=positions.dtype)
    want_s = torch.tensor([0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0], dtype=positions.dtype)
    assert torch.equal(batch.fragment_charge, want_c)
    assert torch.equal(batch.fragment_two_s, want_s)
    assert batch.fragment_to_batch.tolist() == [0, 0, 1, 1, 2, 2, 2]

    # and it stays optional, for models and callers that carry no fragment state
    plain = subset_batch(positions, numbers, fragment_idx, subsets)
    assert plain.fragment_charge is None and plain.fragment_two_s is None


def test_the_independence_check_fires_when_subsets_can_see_each_other(w3):
    """The guard inside ``mbe_decompose`` catches a model that pools across systems.

    It replaced an assertion comparing ``sum_k E^(k)`` against ``E(full)``, which is an
    algebraic identity of the Moebius inversion over the same lookup dict and so could not
    fail for any model at all. This one re-evaluates the full cluster on its own and compares
    it against the copy that rode in the batched subsets, which is the assumption the whole
    decomposition rests on.
    """
    inner = additive_model()

    class LeaksAcrossSystems(torch.nn.Module):
        """Adds a term depending on the whole batch, which no per-system model may do."""

        def forward(self, batch):
            out = inner(batch)
            return type("O", (), {
                "energy": out.energy + 1e-3 * batch.positions.norm(),
            })()

    with pytest.raises(RuntimeError, match="not independent"):
        mbe_decompose(LeaksAcrossSystems(), *frame(w3), split_components=False)

    # the honest model passes the same guard
    mbe_decompose(inner, *frame(w3), split_components=False)


# ---------------------------------------------------------------------------
# the expansion itself
# ---------------------------------------------------------------------------

def test_additive_model_has_no_many_body(w3):
    """A pair sum with per-species coefficients: 3-body must be exactly zero."""
    model = additive_model()
    res = mbe_decompose(model, *frame(w3))
    assert res.by_order[3].abs().max() < 1e-15
    assert torch.allclose(res.two_body, res.total, atol=1e-15)
    assert res.many_body.abs().max() < 1e-15


def test_environment_model_has_many_body(w3):
    """Environment-aware coefficients make the same pair sum non-additive."""
    model = environment_model()
    res = mbe_decompose(model, *frame(w3))
    assert res.by_order[3].abs().item() > 1e-9


def test_expansion_sums_to_the_total(w3):
    """``E = sum_k E^(k)`` exactly.

    A property of the Moebius inversion rather than of the model, which is why
    ``mbe_decompose`` no longer checks it internally -- it cannot fail, so it was not a guard.
    Pinned here, where a truthful identity test belongs.
    """
    for model in (additive_model(), environment_model()):
        res = mbe_decompose(model, *frame(w3))
        rebuilt = sum(res.by_order.values())
        assert torch.allclose(rebuilt, res.total, atol=1e-12)


def test_two_body_equals_explicit_dimer_sum(w3):
    """E^(2) must equal the sum over isolated dimers, since dE(S)=E(S) at order 2."""
    model = environment_model()
    positions, numbers, fragment_idx = frame(w3)
    res = mbe_decompose(model, positions, numbers, fragment_idx)

    total = 0.0
    with torch.no_grad():
        for pair in ((0, 1), (0, 2), (1, 2)):
            batch = subset_batch(positions, numbers, fragment_idx, [pair])
            total += float(model(batch).energy)
    assert res.two_body.item() == pytest.approx(total, abs=1e-12)


def test_components_sum_to_the_total(w3):
    """The ff/corr split is a partition of the same decomposition, order by order."""
    res = mbe_decompose(environment_model(), *frame(w3))
    assert res.components is not None and set(res.components) == {"ff", "corr"}
    assert torch.allclose(
        res.components["ff"].total + res.components["corr"].total, res.total, atol=1e-12
    )
    for k in res.by_order:
        assert torch.allclose(
            res.components["ff"].by_order[k] + res.components["corr"].by_order[k],
            res.by_order[k], atol=1e-12,
        )


def test_separated_fragments_have_no_interaction(w3):
    """Pull one monomer far away: its pair and triple terms go to exactly zero."""
    model = environment_model()
    positions, numbers, fragment_idx = frame(w3)
    positions = positions.clone()
    positions[fragment_idx == 2] += 60.0
    res = mbe_decompose(model, positions, numbers, fragment_idx)
    assert res.by_order[3].abs().item() < 1e-14


def test_intra_fragment_features_are_additive(w3):
    """The ablation switch is an independent route to a strictly additive model."""
    feat, disp = make_model(correction=False, r0_init=1.5, learn_r0=False, randomize=True)
    model = DispersionModel(feat, disp, intra_fragment=True)
    res = mbe_decompose(model, *frame(w3))
    assert res.by_order[3].abs().max() < 1e-14


def test_mbe_dataset_stacks_frames(w3):
    res = mbe_dataset(environment_model(), w3, range(4))
    assert res.total.shape == (4,)
    assert res.by_order[2].shape == (4,)
    assert res.n_fragments.tolist() == [3, 3, 3, 3]
    assert torch.allclose(sum(res.by_order.values()), res.total, atol=1e-12)


def test_max_order_truncation(w3):
    res = mbe_decompose(environment_model(), *frame(w3), max_order=2)
    assert set(res.by_order) == {2}
