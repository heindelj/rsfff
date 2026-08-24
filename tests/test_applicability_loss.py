"""The decomposition-quality term: what it fits, and what it must refuse to claim.

The score says which fragmentation of a geometry is the best description of it. There is no
label for that directly -- what there is, is ALMO-EDA's ``E_pol + E_ct``, the relaxation from
frozen fragments to the true wavefunction, and the physical statement that the decomposition
needing the least relaxation is the right one.

The metric is as load-bearing as the loss here. ``app_acc`` is the number a human reads to
decide whether the head learned anything, and its first version reported **1.0 for an
untrained head**: a zero-initialized readout scores every decomposition identically, and a
"is this frame at the maximum" test marks all of them. ``test_accuracy_is_zero_for_a_model_
with_no_opinion`` is that bug, pinned.
"""

from __future__ import annotations

import types

import pytest
import torch

from rsfff.ff.units import KJMOL_PER_HARTREE
from rsfff.train.loss import applicability_loss


def make(scores, pol_ct_kjmol, group_id, n_frag_per_frame=1):
    """A minimal ``(out, batch)`` pair: one fragment per frame unless told otherwise."""
    scores = torch.tensor(scores, dtype=torch.float64)
    n_frames = len(group_id)
    out = types.SimpleNamespace(applicability=scores)
    per_frame = torch.tensor(pol_ct_kjmol, dtype=torch.float64) / KJMOL_PER_HARTREE
    batch = types.SimpleNamespace(
        n_systems=n_frames,
        fragment_to_batch=torch.repeat_interleave(
            torch.arange(n_frames), n_frag_per_frame
        ),
        group_id=torch.tensor(group_id, dtype=torch.long),
        # Split arbitrarily between the two components; only the sum is read.
        eda={"pol": per_frame * 0.4, "ct": per_frame * 0.6},
    )
    return out, batch


def test_it_prefers_the_least_perturbed_decomposition():
    """Two decompositions of one geometry; the one with smaller |E_pol + E_ct| is the target."""
    out, batch = make([0.0, 0.0], [-100.0, -400.0], [7, 7])
    terms, metrics = applicability_loss(out, batch, weight=1.0, temperature=50.0)
    assert "applicability" in terms
    # Uniform scores against a near-one-hot target: the loss is ~ -log(0.5) = 0.693.
    assert float(terms["applicability"]) == pytest.approx(0.693, abs=0.01)

    # Score the low-perturbation frame higher and the loss must fall.
    better, _ = make([5.0, 0.0], [-100.0, -400.0], [7, 7])
    lower, _ = applicability_loss(better, batch, weight=1.0, temperature=50.0)
    assert float(lower["applicability"]) < float(terms["applicability"])
    # Score it the wrong way round and it must rise.
    worse, _ = make([0.0, 5.0], [-100.0, -400.0], [7, 7])
    higher, _ = applicability_loss(worse, batch, weight=1.0, temperature=50.0)
    assert float(higher["applicability"]) > float(terms["applicability"])


def test_accuracy_is_zero_for_a_model_with_no_opinion():
    """A zero-init head scores every decomposition alike. That is 0% right, not 100%."""
    out, batch = make([0.0, 0.0, 0.0], [-100.0, -400.0, -500.0], [1, 1, 1])
    _terms, metrics = applicability_loss(out, batch, weight=1.0)
    assert metrics["app_acc"] == 0.0


def test_accuracy_counts_a_correct_unique_argmax():
    out, batch = make([5.0, 0.0, 0.0], [-100.0, -400.0, -500.0], [1, 1, 1])
    _terms, metrics = applicability_loss(out, batch, weight=1.0)
    assert metrics["app_acc"] == 1.0

    out, batch = make([0.0, 5.0, 0.0], [-100.0, -400.0, -500.0], [1, 1, 1])
    _terms, metrics = applicability_loss(out, batch, weight=1.0)
    assert metrics["app_acc"] == 0.0


def test_uncontested_geometries_contribute_nothing():
    """A neutral water cluster has one fragmentation; there is no competition to learn."""
    out, batch = make([3.0], [-20.0], [4])
    terms, metrics = applicability_loss(out, batch, weight=1.0)
    assert terms == {} and metrics == {}


def test_contested_and_uncontested_can_share_a_batch():
    """The real corpus mixes them: ion clusters compete, water clusters do not."""
    out, batch = make([1.0, 0.0, 9.0], [-100.0, -400.0, -20.0], [1, 1, 4])
    terms, metrics = applicability_loss(out, batch, weight=1.0)
    # One contested geometry, answered right; the lone water neither helps nor hurts.
    assert metrics["app_acc"] == 1.0
    solo, _ = make([1.0, 0.0], [-100.0, -400.0], [1, 1])
    alone, _ = applicability_loss(solo, batch=make([1.0, 0.0], [-100.0, -400.0], [1, 1])[1],
                                  weight=1.0)
    assert float(terms["applicability"]) == pytest.approx(float(alone["applicability"]))


def test_only_relative_scores_matter():
    """Nothing pins the scale, because nothing downstream uses it."""
    a, batch = make([1.0, 0.0], [-100.0, -400.0], [2, 2])
    b, _ = make([101.0, 100.0], [-100.0, -400.0], [2, 2])
    first, _ = applicability_loss(a, batch, weight=1.0)
    second, _ = applicability_loss(b, batch, weight=1.0)
    assert float(first["applicability"]) == pytest.approx(float(second["applicability"]))


def test_temperature_softens_the_target():
    """A hot target is closer to uniform, so a uniform prediction is penalized less."""
    out, batch = make([0.0, 0.0], [-100.0, -400.0], [3, 3])
    cold, _ = applicability_loss(out, batch, weight=1.0, temperature=10.0)
    hot, _ = applicability_loss(out, batch, weight=1.0, temperature=100000.0)
    # Against a near-uniform target, a uniform prediction is nearly optimal.
    assert float(hot["applicability"]) < float(cold["applicability"])
    assert float(hot["applicability"]) == pytest.approx(0.693, abs=0.01)


def test_the_score_is_pooled_over_a_frames_fragments():
    """A frame is a decomposition, but the head emits one score per *fragment*."""
    out, batch = make(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [-100.0, -400.0], [5, 5], n_frag_per_frame=3
    )
    _terms, metrics = applicability_loss(out, batch, weight=1.0)
    assert metrics["app_acc"] == 1.0


def test_it_is_off_without_a_weight_or_without_groups():
    out, batch = make([1.0, 0.0], [-100.0, -400.0], [2, 2])
    assert applicability_loss(out, batch, weight=0.0) == ({}, {})
    batch.group_id = None
    assert applicability_loss(out, batch, weight=1.0) == ({}, {})


def test_a_missing_eda_component_raises_rather_than_silently_skipping():
    out, batch = make([1.0, 0.0], [-100.0, -400.0], [2, 2])
    del batch.eda["ct"]
    with pytest.raises(KeyError, match="eda_ct"):
        applicability_loss(out, batch, weight=1.0)


def test_gradients_flow_to_the_scores():
    scores = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    out, batch = make([0.0, 0.0], [-100.0, -400.0], [6, 6])
    out.applicability = scores
    terms, _ = applicability_loss(out, batch, weight=1.0)
    terms["applicability"].backward()
    assert scores.grad is not None
    # The under-scored good decomposition must be pushed up and the other down.
    assert scores.grad[0] < 0 < scores.grad[1]
