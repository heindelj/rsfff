"""Distances between structures in the model's own descriptor (:mod:`rsfff.md.similarity`).

Two tests carry the weight here.

**Invariance.** ``LambdaFeatures`` carries lambda=1 and lambda=2 blocks alongside the
invariants, and they are *equivariant*. Letting either into the metric means a rotated copy of
a structure is no longer similar to itself -- a failure that is invisible on any real pair of
structures, because they differ in orientation *and* in geometry and the score is merely
"lower". `test_similarity_is_invariant_...` is the only thing that separates the two.

**Separation.** A metric can be perfectly invariant and still useless, so the second test pins
the ordering the metric exists to produce: two geometries of the same hydrogen-bond topology
must score above two of different topologies. Real structures, not synthetic ones, because
what "different enough" means is a property of the data and not of the code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from rsfff.md import FeatureMetric, load_mediated_model

CHECKPOINT = Path("checkpoints/ion_mediator_v4_full/best.pt")
H3O = Path("data/hydronium_clusters_ccdb/asp-H2O_3--H3O+.xyz")
H3O4 = Path("data/hydronium_clusters_ccdb/asp-H2O_4--H3O+.xyz")
OH = Path("data/hydroxide_clusters/jp5b03893_si_002.xyz")

requires_data = pytest.mark.skipif(
    not (CHECKPOINT.exists() and H3O.exists() and OH.exists()),
    reason="needs the trained checkpoint and the cluster sets",
)


@pytest.fixture(scope="module")
def fitted():
    """A metric, plus the isomer set it was fitted over."""
    if not (CHECKPOINT.exists() and H3O.exists() and OH.exists()):
        pytest.skip("needs the trained checkpoint and the cluster sets")
    torch.set_default_dtype(torch.float64)
    model, _cfg, _state = load_mediated_model(str(CHECKPOINT))
    isomers = read(str(H3O), index=":")
    hydroxide = read(str(OH), index=":")
    reference = ([(a, 1) for a in isomers]
                 + [(a, 1) for a in read(str(H3O4), index=":")[:3]]
                 + [(hydroxide[k], -1) for k in range(11, 18)])
    return FeatureMetric.fit(model, reference), isomers, hydroxide


# ---------------------------------------------------------------------------------------
# Invariance
# ---------------------------------------------------------------------------------------

@requires_data
def test_similarity_is_invariant_to_rotation_translation_permutation(fitted):
    """**The test that keeps the equivariant blocks out of the metric.**

    Rotating, translating and relabelling a structure changes nothing about it, so the score
    against the original must be exactly 1, not merely close. A metric that had picked up
    ``vec_feats`` or ``equiv_feats`` would score a rotated copy as *different*, and nothing
    else in this file would notice -- every other comparison is between structures that differ
    in orientation and geometry at once, where a depressed score looks like a real answer.
    """
    metric, isomers, _oh = fitted
    a = isomers[0]

    rng = np.random.default_rng(0)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    b = a.copy()
    b.set_positions(a.get_positions() @ rot.T + np.array([3.0, -2.0, 5.0]))
    b = b[rng.permutation(len(b))]

    score = metric.similarity(a, b, 1, 1).score
    assert score == pytest.approx(1.0, abs=1e-9), (
        f"a rotated, translated, permuted copy scored {score:.10f} against itself; the metric "
        f"is reading an equivariant feature block or is order-dependent"
    )


@requires_data
def test_a_structure_is_identical_to_itself(fitted):
    metric, isomers, _oh = fitted
    assert metric.similarity(isomers[0], isomers[0], 1, 1).score == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------------------

def _sorted_oo(atoms) -> np.ndarray:
    p, z = atoms.get_positions(), atoms.get_atomic_numbers()
    o = p[z == 8]
    d = np.linalg.norm(o[:, None] - o[None], axis=-1)
    return np.sort(d[np.triu_indices(len(o), 1)])


@requires_data
def test_same_topology_scores_above_different_topology(fitted):
    """The ordering the metric exists to produce.

    ``asp-H2O_3`` holds five isomers of H3O+(H2O)3. Frames 0 and 1 are the same Eigen cation
    (three waters at O-O 2.59, no water-water contacts) and differ by 0.11 Angstrom; frames
    2-4 are genuinely different topologies, with O-O spectra up to 1.7 Angstrom away. Any
    useful metric has to put the first pair above the rest, and by a margin -- an l2-normalized
    cosine on these same features scores them 1.0000 against 0.9968, a separation of 0.003
    that no threshold could use. Per-channel standardization is what opens the gap.
    """
    metric, isomers, _oh = fitted
    base = _sorted_oo(isomers[0])
    same, different = [], []
    for k in range(1, len(isomers)):
        spread = float(np.abs(_sorted_oo(isomers[k]) - base).max())
        score = metric.similarity(isomers[0], isomers[k], 1, 1).score
        (same if spread < 0.2 else different).append(score)

    assert same and different, "the isomer set no longer contains both cases"
    assert min(same) > max(different) + 0.1, (
        f"same-topology scores {min(same):.4f} against different-topology {max(different):.4f}; "
        f"the metric does not resolve hydrogen-bond topology"
    )


@requires_data
def test_fragments_only_match_their_own_composition(fitted):
    """An H3O+ is never paired with an H2O, however close the descriptors are.

    Without the constraint the assignment lowers its total cost by pairing the ion with a
    water, and the resulting score is an average over a correspondence that does not mean
    anything. Checked on a hydronium against a hydroxide cluster, where the ion on each side
    has no counterpart at all and must simply go unmatched.
    """
    metric, isomers, hydroxide = fitted
    a = metric.fragment_descriptors(isomers[0], 1)
    b = metric.fragment_descriptors(hydroxide[12], -1)
    m = metric.match(a, b)

    for i, j, _s in m.pairs:
        assert a.compositions[i] == b.compositions[j], (
            f"matched {a.compositions[i]} to {b.compositions[j]}"
        )
    assert m.unmatched_a or m.unmatched_b, "the ion on each side has no counterpart"
    assert 0.0 < m.score < 1.0


@requires_data
def test_unmatched_fragments_lower_the_score(fitted):
    """A 3-water cluster must not look identical to a 4-water one.

    The score divides by the *larger* fragment count, so surplus fragments dilute it. Dividing
    by the number of matched pairs instead would make every cluster a perfect match for any
    larger cluster that contains it.
    """
    metric, isomers, _oh = fitted
    bigger = read(str(H3O4), index="0")
    m = metric.similarity(isomers[0], bigger, 1, 1)
    assert len(m.unmatched_b) == 1
    assert m.score < 0.9


# ---------------------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------------------

@requires_data
def test_novelty_ranks_the_stranded_structures_above_the_clean_ones():
    """**The test that says the metric is measuring something real.**

    The first harvest produced 5511 structures, 40% of them with a proton stranded 1.3-1.7
    Angstrom from any oxygen. Those are exactly the frames not worth labelling, and they were
    found with a geometric rule. If a novelty score computed against the training corpus does
    *not* also rank them as unusual, then it is not detecting off-manifold geometry and the
    geometric guard is doing all of the work.

    Compared within one cluster size, because a size mismatch leaves fragments unmatched and
    dominates the score.
    """
    corpus_path = Path("data/wb97mv_tzvpd/w2_h3o+_wb97mv_tzvpd.xyz")
    harvest_path = Path("qchem_roundtrip/biased_sampling/h3o+_w2/transition_structures.xyz")
    if not (CHECKPOINT.exists() and corpus_path.exists() and harvest_path.exists()):
        pytest.skip("needs the checkpoint, the training corpus and a harvest to score")

    torch.set_default_dtype(torch.float64)
    model, _cfg, _state = load_mediated_model(str(CHECKPOINT))
    corpus = read(str(corpus_path), index=":")[:25]
    metric = FeatureMetric.fit(model, [(a, 1) for a in corpus])
    reference = [metric.fragment_descriptors(a, 1) for a in corpus]

    harvest = read(str(harvest_path), index=":")

    def worst_oh(atoms) -> float:
        p, z = atoms.get_positions(), atoms.get_atomic_numbers()
        o, h = p[z == 8], p[z == 1]
        return float(np.linalg.norm(h[:, None] - o[None], axis=-1).min(axis=1).max())

    order = np.argsort([worst_oh(a) for a in harvest])
    clean, stranded = order[:12], order[-12:]

    def novelty(idx):
        return np.array([
            metric.novelty(metric.fragment_descriptors(harvest[i], 1), reference)[0]
            for i in idx
        ])

    n_clean, n_stranded = novelty(clean), novelty(stranded)
    assert n_stranded.mean() > n_clean.mean() + 0.02, (
        f"stranded frames scored {n_stranded.mean():.4f} novel against {n_clean.mean():.4f} "
        f"for clean ones; the descriptor is not seeing the pathology"
    )


@requires_data
def test_farthest_point_selection_spreads(fitted):
    """Picking what to label: the selection must beat taking the first N.

    A harvest is heavily autocorrelated, so the point of farthest-point sampling is that the
    chosen structures are less like each other than an arbitrary slice of the same size.
    """
    metric, isomers, _oh = fitted
    descriptors = [metric.fragment_descriptors(a, 1) for a in isomers]
    picked = metric.farthest_point(descriptors, 3)

    assert len(picked) == len(set(picked)) == 3

    def mean_pairwise(idx):
        vals = [metric.match(descriptors[i], descriptors[j]).score
                for k, i in enumerate(idx) for j in idx[k + 1:]]
        return float(np.mean(vals))

    assert mean_pairwise(picked) < mean_pairwise(list(range(3))) + 1e-9, (
        "farthest-point selection is no more spread out than the first three frames"
    )


@requires_data
def test_novelty_is_zero_against_a_reference_containing_the_structure(fitted):
    """A structure that *is* in the reference set has nothing novel about it."""
    metric, isomers, _oh = fitted
    reference = [metric.fragment_descriptors(a, 1) for a in isomers]
    system, fragment = metric.novelty(reference[2], reference)
    assert system == pytest.approx(0.0, abs=1e-9)
    assert fragment == pytest.approx(0.0, abs=1e-9)
