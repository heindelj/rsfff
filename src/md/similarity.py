"""How similar are two structures, in the model's own descriptor?

The corpus this model is fitted on is small and the sampler that grows it produces thousands of
frames, many of them near-duplicates and many of them off-manifold. Both problems are questions
about *distance between structures*, and the useful place to measure that distance is not
Cartesian space -- two isomers with the same energy can be far apart in RMSD, and a rotated copy
of one frame is infinitely far from it -- but the descriptor the model actually reads.

That descriptor is already sitting there. :meth:`FragmentExpertModel.emit` returns two per-atom
slots (``docs/fff_v2.md`` §3):

* ``h``   -- the power spectrum over *intra*-fragment edges. What the fragment would look like
             with the rest of the system deleted. Provably unchanged by neighbours
             (``tests/test_slots.py::test_fragment_slot_is_unchanged_by_neighbours``).
* ``eta`` -- the power spectrum over *cross*-fragment edges. What was deleted, seen from this
             atom. Exactly zero for an isolated fragment, not approximately.

So the two questions the model itself distinguishes -- "is this the same molecule" and "is it in
the same surroundings" -- are already separate axes, and a similarity built on them inherits that
split for free. That is why this reads the model's features rather than, say, a generic SOAP.

Three decisions worth stating
-----------------------------
**Only the lambda=0 block.** ``LambdaFeatures`` also carries ``vec_feats`` (lambda=1) and
``equiv_feats`` (lambda=2), and they are *equivariant*, not invariant. Feeding them to a metric
raw compares orientations: a rotated copy of a structure would score as dissimilar to itself.
:func:`atom_features` takes ``inv_feats`` and nothing else, and
``tests/test_similarity.py::test_similarity_is_invariant_to_rotation_translation_permutation``
is what keeps it that way.

**Standardize, always.** Nothing between the featurizer and the heads normalizes anything --
no l2, no LayerNorm, no per-channel scaling -- and channel magnitudes span orders of magnitude.
A cosine on raw features is a cosine on the half-dozen loudest channels. :meth:`FeatureMetric.fit`
therefore takes per-channel mean and standard deviation over a reference set, and every distance
downstream is computed on z-scored features.

**Pool per species, match per fragment.** Atoms have no canonical order, so a structure-level
comparison has to be permutation invariant. Pooling per (fragment, centre species) handles the
atoms; a composition-constrained best-match assignment handles the fragments. The alternative --
averaging everything into one vector -- is permutation invariant too, but it cannot say *which*
fragment differs, and that is usually the thing you wanted to know.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..ff.mixture_model import intra_pairs_unsorted
from .assign import enumerate_group

__all__ = [
    "FeatureMetric",
    "FragmentDescriptors",
    "Match",
    "atom_features",
]

@dataclass(frozen=True)
class FragmentDescriptors:
    """One structure, reduced to a descriptor per fragment.

    ``vectors`` is ``(F, D)`` with ``D = 2 * n_species * width``: for each of the fragment and
    environment slots, the mean over that fragment's atoms of each species, concatenated. A
    species absent from a fragment contributes zeros, which is what makes descriptors of
    different compositions the same width and therefore comparable at all -- though
    :meth:`FeatureMetric.match` will refuse to match across compositions anyway.
    """

    vectors: np.ndarray                 # (F, D) z-scored
    compositions: tuple[str, ...]       # (F,) e.g. "H2O", "H3O+1"
    charges: np.ndarray                 # (F,) formal charge per fragment
    n_atoms: np.ndarray                 # (F,)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])


@dataclass(frozen=True)
class Match:
    """The result of comparing two structures."""

    score: float                                    # system similarity in [0, 1]
    pairs: tuple[tuple[int, int, float], ...]       # (fragment in a, fragment in b, similarity)
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]

    def __repr__(self) -> str:                      # pragma: no cover
        return (f"Match(score={self.score:.4f}, {len(self.pairs)} pairs, "
                f"{len(self.unmatched_a)}+{len(self.unmatched_b)} unmatched)")


def atom_features(model, atoms, total_charge: int, *, base=None):
    """``(h, eta, species, fragment_idx, fragment_charge)`` for one structure, as numpy.

    Runs the enumeration to get a fragmentation and then :meth:`~FragmentExpertModel.emit`,
    which builds the descriptors and nothing else -- no pair list, no solve, no energy. The
    *reference* decomposition (index 0) is used: a similarity measure should describe the
    geometry, not the mediator's opinion about it, and the competing decompositions of one
    geometry share every atom position anyway.
    """
    dtype = next(model.parameters()).dtype
    pos = torch.as_tensor(np.asarray(atoms.get_positions()), dtype=dtype)
    z = torch.as_tensor(np.asarray(atoms.get_atomic_numbers()), dtype=torch.long)

    group = enumerate_group(pos, z, int(total_charge), base=base)
    batch = group.batch(0)
    frag = batch.fragment_idx
    with torch.no_grad():
        em = model.emit(batch, frag, bond_index=intra_pairs_unsorted(frag))

    p_frag = int(em.iso.inv_feats.shape[-1])
    state = int(p_frag - (int(em.joined.inv_feats.shape[-1]) - p_frag))
    h = em.iso.inv_feats[:, : p_frag - state]        # drop the (Q, 2S, n) block
    eta = em.joined.inv_feats[:, p_frag:]
    return (h.cpu().numpy(), eta.cpu().numpy(), z.cpu().numpy(),
            frag.cpu().numpy(), group.atom_charge[0].cpu().numpy())


def _composition(z: np.ndarray, charge: float) -> str:
    """``"H2O"``, ``"H3O+1"``, ``"HO-1"`` -- a fragment's identity as a matching key."""
    counts = {int(s): int((z == s).sum()) for s in sorted(set(z.tolist()), reverse=True)}
    symbols = {1: "H", 8: "O"}
    body = "".join(f"{symbols.get(s, f'Z{s}')}{n if n > 1 else ''}"
                   for s, n in sorted(counts.items(), key=lambda kv: -kv[0]))
    q = int(round(charge))
    return body if q == 0 else f"{body}{q:+d}"


class FeatureMetric:
    """Distances between structures in the model's descriptor space.

    Build one with :meth:`fit` over a reference set -- the training corpus is the natural
    choice, since "how far is this from what the model was fitted on" is the question that
    matters for curation. The statistics only set the scale; any reasonably diverse set works,
    but two metrics fitted on different sets are not comparable to each other.
    """

    def __init__(self, model, mean: np.ndarray, std: np.ndarray, *, species=(1, 8)) -> None:
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.species = tuple(int(s) for s in species)

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def fit(cls, model, frames, *, species=(1, 8)) -> "FeatureMetric":
        """Per-channel mean/std of ``[h | eta]`` over ``frames``.

        ``frames`` is an iterable of ``(atoms, total_charge)``. A near-constant channel gets a
        floor on its standard deviation rather than being dropped: dividing by ~0 would turn
        numerical noise into the dominant axis of the metric.
        """
        rows = []
        for atoms, charge in frames:
            h, eta, _z, _f, _q = atom_features(model, atoms, charge)
            rows.append(np.concatenate([h, eta], axis=1))
        if not rows:
            raise ValueError("FeatureMetric.fit needs at least one reference frame")
        stacked = np.concatenate(rows, axis=0)
        std = stacked.std(axis=0)
        # The floor is relative to the typical channel, so it scales with the descriptor rather
        # than being an absolute number that happens to suit one checkpoint.
        std = np.maximum(std, 1e-3 * float(np.median(std[std > 0])) if (std > 0).any() else 1.0)
        return cls(model, stacked.mean(axis=0), std, species=species)

    # -- descriptors ----------------------------------------------------------------------

    def fragment_descriptors(self, atoms, total_charge: int) -> FragmentDescriptors:
        """Pool a structure into one z-scored vector per fragment.

        Mean over the atoms of each species separately, for each slot. Species has to be kept
        as its own block rather than averaged over: with ``density_channels`` set, the
        featurizer has already mixed the ``(Z, n)`` axis into learned channels, so the centre
        atom's own species is the only species resolution the descriptor still has.
        """
        h, eta, z, frag, qa = atom_features(self.model, atoms, total_charge)
        raw = np.concatenate([h, eta], axis=1)
        scaled = (raw - self.mean) / self.std

        n_frag = int(frag.max()) + 1 if frag.size else 0
        width = scaled.shape[1]
        out = np.zeros((n_frag, len(self.species) * width), dtype=np.float64)
        comps, charges, sizes = [], np.zeros(n_frag), np.zeros(n_frag, dtype=int)
        for f in range(n_frag):
            in_f = frag == f
            sizes[f] = int(in_f.sum())
            charges[f] = float(qa[in_f][0]) if sizes[f] else 0.0
            comps.append(_composition(z[in_f], charges[f]))
            for k, s in enumerate(self.species):
                sel = in_f & (z == s)
                if sel.any():
                    out[f, k * width:(k + 1) * width] = scaled[sel].mean(axis=0)
        return FragmentDescriptors(out, tuple(comps), charges, sizes)

    # -- comparison -----------------------------------------------------------------------

    @staticmethod
    def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``(Fa, Fb)`` similarity in ``[0, 1]`` from the z-scored Euclidean distance.

        ``1 / (1 + d / sqrt(D))``: monotone in the distance, unbounded input, bounded output,
        and the ``sqrt(D)`` makes the scale independent of the descriptor width so a threshold
        chosen on one checkpoint still means something on another. Not a cosine -- these are
        z-scored, so they are centred near the origin and an angle between them is noise.
        """
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return 1.0 / (1.0 + d / np.sqrt(a.shape[1]))

    def match(self, a: FragmentDescriptors, b: FragmentDescriptors) -> Match:
        """Best-match the fragments of ``a`` to those of ``b``; mean similarity is the score.

        Compositions must agree: an H3O+ is never matched to an H2O, however close their
        descriptors happen to be. Without that constraint the assignment will happily pair an
        ion with a water to lower the total cost, and the resulting "similarity" says nothing.
        Unequal fragment counts are allowed -- the surplus is reported as unmatched and counted
        against the score, so a 3-water cluster does not look identical to a 4-water one.
        """
        from scipy.optimize import linear_sum_assignment

        if len(a) == 0 or len(b) == 0:
            return Match(0.0, (), tuple(range(len(a))), tuple(range(len(b))))

        sim = self._pairwise(a.vectors, b.vectors)
        allowed = np.array([[ca == cb for cb in b.compositions] for ca in a.compositions])
        cost = np.where(allowed, 1.0 - sim, 1e6)
        rows, cols = linear_sum_assignment(cost)

        pairs = [(int(i), int(j), float(sim[i, j]))
                 for i, j in zip(rows, cols) if allowed[i, j]]
        matched_a = {i for i, _j, _s in pairs}
        matched_b = {j for _i, j, _s in pairs}
        # Divide by the larger fragment count, so unmatched fragments dilute the score instead
        # of being quietly ignored.
        denom = max(len(a), len(b))
        score = sum(s for _i, _j, s in pairs) / denom if denom else 0.0
        return Match(
            float(score), tuple(pairs),
            tuple(i for i in range(len(a)) if i not in matched_a),
            tuple(j for j in range(len(b)) if j not in matched_b),
        )

    def similarity(self, a, b, charge_a: int, charge_b: int) -> Match:
        """Compare two ASE structures directly."""
        return self.match(self.fragment_descriptors(a, charge_a),
                          self.fragment_descriptors(b, charge_b))

    # -- scalable selection ---------------------------------------------------------------

    def system_descriptor(self, atoms, total_charge: int) -> np.ndarray | None:
        """One fixed-length vector per structure, for selection over sets too big to match.

        :meth:`match` is the honest comparison -- it says *which* fragment differs -- but it
        solves an assignment per pair, so a farthest-point pass over 15000 structures is 10^8
        Hungarian solves and is not going to happen. This is the cheap surrogate: pool the
        fragment descriptors by composition, so an H3O+(H2O)3 becomes
        ``[mean over the three waters | the hydronium]`` and plain Euclidean distance in that
        space stands in for the matched one.

        What it gives up is real. Pooling the waters means two isomers that differ by swapping
        which water is where look identical, and it carries no fragment count, so it can only
        be compared *within* one composition and size. Both are handled by stratifying the
        selection rather than by making the vector cleverer -- a longer vector would still be a
        surrogate, and the stratification is needed anyway.

        Returns ``None`` when a composition is missing, which is what makes the strata safe to
        assemble by simply dropping the odd frame out.
        """
        desc = self.fragment_descriptors(atoms, total_charge)
        if len(desc) == 0:
            return None
        slots: dict[str, list[np.ndarray]] = {}
        for i, comp in enumerate(desc.compositions):
            slots.setdefault(comp, []).append(desc.vectors[i])
        # Composition order has to be canonical or two frames of the same cluster produce
        # vectors whose blocks are in different places.
        return np.concatenate([np.mean(slots[c], axis=0) for c in sorted(slots)])

    @staticmethod
    def farthest_point_vectors(vectors: np.ndarray, n: int, *, seed: int = 0) -> list[int]:
        """Greedy farthest-point sampling on plain vectors, vectorized.

        Same 2-approximation as :meth:`farthest_point`, but each round is one ``(N, D)``
        broadcast instead of ``N`` assignment problems, so it runs on tens of thousands of
        structures instead of hundreds.
        """
        if n <= 0 or len(vectors) == 0:
            return []
        n = min(n, len(vectors))
        rng = np.random.default_rng(seed)
        first = int(rng.integers(len(vectors)))
        chosen = [first]
        dist = np.linalg.norm(vectors - vectors[first], axis=1)
        for _ in range(n - 1):
            dist[chosen] = -1.0
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(vectors - vectors[nxt], axis=1))
        return chosen

    # -- curation -------------------------------------------------------------------------

    def novelty(self, descriptors, reference: list[FragmentDescriptors]) -> tuple[float, float]:
        """``(system novelty, worst fragment novelty)`` against a reference set.

        ``1 - max_r similarity(descriptors, r)``: zero when the structure is indistinguishable
        from something in the reference set, approaching one when nothing in it is close. The
        second number is the same for the single worst-matched fragment, which is what tells
        you *where* a structure is unusual -- a cluster can be entirely ordinary except for one
        fragment that has come apart.
        """
        best_system, worst_fragment = 0.0, 1.0
        for ref in reference:
            m = self.match(descriptors, ref)
            if m.score > best_system:
                best_system = m.score
                worst_fragment = min((s for _i, _j, s in m.pairs), default=0.0)
        return 1.0 - best_system, 1.0 - worst_fragment

    def farthest_point(self, descriptors: list[FragmentDescriptors], n: int,
                       *, seed: int = 0) -> list[int]:
        """Indices of ``n`` maximally spread structures, greedy farthest-point sampling.

        What to spend Q-Chem time on. A harvest is heavily autocorrelated -- consecutive frames
        of one trajectory segment are nearly the same structure -- so labelling a random subset
        wastes most of the budget on duplicates. Greedy FPS is the standard 2-approximation and
        needs no clustering.
        """
        if n <= 0 or not descriptors:
            return []
        n = min(n, len(descriptors))
        rng = np.random.default_rng(seed)
        chosen = [int(rng.integers(len(descriptors)))]
        dist = np.array([1.0 - self.match(d, descriptors[chosen[0]]).score
                         for d in descriptors])
        while len(chosen) < n:
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, [1.0 - self.match(d, descriptors[nxt]).score
                                     for d in descriptors])
            dist[chosen] = -1.0
        return chosen
