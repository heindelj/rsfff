"""The soft partition: one scalar per edge, and everything the membership weights touch.

``docs/fff_v2.md`` v4. A fragmentation is a partition of the frame's *edges* into intra and
inter, and every candidate assignment of one geometry partitions the **same** edge set --
``A_intra,m + A_cross,m = A_full``, with ``A_full`` independent of ``m``. What a mixture has to
choose is therefore not a descriptor, a parameter or a latent. It is where the boundary sits::

    s_e = sum_m w_m [ frag_m(i) == frag_m(j) ]        in [0, 1]

That single scalar is the whole mixing operator, and it already had two of its three jobs
before this module existed: ``routing_weight`` decided how intra a *pair* was for the energy
accounting, and ``scaled_compliance`` decided how open a *channel* was for the SQE solve --
the same sum, written twice, over two different index lists. The third job, the one that was
still a hard boolean, is the featurizer's edge mask. One definition now serves all three.

Why the partition and not its consequences
------------------------------------------
The density is linear in the edge sum but the descriptor is not: the power spectrum is
quadratic, so a convex combination of two finished descriptors is the spectrum of no density
at all. ``P(A(s))`` is a genuine power spectrum for every ``s``. Every positivity constraint,
prior and physical form in the parameter heads keeps its meaning along the whole path, which is
exactly what a halfway *parameter* set failed to do in v2 -- measured, 162 kJ/mol of excursion
beyond the interval spanned by the two vertices.

Be precise about the size of that claim. This removes the **structural** off-manifold problem:
every intermediate is a real descriptor. It does not remove the **statistical** one: a
fractional ``s_e`` is a region of descriptor space that no definite-fragmentation training
frame visits. That gap closes with data, and it is the right dependency to have -- ALMO-EDA
labels fragments at definite ``(Q, S)``, densely, so the constraint is trained and then lifted.

The isolated guarantee is untouched
-----------------------------------
A lone fragment has ``s_e = 1`` on every one of its edges, so ``1 - s_e`` is exactly zero, the
cross density is an exact zero, and ``eta`` is exactly zero. Not small, not zero by an
anchoring subtraction -- zero because the sum is empty, for the same reason it was before.

At a one-hot membership ``s_e`` is exactly 0 or 1 and every consumer reproduces the definite
fragmentation. That is Invariant 1, and it is what ``tests/test_mediator.py`` pins.
"""

from __future__ import annotations

import torch

__all__ = ["element_counts", "mixed_state", "soft_partition"]


def soft_partition(
    fragments: torch.Tensor,       # (M, N) long, one row per decomposition
    weights: torch.Tensor,         # (M,) the membership, a partition of unity
    index: torch.Tensor,           # (2, P) any list of atom pairs
) -> torch.Tensor:
    """``s = sum_m w_m [frag_m(i) == frag_m(j)]`` -- how intra each pair is. ``(P,)``.

    ``index`` is deliberately untyped as to what it *means*: the frame's SOAP edges, the
    force field's pair list and the SQE channel graph are three different index lists that
    need the same number, and computing it three ways is how they drift apart. Pass whichever
    one is being weighed.

    **This is where energy crosses the boundary between ``fragment_energy`` and the EDA
    channels**, so a hard membership test under a soft ``w`` is not merely inconsistent -- it
    is a discontinuity in the accounting itself. Since ``sum_m w_m = 1``, every pair is still
    counted exactly once across the intra and inter buckets: ``s + (1 - s) = 1`` identically,
    for any weights and any geometry.
    """
    if fragments.dim() != 2:
        raise ValueError(
            f"fragments must be (M, N), one row per decomposition, got "
            f"{tuple(fragments.shape)}"
        )
    w = weights.reshape(-1)
    if w.shape[0] != fragments.shape[0]:
        raise ValueError(
            f"soft_partition got {fragments.shape[0]} decompositions and {w.shape[0]} "
            f"weights; the membership is a partition of unity over the decompositions and "
            f"must have one entry per row of `fragments`"
        )
    i, j = index[0], index[1]
    same = (fragments[:, i] == fragments[:, j]).to(w.dtype)              # (M, P)
    return (w.reshape(-1, 1) * same).sum(0)


def element_counts(
    fragment_idx: torch.Tensor,    # (N,) long
    species_idx: torch.Tensor,     # (N,) long, values in [0, n_species)
    n_species: int,
) -> torch.Tensor:
    """``(N, n_species)`` the element census of each atom's fragment, gathered per atom.

    The composition of the fragment an atom belongs to, as **counts** rather than a one-hot
    over composition names. Counts because they are what a mixture can average: "half an H3O+
    and half an H2O" is 2.5 hydrogens, a real input to the state embedding, where a one-hot
    over ``{"H2O", "H3O"}`` is two discrete labels with nothing defined between them. The
    same argument as fractional charge, applied to topology.
    """
    n_frag = int(fragment_idx.max()) + 1 if fragment_idx.numel() else 0
    one_hot = torch.zeros(
        fragment_idx.shape[0], int(n_species),
        dtype=torch.get_default_dtype(), device=fragment_idx.device,
    )
    one_hot.scatter_(1, species_idx.reshape(-1, 1), 1.0)
    per_fragment = one_hot.new_zeros(n_frag, int(n_species)).index_add_(
        0, fragment_idx, one_hot
    )
    return per_fragment[fragment_idx]


def mixed_state(
    fragments: torch.Tensor,       # (M, N) long
    weights: torch.Tensor,         # (M,)
    atom_charge: torch.Tensor,     # (M, N) the host fragment's formal charge, per atom
    atom_two_s: torch.Tensor,      # (M, N) likewise
    species_idx: torch.Tensor,     # (N,)
    n_species: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(Q*, 2S*, n*)`` per atom: the fragment state, mixed at the **input**.

    Every entry is ``sum_m w_m (that decomposition's value)``, which is the same convex
    combination :func:`soft_partition` applies to edges, applied to the fragment-level labels.

    **Mixing the inputs and never the outputs is the point.** ``FragmentStateEmbedding`` is an
    MLP over continuous ``(Q, 2S, n)``, so ``E(Q*)`` is a genuine output of that network at a
    real input -- a fractional charge is a point the net can be asked about, and a proton
    halfway between two waters genuinely carries one. Averaging two *embeddings* instead would
    put the decoder at a point in latent space that nothing produced, which is the failure this
    design exists to escape and which no amount of training data fixes.

    ``atom_charge``/``atom_two_s`` arrive per atom rather than per fragment because the
    fragment numbering differs between decompositions and there is no common indexing to state
    them in; :class:`rsfff.ff.mixture_model.MixtureGroup` already carries them this way for
    :class:`rsfff.ff.mediator.MediatorHead`.
    """
    w = weights.reshape(-1)
    dtype = torch.get_default_dtype()
    q = (w.reshape(-1, 1) * atom_charge.to(dtype)).sum(0)                # (N,)
    two_s = (w.reshape(-1, 1) * atom_two_s.to(dtype)).sum(0)             # (N,)
    counts = torch.zeros(
        fragments.shape[1], int(n_species), dtype=dtype, device=fragments.device
    )
    for m in range(fragments.shape[0]):
        counts = counts + w[m] * element_counts(fragments[m], species_idx, n_species)
    return q, two_s, counts
