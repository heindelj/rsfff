"""Pair lists for the explicit force-field terms.

Distinct from two other neighbor structures already in the repo, and deliberately so:

- ``LambdaFeatures.edge_index`` is the featurizer's ``radius_graph`` at the *feature*
  cutoff. Force-field terms reach further than the descriptor does, so they need their
  own list.
- ``Batch.bond_index`` is the SQE charge-transfer channel graph, which comes from the
  diabatic assignment and never from a distance rule.

Every list here is **undirected** (``i < j``, each pair once) so a pair energy can be summed
without a factor of a half. Three builders, for two different eras of the model:

- :func:`inter_fragment_pairs` drops same-fragment pairs at construction, which is what makes
  the sum an interaction energy directly comparable to an EDA component. Used by the
  per-term modules (:mod:`rsfff.ff.dispersion`, :mod:`rsfff.ff.pauli`,
  :mod:`rsfff.ff.electrostatics`).
- :func:`union_pairs` drops nothing, and hands the intra/inter question to a *learned*
  per-pair range separation instead of a boolean mask. Used by :mod:`rsfff.ff.unified`.
- :func:`intra_fragment_pairs` / :func:`intra_fragment_channels` enumerate inside each
  fragment with no distance test anywhere, which does *not* violate the rule above -- see
  :func:`intra_fragment_channels` for why enumerating is equivalent to supplying the covalent
  graph.
"""

from __future__ import annotations

import torch
from torch_cluster import radius_graph


def inter_fragment_pairs(
    positions: torch.Tensor,               # (N, 3) Angstrom
    batch_idx: torch.Tensor,               # (N,) long, frame id per atom
    cutoff: float,                         # Angstrom
    *,
    fragment_idx: torch.Tensor | None = None,   # (N,) long, batch-global
    max_num_neighbors: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected pairs within ``cutoff``, optionally restricted to different fragments.

    Returns ``(pair_index (2, P) with pair_index[0] < pair_index[1], r (P,))``.

    Grouping is by ``batch_idx``, **not** ``fragment_idx`` -- this is the one place in the
    codebase that wants cross-fragment edges, so grouping by fragment would return exactly
    the pairs we intend to discard. The inter-fragment restriction is applied afterwards as
    a mask; it is safe across frames because ``MoleculeDataset.flat_batch`` re-offsets
    ``fragment_idx`` to be batch-global (and ``radius_graph`` has already excluded
    cross-frame edges anyway).

    ``r`` is computed here and returned rather than recomputed by each caller, so every
    switching function sees the same distance and shares one autograd graph. It is not
    clamped: ``loop=False`` guarantees ``r > 0``, and a clamp would only hide a bug.
    """
    if fragment_idx is not None and not bool(
        torch.all(fragment_idx[1:] >= fragment_idx[:-1])
    ):
        raise ValueError(
            "inter_fragment_pairs needs a non-decreasing fragment_idx (atoms grouped by "
            "fragment); got an interleaved ordering, which would silently mis-mask pairs"
        )

    # torch_cluster has CPU+CUDA kernels but no MPS kernel; fall back to CPU on MPS.
    # max_num_neighbors is explicit because the default (32) silently *truncates*, which
    # a long-range FF cutoff will hit even on modest clusters.
    if positions.device.type == "mps":
        edge = radius_graph(
            positions.detach().cpu(), r=cutoff, batch=batch_idx.cpu(), loop=False,
            max_num_neighbors=max_num_neighbors,
        ).to(positions.device)
    else:
        edge = radius_graph(
            positions.detach(), r=cutoff, batch=batch_idx, loop=False,
            max_num_neighbors=max_num_neighbors,
        )

    mask = edge[0] < edge[1]                       # radius_graph is directed; keep i < j
    if fragment_idx is not None:
        mask = mask & (fragment_idx[edge[0]] != fragment_idx[edge[1]])
    pair_index = edge[:, mask]

    r = (positions[pair_index[0]] - positions[pair_index[1]]).norm(dim=-1)
    return pair_index, r


def union_pairs(
    positions: torch.Tensor,               # (N, 3) Angstrom
    batch_idx: torch.Tensor,               # (N,) long, frame id per atom
    fragment_idx: torch.Tensor,            # (N,) long, batch-global, non-decreasing
    cutoff: float,                         # Angstrom, the largest any channel needs
    *,
    max_num_neighbors: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One pair list for every channel: ``(pair_index (2,P), r (P,), is_intra (P,), pair_frag (P,))``.

    The neighbor search within ``cutoff`` **with no fragment mask**, unioned with the
    complete intra-fragment enumeration. Two properties this buys, each of which the
    unified model depends on:

    1. **Every pair carries the classical backbone.** :func:`inter_fragment_pairs` drops
       same-fragment pairs at construction, which means a same-fragment pair at 8 Angstrom
       gets no electrostatics at all -- fine for water, wrong for any fragment bigger than a
       monomer. Here nothing is dropped; how much of the classical form is switched on is
       decided downstream by the learned range separation, per pair and per channel.
    2. **The intra pairs still have no pair-list radius.** They come from
       :func:`intra_fragment_channels`, not from the neighbor search, so a stretched bond can
       never be dropped by ``cutoff`` or silently truncated by ``max_num_neighbors``. That is
       the same argument :func:`intra_fragment_pairs` makes, preserved through the union.

    ``is_intra`` and ``pair_frag`` are **routing**, not physics: they say which training label
    a pair's energy is compared against (``fragment_energy`` for intra, the ``eda_*``
    components for inter), because that is what those labels mean. They never gate the
    functional form. ``pair_frag`` is the fragment id for intra pairs and ``-1`` for inter
    pairs, so a per-fragment pool must mask on ``is_intra`` first.

    Output is sorted lexicographically by ``(i, j)`` and each pair appears exactly once with
    ``i < j``, so a pair energy sums without a factor of a half.
    """
    if fragment_idx is None:
        raise ValueError(
            "union_pairs needs a fragment partition to route pairs to their labels; "
            "the extxyz needs a `fragment_idx` column or a diabatic state library"
        )
    if not bool(torch.all(fragment_idx[1:] >= fragment_idx[:-1])):
        raise ValueError(
            "union_pairs needs a non-decreasing fragment_idx (atoms grouped by fragment); "
            "got an interleaved ordering, which intra_fragment_channels cannot enumerate"
        )

    # torch_cluster has CPU+CUDA kernels but no MPS kernel; fall back to CPU on MPS. The
    # list is built from detached positions -- only `r` below carries gradient.
    if positions.device.type == "mps":
        edge = radius_graph(
            positions.detach().cpu(), r=cutoff, batch=batch_idx.cpu(), loop=False,
            max_num_neighbors=max_num_neighbors,
        ).to(positions.device)
    else:
        edge = radius_graph(
            positions.detach(), r=cutoff, batch=batch_idx, loop=False,
            max_num_neighbors=max_num_neighbors,
        )
    edge = edge[:, edge[0] < edge[1]]          # radius_graph is directed; keep i < j
    intra, _ = intra_fragment_channels(fragment_idx)

    # Union by a single integer key per pair. n_atoms^2 stays far inside int64 for any
    # batch that fits in memory, and `unique` sorts, so the output order is deterministic
    # and independent of how radius_graph happened to emit its edges.
    n_atoms = positions.shape[0]
    both = torch.cat((edge, intra), dim=1)
    keys = both[0] * n_atoms + both[1]
    _, inverse = torch.unique(keys, return_inverse=True)
    # Scatter an arange through `inverse`: whichever source row lands last wins, and any
    # representative of a duplicated key is as good as another since the pairs are equal.
    pick = torch.empty(int(inverse.max()) + 1 if inverse.numel() else 0,
                       dtype=torch.long, device=keys.device)
    pick[inverse] = torch.arange(keys.numel(), device=keys.device)
    pair_index = both[:, pick]

    i, j = pair_index[0], pair_index[1]
    r = (positions[i] - positions[j]).norm(dim=-1)
    is_intra = fragment_idx[i] == fragment_idx[j]
    pair_frag = torch.where(is_intra, fragment_idx[i], torch.full_like(fragment_idx[i], -1))
    return pair_index, r, is_intra, pair_frag


#: ``(fragment_idx, version, result)`` for the last few distinct inputs. Small because the
#: only hit rate that matters is *within* one forward: the channel graph is a function of
#: ``fragment_idx`` alone, and one training step asks for it nine times (fifteen with charge
#: transfer) across :func:`union_pairs`, :func:`union_channels` and the two response solves,
#: every time from the same batch. Three batches are live at once at most -- the cluster
#: minibatch, the monomer anchor and the free-atom systems -- so four entries covers a step
#: with room to spare, and each entry is a few kilobytes of ``long``.
_CHANNEL_CACHE: list[tuple[torch.Tensor, int, tuple[torch.Tensor, torch.Tensor]]] = []
_CHANNEL_CACHE_SIZE = 4


def _cache_lookup(fragment_idx: torch.Tensor):
    """The cached graph for *this exact tensor*, or None.

    Keyed on object identity **and** ``_version``, not on value. Identity is what makes the
    lookup O(1) rather than a comparison against every entry, and the version counter is what
    makes it correct if a caller ever mutates a ``fragment_idx`` in place -- a value-equality
    key would be both slower and, on a tensor that had been written through, wrong in the same
    way. Entries hold a strong reference to the key, which is what keeps a freed tensor's
    address from being reused by a different one and silently matching.
    """
    for key, version, result in _CHANNEL_CACHE:
        if key is fragment_idx and version == key._version:
            return result
    return None


def _cache_store(fragment_idx: torch.Tensor, result) -> None:
    _CHANNEL_CACHE.append((fragment_idx, fragment_idx._version, result))
    del _CHANNEL_CACHE[:-_CHANNEL_CACHE_SIZE]


def intra_fragment_channels(
    fragment_idx: torch.Tensor,            # (N,) long, batch-global, non-decreasing
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every ``i<j`` pair *inside* each fragment, as an SQE channel graph.

    Returns ``(bond_index (2, Nb), bond_batch (Nb,))`` where ``bond_batch`` is the
    **fragment** id of each channel -- the solve this feeds is grouped by fragment, not by
    frame.

    No distance enters, so this does not break the rule that channels never come from a
    distance rule: it is the complete graph on each fragment's atoms.

    **Why enumerating is as good as supplying the covalent graph.** SQE writes
    ``q = q0 + B p``, and the column space of the incidence matrix ``B`` is
    ``{x : sum x = 0}`` for *any* connected graph on the fragment. So the covalent graph and
    the complete graph reach exactly the same charge distributions; they differ only in the
    channel-energy term ``1/2 sum_e kappa_e p_e^2``, i.e. in which distribution is cheapest.
    The compliance head therefore *learns* the topology, and the covalent graph is recovered
    exactly as the ``s -> 0`` limit of the extra channels -- verified on water, where
    zeroing the H-H compliance reproduces the O-H-only charges bit for bit.

    Cost is O(n^2) in the fragment size, which is right for monomers and wrong for a
    macromolecule; a fragment big enough for that to matter wants a supplied graph.

    **Built without a Python loop over fragments.** It used to be one, with an ``int(counts[f])``
    per iteration, and at ~6.5 microseconds per fragment that was the single largest pure-overhead
    item in a training step: 5.8 ms per call at 900 fragments, called nine times a step (fifteen
    with charge transfer on), for 52 ms a step at batch 256. The vectorized form below enumerates
    the upper triangle of the *largest* fragment once and masks each fragment down to its own
    count, which is ~100x faster and returns a bit-identical result -- masking ``triu(n_max)`` by
    ``b < n`` keeps the surviving pairs in row-major order, which is exactly ``triu(n)``.

    The padding is ``n_frag * n_max^2 / 2`` entries before the mask, so uniform fragment sizes
    (a water cluster) cost nothing and wildly ragged ones (one macromolecule among monomers)
    would waste the difference. That is the same regime the O(n^2) caveat above already excludes.

    The result is memoized on the input tensor -- see :data:`_CHANNEL_CACHE` -- because one
    forward asks for the same graph several times over. The returned tensors are integer index
    buffers with no autograd history, so handing the same objects to several callers is safe;
    nothing downstream writes through them.
    """
    cached = _cache_lookup(fragment_idx)
    if cached is not None:
        return cached
    result = _intra_fragment_channels(fragment_idx)
    _cache_store(fragment_idx, result)
    return result


def _intra_fragment_channels(
    fragment_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The uncached enumeration. See :func:`intra_fragment_channels`."""
    if fragment_idx.numel() and bool((fragment_idx[1:] < fragment_idx[:-1]).any()):
        raise ValueError(
            "intra_fragment_channels needs a non-decreasing fragment_idx (atoms grouped by "
            "fragment), which is what MoleculeDataset.flat_batch produces"
        )
    device = fragment_idx.device
    n_frag = int(fragment_idx.max()) + 1 if fragment_idx.numel() else 0
    counts = torch.bincount(fragment_idx, minlength=n_frag)
    n_max = int(counts.max()) if n_frag else 0
    if n_max < 2:                     # every fragment is a lone atom: no channel anywhere
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return torch.zeros(2, 0, dtype=torch.long, device=device), empty

    offsets = (torch.cumsum(counts, 0) - counts).unsqueeze(1)      # (n_frag, 1)
    a, b = torch.triu_indices(n_max, n_max, offset=1, device=device)
    # `a < b` throughout, so `b < n` implies `a < n` and one comparison selects the pairs that
    # fit inside each fragment. Fragments with fewer than two atoms select nothing.
    keep = b.unsqueeze(0) < counts.unsqueeze(1)                     # (n_frag, K)
    rows = (offsets + a.unsqueeze(0))[keep]
    cols = (offsets + b.unsqueeze(0))[keep]
    owner = torch.arange(n_frag, device=device).unsqueeze(1).expand(-1, a.numel())[keep]
    return torch.stack((rows, cols)), owner


def intra_fragment_pairs(
    positions: torch.Tensor,               # (N, 3)
    fragment_idx: torch.Tensor,            # (N,) long, batch-global, non-decreasing
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every ``i<j`` pair inside each fragment, with distances: ``(pair_index, r, pair_frag)``.

    The intra-fragment complement of :func:`inter_fragment_pairs`, but built on
    :func:`intra_fragment_channels` rather than on a neighbor search, for two reasons:

    1. **No cutoff, by design.** A bond term must not depend on a pair-list radius. An
       intramolecular H-H sits at ~1.5 A while an intermolecular O-H hydrogen bond sits at
       ~1.8, so no distance separates the two classes -- only the fragment mask does, and a
       ``max_num_neighbors`` truncation could silently drop a bond.
    2. ``pair_frag`` is the **fragment** id, which is what a per-fragment energy pools over.

    ``r`` is recomputed from the live ``positions``, so autograd -- and hence forces --
    flows through it. Same O(n^2) caveat as :func:`intra_fragment_channels`.
    """
    pair_index, pair_frag = intra_fragment_channels(fragment_idx)
    r = (positions[pair_index[0]] - positions[pair_index[1]]).norm(dim=-1)
    return pair_index, r, pair_frag


def union_channels(
    positions: torch.Tensor,               # (N, 3) Angstrom
    batch_idx: torch.Tensor,               # (N,) long, frame id per atom
    fragment_idx: torch.Tensor,            # (N,) long, batch-global, non-decreasing
    cutoff: float,                         # Angstrom; 0 disables the radius half entirely
    *,
    max_num_neighbors: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The charge-transfer channel graph: ``(bond_index (2,Nb), bond_batch (Nb,), from_radius (Nb,))``.

    The channel-graph analogue of :func:`union_pairs`, and it exists for the same reason: at
    the CT level there is no fragment distinction to make, because that is precisely the
    constraint being lifted. So the complete intra-fragment enumeration is unioned with a
    radius graph over *all* pairs, and charge is free to flow anywhere the compliance head
    lets it.

    ``bond_batch`` is the **frame**, not the fragment -- unlike
    :func:`intra_fragment_channels`. With inter-fragment channels open, charge conserves per
    frame and the connected component of the channel graph is the frame, so that is the block
    the solve is grouped by. Passing ``cutoff = 0`` gives exactly the intra-fragment channels
    with frame grouping, which is the polarized level: same edges, no charge crossing a
    fragment boundary, but pooled where the coupled solve needs them.

    ``from_radius`` marks the channels that entered through the neighbor search and **must**
    therefore be enveloped to zero at the cutoff -- otherwise a channel appears
    discontinuously as two molecules approach and the forces jump. Channels from the intra
    enumeration are marked ``False`` even when the radius graph also found them, because they
    are asserted to exist at any distance: enveloping one would let a stretched bond's channel
    vanish, which is the property :func:`intra_fragment_channels` is built to protect.

    Note this does *not* reintroduce a distance rule for bonds, the thing
    :func:`intra_fragment_channels` warns against. The bonded channels still come from the
    fragmentation. What the radius supplies is a candidate *set* for charge transfer, whose
    members are then switched on or off by the learned compliance -- and closing a channel is
    the bounded limit ``s -> 0``, so a generous set costs accuracy nothing.
    """
    if fragment_idx is None:
        raise ValueError("union_channels needs a fragment partition to enumerate bonds from")
    intra, _ = intra_fragment_channels(fragment_idx)
    device = fragment_idx.device
    if not cutoff > 0.0:
        return (
            intra,
            batch_idx[intra[0]] if intra.shape[1] else torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(intra.shape[1], dtype=torch.bool, device=device),
        )

    # Same MPS fallback as `union_pairs`: torch_cluster has no MPS kernel, and the list is
    # built from detached positions because only the compliance carries gradient here.
    if positions.device.type == "mps":
        edge = radius_graph(
            positions.detach().cpu(), r=cutoff, batch=batch_idx.cpu(), loop=False,
            max_num_neighbors=max_num_neighbors,
        ).to(device)
    else:
        edge = radius_graph(
            positions.detach(), r=cutoff, batch=batch_idx, loop=False,
            max_num_neighbors=max_num_neighbors,
        )
    edge = edge[:, edge[0] < edge[1]]

    n_atoms = positions.shape[0]
    both = torch.cat((edge, intra), dim=1)
    keys = both[0] * n_atoms + both[1]
    _, inverse = torch.unique(keys, return_inverse=True)
    pick = torch.empty(
        int(inverse.max()) + 1 if inverse.numel() else 0, dtype=torch.long, device=device
    )
    pick[inverse] = torch.arange(keys.numel(), device=device)
    bond_index = both[:, pick]

    intra_keys = intra[0] * n_atoms + intra[1]
    final_keys = bond_index[0] * n_atoms + bond_index[1]
    from_radius = ~torch.isin(final_keys, intra_keys)
    return bond_index, batch_idx[bond_index[0]], from_radius
