"""Pair lists for the explicit force-field terms.

Distinct from two other neighbor structures already in the repo, and deliberately so:

- ``LambdaFeatures.edge_index`` is the featurizer's ``radius_graph`` at the *feature*
  cutoff. Force-field terms reach further than the descriptor does, so they need their
  own list.
- ``Batch.bond_index`` is the SQE charge-transfer channel graph, which comes from the
  diabatic assignment and never from a distance rule.

The list here is **undirected** (``i < j``, each pair once) so a pair energy can be
summed without a factor of a half, and optionally **inter-fragment only**, which is what
makes the sum an interaction energy comparable to an EDA component.

:func:`intra_fragment_channels` builds a channel graph too, and does *not* violate the rule
above: it is a complete **enumeration** of the pairs inside each fragment, with no distance
test anywhere. See its docstring for why enumerating is equivalent to supplying the covalent
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
    """
    if fragment_idx.numel() and bool((fragment_idx[1:] < fragment_idx[:-1]).any()):
        raise ValueError(
            "intra_fragment_channels needs a non-decreasing fragment_idx (atoms grouped by "
            "fragment), which is what MoleculeDataset.flat_batch produces"
        )
    n_frag = int(fragment_idx.max()) + 1 if fragment_idx.numel() else 0
    counts = torch.bincount(fragment_idx, minlength=n_frag)
    offsets = torch.cumsum(counts, 0) - counts

    rows, cols, owner = [], [], []
    for f in range(n_frag):
        n = int(counts[f])
        if n < 2:                     # a lone atom has no channel
            continue
        a, b = torch.triu_indices(n, n, offset=1, device=fragment_idx.device)
        rows.append(a + offsets[f])
        cols.append(b + offsets[f])
        owner.append(torch.full((a.numel(),), f, dtype=torch.long, device=fragment_idx.device))

    if not rows:
        empty = torch.zeros(0, dtype=torch.long, device=fragment_idx.device)
        return torch.zeros(2, 0, dtype=torch.long, device=fragment_idx.device), empty
    return torch.stack((torch.cat(rows), torch.cat(cols))), torch.cat(owner)


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
