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
