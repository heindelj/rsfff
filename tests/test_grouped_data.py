"""Loading a geometry's several decompositions, and keeping them together afterwards.

A multi-fragmentation file turns one geometry into two or three training frames. They are
*not* independent samples -- they share every nucleus -- so two things must hold everywhere
downstream, and neither is visible in a loss curve if it breaks:

* the train/val split must not put one decomposition of a geometry in training and another in
  validation, which would report a memorized geometry as held out;
* a minibatch must contain a geometry's decompositions all together or not at all, because
  the applicability term compares them against each other.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsfff.ff.pairs import union_pairs
from rsfff.train.data import (
    frame_fragmentations,
    load_cluster_datasets,
    split_indices_grouped,
)
from rsfff.train.train import _iter_minibatches

ION = [f"data/wb97mv_tzvpd/{s}_wb97mv_tzvpd.xyz" for s in
       ("w1_h3o+", "w2_h3o+", "w1_oh-", "w2_oh-")]
WATER = ["data/wb97mv_tzvpd/w2_wb97mv_tzvpd.xyz"]


@pytest.fixture(scope="module")
def ions():
    return load_cluster_datasets(ION, dtype=torch.float64)


def test_frame_fragmentations_reads_the_header():
    assert frame_fragmentations(ION[0]) == 2      # w1: two placements of the charge
    assert frame_fragmentations(ION[1]) == 3      # w2: three
    assert frame_fragmentations(WATER[0]) == 1    # neutral water: one


def test_every_decomposition_becomes_a_frame(ions):
    # 100 + 100 frames at 2 decompositions, 100 + 99 at 3.
    assert len(ions) == 2 * (100 + 100) + 3 * (100 + 99)
    sizes = torch.bincount(torch.bincount(ions._group_id))
    assert int(sizes[2]) == 200 and int(sizes[3]) == 199


def test_frames_of_one_geometry_share_a_group_and_a_geometry(ions):
    """The group is the claim that these frames are the same nuclei. Check the nuclei."""
    for group in (0, 1, 250):
        members = (ions._group_id == group).nonzero().squeeze(-1)
        assert members.numel() > 1
        batch = ions.flat_batch(members)
        n = int(ions._counts[members[0]])
        first = batch.positions[:n]
        for m in range(1, members.numel()):
            other = batch.positions[m * n : (m + 1) * n]
            # Same atoms, possibly reordered by the per-decomposition sort.
            assert np.allclose(
                np.sort(np.linalg.norm(first[:, None] - first[None], axis=-1).ravel()),
                np.sort(np.linalg.norm(other[:, None] - other[None], axis=-1).ravel()),
                atol=1e-9,
            )


def test_decompositions_of_one_geometry_really_differ(ions):
    """Otherwise the grouping is bookkeeping over identical frames."""
    members = (ions._group_id == 250).nonzero().squeeze(-1)
    batch = ions.flat_batch(members)
    n = int(ions._counts[members[0]])
    charges = [batch.fragment_charge[i * 3 : (i + 1) * 3] for i in range(members.numel())]
    assert not torch.equal(charges[0], charges[1])
    assert not torch.equal(batch.eda["ct"][0], batch.eda["ct"][1])
    del n


def test_every_loaded_frame_is_grouped_by_fragment(ions):
    """The pair builder refuses an interleaved partition; the loader must have re-sorted."""
    batch = ions.flat_batch(range(len(ions)))
    assert bool((batch.fragment_idx[1:] >= batch.fragment_idx[:-1]).all())
    union_pairs(batch.positions, batch.batch_idx, batch.fragment_idx, 12.0)


def test_a_mixed_corpus_needs_no_special_casing():
    """Neutral water clusters have one fragmentation and load beside the ion clusters."""
    mixed = load_cluster_datasets(ION + WATER, dtype=torch.float64)
    counts = torch.bincount(torch.bincount(mixed._group_id))
    assert int(counts[1]) > 0 and int(counts[2]) == 200 and int(counts[3]) == 199


def test_selecting_one_fragmentation_still_works():
    only_first = load_cluster_datasets(ION, dtype=torch.float64, fragmentations=0)
    assert len(only_first) == 399
    assert int(torch.bincount(only_first._group_id).max()) == 1


def test_group_ids_do_not_collide_across_files(ions):
    """Each file's geometries must get their own ids, or two systems would be 'the same'."""
    per_group_size = torch.bincount(ions._group_id)
    assert int(per_group_size.min()) >= 2      # no group left with a single stray frame
    assert int(ions._group_id.max()) + 1 == 399


# ---------------------------------------------------------------------------------------
# the split and the batching
# ---------------------------------------------------------------------------------------

def test_the_split_never_straddles_a_geometry(ions):
    train, val = split_indices_grouped(ions._group_id, 0.1, seed=0)
    assert train.numel() + val.numel() == len(ions)
    assert not (set(ions._group_id[train].tolist()) & set(ions._group_id[val].tolist()))


def test_a_frame_wise_split_would_have_straddled(ions):
    """The failure this exists to prevent, demonstrated rather than asserted."""
    from rsfff.train.data import split_indices

    train, val = split_indices(len(ions), 0.1, seed=0)
    overlap = set(ions._group_id[train].tolist()) & set(ions._group_id[val].tolist())
    assert overlap, "the naive split happened not to leak; pick another seed"


def test_the_split_is_deterministic(ions):
    a = split_indices_grouped(ions._group_id, 0.1, seed=3)
    b = split_indices_grouped(ions._group_id, 0.1, seed=3)
    c = split_indices_grouped(ions._group_id, 0.1, seed=4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[1], c[1])


@pytest.mark.parametrize("shuffle", [False, True])
def test_minibatches_keep_a_geometry_whole(ions, shuffle):
    train, _ = split_indices_grouped(ions._group_id, 0.1, seed=0)
    sizes_in_dataset = torch.bincount(ions._group_id)

    seen_total = 0
    for mb in _iter_minibatches(
        train, 64, shuffle=shuffle, seed=0, group_id=ions._group_id
    ):
        seen_total += mb.numel()
        present = torch.bincount(ions._group_id[mb], minlength=len(sizes_in_dataset))
        partial = (present > 0) & (present != sizes_in_dataset)
        assert not bool(partial.any()), "a geometry was split across minibatches"
    assert seen_total == train.numel()


def test_minibatches_stay_near_the_requested_size(ions):
    train, _ = split_indices_grouped(ions._group_id, 0.1, seed=0)
    sizes = [
        mb.numel()
        for mb in _iter_minibatches(train, 64, shuffle=True, seed=0, group_id=ions._group_id)
    ]
    # Packed until full, so a batch overshoots by at most one group (3 frames here).
    assert max(sizes) <= 64 + 3
    assert min(sizes[:-1]) > 64 - 3      # the last one is the remainder


def test_without_group_ids_the_batching_is_unchanged():
    """The ungrouped path must stay exactly what it was for every other dataset."""
    idx = torch.arange(10)
    plain = [mb.tolist() for mb in _iter_minibatches(idx, 4, shuffle=False, seed=0)]
    assert plain == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
