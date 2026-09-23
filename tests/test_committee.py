"""Resolving a committee directory to its members.

The load-bearing test is :func:`test_staged_layout_takes_the_stage_that_ran_last`, and it is
load-bearing because the obvious implementation fails it: ``"full" < "isolated"``, so sorting
the stage directory names picks ``_isolated`` -- a real, loadable film model whose environment
path was never trained. That is a silent wrong answer, not an error, which is exactly the kind
this suite exists to catch.
"""

import json

import pytest
import torch

from rsfff.train.committee import COMMITTEE_MANIFEST, committee_members
from rsfff.train.config import Config, StageConfig


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_plain_checkpoint_is_a_one_member_committee(tmp_path):
    """Iteration 0: a single starting model, and no spread to measure."""
    checkpoint = touch(tmp_path / "best.pt")
    assert committee_members(checkpoint) == [checkpoint]


def test_missing_path_is_not_checked_here(tmp_path):
    """A non-directory is passed straight through; the loader that opens it errors better."""
    assert committee_members(tmp_path / "nope.pt") == [tmp_path / "nope.pt"]


def test_flat_layout(tmp_path):
    expected = [touch(tmp_path / f"member_{i:02d}" / "best.pt") for i in range(3)]
    assert committee_members(tmp_path) == expected


def write_checkpoint(path, stages):
    """A checkpoint carrying only the field the resolver reads: ``config.stages``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = Config(stages=[StageConfig(name=name) for name in stages])
    torch.save({"config": config}, path)
    return path


def test_staged_layout_takes_the_stage_that_ran_last(tmp_path):
    """The order comes from ``config.stages``, which is ``isolated`` then ``full``.

    Note the direction of the alphabetical trap: ``_full`` sorts *before* ``_isolated``, so a
    resolver that sorted names would return the isolated checkpoints and every figure drawn
    from them would show a model with no environment path.
    """
    stages = ["isolated", "full"]
    wanted = []
    for i in range(3):
        member = tmp_path / f"member_{i:02d}"
        # written full-first so a modification-time rule would also get this wrong
        wanted.append(write_checkpoint(member / f"member_{i:02d}_full" / "best.pt", stages))
        write_checkpoint(member / f"member_{i:02d}_isolated" / "best.pt", stages)
    assert committee_members(tmp_path) == wanted


def test_staged_layout_falls_back_to_modification_time(tmp_path):
    """A checkpoint too old to carry a config still resolves, by write order."""
    for i in range(2):
        member = tmp_path / f"member_{i:02d}"
        touch(member / f"member_{i:02d}_isolated" / "best.pt")
    wanted = [
        touch(tmp_path / f"member_{i:02d}" / f"member_{i:02d}_full" / "best.pt")
        for i in range(2)
    ]
    assert committee_members(tmp_path) == wanted


def test_single_stage_per_member_needs_no_order(tmp_path):
    """One stage directory per member: nothing to disambiguate, and no checkpoint read."""
    wanted = [
        touch(tmp_path / f"member_{i:02d}" / f"member_{i:02d}_full" / "best.pt")
        for i in range(3)
    ]
    assert committee_members(tmp_path) == wanted


def test_manifest_wins_and_fixes_the_order(tmp_path):
    """A glob can only order by name; the manifest is the only source of member order."""
    paths = [touch(tmp_path / f"member_{i:02d}" / "best.pt") for i in range(3)]
    shuffled = [paths[2], paths[0], paths[1]]
    (tmp_path / COMMITTEE_MANIFEST).write_text(
        json.dumps({"members": [{"checkpoint": str(p)} for p in shuffled]})
    )
    assert committee_members(tmp_path) == shuffled


def test_manifest_paths_are_rerooted_after_a_sync(tmp_path):
    """The manifest records absolute paths on the machine that trained the committee.

    A Perlmutter fit writes `/global/cfs/cdirs/.../member_00/member_00_full/best.pt`, and
    nothing at that path exists once the committee is copied down. Only the tail from
    `member_NN` onward is portable.
    """
    local = [
        touch(tmp_path / f"member_{i:02d}" / f"member_{i:02d}_full" / "best.pt")
        for i in range(3)
    ]
    remote = [
        f"/global/cfs/cdirs/m3196/runs/film_committee/member_{i:02d}/member_{i:02d}_full/best.pt"
        for i in range(3)
    ]
    (tmp_path / COMMITTEE_MANIFEST).write_text(
        json.dumps({"members": [{"checkpoint": p} for p in remote]})
    )
    assert committee_members(tmp_path) == local


def test_manifest_pointing_outside_is_honoured_when_it_exists(tmp_path):
    """A manifest may deliberately point elsewhere -- a shared store, a borrowed member."""
    elsewhere = touch(tmp_path / "store" / "member_00" / "best.pt")
    root = tmp_path / "committee"
    root.mkdir()
    (root / COMMITTEE_MANIFEST).write_text(
        json.dumps({"members": [{"checkpoint": str(elsewhere)}]})
    )
    assert committee_members(root) == [elsewhere]


def test_manifest_entry_that_resolves_nowhere_raises(tmp_path):
    (tmp_path / COMMITTEE_MANIFEST).write_text(
        json.dumps({"members": [{"checkpoint": "/remote/member_00/member_00_full/best.pt"}]})
    )
    with pytest.raises(FileNotFoundError, match="does not exist either"):
        committee_members(tmp_path)


def test_empty_tree_says_the_files_are_missing(tmp_path):
    """The failure mode of a sync that copied directories but not their contents."""
    for i in range(4):
        (tmp_path / f"member_{i:02d}" / f"member_{i:02d}_full").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="without their files"):
        committee_members(tmp_path)


def test_empty_directory_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no committee members"):
        committee_members(tmp_path)
