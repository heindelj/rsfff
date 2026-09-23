"""Resolving a committee directory to its member checkpoints.

A committee is several fits of the same architecture on the same data, differing in their
initialization; the active-learning loop selects and stops on their disagreement
(``qchem_roundtrip/active_learning/committee.py``). This module is only the *addressing*
half of that -- which files are the members, and in what order -- because that is the part
every consumer needs and the part with a trap in it.

The trap is the staged layout. A staged film fit writes one run directory per stage:

    runs/<committee>/member_00/member_00_isolated/best.pt
    runs/<committee>/member_00/member_00_full/best.pt

The fitted model is the one from the **last stage that ran**, and the stages ran in the order
the config declares (``isolated`` then ``full``). Sorting the directory names does *not*
recover that order -- ``"full" < "isolated"`` alphabetically, so the obvious "take the last
sorted stage" picks precisely the wrong file. It picks it silently, too: an ``_isolated``
checkpoint is a real, loadable film model whose environment and cross embedders were never
trained, so the result is a model with no environment path at all, which reads as a bad fit
rather than as the wrong file.

The order therefore comes from the checkpoint's own embedded ``config.stages``, with the file
modification time as the fallback when a checkpoint is too old to carry one.

``Committee.load`` in the active-learning package globs only ``member_*/best.pt`` and so does
not see the staged layout at all; that is why this resolution lives here rather than being
imported from there.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["COMMITTEE_MANIFEST", "committee_members", "data_root_for",
           "localize_data_paths"]

#: Written by the train stage into the committee directory.
COMMITTEE_MANIFEST = "committee.json"


def _longest_suffix_under(path: Path, root: Path) -> Path | None:
    """``path`` re-rooted at ``root``, keeping the longest trailing subpath that exists.

    Longest and not merely the basename: a bare basename can match the wrong file, and here
    it would -- ``w4_wb97mv_tzvpd.xyz`` exists under both ``data/clusters/original/`` and, by
    another name, in the benchmark set. Keeping as much of the recorded structure as still
    resolves locally is what makes the match unambiguous.
    """
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for i in range(len(parts)):
        candidate = root.joinpath(*parts[i:])
        if candidate.exists():
            return candidate
    return None


def data_root_for(committee) -> Path | None:
    """The local directory that plays the role the training machine's run root played.

    The staged bundle is laid out as ``<root>/data/`` beside ``<root>/runs/<committee>/``
    (``qchem_roundtrip/train/scripts/stage_data.sh``), so walking up from the committee until
    a sibling ``data/`` appears recovers ``<root>`` without anything having to be configured.
    ``None`` when no such ancestor exists, which is not an error -- a committee need not have
    been synced with its data.
    """
    here = Path(committee).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir():
            return parent
    return None


def localize_data_paths(config, *roots) -> list[tuple[str, str]]:
    """Rewrite ``config.data``'s file paths onto the local tree. Returns what it changed.

    A checkpoint embeds the config it was trained under, and that config's data paths are
    absolute **on the training machine** -- ``/global/cfs/cdirs/.../train/data/...`` for a
    Perlmutter fit. Rebuilding the model reads ``reference_energies`` from it, so a synced
    checkpoint cannot be loaded at all until these are pointed somewhere real.

    Mutates ``config.data`` in place and returns the ``(before, after)`` pairs, so a caller
    can print exactly which paths moved rather than silently evaluating a model against data
    it did not name. Paths that already exist are left alone, and paths that resolve nowhere
    are left alone too -- the consumer that opens them gives the better error, and not every
    field is needed by every caller.
    """
    data = getattr(config, "data", None)
    if data is None:
        return []
    candidates = [Path(r) for r in roots if r is not None]
    changed: list[tuple[str, str]] = []

    def fix(value):
        if isinstance(value, list):
            return [fix(v) for v in value]
        if not isinstance(value, str) or Path(value).exists():
            return value
        for root in candidates:
            local = _longest_suffix_under(Path(value), root)
            if local is not None:
                changed.append((value, str(local)))
                return str(local)
        return value

    for field, value in vars(data).items():
        if value is None:
            continue
        fixed = fix(value)
        if fixed != value:
            setattr(data, field, fixed)
    return changed


def _relocate(checkpoint: Path, root: Path) -> Path:
    """A manifest checkpoint path, re-rooted at the local committee directory if need be.

    The manifest records **absolute paths on the machine that trained the committee** -- for
    a Perlmutter fit, ``/global/cfs/cdirs/.../member_00/member_00_full/best.pt`` -- and those
    do not exist after the committee is synced down. Only the tail from ``member_NN`` onward
    is portable, so that is what gets joined to the local directory.

    Preferring the recorded path when it does exist keeps a manifest that points somewhere
    deliberate (a shared checkpoint store, a member borrowed from another run) working.
    """
    if checkpoint.exists():
        return checkpoint
    for i, part in enumerate(checkpoint.parts):
        if part.startswith("member_"):
            local = root.joinpath(*checkpoint.parts[i:])
            if local.exists():
                return local
            raise FileNotFoundError(
                f"{COMMITTEE_MANIFEST} in {root} lists {checkpoint}, which does not exist; "
                f"re-rooting it locally gives {local}, which does not exist either"
            )
    raise FileNotFoundError(
        f"{COMMITTEE_MANIFEST} in {root} lists {checkpoint}, which does not exist and has no "
        f"member_* component to re-root against the local directory"
    )


def _stage_order(checkpoint: Path) -> list[str] | None:
    """The stage names of the fit in the order they ran, from the checkpoint's own config.

    ``None`` when the checkpoint carries no config, declares no stages, or cannot be read.
    ``torch`` is imported lazily so that merely addressing a committee does not pay for it.
    """
    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:       # noqa: BLE001 -- an unreadable candidate falls back, not fails
        return None
    config = state.get("config") if isinstance(state, dict) else None
    stages = getattr(config, "stages", None)
    return [stage.name for stage in stages] if stages else None


def _stage_name(checkpoint: Path) -> str:
    """``member_00/member_00_full/best.pt`` -> ``full``."""
    member, stage = checkpoint.parent.parent.name, checkpoint.parent.name
    prefix = f"{member}_"
    return stage[len(prefix):] if stage.startswith(prefix) else stage


def _final_stage(candidates: list[Path], order: list[str] | None) -> Path:
    """The checkpoint of the last stage among one member's stage directories."""
    if len(candidates) == 1:
        return candidates[0]
    if order:
        known = [c for c in candidates if _stage_name(c) in order]
        if known:
            return max(known, key=lambda c: order.index(_stage_name(c)))
    # No declared order to go on. The stages ran in sequence, so the newest file is the last
    # stage -- true of a fit in place, and of a sync that preserves times.
    return max(candidates, key=lambda c: c.stat().st_mtime)


def committee_members(path) -> list[Path]:
    """The member checkpoints of a committee, in member order.

    Three layouts, in order of authority:

    1. a :data:`COMMITTEE_MANIFEST` -- what the train stage writes, and the only source that
       fixes the member *order*; everything below is a sorted glob and so orders by name. Its
       paths are absolute on the *training* machine and are re-rooted here (:func:`_relocate`);
    2. ``member_NN/<stage>/best.pt``, the nested layout a staged fit produces, from which the
       last stage that ran is taken per member (see the module docstring -- this is **not**
       the last sorted name);
    3. ``member_NN/best.pt``, the flat layout.

    A path that is not a directory comes back as a one-member list, matching
    ``Committee.load``'s handling of iteration 0: one starting model, no spread to measure.
    Its existence is not checked here -- the loader that opens it gives the better error.

    Raises ``FileNotFoundError`` when a directory holds no members, which in practice most
    often means a sync copied the directory tree without the files in it.
    """
    path = Path(path)
    if not path.is_dir():
        return [path]

    manifest = path / COMMITTEE_MANIFEST
    if manifest.exists():
        entries = json.loads(manifest.read_text())["members"]
        return [_relocate(Path(entry["checkpoint"]), path) for entry in entries]

    nested = sorted(path.glob("member_*/*/best.pt"))
    if nested:
        by_member: dict[str, list[Path]] = {}
        for checkpoint in nested:
            by_member.setdefault(checkpoint.parent.parent.name, []).append(checkpoint)
        # One read, not one per member: every member of a committee is the same staged fit.
        order = _stage_order(nested[0])
        return [_final_stage(stages, order) for _member, stages in sorted(by_member.items())]

    flat = sorted(path.glob("member_*/best.pt"))
    if flat:
        return flat

    n_dirs = sum(1 for p in path.glob("member_*") if p.is_dir())
    hint = (
        f" {n_dirs} member_* director(ies) are present but hold no best.pt, which usually "
        f"means a sync copied directories without their files."
        if n_dirs else ""
    )
    raise FileNotFoundError(
        f"no committee members under {path}: expected {COMMITTEE_MANIFEST}, "
        f"member_*/best.pt, or member_*/<stage>/best.pt.{hint}"
    )
