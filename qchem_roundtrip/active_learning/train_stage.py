"""Train stage: refit the film model as a committee of independent fits.

One model per iteration would be enough to sample with and useless to select with -- a single
network has no way to say where it is unsupported. So each iteration fits ``n_members`` models
that differ only in their initialization, on one shared train/holdout split, and the loop uses
their disagreement (see :mod:`committee`) both to choose what to label next and to say whether
it has converged.

Each member is a separate process running ``rsfff.train.train_film`` on a config this stage
writes out in full::

    <ctx.output>/
        committee.json                 members, their checkpoints, the split seed, val losses
        member_00/
            config.resolved.yaml       exactly what was fitted -- rerunnable by hand
            train.log
            member_00/best.pt          (train_film's own checkpoint_root/run_name layout)
        member_01/ ...

A member that finished writes ``done.json`` and is never refitted, so a driver killed by the
wall clock loses only what was in flight. A separate process per member is not just for
parallelism: ``train_film`` sets the global torch default dtype and builds module-level state,
and fitting four models in one interpreter would have them share it.

Warm starting
-------------
``warm_start=True`` points each member's ``train.init_from`` at the *same member index* of the
iteration before, so member 2 always continues member 2. That is what makes the size walk
affordable -- each iteration extends the data rather than replacing it, and a fit that already
knows dimers does not need to relearn them to fit pentamers.

It has a cost worth watching. Members that all descend from one starting checkpoint begin
correlated, and a committee that agrees because its members share a history rather than
because the data pins the answer will understate its own uncertainty. The holdout coverage
reported by the assess stage is the check on that: if the true value falls inside the
committee's interval far more often than 95% of the time, the spread has stopped meaning what
it should, and the fix is ``warm_start=False`` for an iteration.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from easyal import Train, iter_extxyz, write_extxyz

from committee import MANIFEST
from common import REPO_ROOT
from label_stage import QCHEM_TRAINING

__all__ = ["CommitteeTrain", "DEFAULT_TRAINING_CONFIG"]

#: The config this model is fitted with. Only a default -- ``config=`` overrides it.
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "water_film.yaml"


def _set_dotted(tree: dict, dotted: str, value) -> None:
    node = tree
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


class CommitteeTrain(Train):
    """Fit ``n_members`` film models on everything labeled so far.

    ``config``        the training YAML to start from. Defaults to the repository's
                      ``configs/water_film.yaml`` -- the config this model is fitted with --
                      which is found through ``$RSFFF_REPO`` or the installed rsfff
    ``n_members``     committee size (default 4)
    ``seed``          member ``k`` fits with ``train.seed = seed + k`` (default 0)
    ``split_seed``    ``data.seed``, shared by every member so they have one holdout
    ``warm_start``    continue each member from the same index of the previous iteration
    ``parallel``      members at once: an integer, or ``"auto"`` (one per visible GPU, else 1)
    ``overrides``     dotted keys written into every member's config, e.g.
                      ``{"train.epochs": 200, "device": "cuda"}``
    ``python``        interpreter for the member processes (default: this one)
    """

    requires = QCHEM_TRAINING
    output = "committee"

    def __init__(self, **params):
        params.setdefault("config", str(DEFAULT_TRAINING_CONFIG))
        params.setdefault("n_members", 4)
        params.setdefault("seed", 0)
        params.setdefault("split_seed", 0)
        params.setdefault("warm_start", True)
        params.setdefault("parallel", "auto")
        params.setdefault("overrides", {})
        params.setdefault("python", sys.executable)
        super().__init__(**params)

    # --- pieces -----------------------------------------------------------------------------

    def _n_parallel(self, n_members: int) -> int:
        requested = self.params["parallel"]
        if requested != "auto":
            return max(1, min(int(requested), n_members))
        try:
            import torch
            visible = torch.cuda.device_count()
        except Exception:
            visible = 0
        return max(1, min(visible or 1, n_members))

    @staticmethod
    def _previous_members(model_path) -> list[Path]:
        """The last iteration's member checkpoints, in order, or ``[]``."""
        if model_path is None:
            return []
        path = Path(model_path)
        if path.is_file():
            return [path]
        manifest = path / MANIFEST
        if manifest.exists():
            return [Path(e["checkpoint"]) for e in json.loads(manifest.read_text())["members"]]
        return sorted(path.glob("member_*/*/best.pt"))

    @staticmethod
    def _checkpoint_of(member: Path, tree: dict) -> Path:
        """Where ``train_film`` left this member's best checkpoint.

        Not simply ``<checkpoint_root>/<run_name>/best.pt``: a config with ``stages:`` fits
        them in sequence and each writes to ``<run_name>_<stage>``, so the one to keep is the
        last stage's. The config says which that is, and a newest-file search is the backstop
        for a layout neither branch predicted.
        """
        root = Path(tree["checkpoint_root"])
        run = str(tree["run_name"])
        stages = tree.get("stages") or []
        names = [f"{run}_{stage['name']}" for stage in stages][-1:] + [run]
        for name in names:
            candidate = root / name / "best.pt"
            if candidate.exists():
                return candidate
        found = sorted(root.rglob("best.pt"), key=lambda q: q.stat().st_mtime)
        if found:
            return found[-1]
        raise FileNotFoundError(
            f"no best.pt under {root} (looked for {', '.join(names)}); see "
            f"{member / 'train.log'}"
        )

    def _member_config(self, ctx, index: int, dataset: Path, parents: list[Path]) -> dict:
        template = Path(self.params["config"] or DEFAULT_TRAINING_CONFIG)
        if not template.exists():
            raise FileNotFoundError(
                f"training config {template} does not exist. Pass config=<yaml> (or "
                f"--train-config), or point $RSFFF_REPO at a checkout that has "
                f"configs/water_film.yaml."
            )
        tree = copy.deepcopy(yaml.safe_load(template.read_text()) or {})
        data = tree.setdefault("data", {})
        # Every path in the template is relative to the repo it was written in, and each
        # member runs in its own directory, so they are resolved to absolute paths here.
        # resolve_data_path alone is not enough: it returns a relative path unchanged when it
        # exists relative to *this* process's directory, which is not the member's.
        from rsfff.train.data import resolve_data_path

        def absolute(value):
            if isinstance(value, list):
                return [absolute(v) for v in value]
            return str(Path(resolve_data_path(value)).resolve())

        for key in ("reference_energies", "atomic_reference_states", "isolated_species",
                    "diabatic_states", "monomer_path", "large_path"):
            if data.get(key):
                data[key] = absolute(data[key])
        data["path"] = [str(dataset)]
        data["seed"] = int(self.params["split_seed"])
        train = tree.setdefault("train", {})
        train["seed"] = int(self.params["seed"]) + index
        if self.params["warm_start"] and parents:
            train["init_from"] = str(parents[min(index, len(parents) - 1)])
        tree["run_name"] = f"member_{index:02d}"
        tree["checkpoint_root"] = str(ctx.output / f"member_{index:02d}")
        for dotted, value in dict(self.params["overrides"]).items():
            _set_dotted(tree, dotted, value)
        return tree

    # --- the stage --------------------------------------------------------------------------

    def run(self, ctx):
        n_members = int(self.params["n_members"])
        ctx.output.mkdir(parents=True, exist_ok=True)

        # 1. one file to fit: the initial data plus every labeled set so far
        dataset = ctx.scratch / "training_data.extxyz"
        if not dataset.exists():
            n_frames = 0
            with open(dataset, "w") as fh:
                for source in ctx.training_data:
                    for frame in iter_extxyz(source):
                        write_extxyz(fh, [frame], append=True)
                        n_frames += 1
            ctx.note(f"{n_frames} frames from {len(ctx.training_data)} file(s) -> {dataset}")
        ctx.log(n_training_frames=sum(1 for _ in iter_extxyz(dataset)),
                n_training_files=len(ctx.training_data))

        parents = self._previous_members(ctx.model)
        if self.params["warm_start"] and not parents:
            ctx.note("no previous committee to warm start from; fitting from scratch")

        # 2. the members
        pending = []
        for index in range(n_members):
            member = ctx.output / f"member_{index:02d}"
            member.mkdir(parents=True, exist_ok=True)
            if (member / "done.json").exists():
                continue
            resolved = member / "config.resolved.yaml"
            resolved.write_text(yaml.safe_dump(
                self._member_config(ctx, index, dataset, parents), sort_keys=False))
            pending.append((index, member, resolved))
        if not pending:
            ctx.note(f"all {n_members} members already trained")

        lanes = self._n_parallel(len(pending)) if pending else 1
        ctx.note(f"training {len(pending)} of {n_members} member(s), {lanes} at a time")
        for start in range(0, len(pending), lanes):
            batch = pending[start:start + lanes]
            running = []
            trees = {}
            for lane, (index, member, resolved) in enumerate(batch):
                trees[index] = yaml.safe_load(resolved.read_text())
                env = dict(os.environ)
                if lanes > 1:
                    env["CUDA_VISIBLE_DEVICES"] = str(lane)   # each member sees "cuda:0"
                log = open(member / "train.log", "w")
                t0 = time.time()
                running.append((index, member, t0, log, subprocess.Popen(
                    [self.params["python"], "-m", "rsfff.train.train_film", str(resolved)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(member))))
            for index, member, t0, log, process in running:
                code = process.wait()
                log.close()
                elapsed = round(time.time() - t0, 1)
                if code != 0:
                    tail = (member / "train.log").read_text().splitlines()[-15:]
                    raise RuntimeError(
                        f"member {index:02d} exited {code} after {elapsed}s; last lines of "
                        f"{member / 'train.log'}:\n  " + "\n  ".join(tail)
                    )
                checkpoint = self._checkpoint_of(member, trees[index])
                (member / "done.json").write_text(json.dumps(
                    {"member": index, "checkpoint": str(checkpoint),
                     "elapsed_seconds": elapsed}, indent=2) + "\n")
                ctx.note(f"member {index:02d} done in {elapsed}s")

        # 3. the manifest the rest of the loop reads
        members = []
        for index in range(n_members):
            done = ctx.output / f"member_{index:02d}" / "done.json"
            members.append(json.loads(done.read_text()))
        (ctx.output / MANIFEST).write_text(json.dumps({
            "n_members": n_members,
            "split_seed": int(self.params["split_seed"]),
            "seed": int(self.params["seed"]),
            "warm_start": bool(self.params["warm_start"]),
            "warm_started_from": [str(p) for p in parents] if self.params["warm_start"] else [],
            "training_data": [str(p) for p in ctx.training_data],
            "members": members,
        }, indent=2) + "\n")
        return {"n_members": n_members,
                "seconds": round(sum(m["elapsed_seconds"] for m in members), 1)}
