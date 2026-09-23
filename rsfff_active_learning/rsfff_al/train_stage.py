"""Train stage: export the training set from the store, fit a committee, hand it on.

Every iteration's fit is self-contained in its stage directory::

    iter_NNN/train/
        data/eda/w{n}.xyz  data/force/w{n}.xyz  data/manifest.json     the store export
        config.yaml                                                     what is fitted
        committee/  committee.json  member_NN/...                       the output
        scratch/submitted.json  slurm-<id>.out                          when run as a batch job

The export takes every done water force(+EDA) job at the reference level from the store:
legacy data, the labels of this loop up to this iteration (``al_loop`` = the loop directory's
name), and any extra ``loops``. Frames with an EDA feed the main cluster stream, frames
without it the force-only stream (``data.force_path``), and the monomer set and atomic
references ride along as anchors in every fit.

``mode="sbatch"`` (default) queues ``train/scripts/committee.slurm`` on a 4-GPU node and
reports ``Pending`` until ``committee/committee.json`` appears; ``mode="local"`` runs
``train_committee.py`` in this process's allocation (an interactive GPU node).

Warm start: member k continues member k of the committee the iteration started from
(``ctx.model``), with the ``isolated`` stage dropped and ``warm_epochs`` of the ``full``
stage. ``cold_every = m`` refits from scratch every m-th iteration instead, so members do
not end up agreeing because of a shared history.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from easyal import Pending, Train

from .label import LABELED, _store_path

__all__ = ["CommitteeTrain"]

AL_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = AL_ROOT / "train"


class CommitteeTrain(Train):
    """Parameters

    ``store``            the cc_workers store (or ``$RSFFF_STORE``)
    ``template``         base YAML (default ``train/water_film_al.yaml``)
    ``members``          committee size (4)
    ``mode``             ``sbatch`` | ``local``
    ``sbatch``           extra sbatch arguments (default ``-A m3196 -q regular -t 08:00:00``)
    ``warm_start``       continue from ``ctx.model`` member by member (True)
    ``warm_epochs``      epochs of the ``full`` stage when warm starting (40)
    ``cold_every``       fit from scratch every this many iterations (0: never)
    ``loops``            extra ``al_loop`` names whose labels count (default: this loop only)
    ``legacy``           include store jobs without AL tags (True)
    ``max_waters``       drop larger clusters from the export (None)
    ``set``              list of ``key=value`` overrides passed to train_committee.py
    ``device``           ``cuda`` (default) | ``cpu``
    """

    name = "train"
    output = "committee"
    requires = LABELED

    def run(self, ctx):
        from .dataset import export_training

        p = ctx.params
        out = ctx.output
        if (out / "committee.json").exists():
            return self._metrics(out)

        data = ctx.dir / "data"
        if not (data / "manifest.json").exists():
            loops = [ctx.root.name, *(p.get("loops") or [])]
            manifest = export_training(_store_path(p), data, loops=loops, upto=ctx.iteration,
                                       legacy=p.get("legacy", True),
                                       max_waters=p.get("max_waters"), log=ctx.note)
            ctx.log(data=manifest["totals"],
                    data_files={k: v["n_frames"] for k, v in manifest["files"].items()})
        config = self._write_config(ctx, data)
        cmd = self._command(ctx, config, out)

        if p.get("mode", "sbatch") == "local":
            ctx.note("training locally: " + " ".join(cmd))
            log = ctx.dir / "train_committee.log"
            with open(log, "w") as fh:
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            if code != 0:
                raise RuntimeError(f"train_committee.py exited {code}; see {log}")
            return self._metrics(out)

        submitted = ctx.scratch / "submitted.json"
        if submitted.exists():
            job = json.loads(submitted.read_text())["job_id"]
            if _in_queue(job):
                raise Pending(f"committee training job {job} is queued or running")
            raise RuntimeError(f"training job {job} left the queue without writing "
                               f"{out / 'committee.json'}; see {ctx.dir}/slurm-{job}.out. "
                               f"Delete scratch/submitted.json to resubmit (finished members "
                               f"are kept).")
        sb = ["sbatch", "--parsable", "--job-name", f"rsfff_al_train_{ctx.root.name}",
              "--output", str(ctx.dir / "slurm-%j.out"),
              *shlex.split(p.get("sbatch", "-A m3196 -q regular -t 08:00:00")),
              str(TRAIN_DIR / "scripts" / "committee.slurm"), *cmd[2:]]
        proc = subprocess.run(sb, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr}")
        job = proc.stdout.strip().split(";")[0]
        submitted.write_text(json.dumps({"job_id": job, "command": sb}))
        ctx.note(f"submitted committee training job {job}")
        raise Pending(f"committee training job {job} submitted")

    # ------------------------------------------------------------------------------------
    def _warm(self, ctx) -> bool:
        p = ctx.params
        if not p.get("warm_start", True) or ctx.model is None:
            return False
        every = int(p.get("cold_every", 0) or 0)
        return not (every and ctx.iteration > 0 and ctx.iteration % every == 0)

    def _write_config(self, ctx, data: Path) -> Path:
        p = ctx.params
        template = Path(p.get("template") or TRAIN_DIR / "water_film_al.yaml")
        tree = yaml.safe_load(template.read_text())
        d = tree.setdefault("data", {})
        eda = sorted(str(f) for f in (data / "eda").glob("*.xyz"))
        force = sorted(str(f) for f in (data / "force").glob("*.xyz"))
        if not eda:
            raise RuntimeError(f"the export has no EDA frames ({data}); the main stream "
                               f"cannot be empty")
        d["path"] = eda
        d["force_path"] = force or None
        for key in ("reference_energies", "monomer_path"):
            value = d.get(key)
            if value and not Path(value).is_absolute():
                d[key] = str((template.parent / value).resolve())
        if not force:
            tree.setdefault("film", {})["force_stream_weight"] = 0.0
        if self._warm(ctx):
            stages = [s for s in tree.get("stages") or [] if s.get("name") != "isolated"]
            for s in stages:
                s.setdefault("train", {})["epochs"] = int(p.get("warm_epochs", 40))
            tree["stages"] = stages
        path = ctx.dir / "config.yaml"
        path.write_text(yaml.safe_dump(tree, sort_keys=False))
        ctx.track("config", path)
        return path

    def _command(self, ctx, config: Path, out: Path) -> list[str]:
        p = ctx.params
        cmd = [sys.executable, str(TRAIN_DIR / "train_committee.py"), "--config", str(config),
               "--out", str(out), "--members", str(int(p.get("members", 4))),
               "--seed", str(int(p.get("seed", 0)) + 100 * ctx.iteration),
               "--device", str(p.get("device", "cuda"))]
        if self._warm(ctx):
            cmd += ["--init-from", str(ctx.model)]
        for item in p.get("set") or []:
            cmd += ["--set", item]
        return cmd

    @staticmethod
    def _metrics(out: Path) -> dict:
        m = json.loads((out / "committee.json").read_text())
        return {"val_losses": [x.get("val_loss") for x in m["members"]],
                "warm_start": m.get("warm_start")}


def _in_queue(job: str) -> bool:
    proc = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T"], capture_output=True,
                          text=True)
    return proc.returncode == 0 and bool(proc.stdout.strip())
