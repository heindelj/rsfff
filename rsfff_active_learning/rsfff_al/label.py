"""Label stage: selected frames -> Q-Chem jobs in the cc_workers store -> training frames.

    label = QChemStoreLabel(store="/global/cfs/cdirs/m3196/heindelj/rsfff_store",
                            submit=["ccw submit {store} --site perlmutter_cpu --target 16"])

``run`` is repeatable, which is what easyAL's ``Pending`` asks for:

1. every selected frame becomes a ``force`` spec and, when its ``labels`` say so, an ``eda2``
   spec (:func:`rsfff_al.store_io.specs_for`), added to the store. Adding is idempotent --
   a spec is its hash -- so a rerun adds nothing, and a geometry the store already holds
   (from qchem_roundtrip, another loop, an earlier attempt) is not recomputed.
   Priority is the atom count, so the long jobs start first and the tail is short.
2. the ``submit`` commands run once (``{store}`` is substituted; cwd is the store), after
   the cc_workers source is copied to ``<store>/_code`` for the batch runners; the ``sync``
   commands run on every call.
3. while any job is pending or running: ``Pending``. With ``wait_seconds`` it polls that
   long first (for an interactive allocation where the runners are local).
4. once all are finished: jobs run with ``--no-parse`` are parsed here, and every frame whose
   force job is done becomes a training frame (with its EDA when that is done too; an EDA
   that failed demotes the frame to force-only rather than losing it). ``max_failed_fraction``
   of frames without a force result stops the stage.

Outputs: ``labeled.extxyz`` (every frame, for the record), ``labeled_eda.extxyz`` and
``labeled_force.extxyz`` beside it (the two kinds the trainer keeps apart), and
``scratch/jobs.json`` (frame -> job ids) and ``scratch/dropped.json``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from easyal import Contract, Label, Pending

from .stages import SELECTED

__all__ = ["QChemStoreLabel", "LABELED"]

LABELED = Contract(info=("energy", "n_fragments", "traj_id"),
                   arrays=("forces", "fragment_idx", "mulliken_charges"))


def _store_path(p) -> Path:
    path = p.get("store") or os.environ.get("RSFFF_STORE")
    if not path:
        raise ValueError("QChemStoreLabel needs store= (or $RSFFF_STORE)")
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def _run_hooks(commands, store: Path, ctx, what: str):
    for cmd in commands or []:
        line = cmd.format(store=shlex.quote(str(store)))
        proc = subprocess.run(line, shell=True, cwd=store, capture_output=True, text=True)
        ctx.note(f"{what}: {line} -> exit {proc.returncode}"
                 + (f"; {proc.stdout.strip()[-400:]}" if proc.stdout.strip() else "")
                 + (f"; stderr: {proc.stderr.strip()[-400:]}" if proc.returncode else ""))
        if proc.returncode != 0:
            raise RuntimeError(f"{what} hook failed: {line}\n{proc.stderr[-2000:]}")


def _install_code(store: Path) -> Path:
    """Copy the imported cc_workers package to ``<store>/_code/cc_workers``, which is what
    the site batch script puts on ``PYTHONPATH`` (``ccw push`` does the same with rsync)."""
    import shutil

    import cc_workers

    src = Path(cc_workers.__file__).parent
    dst = store / "_code" / "cc_workers"
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "output.out"))
    return dst


class QChemStoreLabel(Label):
    """Force (+ EDA) labels through a cc_workers store. See the module docstring.

    ``store``                path of the store (or ``$RSFFF_STORE``); created if missing
    ``submit``, ``sync``     shell commands (``{store}`` substituted), run in the store
    ``wait_seconds``         keep polling this long before reporting Pending (default 0)
    ``poll_seconds``         between polls (default 60)
    ``max_failed_fraction``  of frames without a force result that is tolerated (0.1)
    ``tags``                 extra tags on every job (the loop's name is added as ``al_loop``)
    """

    requires = SELECTED
    produces = LABELED

    def run(self, ctx):
        from cc_workers.common import store as st
        from cc_workers.common import sync

        from .store_io import specs_for, training_frame

        p = ctx.params
        store_dir = _store_path(p)
        store = st.Store(store_dir)
        ctx.track("store", store_dir / "store.json")
        jobs_file = ctx.scratch / "jobs.json"
        frames = ctx.read()

        # 1. specs into the store (idempotent)
        if jobs_file.exists():
            jobs = json.loads(jobs_file.read_text())
        else:
            tags = {"al_loop": ctx.root.name, "al_iteration": ctx.iteration, **(p.get("tags") or {})}
            jobs, created = [], 0
            for i, frame in enumerate(frames):
                entry = {"frame": i}
                n_atoms = len(frame["arrays"]["species"])
                for calc, spec in specs_for(frame, tags=tags).items():
                    job, new = store.add(spec, priority=n_atoms)
                    entry[calc] = job.id
                    created += int(new)
                jobs.append(entry)
            jobs_file.write_text(json.dumps(jobs, indent=0))
            n_eda = sum("eda2" in e for e in jobs)
            ctx.log(n_frames=len(frames), n_force=len(jobs), n_eda=n_eda, n_new_jobs=created,
                    store=str(store_dir))
            ctx.note(f"{len(jobs)} force + {n_eda} eda2 specs; {created} new to the store")

        # 2. hand the work to the runners, once
        submitted = ctx.scratch / "submitted.json"
        if not submitted.exists():
            _install_code(store_dir)             # <store>/_code/cc_workers for batch runners
            _run_hooks(p.get("submit"), store_dir, ctx, "submit")
            submitted.write_text(json.dumps({"time": time.time()}))

        # 3. wait
        deadline = time.time() + float(p.get("wait_seconds", 0))
        while True:
            _run_hooks(p.get("sync"), store_dir, ctx, "sync")
            counts, states = self._states(store, jobs)
            open_jobs = counts.get(st.PENDING, 0) + counts.get(st.RUNNING, 0)
            if open_jobs == 0:
                break
            if time.time() >= deadline:
                ctx.log(job_states=counts)
                raise Pending(f"{open_jobs} of {sum(counts.values())} Q-Chem jobs open "
                              f"({counts})")
            time.sleep(float(p.get("poll_seconds", 60)))

        # 4. collect
        out, dropped, demoted = [], [], 0
        for entry in jobs:
            frame = frames[entry["frame"]]
            fjob = store.get("qchem", entry["force"])
            if states[entry["force"]] != st.DONE:
                dropped.append({"frame": entry["frame"], "force": entry["force"],
                                "reason": fjob.status().get("reason", states[entry["force"]])})
                continue
            if not fjob.result_path.exists():
                sync.parse_job(fjob)
            eda_res = None
            if "eda2" in entry:
                ejob = store.get("qchem", entry["eda2"])
                if states[entry["eda2"]] == st.DONE:
                    if not ejob.result_path.exists():
                        sync.parse_job(ejob)
                    eda_res = ejob.load_result()
            record = fjob.record()       # a reused job may carry another request's tags
            record["tags"] = {**record.get("tags", {}),
                              "fragment_idx": [int(v) for v in frame["arrays"]["fragment_idx"]]}
            try:
                tf = training_frame(fjob.load_result(), eda_res, spec_record=record,
                                    extra_info=self._lineage(frame, ctx))
            except ValueError as exc:
                if eda_res is None:
                    dropped.append({"frame": entry["frame"], "force": entry["force"],
                                    "reason": str(exc)})
                    continue
                ctx.note(f"frame {entry['frame']}: {exc}; kept force-only")
                tf = training_frame(fjob.load_result(), None, spec_record=record,
                                    extra_info=self._lineage(frame, ctx))
                eda_res = None
            if "eda2" in entry and eda_res is None:
                demoted += 1
            out.append(tf)

        (ctx.scratch / "dropped.json").write_text(json.dumps(dropped, indent=1))
        frac = len(dropped) / max(len(jobs), 1)
        ctx.log(n_labeled=len(out), n_dropped=len(dropped), n_demoted_to_force_only=demoted,
                job_states=counts,
                n_with_eda=sum(1 for f in out if "eda_int" in f["info"]))
        if frac > float(p.get("max_failed_fraction", 0.1)):
            raise RuntimeError(f"{len(dropped)} of {len(jobs)} frames have no force result "
                               f"(> max_failed_fraction); see scratch/dropped.json")
        if not out:
            raise RuntimeError("no labeled frames")
        from easyal import write_extxyz

        eda = [f for f in out if "eda_int" in f["info"]]
        force_only = [f for f in out if "eda_int" not in f["info"]]
        write_extxyz(ctx.dir / "labeled_eda.extxyz", eda)
        write_extxyz(ctx.dir / "labeled_force.extxyz", force_only)
        return out

    @staticmethod
    def _states(store, jobs):
        states, counts = {}, {}
        for entry in jobs:
            for calc in ("force", "eda2"):
                if calc in entry:
                    s = store.get("qchem", entry[calc]).status()["state"]
                    states[entry[calc]] = s
                    counts[s] = counts.get(s, 0) + 1
        return counts, states

    @staticmethod
    def _lineage(frame, ctx) -> dict:
        info = frame["info"]
        keep = ("traj_id", "time_fs", "phase", "sigma_energy", "sigma_forces", "select_rank",
                "fs_to_failure", "committee_mean_energy")
        out = {k: info[k] for k in keep if k in info}
        out.update(al_loop=ctx.root.name, al_iteration=ctx.iteration,
                   split_group=str(info.get("traj_id", "")) or None)
        return {k: v for k, v in out.items() if v is not None}
