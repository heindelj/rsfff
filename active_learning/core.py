"""The campaign machinery: stages, standard locations, and provenance.

An active-learning campaign is a sequence of iterations. Each iteration runs the same ordered
stages (``build -> sample -> label -> train`` by default), and every stage gets its own
directory::

    <campaign>/
        campaign.json               what the campaign started from (checkpoint, base data)
        events.jsonl                append-only log of every stage start/finish/pending/failure
        summary.json                one row per (iteration, stage): status + metrics
        iter_000/
            build/   stage.json  structures.extxyz
            sample/  stage.json  candidates.extxyz  trajectories/
            label/   stage.json  selected.extxyz    dataset/
            train/   stage.json  best.pt            config.yaml
            _superseded/<stage>_<timestamp>/        old outputs of a stage that was re-run
        iter_001/
            ...

A stage implementation only has to write its declared outputs (``Stage.outputs``) into
``ctx.output(key)``. The next stage finds them with ``ctx.input(key)``. Everything else --
hashing what was read and written, the code version, the model the stage started from, the
cumulative training set, timings, metrics -- is recorded in ``stage.json`` by the campaign.

Provenance is collected automatically for anything reached through the context
(``ctx.input``, ``ctx.checkpoint``, ``ctx.training_data``, ``ctx.history``). Files a stage reads
some other way (a YAML config, a Q-Chem job directory) should be declared with
``ctx.add_input`` so they are hashed too, and a stage is re-run when any recorded input changes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Packages whose installed version is recorded with every stage. Looked up through
#: ``importlib.metadata`` so recording them never imports torch.
TRACKED_PACKAGES = ("rsfff", "torch", "numpy", "ase", "e3nn", "cuequivariance-torch", "pyscf")

STATUS_COMPLETE = "complete"
STATUS_PENDING = "pending"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"


class StagePending(Exception):
    """Raise from ``Stage.run`` when the stage is waiting on work outside this process.

    The typical case is the label stage after it has written Q-Chem inputs: nothing more can
    happen until the jobs have run on Perlmutter and the outputs are synced back. The campaign
    records ``status = pending`` with this message and stops. Running the campaign again calls
    ``run`` on the same stage directory, so ``run`` must be safe to call repeatedly (write
    inputs only if missing, then check for outputs).
    """


class ProvenanceError(RuntimeError):
    """The campaign's recorded starting point no longer matches what is on disk."""


# --------------------------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------------------------

def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _count_xyz_frames(path: Path) -> int | None:
    """Frame count of an (ext)xyz file without parsing it; None if it does not look like one."""
    try:
        n = 0
        with open(path) as fh:
            while True:
                line = fh.readline()
                if not line:
                    return n
                if not line.strip():
                    continue
                natoms = int(line.split()[0])
                fh.readline()
                for _ in range(natoms):
                    fh.readline()
                n += 1
    except (ValueError, IndexError, UnicodeDecodeError):
        return None


class _HashCache:
    """sha256 keyed on (size, mtime) so unchanged multi-GB trajectories are hashed once."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self._data: dict[str, Any] = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}
        self._dirty = False

    def sha256(self, path: Path) -> str:
        st = path.stat()
        key = str(path.resolve())
        hit = self._data.get(key)
        if hit and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
            return hit["sha256"]
        digest = _sha256_file(path)
        self._data[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        self._dirty = True
        return digest

    def save(self) -> None:
        if self._dirty:
            _write_json(self.path, self._data)
            self._dirty = False


def fingerprint(path: Path, cache: _HashCache | None = None, relative_to: Path | None = None) -> dict:
    """A JSON-able description of a file or directory that changes whenever its content does.

    Directories hash the sorted list of (relative path, file hash) pairs, so a renamed or added
    file changes the fingerprint too. ``.xyz``/``.extxyz`` files also record their frame count.
    """
    path = Path(path)
    shown = str(path)
    if relative_to is not None:
        try:
            shown = str(path.resolve().relative_to(relative_to.resolve()))
        except ValueError:
            pass
    sha = cache.sha256 if cache is not None else _sha256_file
    if not path.exists():
        return {"path": shown, "exists": False}
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith("."))
        h = hashlib.sha256()
        n_frames = 0
        for p in files:
            h.update(f"{p.relative_to(path)}\0{sha(p)}\n".encode())
            if p.suffix in (".xyz", ".extxyz"):
                n_frames += _count_xyz_frames(p) or 0
        out = {"path": shown, "kind": "dir", "sha256": h.hexdigest(), "n_files": len(files),
               "bytes": sum(p.stat().st_size for p in files)}
        if n_frames:
            out["n_frames"] = n_frames
        return out
    out = {"path": shown, "kind": "file", "sha256": sha(path), "bytes": path.stat().st_size}
    if path.suffix in (".xyz", ".extxyz"):
        n = _count_xyz_frames(path)
        if n is not None:
            out["n_frames"] = n
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _jsonable(obj: Any) -> Any:
    """What ``obj`` looks like after a trip through stage.json (Paths become strings, tuples
    lists), so live parameters compare equal to recorded ones."""
    return json.loads(json.dumps(obj, default=str))


# --------------------------------------------------------------------------------------------
# code / environment provenance
# --------------------------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def code_state(repo: Path, patch_path: Path | None = None) -> dict:
    """Commit, branch and dirtiness of ``repo``. Uncommitted changes to tracked files are
    written to ``patch_path`` so the exact code a stage ran is recoverable as
    ``git checkout <commit> && git apply code.patch``. Untracked files are listed, not saved."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return {"repo": str(repo), "git": False}
    diff = _git(repo, "diff", "HEAD", "--binary") or ""
    untracked = (_git(repo, "ls-files", "--others", "--exclude-standard", "--", "*.py") or "")
    state = {
        "repo": str(repo),
        "commit": commit.strip(),
        "branch": (_git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
        "dirty": bool(diff),
        "untracked_py": [line for line in untracked.splitlines() if line],
    }
    if diff:
        state["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        if patch_path is not None:
            patch_path.write_text(diff)
            state["patch"] = patch_path.name
    return state


def environment_state() -> dict:
    from importlib import metadata

    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": versions,
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------

class Stage(ABC):
    """One step of an iteration. Subclass, set ``name`` and ``outputs``, implement ``run``.

    ``outputs`` maps a key to a path relative to the stage directory; a trailing ``/`` marks a
    directory. Keys must be unique across the stages of a campaign, because downstream stages
    look them up by key alone (``ctx.input("candidates")``).

    Constructor keyword arguments are stored in ``self.params`` and recorded verbatim in
    ``stage.json``, so put every knob that changes the result there (temperatures, number of
    frames to select, epochs, ...) rather than in module globals.
    """

    name: ClassVar[str]
    outputs: ClassVar[Mapping[str, str]] = {}

    def __init__(self, **params: Any):
        self.params = dict(params)

    @abstractmethod
    def run(self, ctx: "StageContext") -> None:
        """Do the work and write every declared output to ``ctx.output(key)``."""

    def validate(self, ctx: "StageContext") -> None:
        """Called after ``run`` returns. Raise to mark the stage failed. The default only checks
        that the declared outputs exist; subclasses add format checks and summary metrics."""
        missing = [key for key in self.outputs if not ctx.output(key).exists()]
        if missing:
            raise FileNotFoundError(
                f"stage {self.name!r} finished without writing: "
                + ", ".join(f"{k} -> {ctx.output(k)}" for k in missing)
            )

    def describe(self) -> dict:
        cls = type(self)
        return {"class": f"{cls.__module__}.{cls.__qualname__}", "params": _jsonable(self.params)}


@dataclass
class StageContext:
    """Everything a stage needs, and the recorder of everything it touched."""

    campaign: "Campaign"
    iteration: int
    stage: Stage
    dir: Path
    record: dict = field(default_factory=dict)

    # --- locations -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.stage.name

    @property
    def root(self) -> Path:
        """The campaign directory."""
        return self.campaign.root

    @property
    def scratch(self) -> Path:
        """A per-stage working directory for intermediates that are not outputs."""
        path = self.dir / "scratch"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output(self, key: str) -> Path:
        """Standard location of one of this stage's outputs. Parent directories are created;
        directory outputs (declared with a trailing ``/``) are created too."""
        try:
            rel = self.stage.outputs[key]
        except KeyError:
            raise KeyError(f"stage {self.name!r} declares no output {key!r}; "
                           f"declared: {sorted(self.stage.outputs)}") from None
        path = self.dir / rel.rstrip("/")
        if rel.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def input(self, key: str, *, iteration: int | None = None) -> Path:
        """An output of an earlier stage in this iteration (or of ``iteration``), by key.

        The upstream stage must be complete. The file's fingerprint is recorded as an input of
        this stage.
        """
        it = self.iteration if iteration is None else iteration
        owner = self.campaign.owner_of(key)
        if it == self.iteration and self.campaign.stage_index(owner.name) >= self.campaign.stage_index(self.name):
            raise ValueError(f"{self.name!r} cannot read {key!r}: {owner.name!r} does not run before it")
        self.campaign.require_complete(it, owner.name)
        path = self.campaign.stage_dir(it, owner.name) / owner.outputs[key].rstrip("/")
        self._record_input(f"{owner.name}.{key}" + (f"@{it}" if it != self.iteration else ""), path)
        return path

    @property
    def checkpoint(self) -> Path | None:
        """The model this iteration starts from: the previous iteration's trained checkpoint,
        or the campaign's initial checkpoint in iteration 0."""
        path = self.campaign.checkpoint_for(self.iteration)
        if path is not None:
            self._record_input("checkpoint", path)
        return path

    def training_data(self, *, include_current: bool = True) -> list[Path]:
        """The cumulative training set: the campaign's base data followed by the labeled
        ``dataset/`` directory of every completed label stage up to this iteration."""
        paths = self.campaign.training_data(self.iteration, include_current=include_current)
        self.record["training_data"] = [
            fingerprint(p, self.campaign.hashes, self.campaign.repo) | {"abspath": str(p)}
            for p in paths
        ]
        self.record["training_data_include_current"] = include_current
        return paths

    def history(self, stage_name: str) -> list[Path]:
        """Directories of ``stage_name`` in every earlier iteration where it completed. Useful for
        growing clusters from last iteration's structures or de-duplicating selections."""
        dirs = []
        for it in range(self.iteration):
            rec = self.campaign.read_record(it, stage_name)
            if rec and rec.get("status") == STATUS_COMPLETE:
                d = self.campaign.stage_dir(it, stage_name)
                dirs.append(d)
                self._record_input(f"{stage_name}@{it}", d)
        return dirs

    # --- recording -------------------------------------------------------------------------

    def add_input(self, name: str, path: str | os.PathLike) -> Path:
        """Declare an external file or directory this stage read (a config, a job bundle)."""
        path = Path(path)
        self._record_input(name, path)
        return path

    def add_output(self, name: str, path: str | os.PathLike) -> Path:
        """Declare a product written outside the stage directory (e.g. Q-Chem job dirs)."""
        path = Path(path)
        self.record.setdefault("extra_outputs", {})[name] = str(path)
        return path

    def log_metrics(self, **metrics: Any) -> None:
        """Record performance numbers for this stage. Later calls update earlier keys."""
        self.record.setdefault("metrics", {}).update(metrics)

    def log_params(self, **params: Any) -> None:
        """Record parameters decided at run time (a seed, a resolved config path, ...)."""
        self.record.setdefault("runtime_params", {}).update(params)

    def note(self, text: str) -> None:
        self.record.setdefault("notes", []).append({"time": _now(), "text": text})
        print(f"[{self.name} it{self.iteration}] {text}", flush=True)

    def _record_input(self, name: str, path: Path) -> None:
        self.record.setdefault("inputs", {})[name] = fingerprint(
            path, self.campaign.hashes, self.campaign.repo
        ) | {"abspath": str(Path(path).resolve())}


# --------------------------------------------------------------------------------------------
# campaign
# --------------------------------------------------------------------------------------------

class Campaign:
    """Runs stages over iterations and keeps the books.

    Parameters
    ----------
    root
        Campaign directory. Created if missing.
    stages
        Stage instances in execution order.
    initial_checkpoint
        The model iteration 0 starts from.
    base_data
        Training files that predate the campaign (e.g. the ``data.path`` list of the config the
        initial checkpoint was trained with). Always first in ``ctx.training_data()``.
    repo
        The repository whose git state is recorded. Defaults to this rsfff checkout.

    The starting point is frozen in ``campaign.json`` on first use; constructing the campaign
    later with a different checkpoint or base data raises ``ProvenanceError`` rather than
    silently rewriting history (pass ``allow_changed_start=True`` to accept and log the change).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        stages: Sequence[Stage],
        *,
        initial_checkpoint: str | os.PathLike | None = None,
        base_data: Iterable[str | os.PathLike] = (),
        repo: str | os.PathLike = REPO_ROOT,
        description: str = "",
        allow_changed_start: bool = False,
    ):
        self.root = Path(root).resolve()
        self.repo = Path(repo).resolve()
        self.stages = list(stages)
        self.description = description
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}")
        keys: dict[str, str] = {}
        for s in self.stages:
            for k in s.outputs:
                if k in keys:
                    raise ValueError(f"output key {k!r} declared by both {keys[k]!r} and {s.name!r}")
                keys[k] = s.name
        self.initial_checkpoint = Path(initial_checkpoint).resolve() if initial_checkpoint else None
        self.base_data = [Path(p).resolve() for p in base_data]
        self.root.mkdir(parents=True, exist_ok=True)
        self.hashes = _HashCache(self.root / ".hash_cache.json")
        self._init_manifest(allow_changed_start)

    # --- layout ----------------------------------------------------------------------------

    def iteration_dir(self, iteration: int) -> Path:
        return self.root / f"iter_{iteration:03d}"

    def stage_dir(self, iteration: int, stage_name: str) -> Path:
        return self.iteration_dir(iteration) / stage_name

    def stage_index(self, stage_name: str) -> int:
        for i, s in enumerate(self.stages):
            if s.name == stage_name:
                return i
        raise KeyError(f"no stage named {stage_name!r}")

    def owner_of(self, key: str) -> Stage:
        for s in self.stages:
            if key in s.outputs:
                return s
        raise KeyError(f"no stage declares an output {key!r}")

    def declares(self, key: str) -> bool:
        return any(key in s.outputs for s in self.stages)

    def read_record(self, iteration: int, stage_name: str) -> dict | None:
        path = self.stage_dir(iteration, stage_name) / "stage.json"
        return _read_json(path) if path.exists() else None

    def require_complete(self, iteration: int, stage_name: str) -> dict:
        rec = self.read_record(iteration, stage_name)
        if not rec or rec.get("status") != STATUS_COMPLETE:
            status = rec.get("status") if rec else "not run"
            raise RuntimeError(f"iteration {iteration} stage {stage_name!r} is {status}, not complete")
        return rec

    def iterations(self) -> list[int]:
        out = []
        for d in self.root.glob("iter_*"):
            try:
                out.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return sorted(out)

    # --- model and data lineage ------------------------------------------------------------

    def checkpoint_for(self, iteration: int) -> Path | None:
        """Model an iteration starts from. Looks for the last stage that declares a
        ``checkpoint`` output in the previous iteration."""
        if iteration == 0 or not self.declares("checkpoint"):
            return self.initial_checkpoint
        owner = self.owner_of("checkpoint")
        self.require_complete(iteration - 1, owner.name)
        return self.stage_dir(iteration - 1, owner.name) / owner.outputs["checkpoint"].rstrip("/")

    def training_data(self, iteration: int, *, include_current: bool = True) -> list[Path]:
        paths = list(self.base_data)
        if not self.declares("dataset"):
            return paths
        owner = self.owner_of("dataset")
        last = iteration if include_current else iteration - 1
        for it in range(last + 1):
            rec = self.read_record(it, owner.name)
            if rec and rec.get("status") == STATUS_COMPLETE:
                d = self.stage_dir(it, owner.name) / owner.outputs["dataset"].rstrip("/")
                paths.extend(sorted(p for p in d.iterdir()
                                    if p.suffix in (".xyz", ".extxyz") and not p.name.startswith(".")))
        return paths

    # --- manifest --------------------------------------------------------------------------

    def _start_state(self) -> dict:
        fp = lambda p: fingerprint(p, self.hashes, self.repo) | {"abspath": str(p)}  # noqa: E731
        return {
            "initial_checkpoint": fp(self.initial_checkpoint) if self.initial_checkpoint else None,
            "base_data": [fp(p) for p in self.base_data],
        }

    def _init_manifest(self, allow_changed: bool) -> None:
        path = self.root / "campaign.json"
        start = self._start_state()
        pipeline = _jsonable([s.describe() | {"name": s.name, "outputs": dict(s.outputs)}
                              for s in self.stages])
        if not path.exists():
            _write_json(path, {
                "name": self.root.name,
                "description": self.description,
                "created": _now(),
                "start": start,
                "pipeline": pipeline,
                "code": code_state(self.repo),
                "environment": environment_state(),
                "start_changes": [],
            })
            self._event(event="created", start=start)
            self.hashes.save()
            return
        manifest = _read_json(path)

        def strip(state):  # compare content, not where the file happens to be mounted
            if state is None:
                return None
            if isinstance(state, list):
                return [strip(s) for s in state]
            return {k: v for k, v in state.items() if k not in ("abspath", "path")}

        old = manifest["start"]
        if strip(old["initial_checkpoint"]) != strip(start["initial_checkpoint"]) or \
                strip(old["base_data"]) != strip(start["base_data"]):
            msg = (f"{path}: the campaign was started from a different checkpoint/base data "
                   f"than the one given now")
            if not allow_changed:
                raise ProvenanceError(msg + " (pass allow_changed_start=True to accept and log it)")
            manifest["start_changes"].append({"time": _now(), "old": old, "new": start})
            manifest["start"] = start
            self._event(event="start_changed", old=old, new=start)
        if manifest.get("pipeline") != pipeline:
            manifest.setdefault("pipeline_history", []).append(
                {"time": _now(), "pipeline": manifest.get("pipeline")})
            manifest["pipeline"] = pipeline
        _write_json(path, manifest)
        self.hashes.save()

    def _event(self, **payload: Any) -> None:
        with open(self.root / "events.jsonl", "a") as fh:
            fh.write(json.dumps({"time": _now(), **payload}, default=str) + "\n")

    # --- staleness -------------------------------------------------------------------------

    def stale_reasons(self, iteration: int, stage: Stage) -> list[str]:
        """Why a completed stage would have to run again: its recorded inputs or outputs no
        longer match the files on disk. Empty list means up to date."""
        rec = self.read_record(iteration, stage.name)
        if not rec or rec.get("status") != STATUS_COMPLETE:
            return ["not complete"]
        reasons = []
        recorded = list(rec.get("inputs", {}).items())
        recorded += [(f"training_data {fp['path']}", fp) for fp in rec.get("training_data", [])]
        for name, fp in recorded:
            now = fingerprint(Path(fp["abspath"]), self.hashes, self.repo)
            if now.get("sha256") != fp.get("sha256"):
                reasons.append(f"input {name} changed")
        if "training_data" in rec:
            now = self.training_data(iteration,
                                     include_current=rec.get("training_data_include_current", True))
            if [str(p) for p in now] != [fp["abspath"] for fp in rec["training_data"]]:
                reasons.append("training set membership changed")
        for name, fp in rec.get("outputs", {}).items():
            now = fingerprint(self.stage_dir(iteration, stage.name) / fp["relpath"], self.hashes)
            if now.get("sha256") != fp.get("sha256"):
                reasons.append(f"output {name} changed on disk")
        if rec.get("stage", {}).get("params") != _jsonable(stage.params):
            reasons.append("stage parameters changed")
        return reasons

    # --- running ---------------------------------------------------------------------------

    def run(
        self,
        iterations: int,
        *,
        rerun: Iterable[tuple[int, str]] = (),
        stop_after: tuple[int, str] | None = None,
        on_stale: str = "rerun",
    ) -> str:
        """Bring iterations ``0 .. iterations-1`` up to date, in order.

        Completed, up-to-date stages are skipped. A stage runs when it has never completed, is
        pending, is listed in ``rerun`` as ``(iteration, stage_name)``, or is stale (an input,
        output or parameter changed). ``on_stale`` is ``"rerun"`` (default), ``"keep"`` (warn and
        leave it), or ``"error"``.

        Returns ``"complete"`` when everything requested is done, ``"pending"`` if a stage is
        waiting on external work, or ``"stopped"`` when ``stop_after`` was reached.
        """
        rerun = {(int(i), str(s)) for i, s in rerun}
        with _CampaignLock(self.root):
            for it in range(iterations):
                for stage in self.stages:
                    try:
                        status = self._bring_up_to_date(it, stage, force=(it, stage.name) in rerun,
                                                        on_stale=on_stale)
                    finally:
                        self.write_summary()
                    if status == STATUS_PENDING:
                        return STATUS_PENDING
                    if stop_after == (it, stage.name):
                        return "stopped"
        return STATUS_COMPLETE

    def _bring_up_to_date(self, it: int, stage: Stage, *, force: bool, on_stale: str) -> str:
        rec = self.read_record(it, stage.name)
        status = rec.get("status") if rec else None
        if status == STATUS_COMPLETE and not force:
            reasons = self.stale_reasons(it, stage)
            if not reasons:
                return STATUS_COMPLETE
            msg = f"iteration {it} stage {stage.name!r} is stale: " + "; ".join(reasons)
            if on_stale == "keep":
                print(f"[campaign] warning: {msg} (keeping it)", flush=True)
                return STATUS_COMPLETE
            if on_stale == "error":
                raise ProvenanceError(msg)
            print(f"[campaign] {msg}; re-running", flush=True)
            self._supersede(it, stage.name, reasons)
        elif status == STATUS_COMPLETE and force:
            self._supersede(it, stage.name, ["rerun requested"])
        elif status in (STATUS_FAILED, STATUS_RUNNING):
            # A failed or interrupted attempt: keep its record for the log, reuse the directory
            # so a resumable stage can pick up partial work.
            self._event(iteration=it, stage=stage.name, event="retry", previous_status=status)
        return self._run_stage(it, stage, previous=rec if status in (STATUS_PENDING, STATUS_FAILED, STATUS_RUNNING) else None)

    def _supersede(self, it: int, stage_name: str, reasons: list[str]) -> None:
        src = self.stage_dir(it, stage_name)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = self.iteration_dir(it) / "_superseded" / f"{stage_name}_{stamp}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        self._event(iteration=it, stage=stage_name, event="superseded", reasons=reasons,
                    moved_to=str(dst.relative_to(self.root)))

    def _run_stage(self, it: int, stage: Stage, previous: dict | None) -> str:
        d = self.stage_dir(it, stage.name)
        d.mkdir(parents=True, exist_ok=True)
        attempts = (previous or {}).get("attempts", [])
        if previous:
            attempts = attempts + [{k: previous.get(k) for k in
                                    ("status", "started", "finished", "message", "error")}]
        ctx = StageContext(self, it, stage, d)
        ctx.record.update({
            "iteration": it,
            "stage": stage.describe() | {"name": stage.name},
            "status": STATUS_RUNNING,
            "started": _now(),
            "attempts": attempts,
            "code": code_state(self.repo, d / "code.patch"),
            "environment": environment_state(),
        })
        # Carry forward what an earlier pending attempt already recorded (metrics, notes).
        for key in ("metrics", "notes", "runtime_params", "extra_outputs"):
            if previous and key in previous:
                ctx.record[key] = previous[key]
        _write_json(d / "stage.json", ctx.record)
        self._event(iteration=it, stage=stage.name, event="started")
        print(f"[campaign] iteration {it}: {stage.name}", flush=True)
        t0 = time.perf_counter()
        try:
            if it > 0 or self.initial_checkpoint is not None:
                ctx.checkpoint  # noqa: B018 -- records the starting model for every stage
            stage.run(ctx)
            stage.validate(ctx)
        except StagePending as exc:
            ctx.record.update(status=STATUS_PENDING, message=str(exc))
            print(f"[campaign] iteration {it}: {stage.name} pending -- {exc}", flush=True)
        except BaseException as exc:
            ctx.record.update(status=STATUS_FAILED, message=f"{type(exc).__name__}: {exc}",
                              error=traceback.format_exc())
            raise
        else:
            ctx.record["status"] = STATUS_COMPLETE
            ctx.record["outputs"] = {
                key: fingerprint(d / rel.rstrip("/"), self.hashes) | {"relpath": rel.rstrip("/")}
                for key, rel in stage.outputs.items()
            }
            for name, p in ctx.record.get("extra_outputs", {}).items():
                ctx.record.setdefault("extra_output_fingerprints", {})[name] = fingerprint(
                    Path(p), self.hashes, self.repo)
        finally:
            ctx.record["finished"] = _now()
            ctx.record["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
            _write_json(d / "stage.json", ctx.record)
            self._event(iteration=it, stage=stage.name, event=ctx.record["status"],
                        message=ctx.record.get("message"), metrics=ctx.record.get("metrics"))
            self.hashes.save()
        return ctx.record["status"]

    # --- reporting -------------------------------------------------------------------------

    def summary(self) -> list[dict]:
        rows = []
        for it in self.iterations():
            for stage in self.stages:
                rec = self.read_record(it, stage.name)
                if rec is None:
                    continue
                rows.append({
                    "iteration": it,
                    "stage": stage.name,
                    "status": rec.get("status"),
                    "finished": rec.get("finished"),
                    "elapsed_seconds": rec.get("elapsed_seconds"),
                    "commit": rec.get("code", {}).get("commit", "")[:10],
                    "dirty": rec.get("code", {}).get("dirty"),
                    "message": rec.get("message"),
                    "metrics": rec.get("metrics", {}),
                })
        return rows

    def write_summary(self) -> Path:
        path = self.root / "summary.json"
        _write_json(path, self.summary())
        return path

    def status(self) -> str:
        lines = []
        for row in self.summary():
            metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in row["metrics"].items())
            extra = f"  ({row['message']})" if row["status"] != STATUS_COMPLETE and row["message"] else ""
            lines.append(f"iter {row['iteration']:3d}  {row['stage']:<8} {row['status']:<9} "
                         f"{metrics}{extra}")
        return "\n".join(lines) if lines else "(nothing run yet)"


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


class _CampaignLock:
    """One driver per campaign at a time. A lock left by a killed process names its host and
    pid; delete ``<campaign>/.lock`` once you are sure that process is gone."""

    def __init__(self, root: Path):
        self.path = root / ".lock"

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"campaign is locked: {self.path} ({self.path.read_text().strip()})") from None
        with os.fdopen(fd, "w") as fh:
            fh.write(f"host={socket.gethostname()} pid={os.getpid()} since={_now()}\n")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False
