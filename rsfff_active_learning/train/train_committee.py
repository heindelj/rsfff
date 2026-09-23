#!/usr/bin/env python3
"""Train a committee of film models: N independent fits of one config, one process each.

    python train_committee.py --out runs/film_committee            # 4 members, one per GPU
    python train_committee.py --out runs/smoke --quick --subset 64 --members 2

Ported from ``qchem_roundtrip/train``. The active-learning train stage
(``rsfff_al.train_stage.CommitteeTrain``) runs exactly this script; run by hand it does the
same thing. The member layout, ``done.json`` per finished member and the ``committee.json``
manifest are what ``rsfff_al.committee.Committee.load`` reads. It needs rsfff (with the
force-stream patch when ``data.force_path`` is set) but not easyal.

    <out>/
        committee.json               members, checkpoints, val losses, seeds, provenance
        member_00/
            config.resolved.yaml     exactly what was fitted -- rerunnable by hand with
                                     python -m rsfff.train.train_film config.resolved.yaml
            train.log
            done.json
            member_00_isolated/best.pt
            member_00_full/best.pt   <- the member's checkpoint (last stage)
        member_01/ ...

Members differ only in ``train.seed`` (the initialization); they share ``data.seed`` and so
one train/holdout split. That is what the loop's committee is, and the only thing that makes
their disagreement mean "the data does not pin this down".

Data paths in the config are resolved relative to the config file, so the job directory
(``train/``) carries its own ``data/`` and runs from anywhere. ``data.path`` may be a glob:
the default config trains on ``data/clusters/**/*.xyz``, every cluster file of every group,
with no distinction between them. ``scripts/check_data.py`` runs on those files first and
refuses anything that is not neutral water at one level of theory. ``--subset N`` keeps the first
N frames of every file (for a smoke test); ``--quick`` cuts every stage to one epoch.

A member with ``done.json`` is never refitted, so rerunning the same command after a wall
clock kill finishes only what is missing.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "water_film_al.yaml"
MANIFEST = "committee.json"
#: data keys whose values are file paths (possibly lists), resolved against the config dir
PATH_KEYS = ("path", "reference_energies", "atomic_reference_states", "isolated_species",
             "diabatic_states", "monomer_path", "large_path", "force_path")


# --- config handling ------------------------------------------------------------------------

def set_dotted(tree, dotted: str, value) -> None:
    """``a.b.0.c = value``; integer parts index into lists (``stages.1.train.epochs``)."""
    node = tree
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


#: keys that name sets of cluster files, where a glob pattern is expanded
GLOB_KEYS = ("path", "large_path", "force_path")


def resolve_paths(tree: dict, base: Path) -> None:
    """Make every data path absolute against ``base``; expand globs in ``data.path``.

    ``data.path: [data/clusters/**/*.xyz]`` is how the default config says "every cluster
    file there is training data": a new group is a new directory, not a config edit. Matches
    are sorted, so the file order -- and with it the frame indexing and the seeded split --
    is the same on every machine. A pattern that matches nothing is an error, not an empty
    dataset.
    """
    data = tree.setdefault("data", {})

    def absolute(value, expand=False):
        if isinstance(value, list):
            return [q for v in value for q in absolute(v, expand)]
        p = Path(os.path.expandvars(str(value)))
        p = p if p.is_absolute() else base / p
        if expand and glob.has_magic(str(p)):
            found = sorted(str(Path(q).resolve()) for q in glob.glob(str(p), recursive=True))
            if not found:
                raise FileNotFoundError(f"data pattern {p} matches no files")
            return found
        return [str(p.resolve())]

    for key in PATH_KEYS:
        value = data.get(key)
        if not value:
            continue
        paths = absolute(value, expand=key in GLOB_KEYS)
        data[key] = paths if (isinstance(value, list) or key in GLOB_KEYS) else paths[0]


def data_files(tree: dict) -> list[Path]:
    data = tree.get("data", {})
    out = []
    for key in PATH_KEYS:
        value = data.get(key)
        if not value:
            continue
        out += [Path(v) for v in (value if isinstance(value, list) else [value])]
    return out


def subset_extxyz(src: Path, dst: Path, n_frames: int) -> int:
    """First ``n_frames`` frames of an extxyz file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(src) as fin, open(dst, "w") as fout:
        while kept < n_frames:
            head = fin.readline()
            if not head.strip():
                break
            n_atoms = int(head)
            fout.write(head)
            for _ in range(n_atoms + 1):
                fout.write(fin.readline())
            kept += 1
    return kept


def apply_subset(tree: dict, n_frames: int, out: Path) -> None:
    data = tree["data"]
    for key in ("path", "monomer_path", "large_path", "force_path"):
        value = data.get(key)
        if not value:
            continue
        paths = value if isinstance(value, list) else [value]
        new = []
        for p in paths:   # parent dir in the name: two groups may both have a w4 file
            dst = out / "data_subset" / f"{Path(p).parent.name}__{Path(p).name}"
            if not dst.exists():
                subset_extxyz(Path(p), dst, n_frames)
            new.append(str(dst))
        data[key] = new if isinstance(value, list) else new[0]


def make_quick(tree: dict) -> None:
    tree.setdefault("train", {})["epochs"] = 1
    tree["train"]["eval_every"] = 1
    for stage in tree.get("stages") or []:
        stage.setdefault("train", {})["epochs"] = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- members --------------------------------------------------------------------------------

def previous_members(path) -> list[Path]:
    """Checkpoints to warm start from: a committee directory (by index) or one ``.pt``.
    Manifest paths from another machine resolve to the members under ``path``."""
    if not path:
        return []
    path = Path(path)
    if path.is_file():
        return [path]
    manifest = path / MANIFEST
    if manifest.exists():
        sys.path.insert(0, str(HERE.parent))
        from rsfff_al.committee import member_checkpoints
        return member_checkpoints(path)
    found = sorted(path.glob("member_*/done.json"))
    if found:
        return [Path(json.loads(f.read_text())["checkpoint"]) for f in found]
    raise FileNotFoundError(f"no committee or checkpoint at {path}")


def checkpoint_of(tree: dict) -> Path:
    """Where ``train_film`` left the member's best checkpoint (the last stage's, if staged)."""
    root = Path(tree["checkpoint_root"])
    run = str(tree["run_name"])
    stages = tree.get("stages") or []
    names = [f"{run}_{stage['name']}" for stage in stages][-1:] + [run]
    for name in names:
        candidate = root / name / "best.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no best.pt under {root} (looked for {', '.join(names)})")


def best_val(log: Path) -> float | None:
    """The last ``done; best val loss X`` line -- the final stage's best validation loss."""
    value = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("done; best val loss"):
            try:
                value = float(line.split("best val loss")[1].split(";")[0])
            except ValueError:
                pass
    return value


def visible_gpus() -> list[str]:
    """GPU ids a member may be pinned to, honouring an existing CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return [g.strip() for g in out.stdout.splitlines() if g.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return [g.strip() for g in out.stdout.splitlines() if g.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="training YAML; its data paths are relative to its own directory")
    ap.add_argument("--out", type=Path, required=True, help="committee directory")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="member k fits with train.seed=seed+k")
    ap.add_argument("--split-seed", type=int, default=0, help="data.seed, shared by all")
    ap.add_argument("--device", default="cuda", help="cuda | cpu | auto (default cuda: fail "
                    "loudly rather than fit silently on CPU)")
    ap.add_argument("--parallel", default="auto",
                    help="members at once: an integer, or auto (one per visible GPU)")
    ap.add_argument("--threads", type=int, default=None,
                    help="OMP/MKL threads per member (default: cores // parallel)")
    ap.add_argument("--init-from", default="",
                    help="committee dir or checkpoint; member k warm starts from member k. "
                         "Applied to the first stage only (later stages chain as usual)")
    ap.add_argument("--subset", type=int, default=0,
                    help="keep the first N frames of every data file (smoke tests)")
    ap.add_argument("--quick", action="store_true", help="one epoch per stage")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="dotted override, value parsed as YAML; "
                    "repeatable (e.g. --set train.batch_size=64 --set stages.1.train.epochs=40)")
    ap.add_argument("--skip-data-check", action="store_true",
                    help="do not run scripts/check_data.py on the cluster files first")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="write configs, train nothing")
    args = ap.parse_args(argv)

    config = args.config.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(config.read_text()) or {}
    resolve_paths(base, config.parent)
    missing = [str(p) for p in data_files(base) if not p.exists()]
    if missing:
        print("missing data files (run scripts/stage_data.sh on the laptop and sync?):\n  "
              + "\n  ".join(missing), file=sys.stderr)
        return 2
    if base["data"].get("large_path"):
        print("[committee] note: data.large_path is set -- that is the separate, separately "
              "weighted large-cluster stream. To treat those clusters like any other, list "
              "them under data.path instead.", flush=True)
    cluster_files = [Path(p) for p in base["data"]["path"]]
    if not args.skip_data_check:
        sys.path.insert(0, str(HERE / "scripts"))
        from check_data import main as check_data
        force_files = [str(p) for p in base["data"].get("force_path") or []]
        if check_data([str(p) for p in cluster_files] + ["--quiet"]) != 0 or (
                force_files and check_data(force_files + ["--quiet", "--force-only"]) != 0):
            print("[committee] the cluster data failed scripts/check_data.py; not training "
                  "(--skip-data-check to override)", file=sys.stderr)
            return 2
    if args.subset:
        apply_subset(base, args.subset, out)
    if args.quick:
        make_quick(base)
    base["device"] = args.device
    base.setdefault("data", {})["seed"] = args.split_seed
    for item in args.overrides:
        key, _, raw = item.partition("=")
        set_dotted(base, key.strip(), yaml.safe_load(raw))

    parents = previous_members(args.init_from)

    # 1. every member's config, written before anything runs
    pending = []
    for k in range(args.members):
        member = out / f"member_{k:02d}"
        member.mkdir(exist_ok=True)
        if (member / "done.json").exists():
            continue
        tree = copy.deepcopy(base)
        tree["run_name"] = f"member_{k:02d}"
        tree["checkpoint_root"] = str(member)
        tree.setdefault("train", {})["seed"] = args.seed + k
        if parents:
            parent = str(parents[min(k, len(parents) - 1)].resolve())
            if tree.get("stages"):
                tree["stages"][0].setdefault("train", {})["init_from"] = parent
            else:
                tree["train"]["init_from"] = parent
        resolved = member / "config.resolved.yaml"
        resolved.write_text(yaml.safe_dump(tree, sort_keys=False))
        pending.append((k, member, resolved, tree))

    gpus = visible_gpus() if args.device != "cpu" else []
    if args.device == "cuda" and not gpus:
        print("--device cuda but no GPU is visible (nvidia-smi found none). On Perlmutter "
              "run inside a -C gpu allocation, or pass --device cpu.", file=sys.stderr)
        return 2
    if args.parallel == "auto":
        lanes = max(1, min(len(gpus) or 1, max(len(pending), 1)))
    else:
        lanes = max(1, int(args.parallel))
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    threads = args.threads or max(1, cores // lanes)

    print(f"[committee] {args.members} member(s), {len(pending)} to train, {lanes} at a time, "
          f"{threads} threads each, device={args.device}, gpus={gpus or '-'}", flush=True)
    print(f"[committee] config {config}\n[committee] out    {out}", flush=True)
    if args.dry_run:
        return 0

    # 2. the members, `lanes` at a time, each pinned to its own GPU
    failed = []
    for start in range(0, len(pending), lanes):
        running = []
        for lane, (k, member, resolved, tree) in enumerate(pending[start:start + lanes]):
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[lane % len(gpus)]
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[var] = str(threads)
            env["PYTHONUNBUFFERED"] = "1"
            log = open(member / "train.log", "w")
            print(f"[committee] member {k:02d} -> "
                  f"{'GPU ' + env['CUDA_VISIBLE_DEVICES'] if gpus else 'CPU'}  "
                  f"({member / 'train.log'})", flush=True)
            running.append((k, member, tree, time.time(), log, subprocess.Popen(
                [args.python, "-m", "rsfff.train.train_film", str(resolved)],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(member))))
        for k, member, tree, t0, log, proc in running:
            code = proc.wait()
            log.close()
            elapsed = round(time.time() - t0, 1)
            if code != 0:
                tail = (member / "train.log").read_text(errors="replace").splitlines()[-15:]
                print(f"[committee] member {k:02d} FAILED (exit {code}, {elapsed}s):\n  "
                      + "\n  ".join(tail), file=sys.stderr, flush=True)
                failed.append(k)
                continue
            ckpt = checkpoint_of(tree)
            (member / "done.json").write_text(json.dumps({
                "member": k, "checkpoint": str(ckpt), "elapsed_seconds": elapsed,
                "train_seed": tree["train"]["seed"],
                "val_loss": best_val(member / "train.log"),
            }, indent=2) + "\n")
            print(f"[committee] member {k:02d} done in {elapsed}s -> {ckpt}", flush=True)

    if failed:
        print(f"[committee] {len(failed)} member(s) failed: {failed}; rerun the same command "
              f"to retry only those", file=sys.stderr)
        return 1

    # 3. the manifest rsfff_al.committee reads
    members = [json.loads((out / f"member_{k:02d}" / "done.json").read_text())
               for k in range(args.members)]
    (out / MANIFEST).write_text(json.dumps({
        "n_members": args.members,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "warm_start": bool(parents),
        "warm_started_from": [str(p) for p in parents],
        "config": str(config),
        "config_sha256": sha256(config),
        "training_data": {str(p): sha256(p) for p in data_files(base)},
        "cluster_files": [str(p) for p in base["data"]["path"]],
        "force_files": [str(p) for p in base["data"].get("force_path") or []],
        "subset": args.subset or None,
        "quick": args.quick,
        "overrides": args.overrides,
        "device": args.device,
        "gpus": gpu_names(),
        "host": platform.node(),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "members": members,
    }, indent=2) + "\n")
    vals = [m["val_loss"] for m in members]
    print(f"[committee] wrote {out / MANIFEST}; member val losses: {vals}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
