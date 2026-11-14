"""Train one cmm model: the ChemLab ``train.cmm`` job, and the same run locally.

    python -m rsfff.train.cmm_worker run WORKDIR                                   # behind ChemLab's worker protocol
    python -m rsfff.train.cmm_worker local --data BUNDLE --out RUN [--set key=value ...]
    python -m rsfff.train.cmm_worker config --data BUNDLE --out RUN [--set ...]    # print the training config

The run is :mod:`rsfff.train.cmm` on a **data bundle** -- a directory, or a tar of it, as ``scripts/cmm_bundle.py``
writes it (ct_sets/, walk_eda/, force/, ct/, params/, force_test.txt) -- with the config the settings below make.

Settings (:data:`DEFAULTS`; the job's ``in/settings``, or ``--set``): latent, pairs ("ww", or "/"-separated groups),
probes, film, epochs, seed, lr, batch, force_batch, w_force, device (auto | cuda | cpu), and ``overrides``: further
``table.key=value`` settings of the training config (e.g. "loss.w_curv=0 train.alpha_every=2").

Outputs (``out/`` of the job; the run directory locally): ``run.tar.gz`` (model.pt, metrics.json, predictions.csv,
hull.pt, train.log, config.json and the species table: :func:`rsfff.ff.cmm.load_cmm` opens it), ``metrics.json``,
``train.log``; the report carries the validation / test errors.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

__all__ = ["DEFAULTS", "make_config", "run_training", "main"]

DEFAULTS = {"latent": 6, "pairs": "ww", "probes": True, "film": True, "epochs": 500, "seed": 0, "lr": 1e-3,
            "batch": 128, "force_batch": 64, "w_force": 0.1, "device": "auto", "overrides": ""}


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _settings(given: dict | None) -> dict:
    s = dict(DEFAULTS)
    for k, v in (given or {}).items():
        if k in s and v not in (None, ""):
            s[k] = v
    for k in ("probes", "film"):
        s[k] = _bool(s[k])
    for k in ("latent", "epochs", "seed", "batch", "force_batch"):
        s[k] = int(s[k])
    for k in ("lr", "w_force"):
        s[k] = float(s[k])
    return s


def make_config(data: Path, out: Path, settings: dict | None = None, *, cache_dir: Path | None = None,
                threads: int = 8, device: str = "cpu") -> dict:
    """The training config (:mod:`rsfff.train.cmm`) of one run on the bundle directory ``data``."""
    s = _settings(settings)
    pairs = [x for x in str(s["pairs"]).replace(",", "/").split("/") if x]
    cfg = {"model": {"latent": s["latent"], "pairs": pairs, "probes": s["probes"], "film": s["film"]},
           "data": {"root": str(Path(data).resolve())},
           "loss": {"w_force": s["w_force"]},
           "train": {"out": str(Path(out).resolve()), "epochs": s["epochs"], "seed": s["seed"], "lr": s["lr"],
                     "batch": s["batch"], "force_batch": s["force_batch"], "device": device,
                     "cache_dir": str(cache_dir or (Path(out).resolve().parent / "cache")),
                     "precompute_workers": threads}}
    for kv in shlex.split(str(s["overrides"] or "")):
        key, val = kv.split("=", 1)
        tbl, k = key.split(".", 1)
        from .cmm import parse_value
        cfg.setdefault(tbl, {})[k] = parse_value(val)
    return cfg


def _bundle(data: Path, scratch: Path) -> Path:
    """The bundle directory: ``data`` itself, or its tar extracted under ``scratch``."""
    data = Path(data)
    if data.is_dir():
        return data
    into = scratch / "data"
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(data) as tf:
        tf.extractall(into, filter="data")
    roots = [p for p in into.iterdir()]
    return roots[0] if len(roots) == 1 and roots[0].is_dir() and not (into / "ct_sets").exists() else into


def _device(want: str) -> str:
    if want not in ("auto", "", None):
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def run_training(data, run_dir, settings=None, *, cache_dir=None, threads=None, log=print) -> dict:
    """One run (see the module docstring). Returns the report."""
    t0 = time.time()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    s = _settings(settings)
    bundle = _bundle(Path(data).resolve(), run_dir.parent / "bundle")
    threads = int(threads or os.environ.get("RSFFF_PRECOMPUTE_WORKERS") or os.environ.get("CHEMLAB_THREADS") or 8)
    threads = max(1, min(threads, 64))
    device = _device(s["device"])
    cache = Path(cache_dir or os.environ.get("RSFFF_CACHE_DIR") or (run_dir.parent / "cache"))
    cache.mkdir(parents=True, exist_ok=True)
    cfg = make_config(bundle, run_dir, s, cache_dir=cache, threads=threads, device=device)
    cfg_path = run_dir / "train_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    (run_dir / "settings.json").write_text(json.dumps({**s, "device_used": device, "threads": threads}, indent=1))
    cmd = [sys.executable, "-m", "rsfff.train.cmm", str(cfg_path)]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(threads))
    env.setdefault("MKL_NUM_THREADS", str(threads))
    logf = run_dir / "train.log"
    log(f"cmm_worker: {device}, {threads} threads, cache {cache}")
    with open(logf, "w") as fh:
        fh.write(" ".join(shlex.quote(c) for c in cmd) + "\n\n")
        fh.flush()
        code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(bundle)).returncode
    if code != 0 or not (run_dir / "metrics.json").exists():
        tail = "".join(logf.read_text().splitlines(True)[-40:])
        raise RuntimeError(f"rsfff.train.cmm exited {code}; tail of {logf}:\n{tail}")
    from .cmm import load_config
    tc, base = load_config(cfg_path)
    params = Path(tc.data.params) if Path(tc.data.params).is_absolute() else base / tc.data.params
    if params.exists():
        shutil.copyfile(params, run_dir / params.name)
    with tarfile.open(run_dir / "run.tar.gz", "w:gz") as tar:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p.name != "run.tar.gz" and not p.name.startswith("cache"):
                tar.add(p, arcname=f"run/{p.name}")
    m = json.loads((run_dir / "metrics.json").read_text())
    met, rows = m.get("metrics", {}), m.get("log", [])
    last = rows[-1] if rows else {}
    pick = lambda split: {k: v for k, v in (met.get(split) or {}).items() if k in ("all", "full", "subset")}
    return {"train.loss": last.get("train_loss"), "train.epochs": s["epochs"],
            "rsfff.cmm.val": pick("val"), "rsfff.cmm.test": pick("test"),
            "rsfff.cmm.force_test": {k: v for k, v in (met.get("force_test") or {}).items() if k != "per_file"},
            "rsfff.cmm.val_curv": last.get("val_curv"),
            "rsfff.cmm.device": device, "rsfff.cmm.wall_s": round(time.time() - t0, 1)}


# --- the worker protocol --------------------------------------------------------------------------

def _job_settings(workdir: Path) -> dict:
    p = workdir / "in" / "settings"
    if not p.exists():
        return {}

    def flat(d):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v)
            else:
                yield k, v

    return dict(flat(json.loads(p.read_text())))


def run_job(workdir) -> int:
    workdir = Path(workdir).resolve()
    out = workdir / "out"
    out.mkdir(exist_ok=True)
    data = workdir / "in" / "data"
    if not data.exists():
        raise ValueError("train.cmm: no data input (the bundle tar of scripts/cmm_bundle.py)")
    scratch = Path(os.environ.get("CHEMLAB_JOB_SCRATCH") or (workdir / "scratch"))
    run_dir = scratch / "run"
    s = _job_settings(workdir)
    report = run_training(data, run_dir, s, threads=s.get("cpus"))
    for name in ("run.tar.gz", "metrics.json", "train.log"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out / name)
    (workdir / "report.json").write_text(json.dumps({"data": report}))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["run"], ["verify"]) and len(argv) == 2:
        try:
            return run_job(argv[1])
        except Exception as exc:  # the protocol: a nonzero exit and the reason on stderr
            import traceback

            traceback.print_exc()
            sys.stderr.write(f"rsfff.train.cmm_worker: {exc}\n")
            return 1
    ap = argparse.ArgumentParser(prog="python -m rsfff.train.cmm_worker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["local", "config"])
    ap.add_argument("--data", type=Path, required=True, help="the data bundle: a directory or its tar")
    ap.add_argument("--out", type=Path, required=True, help="the run directory")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help=f"settings over the defaults {DEFAULTS}")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    s = dict(kv.split("=", 1) for kv in a.set)
    if a.mode == "config":
        dev = _settings(s)["device"]
        print(json.dumps(make_config(a.data, a.out, s, cache_dir=a.cache_dir, threads=a.threads or 8,
                                     device="cpu" if dev == "auto" else dev), indent=1))
        return 0
    report = run_training(a.data, a.out, s, cache_dir=a.cache_dir, threads=a.threads)
    (Path(a.out) / "report.json").write_text(json.dumps({"data": report}, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
