"""Train and evaluate one bonding model: the ChemLab ``train.bonding`` job, and the same run locally.

    python -m rsfff.train.bonding_worker run WORKDIR             # behind ChemLab's worker protocol
    python -m rsfff.train.bonding_worker local --config train.yaml --vacuum vacuum.xyz \\
        [--probe probe.xyz] --references atoms.json --out runs/name [--epochs N] [--device auto]

One run (:func:`run_training`):

1. **Split.** With ``holdout_stretch > 0`` the ``scan_stretch`` frames beyond ``r / r_e > holdout_stretch``
   are held out of training (:func:`rsfff.train.bonding_eval.split_stretch`) -- the extrapolation test;
   the rest is the training file, of which ``data.holdout_fraction`` is the validation split as always.
2. **Config.** The ``train_bonding`` YAML with its data paths pointed at this run's files:
   ``data.path`` the training file, ``data.reference_energies`` the atom references, the ``probe``
   stream (``bonding.streams``) the probe file (added when the config has none, dropped when there is
   no probe file; other streams whose files are not there are dropped too), ``checkpoint_root`` /
   ``run_name`` this run's directory; ``epochs`` / ``seed`` / ``device`` override the config's when set.
3. **Train**: ``python -m rsfff.train.train_bonding`` in a subprocess, its log kept.
4. **Evaluate** the best checkpoint (:mod:`rsfff.train.bonding_eval`): the validation split, the
   held-out frames and the probe validation split (interaction energies) -> ``eval.json`` /
   ``summary.txt``; a parity figure of every quantity (train / val / held-out) -> ``parity.png``,
   ``parity_data.npz``.
5. **Pack** ``model.tar.gz``: the checkpoint, the config as run, the training log.

The worker reads its job from the workdir (``in/config`` YAML, ``in/vacuum`` and optional
``in/probe`` extxyz, ``in/references`` JSON, ``in/settings``), runs in the job's scratch, writes the
files above to ``out/`` and the metrics to ``report.json`` (``train.*`` and
``rsfff.bonding.<split>.<metric>``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

__all__ = ["run_training", "write_config", "main"]

#: report metrics of each evaluated split (the ``all`` row of :func:`bonding_eval.summarize`)
METRICS = ("e_mae", "f_mae", "h_rel", "mu", "theta", "alpha", "alpha_iso_rel", "de_int", "df_int")


def write_config(config: Path, out: Path, *, train: Path, references: Path, probe: Path | None,
                 run_dir: Path, epochs: int = 0, seed: int | None = None, device: str | None = None,
                 log=print) -> dict:
    """The run's YAML (see the module docstring). Returns the config as written."""
    import yaml

    cfg = yaml.safe_load(Path(config).read_text()) or {}
    cfg["run_name"] = "model"
    cfg["checkpoint_root"] = str(run_dir / "ckpt")
    if device:
        cfg["device"] = device
    data = cfg.setdefault("data", {})
    data["path"] = [str(train)]
    data["reference_energies"] = str(references)
    bonding = cfg.setdefault("bonding", {})
    streams = []
    batch = int((cfg.get("train") or {}).get("batch_size", 16))
    for st in bonding.get("streams") or []:
        if st.get("name") == "probe":
            if probe is not None:
                streams.append({**st, "path": str(probe)})
            continue
        paths = st.get("path") if isinstance(st.get("path"), list) else [st.get("path")]
        if all(p and Path(p).exists() for p in paths):
            streams.append(st)
        else:
            log(f"bonding_worker: dropping stream {st.get('name')!r} (no file at {paths})")
    if probe is not None and not any(s.get("name") == "probe" for s in streams):
        streams.append({"name": "probe", "path": str(probe), "weight": 1.0, "batch_size": batch})
    bonding["streams"] = streams
    bonding.pop("probe_path", None)
    tr = cfg.setdefault("train", {})
    if epochs and int(epochs) > 0:
        tr["epochs"] = int(epochs)
    if seed is not None:
        tr["seed"] = int(seed)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def _best_val(log_text: str) -> float | None:
    for line in reversed(log_text.splitlines()):
        if line.startswith("done; best val loss"):
            try:
                return float(line.split("best val loss", 1)[1].split(";")[0])
            except ValueError:
                return None
    return None


def run_training(config, vacuum, references, outdir, *, probe=None, epochs: int = 0, seed: int | None = None,
                 holdout_stretch: float = 1.3, device: str | None = None, eval_device: str | None = None,
                 parity: bool = True, parity_frames: int = 150, log=print) -> dict:
    """Split, train, evaluate and pack one model into ``outdir``; returns the report dict."""
    import torch

    from .bonding_eval import (evaluate, headers, load_model, print_summary, split_frames, split_stretch,
                               summarize)

    t0 = time.time()
    outdir = Path(outdir).resolve()
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    vacuum, references = Path(vacuum).resolve(), Path(references).resolve()
    probe = Path(probe).resolve() if probe else None

    # 1. split
    train_xyz, held_xyz = data_dir / "train.xyz", data_dir / "heldout.xyz"
    if holdout_stretch and holdout_stretch > 0:
        n_train, n_held = split_stretch(vacuum, train_xyz, held_xyz, float(holdout_stretch))
    else:
        shutil.copyfile(vacuum, train_xyz)
        n_train, n_held = len(headers(vacuum)), 0
    log(f"bonding_worker: {n_train} training frames, {n_held} held out (r/r_e > {holdout_stretch})")

    # 2. config
    cfg_path = outdir / "config.yaml"
    cfg = write_config(Path(config), cfg_path, train=train_xyz, references=references, probe=probe,
                       run_dir=outdir, epochs=epochs, seed=seed, device=device, log=log)

    # 3. train
    log_path = outdir / "train.log"
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_path, "w") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "rsfff.train.train_bonding", str(cfg_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if not line.lstrip().startswith(("warn", "Warn")) and "UserWarning" not in line:
                sys.stdout.write(line)
        code = proc.wait()
    if code != 0:
        tail = "".join(log_path.read_text().splitlines(True)[-40:])
        raise RuntimeError(f"train_bonding exited {code}; tail of {log_path}:\n{tail}")
    best = outdir / "ckpt" / "model" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"no checkpoint at {best} (did any validation pass run?)")
    report = {"train.model": "rsfff_bonding", "train.loss_val": _best_val(log_path.read_text()),
              "train.epochs": int((cfg.get("train") or {}).get("epochs", 0)),
              "train.size": {"train": n_train, "heldout": n_held},
              "rsfff.bonding.holdout_stretch": float(holdout_stretch or 0.0),
              "rsfff.bonding.train_s": round(time.time() - t0, 1)}

    # 4. evaluate
    if eval_device is None:
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(best, device=eval_device)
    ckcfg = ck["config"]
    summaries = {}
    summaries["val"] = summarize(evaluate(model, train_xyz, idx=split_frames(train_xyz, "val", ckcfg)))
    if n_held:
        summaries["heldout"] = summarize(evaluate(model, held_xyz))
    vac_files = [train_xyz] + ([held_xyz] if n_held else [])
    if probe is not None:
        summaries["probe_val"] = summarize(evaluate(model, probe, idx=split_frames(probe, "val", ckcfg),
                                                    hessian=False, vacuum=vac_files))
        report["train.size"]["probe"] = len(split_frames(probe, "all", ckcfg))
    (outdir / "eval.json").write_text(json.dumps(summaries, indent=1))
    with open(outdir / "summary.txt", "w") as fh:
        for split, s in summaries.items():
            print(f"== {split}", file=fh)
            print_summary(s, file=fh)
            print(file=fh)
    for split, s in summaries.items():
        for m in METRICS:
            if m in s["all"]:
                report[f"rsfff.bonding.{split}.{m}"] = round(float(s["all"][m]), 5)
    log((outdir / "summary.txt").read_text())

    if parity:
        try:
            _parity(model, ckcfg, config, outdir, train_xyz, held_xyz, n_held, probe, vac_files, holdout_stretch,
                    parity_frames)
        except Exception as exc:          # a missing matplotlib must not lose a finished training
            log(f"bonding_worker: no parity figure ({type(exc).__name__}: {exc})")
            report["rsfff.bonding.parity_error"] = f"{type(exc).__name__}: {exc}"

    # 5. pack
    with tarfile.open(outdir / "model.tar.gz", "w:gz") as tar:
        tar.add(best, arcname="best.pt")
        tar.add(cfg_path, arcname="config.yaml")
        tar.add(log_path, arcname="train.log")
    report["rsfff.bonding.wall_s"] = round(time.time() - t0, 1)
    report["rsfff.bonding.device"] = str(ckcfg.device)
    return report


def _parity(model, ckcfg, config, outdir, train_xyz, held_xyz, n_held, probe, vac_files, holdout_stretch,
            parity_frames):
    from .bonding_eval import collect, minimum_energies, plot_parity, save_parity, split_frames

    minima = minimum_energies(model, vac_files)
    sets = {"train": collect(model, train_xyz, idx=split_frames(train_xyz, "train", ckcfg, parity_frames),
                             minima=minima),
            "val": collect(model, train_xyz, idx=split_frames(train_xyz, "val", ckcfg), minima=minima)}
    if n_held:
        sets["held-out"] = collect(model, held_xyz, minima=minima)
    if probe is not None:
        for lab, split in (("train", "train"), ("val", "val")):
            sets[lab].update(collect(model, probe, idx=split_frames(probe, split, ckcfg, parity_frames),
                                     hessian=False, moments=False, vacuum=vac_files))
    save_parity(sets, outdir / "parity_data.npz")
    plot_parity(sets, outdir / "parity.png",
                title=f"{Path(config).stem}" + (f" (held-out: stretch beyond {holdout_stretch} r_e)" if n_held else ""))


# --- the worker protocol --------------------------------------------------------------------------

def _settings(workdir: Path) -> dict:
    p = workdir / "in" / "settings"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())

    def flat(d):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v)
            else:
                yield k, v

    return dict(flat(raw))


def run_job(workdir) -> int:
    workdir = Path(workdir).resolve()
    out = workdir / "out"
    out.mkdir(exist_ok=True)
    s = _settings(workdir)

    def inp(role):
        p = workdir / "in" / role
        return p if p.exists() else None

    config, vacuum, references = inp("config"), inp("vacuum"), inp("references")
    for role, p in (("config", config), ("vacuum", vacuum), ("references", references)):
        if p is None:
            raise ValueError(f"train.bonding: no {role} input")
    scratch = Path(os.environ.get("CHEMLAB_JOB_SCRATCH") or (workdir / "scratch"))
    run_dir = scratch / "run"
    device = str(s.get("device") or "auto")
    report = run_training(
        config, vacuum, references, run_dir, probe=inp("probe"),
        epochs=int(s.get("epochs") or 0), seed=None if s.get("seed") in (None, "") else int(s["seed"]),
        holdout_stretch=float(s.get("holdout_stretch") or 0.0), device=device,
        parity=bool(s.get("parity", True)), parity_frames=int(s.get("parity_frames") or 150),
    )
    # out/ files by role (the manager names an output by its file's stem): model, eval, summary, parity,
    # parity_data, train (the log), config (as run)
    for name in ("model.tar.gz", "eval.json", "summary.txt", "parity.png", "parity_data.npz", "train.log", "config.yaml"):
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
            sys.stderr.write(f"rsfff.train.bonding_worker: {exc}\n")
            return 1
    ap = argparse.ArgumentParser(prog="python -m rsfff.train.bonding_worker local", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["local"])
    ap.add_argument("--config", type=Path, required=True, help="the train_bonding YAML")
    ap.add_argument("--vacuum", type=Path, required=True, help="the vacuum frames (fff_bonding export)")
    ap.add_argument("--probe", type=Path, default=None, help="the probe-charge frames")
    ap.add_argument("--references", type=Path, required=True, help='atom energies {"energies": {...}}')
    ap.add_argument("--out", type=Path, required=True, help="the run directory")
    ap.add_argument("--epochs", type=int, default=0, help="override the config's (0: keep)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--holdout-stretch", type=float, default=1.3, help="hold out stretches beyond r/r_e (0: none)")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda (default: the config's)")
    ap.add_argument("--no-parity", action="store_true")
    a = ap.parse_args(argv)
    report = run_training(a.config, a.vacuum, a.references, a.out, probe=a.probe, epochs=a.epochs, seed=a.seed,
                          holdout_stretch=a.holdout_stretch, device=a.device, parity=not a.no_parity)
    (Path(a.out) / "report.json").write_text(json.dumps({"data": report}, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
