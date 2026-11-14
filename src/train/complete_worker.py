"""Train one complete (ion) model: the ChemLab ``train.complete`` job, and the same run locally.

    python -m rsfff.train.complete_worker run WORKDIR                       # behind ChemLab's worker protocol
    python -m rsfff.train.complete_worker local --data BUNDLE --out RUN [--set key=value ...]
    python -m rsfff.train.complete_worker args  --data BUNDLE --out RUN [--set key=value ...]   # print the command

The run is ``rsfff.train.train_ionct`` with the arguments of the salts runs (research/rsfff train/salts/m37/run.sh,
which this module replaces as the one place they are written) on a **data bundle** -- a directory, or a tar of it,
laid out as ``scripts/complete_bundle.py`` writes it:

    ct_sets/        the folded sets train_ionct reads by name (ct-*.jsonl, seed-<salt>-*.jsonl)
    walk_eda/       the NaCl walk EDAs
    force/          total-energy + force frames (nacl*.xyz)
    ct/             the labelled sets named explicitly: al-nacl-r0, al-nacl-r1, al-nacl-lin, al-nacl-holes,
                    nacl-compressed (.jsonl)
    params/         ions_*.json (the ion table) and wself.json (the bounded field energy's constants)
    force_test.txt  held-out force frames, one '<set>:<name>' per line

Settings (:data:`DEFAULTS`; the job's ``in/settings``, or ``--set``):

    xpol        gram | lam            exchange polarization: the Gram form or the fitted lam / k_perp form
    weights     bool                  gram: the learned magnitudes W1, W2 (--xpol-weights)
    hyperpol    legendre | bounded | bounded-learn | none
    ct_b_pin    acceptor-float | all | donor | donor-scale | donor-shell | none   (default acceptor-float: the cation
                                      acceptor exponents learned within exp(+-0.5) of Fukui, the rest pinned as "all")
    latent      int                   width of the shared latent
    one_latent  bool                  the two-body channels through the shared latent, no two-body branches
    pairs       "ww"                  the channels read (default ww = S^w + T^ww): families ww (plain overlap weights),
                                      ws, ss (dressed), or
                                      exact groups Sw, Ss, Tww, Tws, Tss, e.g. "Ss/Tws"
    probes      bool                  the gated diffuse-probe invariants as trunk inputs
    epochs, seed, device (auto | cuda | cpu), args (extra train_ionct arguments, appended last)

Outputs (``out/`` of the job; the run directory locally): ``run.tar.gz`` (model.pt, metrics.json, predictions.csv,
hull.pt, train.log, the ion table and wself.json, settings.json -- a tarball ``load_complete_model`` opens),
``metrics.json``, ``train.log``; the report carries the validation / test errors.
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

__all__ = ["DEFAULTS", "train_args", "run_training", "main"]

DEFAULTS = {"xpol": "gram", "weights": True, "hyperpol": "legendre", "ct_b_pin": "acceptor-float", "latent": 6, "one_latent": True,
            "pairs": "ww", "probes": True, "epochs": 500, "seed": 0, "device": "auto", "args": "", "lr": 0.001,
            "params": "ions_salts8_cpksb.json", "salts": "lif/kbr/csi"}

POLAR_TEST = ["al-nacl-r1:nacl_polar_002", "al-nacl-r1:nacl_polar_008", "al-nacl-r1:nacl_polar_012",
              "al-nacl-r1:nacl_polar_018", "al-nacl-r1:nacl_polar_022"]
FORCE_TEST = ["nacl11", "al-nacl-r1:nacl_bulk64_005", "al-nacl-r1:nacl_bulk_009", "al-nacl-r1:nacl_bulk_013"]
EXCLUDE = ["nacl_bulk_004", "nacl_bulk_016", "nacl_bulk64_004"] + [f"hole_{k:03d}" for k in range(49, 61)]


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _settings(given: dict | None) -> dict:
    s = dict(DEFAULTS)
    for k, v in (given or {}).items():
        if k in s and v not in (None, ""):
            s[k] = v
        elif k == "args":
            s[k] = v or ""
    for k in ("weights", "one_latent", "probes"):
        s[k] = _bool(s[k])
    for k in ("latent", "epochs", "seed"):
        s[k] = int(s[k])
    s["lr"] = float(s["lr"])
    return s


def train_args(data: Path, out: Path, settings: dict | None = None, *, cache_dir: Path | None = None,
               threads: int = 8, device: str = "cpu") -> list[str]:
    """The ``train_ionct`` argument list of one run on the bundle directory ``data`` (see the module docstring)."""
    s = _settings(settings)
    data, out = Path(data), Path(out)
    D, CT = data / "ct_sets", data / "ct"
    salts = [x for x in str(s["salts"]).split("/") if x]
    pairs = [x for x in str(s["pairs"]).replace(",", "/").split("/") if x]
    ftest = [ln.strip() for ln in (data / "force_test.txt").read_text().split() if ln.strip()]
    fsets = [CT / "al-nacl-r0.jsonl", CT / "al-nacl-r1.jsonl", D / "seed-nacl-cuts.jsonl"] + [D / f"seed-{x}-cuts.jsonl" for x in salts]
    a = ["--ct-sets", D, "--walk-eda", data / "walk_eda", "--salts", *salts,
         "--params", data / "params" / s["params"], "--out", out,
         "--cache-dir", cache_dir or (out.parent / "cache"), "--precompute-workers", threads, "--env", "density",
         "--epochs", s["epochs"], "--seed", s["seed"], "--device", device,
         "--model", "complete", "--ct-form", "direct", "--gap-form", "density", "--tie-c", "--confine", "--xpol",
         "--ct-model", "flux", "--frozen", "--w-pauli", 1, "--w-disp", 1, "--conf-form", "mono", "--pauli-form", "shells",
         "--w-pol", 1, "--w-ct", 1, "--w-chan", 0.3, "--w-alpha", 0.2, "--alpha-every", 4, "--w-smooth", 0.003,
         "--no-polar-almo", "--polar-sets", CT / "al-nacl-r1.jsonl", "--polar-test", *POLAR_TEST,
         "--batch", 128, "--force-batch", 64, "--lr", s["lr"], "--hidden", 64, "--wkb", "smooth+floor", "--ct-max-pair", 3.6,
         "--max-ho-ratio", 1.4, "--env-cap", 3, "--env-sat", 1.2, "--ct-q-cap", 1.5, "--gap-cap", 2,
         "--force-data", data / "force", "--force-sets", *fsets, "--force-test", *FORCE_TEST, *ftest,
         "--w-force", 0.1, "--force-every", 5,
         "--extra-sets", CT / "nacl-compressed.jsonl", CT / "al-nacl-r0.jsonl", CT / "al-nacl-r1.jsonl",
         CT / "al-nacl-lin.jsonl", CT / "al-nacl-holes.jsonl",
         "--exclude-names", *EXCLUDE, "--log-every", 25, "--pin-a0", "--no-aniso",
         # the shared environment
         "--env-channels", "--env-latent", s["latent"], "--env-mode", "shared", "--w-curv", 2e-3,
         "--sr-features", "--env-species", "film", "--tie-disp-range", "--ct-dimers", "like", "--w-env-mid", 0.1,
         "--env-pairs", *pairs, "--ct-b-pin", s["ct_b_pin"]]
    a += ["--env-two-latent"] if s["one_latent"] else ["--env-two-body", "pauli", "disp", "ct"]
    if s["probes"]:
        a += ["--env-ct-gated", "--env-extra-latent"]
    if s["xpol"] == "gram":
        a += ["--xpol-form", "gram"] + (["--xpol-weights"] if s["weights"] else [])
    elif s["xpol"] != "lam":
        raise ValueError(f"xpol {s['xpol']!r}: gram or lam")
    hyp = s["hyperpol"]
    if hyp == "legendre":
        a += ["--hyperpol", "--hyperpol-form", "legendre"]
    elif hyp in ("bounded", "bounded-learn"):
        a += ["--hyperpol", "--hyperpol-form", "bounded", "--wself-params", data / "params" / "wself.json"]
        a += ["--wself-learn"] if hyp == "bounded-learn" else []
    elif hyp != "none":
        raise ValueError(f"hyperpol {hyp!r}: legendre, bounded, bounded-learn or none")
    return [str(x) for x in a] + shlex.split(str(s["args"] or ""))


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
    args = train_args(bundle, run_dir, s, cache_dir=cache, threads=threads, device=device)
    cmd = [sys.executable, "-m", "rsfff.train.train_ionct", *args]
    (run_dir / "settings.json").write_text(json.dumps({**s, "device_used": device, "threads": threads}, indent=1))
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(threads))
    env.setdefault("MKL_NUM_THREADS", str(threads))
    logf = run_dir / "train.log"
    log(f"complete_worker: {device}, {threads} threads, cache {cache}")
    with open(logf, "w") as fh:
        fh.write(" ".join(shlex.quote(c) for c in cmd) + "\n\n")
        fh.flush()
        code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(bundle)).returncode
    if code != 0 or not (run_dir / "metrics.json").exists():
        tail = "".join(logf.read_text().splitlines(True)[-40:])
        raise RuntimeError(f"train_ionct exited {code}; tail of {logf}:\n{tail}")
    for name in (s["params"], "wself.json"):
        if (bundle / "params" / name).exists():
            shutil.copyfile(bundle / "params" / name, run_dir / name)
    with tarfile.open(run_dir / "run.tar.gz", "w:gz") as tar:
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p.name != "run.tar.gz":
                tar.add(p, arcname=f"run/{p.name}")
    m = json.loads((run_dir / "metrics.json").read_text())
    met, rows = m.get("metrics", {}), m.get("log", [])
    last = rows[-1] if rows else {}
    pick = lambda split: {k: v for k, v in (met.get(split) or {}).items() if k in ("all", "full", "subset")}
    return {"train.loss": last.get("train_loss"), "train.epochs": s["epochs"],
            "rsfff.complete.val": pick("val"), "rsfff.complete.test": pick("test"),
            "rsfff.complete.force_test": {k: v for k, v in (met.get("force_test") or {}).items() if k != "per_file"},
            "rsfff.complete.val_curv": last.get("val_curv"),
            "rsfff.complete.device": device, "rsfff.complete.wall_s": round(time.time() - t0, 1)}


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
        raise ValueError("train.complete: no data input (the bundle tar of scripts/complete_bundle.py)")
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
            sys.stderr.write(f"rsfff.train.complete_worker: {exc}\n")
            return 1
    ap = argparse.ArgumentParser(prog="python -m rsfff.train.complete_worker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["local", "args"])
    ap.add_argument("--data", type=Path, required=True, help="the data bundle: a directory or its tar")
    ap.add_argument("--out", type=Path, required=True, help="the run directory")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help=f"settings over the defaults {DEFAULTS}")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    s = dict(kv.split("=", 1) for kv in a.set)
    if a.mode == "args":
        dev = _settings(s)["device"]
        print(" ".join(shlex.quote(x) for x in train_args(a.data, a.out, s, cache_dir=a.cache_dir, threads=a.threads or 8,
                                                          device="cpu" if dev == "auto" else dev)))
        return 0
    report = run_training(a.data, a.out, s, cache_dir=a.cache_dir, threads=a.threads)
    (Path(a.out) / "report.json").write_text(json.dumps({"data": report}, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
