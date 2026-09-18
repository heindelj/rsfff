"""Check everything the loop needs, before spending an allocation finding out.

    python active_learning/scripts/preflight.py \
        --checkpoint checkpoints/water_film_full/best.pt --root $SCRATCH/water_al

Every check is independent and none of them queue anything, so this is ~30 seconds and says
which piece is missing rather than failing three stages in. It exits non-zero if any required
check fails; warnings (a missing ``qchem`` on a login node, say) do not.

The checks are in the order the loop would hit them: the environment, then the model, then the
stages, then the Q-Chem pool. ``--quick`` skips the two that actually compute (packing a dimer
with packmol, and one energy+gradient from the checkpoint).
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get("RSFFF_REPO") or AL_ROOT.parent).resolve()
if str(AL_ROOT) not in sys.path:
    sys.path.insert(0, str(AL_ROOT))

#: torch-cluster ships only an sdist whose setup.py imports torch, so pip cannot build it in
#: an isolated environment -- it has to be built against the torch already installed here.
#: Optional: without it rsfff.neighbors uses its own radius_graph, which gives the same graph.
TORCH_CLUSTER_REMEDY = """
not required -- rsfff.neighbors falls back to radius_graph_torch, same graph, O(N^2) per
graph, indistinguishable at cluster sizes and slower for a large periodic box. To install the
compiled one, on a LOGIN node (compute nodes have no network; it compiles, ~10 min):
    FORCE_ONLY_CPU=1 MAX_JOBS=8 pip install --no-build-isolation torch-cluster
or in the same command as the package:
    FORCE_ONLY_CPU=1 pip install --no-build-isolation -e '.[neighbors]'
for a GPU build instead: FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 (A100), CUDA toolkit loaded
"""

OK, WARN, FAIL = "ok", "warn", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> str:
    _results.append((name, status, detail))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name:<28} {detail}", flush=True)
    return status


def check(name: str, fn, *, required: bool = True, remedy: str = "") -> str:
    """Run ``fn``; it returns a detail string, or raises. ``remedy`` is printed on failure."""
    try:
        detail = fn() or ""
    except Exception as exc:                      # a failing check is a result, not a crash
        status = record(name, FAIL if required else WARN, f"{type(exc).__name__}: {exc}")
        if remedy:
            for line in remedy.strip().splitlines():
                print(f"         {line}")
        return status
    return record(name, OK, detail)


# --- the checks ------------------------------------------------------------------------------

def _python() -> str:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"python {sys.version.split()[0]} is below the required 3.10")
    return f"{sys.version.split()[0]} at {sys.executable}"


def _module(name: str, attr: str = "__version__"):
    def run() -> str:
        module = importlib.import_module(name)
        return f"{getattr(module, attr, '?')}  ({getattr(module, '__file__', '?')})"
    return run


def _easyal_takes_several_samplers() -> str:
    import inspect

    from easyal import ActiveLearning
    source = inspect.getsource(ActiveLearning.__init__)
    if "samples" not in source:
        raise RuntimeError(
            "this easyal predates sample=[...]; a loop with an optimize and a dynamics stage "
            "cannot be built. Update the easyAL checkout and reinstall it."
        )
    return "sample=[...] supported"


def _rsfff_is_complete() -> str:
    import rsfff  # noqa: F401
    for name in ("rsfff.ff.film.model", "rsfff.md.film_driver", "rsfff.train.build_film"):
        importlib.import_module(name)
    backend = importlib.import_module("rsfff.neighbors").BACKEND
    return f"{Path(importlib.import_module('rsfff').__file__).parent}  (neighbors: {backend})"


def _packmol(quick: bool) -> str:
    exe = shutil.which("packmol")
    if exe is None:
        raise RuntimeError("not on PATH: conda install -c conda-forge packmol, or pip install packmol")
    if quick:
        return f"{exe} (not exercised, --quick)"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "water.xyz").write_text(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.9572 0.0 0.0\nH -0.24 0.9266 0.0\n"
        )
        (d / "p.inp").write_text(
            "tolerance 2.0\nfiletype xyz\noutput out.xyz\nseed 1\n\n"
            "structure water.xyz\n  number 2\n  inside sphere 0. 0. 0. 4.\nend structure\n"
        )
        with open(d / "p.inp") as fh:
            proc = subprocess.run([exe], stdin=fh, cwd=d, capture_output=True, text=True)
        if proc.returncode != 0 or not (d / "out.xyz").exists():
            raise RuntimeError(f"packmol failed:\n{(proc.stdout + proc.stderr)[-400:]}")
    return f"{exe}, packed a dimer"


def _checkpoint(path: Path, quick: bool) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    from model import LoadedFilmModel  # noqa: E402  (needs AL_ROOT on sys.path)
    t0 = time.perf_counter()
    model = LoadedFilmModel(path)
    load = time.perf_counter() - t0
    described = model.describe()
    if quick:
        return f"{described['n_parameters']} parameters, loaded in {load:.1f}s"
    import numpy as np
    species = ["O", "H", "H", "O", "H", "H"]
    positions = np.array([
        [0.000, 0.000, 0.000], [0.957, 0.000, 0.000], [-0.240, 0.927, 0.000],
        [2.900, 0.100, 0.000], [3.200, 0.960, 0.300], [3.260, -0.600, 0.480],
    ])
    potential, _ = model.potential(species, positions)
    t0 = time.perf_counter()
    energy, gradient = potential.energy_and_gradient(positions)
    step = time.perf_counter() - t0
    import math
    if not math.isfinite(float(energy[0])):
        raise RuntimeError("the model returned a non-finite energy for a water dimer")
    return (f"{described['n_parameters']} parameters, loaded in {load:.1f}s; dimer "
            f"E = {float(energy[0]):.6f} Ha, |F|max = {abs(gradient).max():.2e}, "
            f"one energy+gradient in {step * 1000:.0f} ms")


def _stages() -> str:
    for name in ("common", "build", "model", "sampling", "label_stage", "workflows"):
        importlib.import_module(name)
    return "build, optimize, dynamics, label import"


def _qcgen_parsers() -> str:
    import common
    common._install_qcgen()
    from qcgen.qchem_eda import parse_eda_output  # noqa: F401
    from qcgen.qchem_force import parse_force_output  # noqa: F401
    return f"loaded from {common._qcgen_dir()} (no pyscf needed)"


def _roundtrip_parser() -> str:
    import common
    parser = common.roundtrip_parser()
    if not hasattr(parser, "build_merged_frame"):
        raise RuntimeError("parse_roundtrip.py has no build_merged_frame")
    return f"{parser.__file__}"


def _pool(root: Path, calculations) -> str:
    import common
    config = root / "config.json"
    if not config.exists():
        raise FileNotFoundError(
            f"{config} does not exist; point --roundtrip-root (or $RSFFF_QCHEM_ROOT) at the "
            f"qchem_roundtrip directory"
        )
    cfg = common.qchem_roundtrip.load_config(config)
    missing = [c for c in calculations if c not in cfg["calculations"]]
    if missing:
        raise RuntimeError(f"config.json has no {missing}; it has {sorted(cfg['calculations'])}")
    for name in calculations:
        template = common.qchem_roundtrip.resolve_config_path(
            config, cfg["calculations"][name]["template"]
        )
        if not template.exists():
            raise FileNotFoundError(f"{name}: template {template} is missing")
    common.qchem_roundtrip.ensure_layout(root, cfg)
    probe = root / list(calculations)[0] / "geoms" / ".preflight"
    probe.write_text("")
    probe.unlink()
    return f"{root} ({', '.join(calculations)}), layout writable"


def _writable(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".preflight"
    probe.write_text("")
    probe.unlink()
    return str(path)


def _command(name: str, hint: str):
    def run() -> str:
        exe = shutil.which(name)
        if exe is None:
            raise RuntimeError(f"not on PATH -- {hint}")
        return exe
    return run


# --- main -------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints" / "water_film_full" / "best.pt")
    p.add_argument("--root", type=Path, default=None, help="where the loop will live")
    p.add_argument("--roundtrip-root", type=Path, default=None,
                   help="the Q-Chem job pool (default $RSFFF_QCHEM_ROOT or <repo>/qchem_roundtrip)")
    p.add_argument("--calculations", nargs="+", default=("eda", "force"))
    p.add_argument("--quick", action="store_true",
                   help="skip the checks that compute (packmol, one model gradient)")
    args = p.parse_args(argv)

    pool = args.roundtrip_root or Path(
        os.environ.get("RSFFF_QCHEM_ROOT") or REPO_ROOT / "qchem_roundtrip")

    print(f"rsfff active-learning preflight\n  repo   {REPO_ROOT}\n  pool   {pool}\n"
          f"  host   {os.uname().nodename}\n")

    check("python", _python)
    check("numpy", _module("numpy"))
    check("scipy", _module("scipy"))
    check("ase", _module("ase"))
    check("torch", _module("torch"))
    check("torch_cluster", _module("torch_cluster", "__file__"), required=False,
          remedy=TORCH_CLUSTER_REMEDY)
    check("e3nn", _module("e3nn"))
    check("easyal", _module("easyal"))
    check("easyal sample=[...]", _easyal_takes_several_samplers)
    check("rsfff (incl. ff.film)", _rsfff_is_complete)
    check("active_learning stages", _stages)
    check("packmol", lambda: _packmol(args.quick))
    check("checkpoint", lambda: _checkpoint(args.checkpoint, args.quick))
    check("qcgen parsers", _qcgen_parsers)
    check("parse_roundtrip", _roundtrip_parser)
    check("q-chem pool", lambda: _pool(pool, tuple(args.calculations)))
    if args.root is not None:
        check("loop root writable", lambda: _writable(args.root))
    check("qchem", _command("qchem", "module load qchem (only the workers need it)"),
          required=False)
    check("sbatch", _command("sbatch", "only needed to queue workers"), required=False)

    failed = [n for n, s, _ in _results if s == FAIL]
    warned = [n for n, s, _ in _results if s == WARN]
    print()
    if failed:
        print(f"{len(failed)} required check(s) failed: {', '.join(failed)}")
        return 1
    print("all required checks passed" + (f"; {len(warned)} warning(s): {', '.join(warned)}"
                                          if warned else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
