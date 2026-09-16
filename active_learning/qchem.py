"""Q-Chem labeling through the existing ``qchem_roundtrip`` bundle.

The label stage does not run Q-Chem itself. It writes nested job directories into the
round-trip tree, where the Perlmutter workers already look (any directory under a calculation
root that contains ``inputs/`` is a job dir)::

    qchem_roundtrip/eda/al_<campaign>/iter_000/{geoms,inputs,outputs,state}
    qchem_roundtrip/force/al_<campaign>/iter_000/{geoms,inputs,outputs,state}

so the usual loop applies unchanged::

    bash qchem_roundtrip/scripts/sync_inputs_up.sh      # laptop -> Perlmutter
    bash qchem_roundtrip/scripts/submit_workers.sh ...  # on Perlmutter
    bash qchem_roundtrip/scripts/sync_outputs_down.sh   # Perlmutter -> laptop

Inputs are rendered from ``qchem_roundtrip/templates/{eda,force}.in`` with the molecule mode of
``qchem_roundtrip/config.json`` -- the same ``$rem`` (wB97M-V/def2-TZVPD, SCF_CONVERGENCE 8,
THRESH 14, ...) as every other job in the corpus, read from the templates rather than copied,
so a template change reaches the campaign and is recorded in its provenance.

Outputs are merged into training extxyz by ``scripts/parse_roundtrip.py``, the same parser that
built ``data/wb97mv_tzvpd``.
"""

from __future__ import annotations

import filecmp
import importlib.util
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .core import REPO_ROOT, StageContext, StagePending

ROUNDTRIP_ROOT = REPO_ROOT / "qchem_roundtrip"
PARSE_SCRIPT = REPO_ROOT / "scripts" / "parse_roundtrip.py"
JOB_KINDS = ("eda", "force")

#: Q-Chem's last line on a normal exit. EDA prints it once per fragment job as well, so a job
#: counts as finished only when it appears in the tail of the file.
_QCHEM_DONE = "Thank you very much for using Q-Chem"


@lru_cache(maxsize=1)
def roundtrip_module():
    """``qchem_roundtrip/scripts/qchem_roundtrip.py`` imported by path (it is stdlib-only)."""
    path = ROUNDTRIP_ROOT / "scripts" / "qchem_roundtrip.py"
    spec = importlib.util.spec_from_file_location("qchem_roundtrip", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("qchem_roundtrip", module)
    spec.loader.exec_module(module)
    return module


def read_frames(path: Path):
    """Frames of an extxyz file as ``qchem_roundtrip.XYZFrame`` (symbols, coords, info dict,
    fragment_idx). No ASE needed; the file must carry a ``Properties=`` header."""
    return roundtrip_module().read_xyz_frames(Path(path))


def check_frames(path: Path, *, fragments: bool) -> int:
    """Raise if any frame of ``path`` lacks what the Q-Chem generator needs. Returns the frame
    count. ``fragments=True`` also requires the EDA fragment definition and builds the
    fragmented ``$molecule`` block as a dry run, which catches non-contiguous fragment indices
    and charge/multiplicity lists of the wrong length."""
    rt = roundtrip_module()
    frames = read_frames(path)
    mode = {"mode": "fragments" if fragments else "plain"}
    for frame in frames:
        rt.build_molecule_block(frame, mode, Path(path))
    return len(frames)


def _calc_config(kind: str, roundtrip_root: Path) -> tuple[Path, dict]:
    rt = roundtrip_module()
    cfg_path = roundtrip_root / "config.json"
    cfg = rt.load_config(cfg_path)
    try:
        calc = cfg["calculations"][kind]
    except KeyError:
        raise KeyError(f"{cfg_path} defines no calculation {kind!r}") from None
    template = rt.resolve_config_path(cfg_path, calc["template"])
    return template, dict(calc.get("molecule", {}))


def job_dir(kind: str, job_name: str, roundtrip_root: Path = ROUNDTRIP_ROOT) -> Path:
    return roundtrip_root / kind / job_name


def write_jobs(
    geometry: Path,
    job_name: str,
    *,
    stem: str,
    kinds: Iterable[str] = JOB_KINDS,
    roundtrip_root: Path = ROUNDTRIP_ROOT,
    ctx: StageContext | None = None,
) -> dict[str, Path]:
    """Render one Q-Chem input per frame of ``geometry`` for each job kind.

    Idempotent: inputs that already exist are left alone, so this is safe to call on every
    resume of a pending label stage. If a *different* geometry file was already staged under
    the same job name it refuses rather than mixing two selections into one bundle.

    ``stem`` names the files: ``geoms/<stem>.extxyz`` and ``inputs/<stem>_frameNNNN.in`` (the
    generator's own convention), which is what ``parse_roundtrip.py`` groups on. The dataset it
    writes is named after the stem, so make the stem unique per campaign iteration.

    With ``ctx`` the templates and config are recorded as inputs and the job dirs as outputs.
    """
    rt = roundtrip_module()
    geometry = Path(geometry)
    frames = rt.read_xyz_frames(geometry)
    dirs: dict[str, Path] = {}
    for kind in kinds:
        template_path, molecule_cfg = _calc_config(kind, roundtrip_root)
        template = template_path.read_text()
        jd = job_dir(kind, job_name, roundtrip_root)
        rt.ensure_job_layout(jd)
        (jd / "geoms").mkdir(exist_ok=True)
        staged = jd / "geoms" / f"{stem}.extxyz"
        if staged.exists() and not filecmp.cmp(staged, geometry, shallow=False):
            raise FileExistsError(
                f"{staged} already holds a different selection; use a new job name or remove it"
            )
        if not staged.exists():
            shutil.copyfile(geometry, staged)
        n_new = 0
        for frame in frames:
            name = rt.input_stem_for_frame(Path(f"{stem}.extxyz"), 2, frame.index)  # always _frameNNNN
            input_path = jd / "inputs" / f"{name}.in"
            if input_path.exists():
                continue
            molecule = rt.build_molecule_block(frame, molecule_cfg, staged)
            rt.atomic_write(input_path, rt.replace_molecule_block(template, molecule))
            rt.write_json_atomic(jd / "state" / "generated" / f"{name}.json", {
                "calculation": kind,
                "geometry": str(staged),
                "input": str(input_path),
                "frame_index": frame.index,
                "frame_metadata": rt.frame_metadata_summary(frame),
                "generated_at": time.time(),
                "generated_by": "active_learning.qchem.write_jobs",
            })
            n_new += 1
        dirs[kind] = jd
        if ctx is not None:
            ctx.add_input(f"qchem_template.{kind}", template_path)
            ctx.add_output(f"qchem_jobs.{kind}", jd)
            if n_new:
                ctx.note(f"wrote {n_new} {kind} input(s) to {jd}")
    if ctx is not None:
        ctx.add_input("qchem_config", roundtrip_root / "config.json")
    return dirs


def _finished(path: Path) -> bool:
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 8192))
        return _QCHEM_DONE.encode() in fh.read()


def job_status(job_dirs: dict[str, Path]) -> dict[str, dict]:
    """Per kind: how many inputs have a finished output, an unfinished one, or none yet.

    Only ``outputs/`` is synced back from Perlmutter (not ``state/``), so a finished job is
    recognized by Q-Chem's closing line. An output without it is either still running or
    crashed; the two are indistinguishable from here, and both are reported as ``unfinished``.
    """
    out = {}
    for kind, jd in job_dirs.items():
        stems = sorted(p.stem for p in (jd / "inputs").glob("*.in"))
        done, unfinished, missing = [], [], []
        for s in stems:
            o = jd / "outputs" / f"{s}.out"
            if not o.exists():
                missing.append(s)
            elif _finished(o):
                done.append(s)
            else:
                unfinished.append(s)
        out[kind] = {"n_inputs": len(stems), "n_done": len(done),
                     "unfinished": unfinished, "missing": missing}
    return out


def wait_for_jobs(job_dirs: dict[str, Path], *, allow_unfinished: bool = False,
                  ctx: StageContext | None = None) -> dict[str, dict]:
    """Raise ``StagePending`` until every input has an output.

    ``allow_unfinished=True`` accepts outputs without Q-Chem's closing line (crashed jobs); the
    parser drops those frames. Use it once you have checked on Perlmutter that nothing is still
    running.
    """
    status = job_status(job_dirs)
    if ctx is not None:
        ctx.log_metrics(**{f"qchem_{k}_done": v["n_done"] for k, v in status.items()},
                        **{f"qchem_{k}_inputs": v["n_inputs"] for k, v in status.items()})
    waiting = {k: len(v["missing"]) + (0 if allow_unfinished else len(v["unfinished"]))
               for k, v in status.items()}
    if any(waiting.values()):
        detail = ", ".join(f"{k}: {v['n_done']}/{v['n_inputs']} done"
                           + (f", {len(v['unfinished'])} unfinished" if v["unfinished"] else "")
                           for k, v in status.items())
        raise StagePending(
            f"waiting on Q-Chem ({detail}). Sync inputs up, run workers, sync outputs down, "
            f"then run the campaign again."
        )
    return status


def parse_jobs(job_dirs: dict[str, Path], out_dir: Path, *, stem: str | None = None,
               roundtrip_root: Path = ROUNDTRIP_ROOT, log_path: Path | None = None,
               strict: bool = False, ctx: StageContext | None = None) -> list[Path]:
    """Merge eda+force outputs into training extxyz in ``out_dir`` with
    ``scripts/parse_roundtrip.py``. Returns the files written."""
    if "force" not in job_dirs:
        raise ValueError("parse_roundtrip.py needs force outputs; EDA alone is not parsed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime_ns for p in out_dir.glob("*.xyz")}
    cmd = [sys.executable, str(PARSE_SCRIPT), "--root", str(roundtrip_root),
           "--force-dir", str(job_dirs["force"]), "--out-dir", str(out_dir)]
    # parse_roundtrip joins --eda-dir onto --root; an absolute path survives os.path.join.
    cmd += ["--eda-dir", str(job_dirs.get("eda", Path(job_dirs["force"]) / "_no_eda"))]
    if stem:
        cmd += ["--stems", stem]
    if strict:
        cmd.append("--strict")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if log_path is not None:
        Path(log_path).write_text(f"$ {' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}")
    if proc.returncode != 0:
        raise RuntimeError(f"parse_roundtrip.py failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    written = sorted(p for p in out_dir.glob("*.xyz")
                     if p not in before or p.stat().st_mtime_ns != before[p])
    if ctx is not None:
        ctx.add_input("parse_roundtrip.py", PARSE_SCRIPT)
        for kind, jd in job_dirs.items():
            ctx.add_input(f"qchem_outputs.{kind}", Path(jd) / "outputs")
        ctx.note(f"parsed {len(written)} dataset file(s): {[p.name for p in written]}")
    return written


def dataset_summary(path: Path) -> dict:
    """Frame count and which label families a parsed dataset carries."""
    frames = read_frames(path)
    keys = set()
    for f in frames:
        keys.update(f.info)
    return {
        "n_frames": len(frames),
        "has_eda": any(k.startswith("eda_") for k in keys),
        "n_atoms_max": max((len(f.symbols) for f in frames), default=0),
    }


__all__ = [
    "ROUNDTRIP_ROOT", "JOB_KINDS", "read_frames", "check_frames", "write_jobs", "job_dir",
    "job_status", "wait_for_jobs", "parse_jobs", "dataset_summary", "roundtrip_module",
]

