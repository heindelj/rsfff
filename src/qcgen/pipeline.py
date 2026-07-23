"""Folder-staged, notebook-drivable reference-data queue.

A molecule advances through explicit stage folders under a single ``root``:

    00_inbox/        raw <name>.xyz               (seeded by the user)
    01_optimized/    <name>.xyz + <name>.json     (optimized geometry + meta)
    02_frequencies/  <name>.npz                   (Hessian, masses, geometry)
    03_wigner/       <name>.extxyz                (N unlabeled sampled geoms)
    04_labeled/      <name>.extxyz                (final labeled dataset)
    _completed/      <name>.xyz                   (original inbox file, archived)
    _failed/         <name>.<stage>.error.log     (traceback for triage)

Each stage function consumes items from its input folder and writes to its
output folder; :func:`run_stage` drives one stage over every pending item,
archiving originals on success and logging tracebacks on failure. Stages are
idempotent and resumable: an item whose output already exists is skipped, so a
Colab session can reconnect and simply re-run the cells.

Run stages in order by calling, in successive notebook cells::

    run_stage(stage_optimize,    cfg)
    run_stage(stage_frequencies, cfg)
    run_stage(stage_wigner,      cfg)
    run_stage(stage_label,       cfg)
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import compute, extxyz, wigner
from .backend import backend_name

# ---------------------------------------------------------------------------
# Stage folder names (ordered)
# ---------------------------------------------------------------------------
INBOX = "00_inbox"
OPTIMIZED = "01_optimized"
FREQUENCIES = "02_frequencies"
WIGNER = "03_wigner"
LABELED = "04_labeled"
COMPLETED = "_completed"
FAILED = "_failed"

_ALL_DIRS = [INBOX, OPTIMIZED, FREQUENCIES, WIGNER, LABELED, COMPLETED, FAILED]

# ---------------------------------------------------------------------------
# Species table: name -> (charge, multiplicity). spin (Nalpha-Nbeta) = mult - 1.
# Extends the psi4 pipeline's CHARGES with multiplicities; open-shell allowed.
# ---------------------------------------------------------------------------
SPECIES = {
    "h2o": (0, 1),
    "ch4": (0, 1),
    "co2": (0, 1),
    "hf": (0, 1),
    "nh3": (0, 1),
    "h2so4": (0, 1),
    "h3o+": (1, 1),
    "hso4-": (-1, 1),
    "oh-": (-1, 1),
    "so42-": (-2, 1),
    "oh_radical": (0, 2),
    "h_atom": (0, 2),
}


def charge_mult_for(name):
    """Look up ``(charge, multiplicity)``; fall back to parsing the file stem.

    Unknown closed-shell stems default to multiplicity 1; a trailing +/-(n)
    encodes the charge (e.g. ``so42-`` -> -2).
    """
    if name in SPECIES:
        return SPECIES[name]
    stem, sign = name, 0
    if stem.endswith("+"):
        sign, stem = +1, stem[:-1]
    elif stem.endswith("-"):
        sign, stem = -1, stem[:-1]
    if sign == 0:
        return 0, 1
    mag = ""
    while stem and stem[-1].isdigit():
        mag, stem = stem[-1] + mag, stem[:-1]
    return sign * (int(mag) if mag else 1), 1


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Everything a stage needs: paths, method, and sampling settings."""

    root: Path
    xc: str = "wB97M-V"
    basis: str = "def2-svpd"
    temperature: float = 300.0
    n_samples: int = 500
    seed: int = 0
    verbose: bool = True

    def __post_init__(self):
        self.root = Path(self.root)

    def dir(self, name):
        return self.root / name


def make_dirs(cfg):
    """Create all stage folders under ``cfg.root``; return the root path."""
    for d in _ALL_DIRS:
        cfg.dir(d).mkdir(parents=True, exist_ok=True)
    return cfg.root


def seed_inbox(cfg, xyz_paths):
    """Copy raw ``.xyz`` files into the inbox (skipping ones already present)."""
    inbox = cfg.dir(INBOX)
    inbox.mkdir(parents=True, exist_ok=True)
    seeded = []
    for p in xyz_paths:
        p = Path(p)
        dst = inbox / p.name
        if not dst.exists():
            shutil.copy(p, dst)
            seeded.append(dst.name)
    return seeded


def _log(cfg, msg):
    if cfg.verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage descriptor + driver
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    """A named transform from one stage folder to the next.

    ``fn(cfg, name) -> None`` performs the work for one item and writes its
    output artifact(s). ``in_dir``/``out_dir`` are stage-folder names;
    ``in_suffix`` selects input files; ``out_name(name)`` returns the primary
    output filename used to decide whether the item is already done.
    """

    name: str
    in_dir: str
    out_dir: str
    in_suffix: str
    out_name: "callable"
    fn: "callable"
    archive_inbox: bool = False
    extra_outputs: list = field(default_factory=list)


def run_stage(stage, cfg):
    """Drive ``stage`` over every pending item in its input folder.

    Returns ``(done, skipped, failed)`` name lists. Idempotent: items whose
    output already exists are skipped; failures are isolated per item.
    """
    make_dirs(cfg)
    in_dir, out_dir = cfg.dir(stage.in_dir), cfg.dir(stage.out_dir)
    done, skipped, failed = [], [], []

    items = sorted(in_dir.glob(f"*{stage.in_suffix}"))
    _log(cfg, f"stage '{stage.name}' [{backend_name()}]: {len(items)} candidate(s) in {stage.in_dir}/")
    for item in items:
        name = item.name[: -len(stage.in_suffix)] if stage.in_suffix else item.stem
        out_path = out_dir / stage.out_name(name)
        if out_path.exists():
            skipped.append(name)
            continue
        t0 = time.time()
        try:
            _log(cfg, f"  {stage.name}: {name} ...")
            stage.fn(cfg, name)
            done.append(name)
            _log(cfg, f"  {stage.name}: {name} done ({time.time() - t0:.1f}s)")
            if stage.archive_inbox:
                _archive_inbox(cfg, name)
        except Exception:  # noqa: BLE001 - isolate per-item failures
            failed.append(name)
            err = cfg.dir(FAILED) / f"{name}.{stage.name}.error.log"
            err.write_text(traceback.format_exc())
            _log(cfg, f"  {stage.name}: {name} FAILED -> {err}")
    _log(cfg, f"stage '{stage.name}': {len(done)} done, {len(skipped)} skipped, {len(failed)} failed")
    return done, skipped, failed


def _archive_inbox(cfg, name):
    """Move the original inbox ``<name>.xyz`` into ``_completed/`` (once)."""
    src = cfg.dir(INBOX) / f"{name}.xyz"
    if src.exists():
        shutil.move(str(src), str(cfg.dir(COMPLETED) / f"{name}.xyz"))


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def _do_optimize(cfg, name):
    charge, mult = charge_mult_for(name)
    spin = mult - 1
    symbols, coords = extxyz.read_xyz(cfg.dir(INBOX) / f"{name}.xyz")
    opt_coords, energy = compute.optimize_geometry(
        symbols, coords, charge, spin, cfg.xc, cfg.basis
    )
    extxyz.write_xyz(
        cfg.dir(OPTIMIZED) / f"{name}.xyz", symbols, opt_coords,
        comment=f"{name} optimized {cfg.xc}/{cfg.basis} energy={energy:.10e}",
    )
    meta = {
        "name": name, "charge": charge, "mult": mult, "spin": spin,
        "xc": cfg.xc, "basis": cfg.basis, "opt_energy": energy,
    }
    (cfg.dir(OPTIMIZED) / f"{name}.json").write_text(json.dumps(meta, indent=2))


def _do_frequencies(cfg, name):
    meta = json.loads((cfg.dir(OPTIMIZED) / f"{name}.json").read_text())
    symbols, coords = extxyz.read_xyz(cfg.dir(OPTIMIZED) / f"{name}.xyz")
    res = compute.harmonic_frequencies(
        symbols, coords, meta["charge"], meta["spin"], cfg.xc, cfg.basis
    )
    np.savez(
        cfg.dir(FREQUENCIES) / f"{name}.npz",
        symbols=np.array(res["symbols"]),
        coords_ang=res["coords_ang"],
        masses_amu=res["masses_amu"],
        hessian=res["hessian"],
        freqs_cm=res["freqs_cm"],
        energy=res["energy"],
        hessian_method=res["hessian_method"],
        charge=meta["charge"], mult=meta["mult"], spin=meta["spin"],
    )


def _do_wigner(cfg, name):
    data = np.load(cfg.dir(FREQUENCIES) / f"{name}.npz", allow_pickle=True)
    symbols = [str(s) for s in data["symbols"]]
    rng = np.random.default_rng(cfg.seed)
    geoms = wigner.sample_wigner(
        data["coords_ang"], data["masses_amu"], data["hessian"],
        n_samples=cfg.n_samples, temperature=cfg.temperature, rng=rng,
    )
    extxyz.write_geoms(
        cfg.dir(WIGNER) / f"{name}.extxyz", symbols, geoms,
        comment_fn=lambda i: (
            f"config_type={name} sample_index={i} "
            f"temperature={cfg.temperature} source=wigner"
        ),
    )


def _do_label(cfg, name):
    charge, mult = charge_mult_for(name)
    spin = mult - 1
    symbols, frames = extxyz.read_geoms(cfg.dir(WIGNER) / f"{name}.extxyz")
    out_path = cfg.dir(LABELED) / f"{name}.extxyz"
    tmp_path = out_path.with_suffix(".extxyz.partial")
    n_ok, n_bad = 0, 0
    with open(tmp_path, "w", buffering=1) as fh:
        for idx, coords in enumerate(frames):
            # Isolate per-frame failures (e.g. a distorted geometry that fails
            # to converge) so one bad sample cannot discard the whole species.
            try:
                data = compute.compute_reference_data(
                    symbols, coords, charge, spin, cfg.xc, cfg.basis
                )
            except Exception:  # noqa: BLE001
                n_bad += 1
                err = cfg.dir(FAILED) / f"{name}.frame{idx}.error.log"
                err.write_text(traceback.format_exc())
                continue
            meta = {
                "name": name, "index": idx, "charge": charge, "mult": mult,
                "method": cfg.xc, "basis": cfg.basis,
                "temperature": cfg.temperature,
            }
            extxyz.write_frame(fh, symbols, coords, data, meta)
            fh.flush()
            os.fsync(fh.fileno())
            n_ok += 1
            if cfg.verbose and (idx + 1) % 25 == 0:
                _log(cfg, f"    {name}: labeled {idx + 1}/{len(frames)}")
    if n_ok == 0:
        os.remove(tmp_path)
        raise RuntimeError(f"{name}: all {len(frames)} frames failed to label")
    _log(cfg, f"    {name}: {n_ok} labeled, {n_bad} skipped")
    os.replace(tmp_path, out_path)  # atomic: only a complete file appears


# Stage descriptors (used by the notebook: run_stage(STAGE_*, cfg)).
STAGE_OPTIMIZE = Stage(
    "optimize", INBOX, OPTIMIZED, ".xyz",
    out_name=lambda n: f"{n}.xyz", fn=_do_optimize,
)
STAGE_FREQUENCIES = Stage(
    "frequencies", OPTIMIZED, FREQUENCIES, ".xyz",
    out_name=lambda n: f"{n}.npz", fn=_do_frequencies,
)
STAGE_WIGNER = Stage(
    "wigner", FREQUENCIES, WIGNER, ".npz",
    out_name=lambda n: f"{n}.extxyz", fn=_do_wigner,
)
STAGE_LABEL = Stage(
    "label", WIGNER, LABELED, ".extxyz",
    out_name=lambda n: f"{n}.extxyz", fn=_do_label, archive_inbox=True,
)

STAGES = [STAGE_OPTIMIZE, STAGE_FREQUENCIES, STAGE_WIGNER, STAGE_LABEL]


def collect(cfg, out_path=None):
    """Concatenate all ``04_labeled/*.extxyz`` into one dataset file.

    Returns ``(out_path, per_species_counts)``.
    """
    out_path = Path(out_path) if out_path else cfg.root / "reference_dataset.extxyz"
    counts = {}
    with open(out_path, "w") as out:
        for p in sorted(cfg.dir(LABELED).glob("*.extxyz")):
            text = p.read_text()
            # Frame count = number of extended-XYZ header lines.
            counts[p.stem] = sum(
                1 for ln in text.splitlines() if ln.startswith("Properties=")
            )
            out.write(text)
    return out_path, counts
