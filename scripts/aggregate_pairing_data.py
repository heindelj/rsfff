#!/usr/bin/env python3
"""Aggregate every labeled structure in ``qchem_roundtrip`` into training files for the
pairing model, and write a manifest of what went where.

    python scripts/aggregate_pairing_data.py                 # everything, into data/pairing/
    python scripts/aggregate_pairing_data.py --only scans    # one set
    python scripts/aggregate_pairing_data.py --dry-run       # counts only

Two kinds of output, matching the two training streams of ``rsfff.train.train_film``:

``data/pairing/eda/``   frames with ALMO-EDA components **and** forces -> ``data.path``
``data/pairing/force/`` frames with energies and forces only          -> ``data.force_path``

Every frame carries a ``split_group`` header (the trajectory or scan it came from), so the
grouped train/val split holds whole trajectories out, and ``fragment_idx`` /
``fragment_charges`` / ``fragment_multiplicities`` for the model's bookkeeping.

Sets
----
transition_structures   reactive-MD sweep frames (``data/cluster_sweep``): forces on every
                        frame; the reaction-centred ALMO-EDA of ``setup_sweep_qchem_jobs.py``
                        (pairA / pairB / dimer splits) merged as three fragmentations of one
                        frame, in the ``read_multifrag_extxyz`` schema, wherever all three
                        finished. Force frames get the **monomer** fragmentation (each H to
                        its nearest O; the ion is the O with three or one hydrogens).
negative_space          the sweep's rejected frames, forces only, monomer fragmentation.
scans                   ``pairing_scans*`` (rigid stretches / bend / proton scans; RKS, and
                        the UKS singlet and triplet sets once they are back), forces only,
                        one fragment carrying the total charge, ``scan`` / ``coord`` headers.
ion_clusters            H3O+/OH- (H2O)n, n = 1..7, EDA + forces, through
                        ``scripts/parse_roundtrip.py`` (monomer fragmentation).
benchmark               the w10-w23 benchmark clusters, EDA + forces, likewise. Already in
                        ``data/wb97mv_tzvpd_large``; regenerated here so one manifest lists it.

Not touched: ``data/wb97mv_tzvpd`` (w2-w5, the w1/w2 ion clusters with every decomposition,
the monomer AIMD and polarizability files) -- those are the existing streams and the
manifest just lists them.

Reading the sweep's EDA
-----------------------
The force job keeps the sweep's atom order; the three EDA jobs reorder atoms as
[donor, shared proton, acceptor, environment] and each is in its own standard nuclear
orientation. The permutation is recovered by matching the ``$molecule`` coordinates of the
two inputs exactly, the rotation by Procrustes on the recentered geometries (checked to
1e-3 Angstrom), and the supersystem energies of the four jobs are required to agree: the EDA's
CT-allowed wavefunction *is* the force job's.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from qcgen.multifrag import (  # noqa: E402
    Fragmentation,
    MultiFragFrame,
    canonical_basis,
    fragmentation_config_type,
    recenter,
    rotate_second_moments,
    rotation_between,
    write_frames,
)
from qcgen.qchem_eda import (  # noqa: E402
    QChemEDAParseError,
    parse_eda_output,
    to_atomic_units as eda_to_atomic_units,
    unique_components,
)
from qcgen.qchem_force import (  # noqa: E402
    parse_force_output,
    to_atomic_units as force_to_atomic_units,
)
from qcgen.qchem_out import QChemParseError  # noqa: E402

ENERGY_ATOL = 1.0e-6          # Hartree, force SCF vs EDA CT-allowed SCF
RMSD_TOL = 1.0e-3             # Angstrom, Procrustes residual
STEM_RE = re.compile(r"^(?P<traj>.+?)_(?P<index>\d+)_s(?P<step>\d+)$")


# ---------------------------------------------------------------------------
# monomer fragmentation


def monomer_fragmentation(symbols, positions):
    """``(fragment_idx, charges)``: each H to its nearest O; the ion is the O with 3 or 1 H.

    Atoms are **not** reordered here; the caller sorts by fragment before writing.
    """
    z = np.array([{"H": 1, "O": 8}.get(s, 0) for s in symbols])
    pos = np.asarray(positions)
    oxygens = np.flatnonzero(z == 8)
    if oxygens.size == 0:
        return np.zeros(len(symbols), dtype=int), [0]
    frag = np.full(len(symbols), -1, dtype=int)
    frag[oxygens] = np.arange(oxygens.size)
    for h in np.flatnonzero(z == 1):
        d = np.linalg.norm(pos[oxygens] - pos[h], axis=1)
        frag[h] = int(np.argmin(d))
    if (frag < 0).any():
        raise ValueError("an atom that is neither O nor H; extend monomer_fragmentation")
    charges = []
    for f in range(oxygens.size):
        n_h = int(((frag == f) & (z == 1)).sum())
        charges.append({3: 1, 2: 0, 1: -1, 0: -2}.get(n_h, n_h - 2))
    return frag, charges


def formula(symbols, mask):
    syms = [s for s, m in zip(symbols, mask) if m]
    counts = {s: syms.count(s) for s in dict.fromkeys(sorted(syms))}
    return "".join(s + (str(c) if c > 1 else "") for s, c in counts.items())


def config_type_of(symbols, frag, charges):
    parts = []
    for f, q in enumerate(charges):
        text = formula(symbols, frag == f)
        parts.append(text + ("+" if q > 0 else "-" if q < 0 else ""))
    if all(p == "H2O" for p in parts):
        return f"w{len(parts)}"
    return "_".join(parts)


# ---------------------------------------------------------------------------
# force-only frames


def fmt(arr) -> str:
    return " ".join(f"{float(v):.12e}" for v in np.asarray(arr, dtype=np.float64).ravel())


def write_force_frame(fh, rec, *, frag, charges, mults, split_group, extra=None):
    """One force-only frame, atoms sorted by fragment, in the loader's schema."""
    order = np.argsort(frag, kind="stable")
    n = rec.n_atoms
    keys = [
        "Properties=species:S:1:pos:R:3:mulliken_charges:R:1:fragment_idx:I:1:forces:R:3",
        f"energy={rec.energy:.12e}",
        f'dipole="{fmt(rec.multipoles["dipole"])}"',
        f'quadrupole="{fmt(rec.multipoles["quadrupole"])}"',
        "multipole_format=tensor",
        f"n_fragments={len(charges)}",
        f"charge={rec.total_charge}",
        f"multiplicity={rec.multiplicity}",
        f'fragment_charges="{" ".join(str(int(c)) for c in charges)}"',
        f'fragment_multiplicities="{" ".join(str(int(m)) for m in mults)}"',
        f"method={rec.method}",
        f"basis={canonical_basis(rec.basis)}",
        f"config_type={config_type_of(rec.symbols, frag, charges)}",
        f"split_group={split_group}",
        f"source={os.path.relpath(rec.path, ROOT)}",
    ]
    for k, v in (extra or {}).items():
        keys.append(f"{k}={v}")
    keys.append("units=atomic")
    fh.write(f"{n}\n{' '.join(keys)}\n")
    for a in order:
        x, y, z = rec.positions[a]
        fx, fy, fz = rec.forces[a]
        fh.write(
            f"{rec.symbols[a]:<3} {x:18.10f} {y:18.10f} {z:18.10f} "
            f"{rec.mulliken_charges[a]:16.10f} {int(frag[a]):4d}"
            f" {fx:22.14e} {fy:22.14e} {fz:22.14e}\n"
        )


S2_RE = re.compile(r"<S\^2>\s*=\s*([-+0-9.]+)")


def load_force(path):
    rec = force_to_atomic_units(parse_force_output(str(path)))
    if not (rec.completed and rec.converged):
        raise QChemParseError("did not complete / converge")
    # the last <S^2> the job printed (unrestricted jobs only): a broken-symmetry singlet on a
    # stretched bond shows up here as ~1, and a triplet as ~2
    text = Path(path).read_text(errors="replace")
    hits = S2_RE.findall(text)
    rec.s2 = float(hits[-1]) if hits else None
    return rec


def stem_parts(stem: str):
    m = STEM_RE.match(stem)
    if not m:
        return stem, 0, 0
    return m["traj"], int(m["index"]), int(m["step"])


# ---------------------------------------------------------------------------
# the sweep sets


def aggregate_sweep_forces(calc_dir: Path, out_path: Path, dry_run: bool):
    """Force outputs of ``force/transition_structures`` or ``force/negative_space``."""
    outputs = sorted(calc_dir.glob("outputs/*.out"))
    n_ok, n_bad, groups = 0, 0, set()
    fh = None if dry_run else open(out_path, "w")
    try:
        for path in outputs:
            traj, index, step = stem_parts(path.stem)
            try:
                rec = load_force(path)
                frag, charges = monomer_fragmentation(rec.symbols, rec.positions)
            except Exception as exc:                        # noqa: BLE001
                n_bad += 1
                print(f"  skip {path.name}: {exc}")
                continue
            if sum(charges) != rec.total_charge:
                n_bad += 1
                print(f"  skip {path.name}: monomer charges {charges} do not sum to {rec.total_charge}")
                continue
            groups.add(traj)
            n_ok += 1
            if fh is not None:
                write_force_frame(
                    fh, rec, frag=frag, charges=charges, mults=[1] * len(charges),
                    split_group=traj, extra={"sweep_index": index, "sweep_step": step},
                )
    finally:
        if fh is not None:
            fh.close()
    return {"frames": n_ok, "skipped": n_bad, "groups": len(groups), "path": str(out_path)}


def _molecule_coords(input_path: Path) -> np.ndarray:
    """The ``$molecule`` coordinates of an input, in file order (fragment separators dropped)."""
    lines = input_path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "$molecule")
    end = next(i for i in range(start, len(lines)) if lines[i].strip().lower() == "$end")
    rows = []
    for ln in lines[start + 1:end]:
        tok = ln.split()
        if len(tok) == 4 and tok[0].isalpha():
            rows.append([float(t) for t in tok[1:]])
    return np.array(rows)


def _permutation(force_in: Path, eda_in: Path) -> np.ndarray:
    """``perm[k]`` = force-order atom of EDA-order atom ``k``, by exact coordinate match."""
    a = _molecule_coords(force_in)
    b = _molecule_coords(eda_in)
    if a.shape != b.shape:
        raise QChemParseError(f"{eda_in.name}: {len(b)} atoms against {len(a)} in the force input")
    perm = np.empty(len(b), dtype=int)
    used = set()
    for k, row in enumerate(b):
        hits = np.flatnonzero(np.all(np.abs(a - row) < 1e-6, axis=1))
        hits = [h for h in hits if h not in used]
        if not hits:
            raise QChemParseError(f"{eda_in.name}: EDA atom {k} has no match in the force input")
        perm[k] = hits[0]
        used.add(hits[0])
    return perm


SPLITS = (("pairA", 0), ("pairB", 1), ("dimer", 2))


def build_sweep_eda_frame(stem: str, force_dir: Path, eda_dir: Path):
    """One transition frame with its three reaction-centred decompositions."""
    force_out = force_dir / "outputs" / f"{stem}.out"
    force_in = force_dir / "inputs" / f"{stem}.in"
    rec_f = load_force(force_out)
    n = rec_f.n_atoms
    positions = recenter(rec_f.symbols, rec_f.positions)
    frags = []
    for split, rank in SPLITS:
        eda_out = eda_dir / "outputs" / f"{stem}_{split}.out"
        eda_in = eda_dir / "inputs" / f"{stem}_{split}.in"
        rec = eda_to_atomic_units(parse_eda_output(str(eda_out)))
        if rec.n_atoms != n:
            raise QChemEDAParseError(f"{split}: {rec.n_atoms} atoms against {n}")
        if not rec.fragment_multipoles:
            raise QChemEDAParseError(f"{split}: no isolated-fragment blocks (SCF_PRINT_FRGM)")
        if abs(rec.energy - rec_f.energy) > ENERGY_ATOL:
            raise QChemEDAParseError(
                f"{split}: CT-allowed energy {rec.energy:.8f} != force energy {rec_f.energy:.8f}"
            )
        perm = _permutation(force_in, eda_in)          # EDA atom k -> force atom perm[k]
        if [rec.symbols[k] for k in range(n)] != [rec_f.symbols[perm[k]] for k in range(n)]:
            raise QChemEDAParseError(f"{split}: element mismatch through the permutation")
        # arrays indexed by force atom a: take EDA atom inv[a]
        inv = np.empty(n, dtype=int)
        inv[perm] = np.arange(n)
        eda_pos = recenter(rec.symbols, rec.positions)[inv]
        rot, _ = rotation_between(eda_pos, positions, rmsd_tol=RMSD_TOL)
        dipoles = np.array([m["dipole"] for m in rec.fragment_multipoles])
        seconds = np.array(
            [unique_components(m["quadrupole"], "quadrupole") for m in rec.fragment_multipoles]
        )
        charges = list(rec.fragment_charges)
        charged = [k for k, q in enumerate(charges) if q != 0]
        frags.append(Fragmentation(
            fragment_idx=np.asarray(rec.fragment_idx)[inv],
            fragment_charges=charges,
            fragment_mults=list(rec.fragment_mults),
            fragment_energies=rec.fragment_energies,
            fragment_dipoles=dipoles @ rot.T,
            fragment_second_moments=rotate_second_moments(seconds, rot),
            fragment_mulliken=np.concatenate(rec.fragment_mulliken)[inv],
            eda=rec.eda,
            rank=rank,
            charge_fragment=charged[0] if charged else 0,
            excess_distance=0.0,
            source=os.path.relpath(eda_out, ROOT),
        ))
    traj, index, step = stem_parts(stem)
    frame = MultiFragFrame(
        symbols=list(rec_f.symbols),
        positions=positions,
        forces=rec_f.forces,
        energy=rec_f.energy,
        mulliken=rec_f.mulliken_charges,
        multipoles=rec_f.multipoles,
        fragmentations=frags,
        total_charge=rec_f.total_charge,
        multiplicity=rec_f.multiplicity,
        method=rec_f.method,
        basis=canonical_basis(rec_f.basis),
        config_type=fragmentation_config_type(rec_f.symbols, frags[2].fragment_idx, frags[2].fragment_charges),
        sample_id=index,
        aimd_step=step,
        source=os.path.relpath(force_out, ROOT),
        extra={"split_group": traj, "sweep_index": index, "sweep_step": step,
               "fragmentation_splits": '"pairA pairB dimer"'},
    )
    return frame


def aggregate_sweep_eda(force_dir: Path, eda_dir: Path, out_path: Path, dry_run: bool):
    stems = sorted({p.stem.rsplit("_", 1)[0] for p in eda_dir.glob("outputs/*.out")})
    frames, n_bad, groups = [], 0, set()
    for stem in stems:
        needed = [eda_dir / "outputs" / f"{stem}_{s}.out" for s, _ in SPLITS]
        if not all(p.exists() for p in needed) or not (force_dir / "outputs" / f"{stem}.out").exists():
            n_bad += 1
            continue
        try:
            frame = build_sweep_eda_frame(stem, force_dir, eda_dir)
        except Exception as exc:                            # noqa: BLE001
            n_bad += 1
            print(f"  skip {stem}: {exc}")
            continue
        frames.append(frame)
        groups.add(stem_parts(stem)[0])
    if not dry_run and frames:
        write_frames(out_path, frames)
    return {"frames": len(frames), "skipped": n_bad, "groups": len(groups), "path": str(out_path)}


# ---------------------------------------------------------------------------
# the scans


def aggregate_scans(rt: Path, out_dir: Path, dry_run: bool):
    """Every ``pairing_scans*`` calculation: one file each, one fragment per frame."""
    from ase.io import read as ase_read

    results = {}
    for calc_dir in sorted(rt.glob("pairing_scans*")):
        if not (calc_dir / "outputs").is_dir():
            continue
        out_path = out_dir / f"{calc_dir.name}.xyz"
        n_ok, n_bad = 0, 0
        fh = None if dry_run else open(out_path, "w")
        try:
            for geom in sorted(calc_dir.glob("geoms/*.xyz")):
                frames = ase_read(geom, index=":")
                for k, atoms in enumerate(frames):
                    out = calc_dir / "outputs" / f"{geom.stem}_frame{k:04d}.out"
                    if not out.exists():
                        n_bad += 1
                        continue
                    try:
                        rec = load_force(out)
                    except Exception as exc:                # noqa: BLE001
                        n_bad += 1
                        print(f"  skip {out.name}: {exc}")
                        continue
                    n_ok += 1
                    if fh is None:
                        continue
                    info = atoms.info
                    extra = {"scan": info.get("scan", geom.stem), "coord": f"{float(info.get('coord', 0.0)):.4f}"}
                    if getattr(rec, "s2", None) is not None:
                        extra["s2"] = f"{rec.s2:.6f}"
                    for key in ("r_oh", "angle_deg", "r_oo", "r_o1h", "r_o2h"):
                        if key in info:
                            extra[key] = info[key]
                    write_force_frame(
                        fh, rec, frag=np.zeros(rec.n_atoms, dtype=int),
                        charges=[rec.total_charge], mults=[rec.multiplicity],
                        split_group=f"{calc_dir.name}:{info.get('scan', geom.stem)}", extra=extra,
                    )
        finally:
            if fh is not None:
                fh.close()
        results[calc_dir.name] = {"frames": n_ok, "skipped": n_bad, "path": str(out_path)}
    return results


# ---------------------------------------------------------------------------
# the nested eda/force bundles, through parse_roundtrip


def aggregate_roundtrip(rt: Path, name: str, out_dir: Path, dry_run: bool):
    eda_dir = {"ion_clusters": "eda/ion_clusters", "benchmark": "eda/benchmark_eda"}[name]
    force_dir = {"ion_clusters": "force/ion_clusters", "benchmark": "force/benchmark_force"}[name]
    cmd = [
        sys.executable, str(ROOT / "scripts" / "parse_roundtrip.py"),
        "--root", str(rt), "--eda-dir", eda_dir, "--force-dir", force_dir,
        "--out-dir", str(out_dir),
    ]
    if dry_run:
        return {"command": " ".join(cmd)}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    written = sorted(p.name for p in out_dir.glob("*.xyz"))
    return {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-1000:], "files": written}


# ---------------------------------------------------------------------------


EXISTING = {
    "water clusters w2-w5 (EDA + forces)": ["data/wb97mv_tzvpd/w{n}_wb97mv_tzvpd.xyz" for n in (2, 3, 4, 5)],
    "ion clusters w1/w2, every decomposition (EDA + forces)": [
        "data/wb97mv_tzvpd/w1_h3o+_wb97mv_tzvpd.xyz", "data/wb97mv_tzvpd/w2_h3o+_wb97mv_tzvpd.xyz",
        "data/wb97mv_tzvpd/w1_oh-_wb97mv_tzvpd.xyz", "data/wb97mv_tzvpd/w2_oh-_wb97mv_tzvpd.xyz",
    ],
    "monomer AIMD (energies + forces)": [
        "data/wb97mv_tzvpd/h2o_wb97mv_tzvpd.xyz", "data/wb97mv_tzvpd/h3o+_wb97mv_tzvpd.xyz",
        "data/wb97mv_tzvpd/oh-_wb97mv_tzvpd.xyz",
    ],
    "monomer multipoles + polarizabilities": [
        "data/wb97mv_tzvpd/h2o_wb97mv_tzvpd_pol.xyz", "data/wb97mv_tzvpd/h2o_opt_wb97mv_tzvpd_pol.xyz",
        "data/wb97mv_tzvpd/h3o+_opt_wb97mv_tzvpd_pol.xyz", "data/wb97mv_tzvpd/oh-_opt_wb97mv_tzvpd_pol.xyz",
    ],
    "large water clusters w10-w23 (EDA + forces)": ["data/wb97mv_tzvpd_large/"],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roundtrip", default=str(ROOT / "qchem_roundtrip"))
    ap.add_argument("--out", default=str(ROOT / "data" / "pairing"))
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["transition_structures", "negative_space", "scans", "ion_clusters", "benchmark"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rt = Path(args.roundtrip)
    out = Path(args.out)
    sets = args.only or ["transition_structures", "negative_space", "scans", "ion_clusters", "benchmark"]
    if not args.dry_run:
        (out / "eda").mkdir(parents=True, exist_ok=True)
        (out / "force").mkdir(parents=True, exist_ok=True)
    manifest: dict = {"existing": EXISTING, "sets": {}}

    if "transition_structures" in sets:
        print("transition_structures: forces")
        manifest["sets"]["transition_structures_force"] = aggregate_sweep_forces(
            rt / "force" / "transition_structures", out / "force" / "transition_structures.xyz", args.dry_run
        )
        print("transition_structures: EDA (pairA / pairB / dimer)")
        manifest["sets"]["transition_structures_eda"] = aggregate_sweep_eda(
            rt / "force" / "transition_structures", rt / "eda" / "transition_structures",
            out / "eda" / "transition_structures.xyz", args.dry_run,
        )
    if "negative_space" in sets:
        print("negative_space: forces")
        manifest["sets"]["negative_space_force"] = aggregate_sweep_forces(
            rt / "force" / "negative_space", out / "force" / "negative_space.xyz", args.dry_run
        )
    if "scans" in sets:
        print("scans")
        manifest["sets"].update({f"{k}_force": v for k, v in aggregate_scans(rt, out / "force", args.dry_run).items()})
    for name in ("ion_clusters", "benchmark"):
        if name in sets:
            print(f"{name}: parse_roundtrip")
            manifest["sets"][f"{name}_eda"] = aggregate_roundtrip(rt, name, out / "eda" / name, args.dry_run)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("stdout_tail", "stderr_tail")}
                      for k, v in manifest["sets"].items()}, indent=1))
    if not args.dry_run:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
