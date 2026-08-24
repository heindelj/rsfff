#!/usr/bin/env python3
"""Stage benchmark water-cluster Q-Chem jobs inside ``qchem_roundtrip``.

The jobs are nested beneath the existing roundtrip calculation folders so the
normal worker/sync scripts find them:

- ``qchem_roundtrip/eda/benchmark_eda``
- ``qchem_roundtrip/force/benchmark_force``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from benchmark_fragment_tools import minimized_oh_groups


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURES = REPO_ROOT / "benchmarks" / "structures"
DEFAULT_ROUNDTRIP = REPO_ROOT / "qchem_roundtrip"


@dataclass(frozen=True)
class JobSpec:
    calc: str
    heading: str
    jobtype: str
    fragmented: bool

    @property
    def root_name(self) -> str:
        return f"{self.calc}/{self.heading}"


JOBS = (
    JobSpec(calc="eda", heading="benchmark_eda", jobtype="eda", fragmented=True),
    JobSpec(calc="force", heading="benchmark_force", jobtype="force", fragmented=False),
)


@dataclass(frozen=True)
class Frame:
    symbols: list[str]
    coords: list[tuple[float, float, float]]
    comment: str
    index: int


def read_xyz_frames(path: Path) -> list[Frame]:
    lines = path.read_text().splitlines()
    frames: list[Frame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i].split()[0])
        if i + natoms + 2 > len(lines):
            raise ValueError(f"{path}: truncated XYZ frame at line {i + 1}")
        comment = lines[i + 1].strip()
        symbols: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for line in lines[i + 2 : i + 2 + natoms]:
            toks = line.split()
            if len(toks) < 4:
                raise ValueError(f"{path}: malformed atom line {line!r}")
            symbols.append(toks[0])
            coords.append((float(toks[1]), float(toks[2]), float(toks[3])))
        frames.append(Frame(symbols=symbols, coords=coords, comment=comment, index=len(frames)))
        i += natoms + 2
    if not frames:
        raise ValueError(f"{path}: no frames found")
    return frames


def validate_ordered_water(frame: Frame, path: Path) -> int:
    if len(frame.symbols) % 3:
        raise ValueError(f"{path}: expected a multiple of 3 atoms")
    n_waters = len(frame.symbols) // 3
    for start in range(0, len(frame.symbols), 3):
        triplet = frame.symbols[start : start + 3]
        if triplet != ["O", "H", "H"]:
            raise ValueError(
                f"{path}: expected ordered water monomers as O H H, found {triplet!r} "
                f"at atom indices {start}-{start + 2}"
            )
    return n_waters


def atom_line(symbol: str, coord: tuple[float, float, float]) -> str:
    x, y, z = coord
    return f"  {symbol:<2} {x:16.8f} {y:16.8f} {z:16.8f}"


def plain_molecule(frame: Frame) -> str:
    lines = ["$molecule", "0 1"]
    lines.extend(atom_line(symbol, coord) for symbol, coord in zip(frame.symbols, frame.coords))
    lines.append("$end")
    return "\n".join(lines)


def sorted_groups(frame: Frame) -> list[list[int]]:
    n_oxygen = sum(1 for symbol in frame.symbols if symbol.upper() == "O")
    return minimized_oh_groups(frame.symbols, frame.coords, [0] * n_oxygen)


def sorted_frame(frame: Frame) -> Frame:
    groups = sorted_groups(frame)
    indices = [idx for group in groups for idx in group]
    return Frame(
        symbols=[frame.symbols[idx] for idx in indices],
        coords=[frame.coords[idx] for idx in indices],
        comment=frame.comment,
        index=frame.index,
    )


def fragmented_molecule(frame: Frame) -> str:
    groups = sorted_groups(frame)
    lines = ["$molecule", "0 1"]
    for group in groups:
        lines.append("--")
        lines.append("0 1")
        for idx in group:
            lines.append(atom_line(frame.symbols[idx], frame.coords[idx]))
    lines.append("$end")
    return "\n".join(lines)


def rem_block(*, jobtype: str, method: str, basis: str, mem_total: int, mem_static: int) -> str:
    if jobtype == "eda":
        lines = [
            "$rem",
            "   JOBTYPE          =  eda",
            "   EDA2             =  1",
            f"   METHOD           =  {method}",
            f"   BASIS            =  {basis}",
            "   SCF_CONVERGENCE  =  8",
            "   THRESH           =  14",
            "   XC_GRID          =  000099000590",
            "   NL_GRID          =  1",
            "   SYMMETRY         =  false",
            "   EDA_BSSE         =  false",
            "   FD_MAT_VEC_PROD  =  false",
            f"   MEM_TOTAL        =  {mem_total}",
            f"   MEM_STATIC       =  {mem_static}",
            "   SCF_PRINT_FRGM   =  true",
            "$end",
        ]
    else:
        lines = [
            "$rem",
            f"   JOBTYPE          =  {jobtype}",
            f"   METHOD           =  {method}",
            f"   BASIS            =  {basis}",
            "   SCF_CONVERGENCE  =  8",
            "   THRESH           =  14",
            "   XC_GRID          =  000099000590",
            "   NL_GRID          =  1",
            "   SYMMETRY         =  false",
            f"   MEM_TOTAL        =  {mem_total}",
            f"   MEM_STATIC       =  {mem_static}",
            "$end",
        ]
    return "\n".join(lines)


def input_text(frame: Frame, job: JobSpec, args: argparse.Namespace) -> str:
    molecule = fragmented_molecule(frame) if job.fragmented else plain_molecule(sorted_frame(frame))
    return (
        molecule
        + "\n\n"
        + rem_block(
            jobtype=job.jobtype,
            method=args.method,
            basis=args.basis,
            mem_total=args.mem_total,
            mem_static=args.mem_static,
        )
        + "\n"
    )


def extxyz_frame(frame: Frame) -> str:
    groups = sorted_groups(frame)
    n_fragments = len(groups)
    fragment_idx = [-1] * len(frame.symbols)
    for frag, group in enumerate(groups):
        for idx in group:
            fragment_idx[idx] = frag
    header = (
        'Properties=species:S:1:pos:R:3:fragment_idx:I:1 charge=0 multiplicity=1 '
        f'n_fragments={n_fragments} '
        f'fragment_charges="{" ".join(["0"] * n_fragments)}" '
        f'fragment_multiplicities="{" ".join(["1"] * n_fragments)}"'
    )
    lines = [str(len(frame.symbols)), header]
    for idx, (symbol, coord) in enumerate(zip(frame.symbols, frame.coords)):
        x, y, z = coord
        lines.append(f"{symbol:<2} {x:16.8f} {y:16.8f} {z:16.8f} {fragment_idx[idx]:d}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str, *, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def calculation_slug(method: str, basis: str) -> str:
    return f"{method.lower().replace('-', '')}_{basis.lower()}"


def frame_stem(path: Path, n_frames: int, frame_index: int, *, method: str, basis: str) -> str:
    stem = path.stem
    slug = calculation_slug(method, basis)
    if "_mp2_avtz" in stem:
        stem = stem.replace("_mp2_avtz", f"_{slug}")
    else:
        stem = f"{stem}_{slug}"
    if n_frames == 1:
        return stem
    return f"{stem}_frame{frame_index:04d}"


def clear_generated(root: Path) -> None:
    for job in JOBS:
        job_root = root / job.root_name
        for rel in ("inputs", "state/generated"):
            target = job_root / rel
            target.mkdir(parents=True, exist_ok=True)
            for path in target.glob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink()


def setup_jobs(args: argparse.Namespace) -> dict:
    paths = (
        [Path(p) for p in args.structures]
        if args.structures
        else sorted(args.structures_dir.glob("*.xyz"))
    )
    if not paths:
        raise FileNotFoundError(f"no XYZ files found in {args.structures_dir}")

    summary = {
        "method": args.method,
        "basis": args.basis,
        "source_structures": [str(path) for path in paths],
        "jobs": [],
    }
    for job in JOBS:
        job_root = args.roundtrip_root / job.root_name
        for sub in ("geoms", "inputs", "outputs", "state/generated", "state/done", "state/failed", "state/locks"):
            (job_root / sub).mkdir(parents=True, exist_ok=True)

    for path in paths:
        frames = read_xyz_frames(path)
        geom_payload = []
        for frame in frames:
            n_waters = validate_ordered_water(frame, path)
            stem = frame_stem(path, len(frames), frame.index, method=args.method, basis=args.basis)
            geom_payload.append(extxyz_frame(frame))
            record = {
                "source": str(path),
                "frame": frame.index,
                "stem": stem,
                "n_waters": n_waters,
            }
            for job in JOBS:
                input_path = args.roundtrip_root / job.root_name / "inputs" / f"{stem}.in"
                written = write_text(input_path, input_text(frame, job, args), overwrite=args.overwrite)
                record[f"{job.heading}_input"] = str(input_path)
                record[f"{job.heading}_written"] = written
            summary["jobs"].append(record)
        text = "".join(geom_payload)
        geom_stem = frame_stem(path, 1, 0, method=args.method, basis=args.basis)
        for job in JOBS:
            write_text(
                args.roundtrip_root / job.root_name / "geoms" / f"{geom_stem}.extxyz",
                text,
                overwrite=args.overwrite,
            )

    manifest = args.roundtrip_root / "benchmark_jobs_manifest.json"
    write_text(manifest, json.dumps(summary, indent=2) + "\n", overwrite=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structures", nargs="*", help="specific benchmark XYZ files")
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--roundtrip-root", type=Path, default=DEFAULT_ROUNDTRIP)
    parser.add_argument("--method", default="wB97M-V")
    parser.add_argument("--basis", default="def2-TZVPD")
    parser.add_argument("--mem-total", type=int, default=64000)
    parser.add_argument("--mem-static", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite:
        clear_generated(args.roundtrip_root)
    summary = setup_jobs(args)
    n_frames = len(summary["jobs"])
    n_inputs = n_frames * len(JOBS)
    n_written = sum(
        int(job.get(f"{spec.heading}_written", False))
        for job in summary["jobs"]
        for spec in JOBS
    )
    print(
        f"prepared {n_frames} benchmark frame(s), {n_inputs} requested Q-Chem input(s), "
        f"{n_written} written under {args.roundtrip_root}/eda and {args.roundtrip_root}/force"
    )


if __name__ == "__main__":
    main()
