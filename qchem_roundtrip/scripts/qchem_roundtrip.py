#!/usr/bin/env python3
"""Q-Chem round-trip input generation and worker utilities.

The directory contract is intentionally simple: each calculation type is a
folder containing ``geoms/``, ``inputs/``, and ``outputs/``. Geometry files are
expanded into Q-Chem inputs from templates, and worker processes atomically claim
inputs before running Q-Chem so multiple queued jobs can share one work pool.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class XYZFrame:
    symbols: list[str]
    coords: list[tuple[float, float, float]]
    comment: str
    info: dict[str, str]
    fragment_idx: list[int] | None
    index: int


@dataclass(frozen=True)
class GeneratedInput:
    calculation: str
    geometry: Path
    input_path: Path
    frame_index: int
    skipped: bool


@dataclass(frozen=True)
class ClaimedJob:
    calculation: str
    calc_dir: Path
    input_path: Path
    output_path: Path
    lock_dir: Path


class RoundtripError(RuntimeError):
    """Raised when the round-trip configuration or data is malformed."""


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            cfg = json.loads(text)
        else:
            cfg = yaml.safe_load(text) or {}
    if "calculations" not in cfg:
        raise RoundtripError(f"{path} must define a 'calculations' mapping")
    cfg["_config_path"] = str(path)
    return cfg


def resolve_config_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def ensure_layout(root: Path, cfg: dict[str, Any]) -> None:
    for calc_name in cfg["calculations"]:
        calc_dir = root / calc_name
        for rel in (
            "geoms",
            "inputs",
            "outputs",
            "state/generated",
            "state/locks",
            "state/done",
            "state/failed",
        ):
            (calc_dir / rel).mkdir(parents=True, exist_ok=True)


def geometry_files(geom_dir: Path) -> list[Path]:
    paths = []
    for suffix in ("*.xyz", "*.extxyz"):
        paths.extend(geom_dir.glob(suffix))
    return sorted(set(paths))


def read_xyz_frames(path: Path) -> list[XYZFrame]:
    lines = path.read_text().splitlines()
    frames: list[XYZFrame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            natoms = int(lines[i].split()[0])
        except (IndexError, ValueError) as exc:
            raise RoundtripError(f"{path}: expected atom count at line {i + 1}") from exc
        if i + 2 + natoms > len(lines):
            raise RoundtripError(f"{path}: truncated XYZ frame starting at line {i + 1}")
        comment = lines[i + 1].strip()
        frame = parse_extxyz_frame(path, lines[i + 2 : i + 2 + natoms], comment, len(frames), i + 3)
        frames.append(frame)
        i += natoms + 2
    if not frames:
        raise RoundtripError(f"{path}: no XYZ frames found")
    return frames


def parse_header_info(path: Path, comment: str) -> dict[str, str]:
    try:
        parts = shlex.split(comment)
    except ValueError as exc:
        raise RoundtripError(f"{path}: malformed extxyz header: {exc}") from exc
    info: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        info[key] = value
    if "Properties" not in info:
        raise RoundtripError(f"{path}: extxyz header must include Properties=...")
    return info


def parse_properties_schema(path: Path, properties: str) -> list[tuple[str, str, int]]:
    toks = properties.split(":")
    if len(toks) % 3 != 0:
        raise RoundtripError(f"{path}: malformed Properties schema")
    schema: list[tuple[str, str, int]] = []
    for idx in range(0, len(toks), 3):
        name, kind, count_text = toks[idx], toks[idx + 1], toks[idx + 2]
        try:
            count = int(count_text)
        except ValueError as exc:
            raise RoundtripError(f"{path}: non-integer Properties count for {name}") from exc
        schema.append((name, kind, count))
    names = {name for name, _, _ in schema}
    if "species" not in names or "pos" not in names:
        raise RoundtripError(f"{path}: extxyz Properties must include species and pos")
    return schema


def parse_typed_values(path: Path, kind: str, values: list[str], line_no: int) -> list[Any]:
    if kind == "S":
        return values
    if kind == "I":
        try:
            return [int(value) for value in values]
        except ValueError as exc:
            raise RoundtripError(f"{path}: non-integer extxyz value at line {line_no}") from exc
    if kind == "R":
        try:
            return [float(value) for value in values]
        except ValueError as exc:
            raise RoundtripError(f"{path}: non-numeric extxyz value at line {line_no}") from exc
    raise RoundtripError(f"{path}: unsupported extxyz Properties type {kind!r}")


def parse_extxyz_frame(
    path: Path,
    atom_lines: list[str],
    comment: str,
    frame_index: int,
    first_atom_line: int,
) -> XYZFrame:
    info = parse_header_info(path, comment)
    schema = parse_properties_schema(path, info["Properties"])
    ncols = sum(count for _, _, count in schema)
    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    fragment_idx: list[int] | None = [] if any(name == "fragment_idx" for name, _, _ in schema) else None
    for row_idx, line in enumerate(atom_lines):
        line_no = first_atom_line + row_idx
        toks = line.split()
        if len(toks) < ncols:
            raise RoundtripError(f"{path}: atom line {line_no} has {len(toks)} fields, expected {ncols}")
        cursor = 0
        row: dict[str, list[Any]] = {}
        for name, kind, count in schema:
            raw_values = toks[cursor : cursor + count]
            cursor += count
            row[name] = parse_typed_values(path, kind, raw_values, line_no)
        species = row["species"]
        pos = row["pos"]
        if len(species) != 1 or len(pos) != 3:
            raise RoundtripError(f"{path}: species:S:1 and pos:R:3 are required")
        symbols.append(str(species[0]))
        coords.append((float(pos[0]), float(pos[1]), float(pos[2])))
        if fragment_idx is not None:
            value = row["fragment_idx"]
            if len(value) != 1:
                raise RoundtripError(f"{path}: fragment_idx must have count 1")
            fragment_idx.append(int(value[0]))
    return XYZFrame(
        symbols=symbols,
        coords=coords,
        comment=comment,
        info=info,
        fragment_idx=fragment_idx,
        index=frame_index,
    )


def input_stem_for_frame(geom: Path, n_frames: int, frame_index: int) -> str:
    if n_frames == 1:
        return geom.stem
    return f"{geom.stem}_frame{frame_index:04d}"


def format_atom_line(symbol: str, coord: tuple[float, float, float]) -> str:
    x, y, z = coord
    return f"  {symbol:<3} {x:16.8f} {y:16.8f} {z:16.8f}"


def required_int(frame: XYZFrame, key: str, path: Path) -> int:
    if key not in frame.info:
        raise RoundtripError(f"{path}: extxyz frame {frame.index} requires header key {key!r}")
    try:
        return int(frame.info[key])
    except ValueError as exc:
        raise RoundtripError(f"{path}: header key {key!r} must be an integer") from exc


def required_int_list(frame: XYZFrame, key: str, path: Path) -> list[int]:
    if key not in frame.info:
        raise RoundtripError(f"{path}: extxyz frame {frame.index} requires header key {key!r}")
    try:
        return [int(value) for value in frame.info[key].split()]
    except ValueError as exc:
        raise RoundtripError(f"{path}: header key {key!r} must be a space-separated integer list") from exc


def build_plain_molecule(frame: XYZFrame, geom_path: Path) -> str:
    charge = required_int(frame, "charge", geom_path)
    mult = required_int(frame, "multiplicity", geom_path)
    lines = ["$molecule", f"{charge} {mult}"]
    lines.extend(format_atom_line(sym, coord) for sym, coord in zip(frame.symbols, frame.coords))
    lines.append("$end")
    return "\n".join(lines)


def build_fragmented_molecule(frame: XYZFrame, geom_path: Path) -> str:
    if frame.fragment_idx is None:
        raise RoundtripError(f"{geom_path}: fragmented molecule mode requires Properties=...:fragment_idx:I:1")
    total_charge = required_int(frame, "charge", geom_path)
    mult = required_int(frame, "multiplicity", geom_path)
    fragment_charges = required_int_list(frame, "fragment_charges", geom_path)
    fragment_mults = required_int_list(frame, "fragment_multiplicities", geom_path)
    fragment_ids = sorted(set(frame.fragment_idx))
    expected_ids = list(range(len(fragment_ids)))
    if fragment_ids != expected_ids:
        raise RoundtripError(
            f"{geom_path}: fragment_idx values must be contiguous from 0; found {fragment_ids}"
        )
    if "n_fragments" in frame.info and required_int(frame, "n_fragments", geom_path) != len(fragment_ids):
        raise RoundtripError(f"{geom_path}: n_fragments does not match fragment_idx values")
    if len(fragment_charges) != len(fragment_ids):
        raise RoundtripError(f"{geom_path}: fragment_charges length must match number of fragments")
    if len(fragment_mults) != len(fragment_ids):
        raise RoundtripError(f"{geom_path}: fragment_multiplicities length must match number of fragments")
    lines = ["$molecule", f"{total_charge} {mult}"]
    seen: set[int] = set()
    for fragment_id in fragment_ids:
        indices = [idx for idx, value in enumerate(frame.fragment_idx) if value == fragment_id]
        seen.update(indices)
        lines.append("--")
        lines.append(f"{fragment_charges[fragment_id]} {fragment_mults[fragment_id]}")
        for idx in indices:
            lines.append(format_atom_line(frame.symbols[idx], frame.coords[idx]))
    if seen != set(range(len(frame.symbols))):
        missing = sorted(set(range(len(frame.symbols))) - seen)
        raise RoundtripError(f"fragment definition does not cover atom indices: {missing}")
    lines.append("$end")
    return "\n".join(lines)


def build_molecule_block(frame: XYZFrame, molecule_cfg: dict[str, Any], geom_path: Path) -> str:
    mode = molecule_cfg.get("mode", "plain")
    if mode == "plain":
        return build_plain_molecule(frame, geom_path)
    if mode == "fragments":
        return build_fragmented_molecule(frame, geom_path)
    raise RoundtripError(f"unsupported molecule mode: {mode}")


def replace_molecule_block(template: str, molecule_block: str) -> str:
    lines = template.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "$molecule")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].strip().lower() == "$end")
    except StopIteration as exc:
        raise RoundtripError("template must contain a $molecule ... $end block") from exc
    new_lines = lines[:start] + molecule_block.splitlines() + lines[end + 1 :]
    return "\n".join(new_lines).rstrip() + "\n"


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def frame_metadata_summary(frame: XYZFrame) -> dict[str, str]:
    keys = (
        "charge",
        "multiplicity",
        "n_fragments",
        "fragment_charges",
        "fragment_multiplicities",
        "config_type",
        "sample_id",
        "source",
    )
    return {key: frame.info[key] for key in keys if key in frame.info}


def generate_inputs(root: Path, cfg: dict[str, Any], *, overwrite: bool = False) -> list[GeneratedInput]:
    config_path = Path(cfg["_config_path"])
    ensure_layout(root, cfg)
    generated: list[GeneratedInput] = []
    defaults = cfg.get("defaults", {})
    for calc_name, calc_cfg in cfg["calculations"].items():
        if not calc_cfg.get("enabled", True):
            continue
        calc_dir = root / calc_name
        template_path = resolve_config_path(config_path, calc_cfg["template"])
        template = template_path.read_text()
        molecule_cfg = dict(defaults.get("molecule", {}))
        molecule_cfg.update(calc_cfg.get("molecule", {}))
        for geom in geometry_files(calc_dir / "geoms"):
            frames = read_xyz_frames(geom)
            for frame in frames:
                stem = input_stem_for_frame(geom, len(frames), frame.index)
                input_path = calc_dir / "inputs" / f"{stem}.in"
                marker_path = calc_dir / "state" / "generated" / f"{stem}.json"
                if input_path.exists() and not overwrite:
                    generated.append(GeneratedInput(calc_name, geom, input_path, frame.index, True))
                    continue
                molecule = build_molecule_block(frame, molecule_cfg, geom)
                rendered = replace_molecule_block(template, molecule)
                atomic_write(input_path, rendered)
                write_json_atomic(
                    marker_path,
                    {
                        "calculation": calc_name,
                        "geometry": str(geom),
                        "input": str(input_path),
                        "frame_index": frame.index,
                        "frame_metadata": frame_metadata_summary(frame),
                        "generated_at": time.time(),
                    },
                )
                generated.append(GeneratedInput(calc_name, geom, input_path, frame.index, False))
    return generated


def calculation_priority(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    calc_name, calc_cfg = item
    return (-int(calc_cfg.get("priority", 0)), calc_name)


def lock_is_stale(lock_dir: Path, stale_seconds: int) -> bool:
    if stale_seconds <= 0:
        return False
    try:
        age = time.time() - lock_dir.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > stale_seconds


def remove_lock(lock_dir: Path) -> None:
    if lock_dir.exists():
        shutil.rmtree(lock_dir)


def claim_next_job(root: Path, cfg: dict[str, Any], *, stale_lock_seconds: int = 172800) -> ClaimedJob | None:
    ensure_layout(root, cfg)
    for calc_name, calc_cfg in sorted(cfg["calculations"].items(), key=calculation_priority):
        if not calc_cfg.get("enabled", True):
            continue
        calc_dir = root / calc_name
        for input_path in sorted((calc_dir / "inputs").glob("*.in")):
            stem = input_path.stem
            output_path = calc_dir / "outputs" / f"{stem}.out"
            done_marker = calc_dir / "state" / "done" / f"{stem}.json"
            failed_marker = calc_dir / "state" / "failed" / f"{stem}.json"
            lock_dir = calc_dir / "state" / "locks" / f"{stem}.lock"
            if output_path.exists() or done_marker.exists() or failed_marker.exists():
                continue
            if lock_dir.exists() and lock_is_stale(lock_dir, stale_lock_seconds):
                remove_lock(lock_dir)
            try:
                lock_dir.mkdir()
            except FileExistsError:
                continue
            write_json_atomic(
                lock_dir / "claim.json",
                {
                    "calculation": calc_name,
                    "input": str(input_path),
                    "output": str(output_path),
                    "claimed_at": time.time(),
                    "hostname": os.uname().nodename,
                    "pid": os.getpid(),
                },
            )
            return ClaimedJob(calc_name, calc_dir, input_path, output_path, lock_dir)
    return None


def run_qchem_job(job: ClaimedJob, qchem_command: str, threads: int, extra_args: list[str]) -> int:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    rel_input = job.input_path.relative_to(job.calc_dir)
    rel_output = job.output_path.relative_to(job.calc_dir)
    cmd = [qchem_command, "-save", "-nt", str(threads), *extra_args, str(rel_input), str(rel_output)]
    started = time.time()
    start_error = ""
    try:
        proc = subprocess.run(cmd, cwd=job.calc_dir)
        returncode = proc.returncode
    except OSError as exc:
        start_error = str(exc)
        returncode = 127
    finished = time.time()
    marker_dir = job.calc_dir / "state" / ("done" if returncode == 0 else "failed")
    marker_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "calculation": job.calculation,
        "input": str(job.input_path),
        "output": str(job.output_path),
        "command": cmd,
        "returncode": returncode,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": finished - started,
    }
    if start_error:
        payload["start_error"] = start_error
    write_json_atomic(marker_dir / f"{job.input_path.stem}.json", payload)
    remove_lock(job.lock_dir)
    return returncode


def worker_loop(
    root: Path,
    cfg: dict[str, Any],
    *,
    qchem_command: str = "qchem",
    threads: int = 128,
    poll_seconds: int = 30,
    idle_timeout_seconds: int = 1800,
    stale_lock_seconds: int = 172800,
    once: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    extra_args = extra_args or []
    last_work = time.time()
    while True:
        generate_inputs(root, cfg)
        job = claim_next_job(root, cfg, stale_lock_seconds=stale_lock_seconds)
        if job is not None:
            print(f"[worker] running {job.calculation}: {job.input_path.name}", flush=True)
            rc = run_qchem_job(job, qchem_command, threads, extra_args)
            print(f"[worker] finished {job.input_path.name} rc={rc}", flush=True)
            last_work = time.time()
            if once:
                return rc
            continue
        idle_for = time.time() - last_work
        if once:
            print("[worker] no runnable inputs", flush=True)
            return 0
        if idle_for >= idle_timeout_seconds:
            print(f"[worker] no work for {idle_timeout_seconds}s; exiting", flush=True)
            return 0
        print(f"[worker] no work found; sleeping {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Round-trip work root.")
    parser.add_argument("--config", type=Path, required=True, help="YAML round-trip config.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create calculation directories.")
    init.set_defaults(func=cmd_init)

    gen = sub.add_parser("generate", help="Generate Q-Chem inputs from geoms/*.xyz.")
    gen.add_argument("--overwrite", action="store_true", help="Regenerate existing inputs.")
    gen.set_defaults(func=cmd_generate)

    worker = sub.add_parser("worker", help="Run the polling Q-Chem worker.")
    worker.add_argument("--qchem-command", default="qchem")
    worker.add_argument("--threads", type=int, default=int(os.environ.get("QCHEM_THREADS", "128")))
    worker.add_argument("--poll-seconds", type=int, default=int(os.environ.get("QCHEM_POLL_SECONDS", "30")))
    worker.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=int(os.environ.get("QCHEM_IDLE_TIMEOUT_SECONDS", "1800")),
    )
    worker.add_argument(
        "--stale-lock-seconds",
        type=int,
        default=int(os.environ.get("QCHEM_STALE_LOCK_SECONDS", "172800")),
    )
    worker.add_argument("--once", action="store_true", help="Generate, run at most one input, then exit.")
    worker.add_argument("--qchem-extra-arg", action="append", default=[], help="Additional qchem argument.")
    worker.set_defaults(func=cmd_worker)
    return parser


def cmd_init(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ensure_layout(args.root, cfg)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    generated = generate_inputs(args.root, cfg, overwrite=args.overwrite)
    n_new = sum(1 for item in generated if not item.skipped)
    n_skipped = sum(1 for item in generated if item.skipped)
    print(f"[generate] wrote {n_new} input(s); skipped {n_skipped} existing input(s)")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    return worker_loop(
        args.root,
        cfg,
        qchem_command=args.qchem_command,
        threads=args.threads,
        poll_seconds=args.poll_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
        stale_lock_seconds=args.stale_lock_seconds,
        once=args.once,
        extra_args=args.qchem_extra_arg,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RoundtripError as exc:
        print(f"qchem-roundtrip: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
