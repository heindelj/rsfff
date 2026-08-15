#!/usr/bin/env python3
"""Stage a qchem_roundtrip upload containing only locally runnable inputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def is_runnable_input(calc_dir: Path, input_path: Path) -> bool:
    stem = input_path.stem
    return not (
        (calc_dir / "outputs" / f"{stem}.out").exists()
        or (calc_dir / "state" / "done" / f"{stem}.json").exists()
        or (calc_dir / "state" / "failed" / f"{stem}.json").exists()
    )


def stage_tree(root: Path, dest: Path) -> tuple[int, int]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for name in ("config.json", "README.md"):
        path = root / name
        if path.exists():
            copy_file(path, dest / name)

    for dirname in ("templates", "scripts"):
        src_dir = root / dirname
        if src_dir.exists():
            shutil.copytree(src_dir, dest / dirname)

    if (root / "aimd" / "geoms").exists():
        shutil.copytree(root / "aimd" / "geoms", dest / "aimd" / "geoms")

    n_inputs = 0
    n_skipped = 0
    for calc_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        inputs_dir = calc_dir / "inputs"
        if not inputs_dir.is_dir():
            continue
        staged_inputs = dest / calc_dir.name / "inputs"
        staged_inputs.mkdir(parents=True, exist_ok=True)
        for keep in inputs_dir.glob(".gitkeep"):
            copy_file(keep, staged_inputs / keep.name)
        for input_path in sorted(inputs_dir.glob("*.in")):
            if is_runnable_input(calc_dir, input_path):
                copy_file(input_path, staged_inputs / input_path.name)
                n_inputs += 1
            else:
                n_skipped += 1
    return n_inputs, n_skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    n_inputs, n_skipped = stage_tree(args.root.resolve(), args.dest.resolve())
    print(f"[stage-runnable-inputs] staged {n_inputs} runnable input(s); skipped {n_skipped} completed input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
