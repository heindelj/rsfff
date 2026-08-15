#!/usr/bin/env python3
"""Report which qchem_roundtrip inputs are runnable versus already completed."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    for calc_dir in sorted(path for path in root.iterdir() if path.is_dir() and (path / "inputs").is_dir()):
        inputs = {path.stem for path in (calc_dir / "inputs").glob("*.in")}
        outputs = {path.stem for path in (calc_dir / "outputs").glob("*.out")}
        done = {path.stem for path in (calc_dir / "state" / "done").glob("*.json")}
        failed = {path.stem for path in (calc_dir / "state" / "failed").glob("*.json")}
        completed = outputs | done | failed
        runnable = inputs - completed
        print(
            f"{calc_dir.name:14s} "
            f"inputs={len(inputs):6d} outputs={len(outputs):6d} "
            f"done={len(done):6d} failed={len(failed):6d} "
            f"runnable={len(runnable):6d} skipped_by_worker={len(inputs & completed):6d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
