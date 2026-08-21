#!/usr/bin/env python3
"""Compute O-H bond lengths and harmonic vibrational frequencies for optimized clusters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations

from benchmark_utils import (
    DEFAULT_CHECKPOINT,
    RSFFFCalculator,
    default_results_dir,
    load_water_model,
    oh_bond_lengths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "structures",
        nargs="*",
        help=(
            "XYZ files to analyze. Defaults to optimized XYZs from optimize_rmsd.py."
        ),
    )
    # Recorded in the result JSON, but no longer load-bearing: the model is rebuilt from the
    # checkpoint's own embedded config. See benchmark_utils.load_water_model.
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default=None, help="recorded in the results; not used to build")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--checkpoint-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-dir", default=str(default_results_dir()))
    ap.add_argument("--delta", type=float, default=0.01, help="finite-difference step, A")
    ap.add_argument("--indices", default=None, help="comma-separated atom indices for testing")
    ap.add_argument(
        "--keep-vib-files",
        action="store_true",
        help="keep ASE finite-difference files under results/vibrations",
    )
    return ap.parse_args()


def default_frequency_structures(results_dir: Path) -> list[Path]:
    opt_dir = results_dir / "optimized_structures"
    optimized = sorted(opt_dir.glob("*_rsfff_opt.xyz"))
    if not optimized:
        raise FileNotFoundError(
            f"no optimized structures found in {opt_dir}; run optimize_rmsd.py first "
            "or pass XYZ paths explicitly"
        )
    return optimized


def parse_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    vib_root = results_dir / "vibrations"
    vib_root.mkdir(parents=True, exist_ok=True)

    paths = (
        [Path(p) for p in args.structures]
        if args.structures
        else default_frequency_structures(results_dir)
    )
    bundle = load_water_model(
        args.config,
        stage=args.stage,
        checkpoint_path=args.checkpoint,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
    )
    calc = RSFFFCalculator(bundle)

    all_rows = []
    bond_rows = []
    for path in paths:
        atoms = read(path)
        atoms.calc = calc
        name = path.stem.replace("_rsfff_opt", "")
        vib_name = str(vib_root / name)
        vib = Vibrations(
            atoms, indices=parse_indices(args.indices), name=vib_name, delta=args.delta
        )
        vib.clean()
        vib.run()
        freqs = np.asarray(vib.get_frequencies(), dtype=complex)
        freq_rows = [
            {
                "mode": i,
                "frequency_cm^-1_real": float(freq.real),
                "frequency_cm^-1_imag": float(freq.imag),
            }
            for i, freq in enumerate(freqs)
        ]
        data = {
            "structure": path.name,
            "source_xyz": str(path),
            "oh_bond_lengths": oh_bond_lengths(atoms),
            "frequencies_cm^-1": freq_rows,
        }
        write_json(results_dir / f"{name}_vibrations.json", data)

        for bond in data["oh_bond_lengths"]:
            bond_rows.append({"structure": path.name, **bond})
        for freq in freq_rows:
            all_rows.append({"structure": path.name, **freq})
        if not args.keep_vib_files:
            vib.clean()
        print(f"{path.name}: {len(freq_rows)} modes")

    with (results_dir / "oh_bond_lengths.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["structure", "oxygen_index", "hydrogen_index", "length_angstrom"],
        )
        writer.writeheader()
        writer.writerows(bond_rows)

    with (results_dir / "vibrational_frequencies.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "structure",
                "mode",
                "frequency_cm^-1_real",
                "frequency_cm^-1_imag",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    write_json(
        results_dir / "vibrational_frequencies.json",
        {
            "config": str(bundle.config_path),
            "config_stage": bundle.config_stage,
            "checkpoint": str(bundle.checkpoint_path),
            "device": str(bundle.device),
            "structures": [str(p) for p in paths],
        },
    )
    print(f"wrote {results_dir / 'vibrational_frequencies.csv'}")


if __name__ == "__main__":
    main()
