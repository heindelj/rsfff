#!/usr/bin/env python3
"""Optimize benchmark water clusters and compare them with MP2/AVTZ references."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS, FIRE

from benchmark_utils import (
    DEFAULT_CHECKPOINT,
    RSFFFCalculator,
    attach_singlepoint_results,
    default_results_dir,
    load_water_model,
    model_report,
    mp2_binding_energy_kcal_mol,
    reference_energy_hartree,
    rmsd_kabsch,
    structure_paths,
    write_json,
)


BASE_COLUMNS = [
    "structure",
    "frame",
    "n_atoms",
    "n_waters",
    "converged",
    "steps",
    "rmsd_angstrom",
    "initial_energy_hartree",
    "optimized_energy_hartree",
    "initial_binding_energy_kcal_mol",
    "optimized_binding_energy_kcal_mol",
    "mp2_reference_energy_hartree",
    "mp2_binding_energy_kcal_mol",
    "final_max_force_ev_per_angstrom",
    "xyz_written",
    "xyz_skip_reason",
    "optimized_xyz",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("structures", nargs="*", help="XYZ files to optimize")
    ap.add_argument("--config", default="configs/water_staged.yaml")
    ap.add_argument("--stage", default="full", help="config stage used to build the model")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--checkpoint-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-dir", default=str(default_results_dir()))
    ap.add_argument("--fmax", type=float, default=0.01, help="ASE force threshold, eV/A")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--optimizer", choices=("bfgs", "fire"), default="bfgs")
    ap.add_argument(
        "--maxstep",
        type=float,
        default=0.3,
        help="maximum optimizer atom displacement per step, Angstrom",
    )
    ap.add_argument(
        "--mp2-water-energy-hartree",
        type=float,
        default=None,
        help="MP2/AVTZ optimized water monomer energy for reference binding energies",
    )
    ap.add_argument(
        "--write-force-threshold",
        type=float,
        default=1.0,
        help="do not write optimized XYZ when final max force exceeds this eV/A value",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    opt_dir = results_dir / "optimized_structures"
    log_dir = results_dir / "logs"
    opt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_water_model(
        args.config,
        stage=args.stage,
        checkpoint_path=args.checkpoint,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
    )
    calc = RSFFFCalculator(bundle)

    rows = []
    for path in structure_paths(args.structures):
        frames = read(path, index=":")
        multi_frame = len(frames) > 1
        for frame_idx, reference in enumerate(frames):
            tag = f"{path.stem}_frame{frame_idx:04d}" if multi_frame else path.stem
            rows.append(
                optimize_frame(
                    reference,
                    path_name=path.name,
                    frame_idx=frame_idx,
                    tag=tag,
                    args=args,
                    bundle=bundle,
                    calc=calc,
                    log_dir=log_dir,
                    opt_dir=opt_dir,
                )
            )

    csv_path = results_dir / "optimization_rmsd.csv"
    extra_columns = sorted({key for row in rows for key in row} - set(BASE_COLUMNS))
    columns = BASE_COLUMNS + extra_columns
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        results_dir / "optimization_rmsd.json",
        {
            "config": str(bundle.config_path),
            "config_stage": bundle.config_stage,
            "checkpoint": str(bundle.checkpoint_path),
            "device": str(bundle.device),
            "mp2_water_energy_hartree": args.mp2_water_energy_hartree,
            "structures": rows,
        },
    )
    print(f"wrote {csv_path}")


def optimize_frame(reference, *, path_name, frame_idx, tag, args, bundle, calc, log_dir, opt_dir):
    optimized = reference.copy()
    optimized.calc = calc
    initial_report = model_report(optimized, bundle)

    logfile = log_dir / f"{tag}_opt.log"
    trajectory = log_dir / f"{tag}_opt.traj"
    opt_cls = BFGS if args.optimizer == "bfgs" else FIRE
    opt = opt_cls(
        optimized,
        logfile=str(logfile),
        trajectory=str(trajectory),
        maxstep=args.maxstep,
    )
    opt.run(fmax=args.fmax, steps=args.steps)

    final_report = model_report(optimized, bundle, with_forces=True)
    rmsd = rmsd_kabsch(reference, optimized)
    xyz_path = opt_dir / f"{tag}_rsfff_opt.xyz"
    max_force = final_report["max_force_ev_per_angstrom"]
    xyz_written = max_force <= args.write_force_threshold
    skip_reason = ""
    if xyz_written:
        optimized.info.clear()
        attach_singlepoint_results(optimized, final_report)
        write(xyz_path, optimized, format="extxyz")
        optimized_xyz = str(xyz_path)
    else:
        skip_reason = (
            f"final max force {max_force:.6g} eV/A exceeds "
            f"{args.write_force_threshold:.6g} eV/A"
        )
        xyz_path.unlink(missing_ok=True)
        optimized_xyz = ""

    row = {
        "structure": path_name,
        "frame": frame_idx,
        "n_atoms": len(reference),
        "n_waters": len(reference) // 3,
        "converged": bool(opt.converged()),
        "steps": int(opt.get_number_of_steps()),
        "rmsd_angstrom": rmsd,
        "initial_energy_hartree": initial_report["total_energy_hartree"],
        "optimized_energy_hartree": final_report["total_energy_hartree"],
        "initial_binding_energy_kcal_mol": initial_report[
            "binding_energy_kcal_mol"
        ],
        "optimized_binding_energy_kcal_mol": final_report[
            "binding_energy_kcal_mol"
        ],
        "mp2_reference_energy_hartree": reference_energy_hartree(reference),
        "mp2_binding_energy_kcal_mol": (
            None
            if args.mp2_water_energy_hartree is None
            else mp2_binding_energy_kcal_mol(reference, args.mp2_water_energy_hartree)
        ),
        "final_max_force_ev_per_angstrom": max_force,
        "xyz_written": xyz_written,
        "xyz_skip_reason": skip_reason,
        "optimized_xyz": optimized_xyz,
    }
    for prefix, report in (("initial", initial_report), ("final", final_report)):
        for name, value in report["eda_kj_mol"].items():
            row[f"{prefix}_eda_{name}_kj_mol"] = value
    print(
        f"{path_name}[{frame_idx}]: RMSD={rmsd:.6f} A, "
        f"steps={row['steps']}, converged={row['converged']}, "
        f"max_force={max_force:.6f} eV/A"
    )
    return row


if __name__ == "__main__":
    main()
