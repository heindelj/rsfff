"""Relax a set of clusters on a trained film model and score them against the references.

    python scripts/optimize_clusters.py checkpoints/water_film_full/best.pt \
        data/kristina_clusters/all_clusters.xyz \
        --out-xyz data/kristina_clusters/all_clusters_film_opt.xyz \
        --out-csv data/kristina_clusters/all_clusters_film_opt.csv

Three numbers per cluster, and each says something different:

RMSD to the reference
    after a Kabsch superposition, so it measures shape and not placement. Reported for all
    atoms and for the oxygens alone -- the O-framework RMSD is the one that says whether the
    hydrogen-bond topology survived relaxation, while the all-atom number also picks up
    rotations of individual waters about their own axes.
binding energy
    ``E_cluster - n * E_monomer`` with ``E_monomer`` the model's *own* relaxed water, so the
    deformation energy of the monomers is inside the binding energy exactly as it is in a
    reference calculation. It is computed at both the reference and the relaxed geometry; the
    gap between them is the model's relaxation energy and is the honest measure of how far the
    optimization had to travel.
max force at the reference geometry
    a per-cluster residual that does not depend on the optimizer having converged. A model
    that reproduces the reference minima has small forces there.

Two force thresholds, deliberately: ``--gtol`` is what the optimizer is *driven* to and is set
below what the model can actually deliver, because pushing until progress stops is free and
leaves the tightest geometry available for a Hessian. ``--force-tol`` is what the ``converged``
column is *judged* against. The gap matters above ~10 waters, where the induction solve's CG
tolerance puts a noise floor of a few 1e-7 Ha/Angstrom on the gradient and no amount of
restarting gets below it; a few 1e-7 Ha/Angstrom shifts a harmonic frequency by far less than
1 cm^-1, so it is a floor worth reporting and not worth fighting.

The fragmentation is O + nearest-two-H, fixed at the input geometry (see
:mod:`rsfff.md.film_driver`).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from rsfff.md.film_driver import (
    FilmPotential,
    load_film_model,
    optimize,
    water_fragment_index,
)
from rsfff.md.xyz import Frame, read_xyz, write_xyz

HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_KJ = 2625.4996394799


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray, weights=None):
    """RMSD after the optimal rigid superposition of ``mobile`` onto ``target``.

    Standard Kabsch: shift both to their (weighted) centroid, take the SVD of the covariance,
    and flip the last singular direction when the rotation would be a reflection. Returns
    ``(rmsd, rotated_mobile)`` with the rotated copy placed on the target's centroid so it can
    be written out directly.
    """
    weights = np.ones(len(mobile)) if weights is None else np.asarray(weights, dtype=float)
    w = (weights / weights.sum())[:, None]
    p = mobile - (w * mobile).sum(0)
    q = target - (w * target).sum(0)

    u, _, vt = np.linalg.svd((w * p).T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = p @ rotation
    rmsd = float(np.sqrt((w[:, 0] * ((aligned - q) ** 2).sum(1)).sum()))
    return rmsd, aligned + (w * target).sum(0)


def relaxed_monomer(model, args):
    """The model's own relaxed water: energy, geometry, and the internal coordinates."""
    positions = np.array([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9273, 0.0]])
    numbers = np.array([8, 1, 1])
    potential = FilmPotential(
        model, numbers, water_fragment_index(positions, numbers),
        with_induction=args.induction,
    )
    result = optimize(potential, positions, gtol=args.gtol, max_iter=args.max_iter)
    r = np.linalg.norm(result.positions[1:] - result.positions[0], axis=1)
    cos_theta = np.dot(*(result.positions[1:] - result.positions[0])) / (r[0] * r[1])
    return result, float(r.mean()), float(np.degrees(np.arccos(cos_theta)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("xyz", type=Path, help="reference structures (plain XYZ, any sizes)")
    parser.add_argument("--out-xyz", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--gtol", type=float, default=1e-7,
                        help="max |dE/dR| the optimizer is driven to, Hartree/Angstrom")
    parser.add_argument("--force-tol", type=float, default=1e-6,
                        help="max |dE/dR| the `converged` column is judged against")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--no-induction", dest="induction", action="store_false",
                        help="drop the coupled induction solve (a pairwise-only relaxation)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    args = parser.parse_args()

    stem = args.xyz.with_suffix("")
    out_xyz = args.out_xyz or Path(f"{stem}_film_opt.xyz")
    out_csv = args.out_csv or Path(f"{stem}_film_opt.csv")

    model, _ = load_film_model(args.checkpoint)
    frames = read_xyz(args.xyz)[: args.limit]
    print(f"{len(frames)} frames from {args.xyz}")

    monomer, r_oh, theta = relaxed_monomer(model, args)
    print(
        f"relaxed monomer: E = {monomer.energy:.8f} Ha, r_OH = {r_oh:.5f} A, "
        f"theta = {theta:.3f} deg, max|g| = {monomer.max_force:.2e}"
    )

    rows, optimized = [], []
    for index, frame in enumerate(frames):
        numbers = frame.numbers
        fragments = water_fragment_index(frame.positions, numbers)
        n_waters = int(fragments.max()) + 1
        potential = FilmPotential(
            model, numbers, fragments, with_induction=args.induction
        )

        e_ref, g_ref = potential.energy_and_gradient(frame.positions)
        start = time.time()
        result = optimize(potential, frame.positions, gtol=args.gtol,
                          max_iter=args.max_iter)
        wall = time.time() - start

        rmsd_all, aligned = kabsch_rmsd(result.positions, frame.positions)
        is_o = numbers == 8
        rmsd_o, _ = kabsch_rmsd(result.positions[is_o], frame.positions[is_o])

        binding = result.energy - n_waters * monomer.energy
        binding_ref = float(e_ref[0]) - n_waters * monomer.energy
        label = frame.comment.lstrip("#").strip() or f"frame_{index}"
        rows.append({
            "index": index,
            "label": label,
            "n_atoms": frame.n_atoms,
            "n_waters": n_waters,
            "rmsd_all_angstrom": rmsd_all,
            "rmsd_oxygen_angstrom": rmsd_o,
            "binding_energy_hartree": binding,
            "binding_energy_kcal_mol": binding * HARTREE_TO_KCAL,
            "binding_energy_kj_mol": binding * HARTREE_TO_KJ,
            "binding_per_water_kcal_mol": binding * HARTREE_TO_KCAL / n_waters,
            "binding_energy_ref_geom_kcal_mol": binding_ref * HARTREE_TO_KCAL,
            "relaxation_energy_kcal_mol": (result.energy - float(e_ref[0]))
                                          * HARTREE_TO_KCAL,
            "energy_opt_hartree": result.energy,
            "energy_ref_geom_hartree": float(e_ref[0]),
            "max_force_ref_geom": float(np.abs(g_ref).max()),
            "max_force_opt": result.max_force,
            "rms_force_opt": result.rms_force,
            "converged": int(result.max_force <= args.force_tol),
            "n_iterations": result.n_iterations,
            "n_evaluations": result.n_evaluations,
            "seconds": wall,
        })
        optimized.append(Frame(
            symbols=list(frame.symbols),
            positions=aligned,
            comment=(f"{label} | E = {result.energy:.10f} Ha | "
                     f"Eb = {binding * HARTREE_TO_KCAL:.4f} kcal/mol | "
                     f"RMSD = {rmsd_all:.4f} A | max|F| = {result.max_force:.2e}"),
        ))
        print(
            f"[{index + 1:3d}/{len(frames)}] {label:<38s} n={n_waters:3d}  "
            f"RMSD {rmsd_all:6.3f} (O {rmsd_o:6.3f})  "
            f"Eb {binding * HARTREE_TO_KCAL:10.3f} kcal/mol  "
            f"max|F| {result.max_force:.1e}  {wall:6.1f} s"
        )

    write_xyz(out_xyz, optimized)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_xyz}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
