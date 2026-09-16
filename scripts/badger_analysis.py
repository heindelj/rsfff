"""Harmonic frequencies on the film model, and the Badger correlation they imply.

    python scripts/badger_analysis.py checkpoints/water_film_full/best.pt \
        data/kristina_clusters/all_clusters_film_opt.xyz

Badger's rule -- a linear relation between an O-H stretching frequency and the O-H bond length
-- holds tightly for water across every hydrogen-bonding environment, which makes it a sharp
test of a force field: reproducing cluster binding energies says nothing about whether the
*coupling* between bond length and force constant is right. Here it is measured, not assumed:

1.  The model's own isolated water is relaxed. Its symmetric and antisymmetric stretches are
    averaged into a reference frequency ``nu_0``, and its bond length is the reference
    ``r_0``. Both references are properties of the model, so the correlation below is
    internally consistent even if the model's absolute monomer frequency is off.
2.  Every structure in the input file gets a Hessian and a normal-mode analysis.
3.  Each mode below ``nu_0`` is displaced by a small step and the resulting change in every
    O-H bond length is measured. A mode that is an O-H stretch moves one or two bonds and
    nothing else, and the bond it moves most is the bond that mode belongs to. That
    assignment is what pairs a frequency with a length.

The discriminator is written out rather than hidden: ``oh_character`` is the sum of squared
bond-length changes under a unit-norm Cartesian step (near 1 for a pure O-H stretch, small for
a bend or a libration) and ``localization`` is the fraction of that carried by the single
assigned bond (near 1 for a local mode, near 0.5 for a symmetric pair). Filter the output CSV
on them rather than trusting the default threshold.

Local modes, which is the analysis to trust
-------------------------------------------
Normal modes stop being the right object above about four waters: the O-H stretches of a ring
or a cage couple into delocalized combinations, and pairing one such mode with one bond length
is meaningless no matter how good the potential is. The ``_local_modes.csv`` output avoids that
entirely. For each O-H in turn, every *other* hydrogen is isotopically substituted (deuterium by
default, ``--heavy-mass`` to change it), which drops the neighbouring stretches out of resonance
and leaves a genuinely local oscillator -- the computational form of the HOD-in-D2O experiment
that established Badger's rule in the first place. Masses do not enter the potential, so all of
this is re-diagonalization of the Hessian that was already computed: the extra cost is
negligible and the resulting correlation covers every O-H rather than the localized subset.

Each O-H also carries a hydrogen-bond descriptor (``is_donor``, and the H...O distance and
O-H...O angle to its acceptor, or to the nearest intermolecular oxygen when it donates to
none), so the donor and free populations can be fitted separately -- they are two different
regimes of the same rule and lumping them together flatters the fit.

The input is any plain XYZ of water structures; it is *assumed to be relaxed on the same
model*, since a Hessian at a non-stationary geometry has no harmonic meaning. The maximum
residual force is reported per structure so a bad input announces itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from rsfff.md.film_driver import (
    FilmPotential,
    hessian,
    hydrogen_bonds,
    load_film_model,
    oh_bond_list,
    optimize,
    water_fragment_index,
)
from rsfff.md.vibrations import bond_displacements, local_mode, vibrational_analysis
from rsfff.md.xyz import read_xyz


def monomer_reference(model, args) -> dict:
    """Relax an isolated water and return its Badger reference point.

    The two highest frequencies of a water monomer are the symmetric and antisymmetric O-H
    stretches; their mean is the reference against which a hydrogen-bonded stretch is shifted.
    Using the mean (rather than either one) is the convention that makes a *local* O-H mode
    the right comparison: the two monomer stretches are the symmetric/antisymmetric
    combinations of two equivalent local oscillators, and their mean is the local-mode
    frequency with the intramolecular coupling averaged out.
    """
    positions = np.array([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9273, 0.0]])
    numbers = np.array([8, 1, 1])
    fragments = water_fragment_index(positions, numbers)
    potential = FilmPotential(model, numbers, fragments, with_induction=args.induction)
    result = optimize(potential, positions, gtol=args.gtol)
    matrix = hessian(potential, result.positions, delta=args.delta, chunk=args.chunk)
    masses = np.array([15.999, 1.008, 1.008])
    modes = vibrational_analysis(matrix, result.positions, masses)
    bend, sym, antisym = modes.frequencies

    # The local reference: HOD, i.e. this same water with one hydrogen substituted. It is the
    # right zero for a decoupled cluster O-H, where the mean of the monomer's symmetric and
    # antisymmetric stretches is the right zero for a coupled normal mode. Keeping both means
    # each correlation is referenced to a frequency measured the same way as its data.
    local = local_mode(
        matrix, result.positions, masses, oh_bond_list(fragments, numbers), 0,
        heavy_mass=args.heavy_mass, step=args.step,
    )
    r = np.linalg.norm(result.positions[1:] - result.positions[0], axis=1)
    return {
        "energy_hartree": result.energy,
        "max_force": result.max_force,
        "r_oh_angstrom": float(r.mean()),
        "theta_degrees": float(np.degrees(np.arccos(
            np.dot(*(result.positions[1:] - result.positions[0])) / (r[0] * r[1])
        ))),
        "bend_cm1": float(bend),
        "symmetric_stretch_cm1": float(sym),
        "antisymmetric_stretch_cm1": float(antisym),
        "reference_frequency_cm1": float(0.5 * (sym + antisym)),
        "heavy_mass_amu": float(args.heavy_mass),
        "local_reference_frequency_cm1": local.frequency,
        "local_reference_localization": local.localization,
    }


def linear_fit(dr, dnu) -> dict:
    """The Badger fit itself: d(nu) against d(r_OH), with the scatter around it."""
    dr, dnu = np.asarray(dr), np.asarray(dnu)
    if dr.size < 3:
        return {"n_points": int(dr.size)}
    slope, intercept = np.polyfit(dr, dnu, 1)
    residual = dnu - (slope * dr + intercept)
    return {
        "n_points": int(dr.size),
        "slope_cm1_per_angstrom": float(slope),
        "intercept_cm1": float(intercept),
        "r_squared": float(1.0 - residual.var() / dnu.var()),
        "pearson_r": float(np.corrcoef(dr, dnu)[0, 1]),
        "rms_residual_cm1": float(np.sqrt((residual**2).mean())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("xyz", type=Path, help="structures relaxed on the same model")
    parser.add_argument("--out-frequencies", type=Path, default=None)
    parser.add_argument("--out-badger", type=Path, default=None)
    parser.add_argument("--out-local-modes", type=Path, default=None)
    parser.add_argument("--out-reference", type=Path, default=None)
    parser.add_argument("--delta", type=float, default=1e-3,
                        help="finite-difference step for the Hessian, Angstrom")
    parser.add_argument("--chunk", type=int, default=16,
                        help="displaced geometries evaluated per forward pass")
    parser.add_argument("--step", type=float, default=0.02,
                        help="step along a normal mode for the bond assignment, Angstrom")
    parser.add_argument("--stretch-threshold", type=float, default=0.5,
                        help="minimum oh_character for a mode to count as an O-H stretch")
    parser.add_argument("--all-stretch-modes", action="store_true",
                        help="keep O-H stretches above the reference frequency too "
                             "(free O-H modes can sit slightly blue of the monomer)")
    parser.add_argument("--heavy-mass", type=float, default=2.014,
                        help="mass (amu) given to every hydrogen but the one being localized; "
                             "2.014 is deuterium, 1e5 decouples completely")
    parser.add_argument("--hbond-distance", type=float, default=2.5,
                        help="maximum H...O distance for an O-H to count as a donor, Angstrom")
    parser.add_argument("--hbond-angle", type=float, default=130.0,
                        help="minimum O-H...O angle for an O-H to count as a donor, degrees")
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument("--no-induction", dest="induction", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stem = args.xyz.with_suffix("")
    out_freq = args.out_frequencies or Path(f"{stem}_frequencies.csv")
    out_badger = args.out_badger or Path(f"{stem}_badger.csv")
    out_local = args.out_local_modes or Path(f"{stem}_local_modes.csv")
    out_reference = args.out_reference or Path(f"{stem}_badger_reference.json")

    model, _ = load_film_model(args.checkpoint)
    reference = monomer_reference(model, args)
    nu_0 = reference["reference_frequency_cm1"]
    r_0 = reference["r_oh_angstrom"]
    print(
        "isolated water on this model:\n"
        f"  r_OH        = {r_0:.5f} A      theta = {reference['theta_degrees']:.3f} deg\n"
        f"  bend        = {reference['bend_cm1']:9.2f} cm^-1\n"
        f"  sym / anti  = {reference['symmetric_stretch_cm1']:9.2f} / "
        f"{reference['antisymmetric_stretch_cm1']:.2f} cm^-1\n"
        f"  reference   = {nu_0:9.2f} cm^-1  (mean of the two stretches)\n"
        f"  HOD local   = {reference['local_reference_frequency_cm1']:9.2f} cm^-1  "
        f"(one H at {args.heavy_mass:g} amu; localization "
        f"{reference['local_reference_localization']:.4f})\n"
    )

    nu_local_0 = reference["local_reference_frequency_cm1"]
    frames = read_xyz(args.xyz)[: args.limit]
    freq_rows, badger_rows, local_rows = [], [], []
    n_above = 0

    for index, frame in enumerate(frames):
        numbers = frame.numbers
        fragments = water_fragment_index(frame.positions, numbers)
        bonds = oh_bond_list(fragments, numbers)
        label = frame.comment.split("|")[0].lstrip("#").strip() or f"frame_{index}"
        potential = FilmPotential(model, numbers, fragments, with_induction=args.induction)

        start = time.time()
        _, gradient = potential.energy_and_gradient(frame.positions)
        matrix = hessian(potential, frame.positions, delta=args.delta, chunk=args.chunk)
        modes = vibrational_analysis(matrix, frame.positions, frame.masses)
        wall = time.time() - start

        r_oh = np.linalg.norm(
            frame.positions[bonds[:, 1]] - frame.positions[bonds[:, 0]], axis=1
        )
        descriptor = hydrogen_bonds(
            frame.positions, bonds, fragments, numbers,
            max_distance=args.hbond_distance, min_angle=args.hbond_angle,
        )
        n_stretch = 0
        for mode_index, (frequency, mode) in enumerate(
            zip(modes.frequencies, modes.modes)
        ):
            unit = mode / np.linalg.norm(mode)
            delta_r = bond_displacements(frame.positions, bonds, unit, step=args.step)
            weight = (delta_r / args.step) ** 2
            character = float(weight.sum())
            best = int(weight.argmax())
            localization = float(weight[best] / weight.sum()) if character > 0 else 0.0
            is_stretch = character >= args.stretch_threshold

            freq_rows.append({
                "structure": index, "label": label, "mode": mode_index,
                "frequency_cm1": float(frequency),
                "reduced_mass_amu": float(modes.reduced_mass[mode_index]),
                "oh_character": character,
                "localization": localization,
                "is_oh_stretch": int(is_stretch),
                "assigned_o": int(bonds[best, 0]) if is_stretch else -1,
                "assigned_h": int(bonds[best, 1]) if is_stretch else -1,
            })
            if not is_stretch:
                continue
            if frequency >= nu_0 and not args.all_stretch_modes:
                n_above += 1
                continue
            n_stretch += 1
            o_index, h_index = int(bonds[best, 0]), int(bonds[best, 1])
            badger_rows.append({
                "structure": index, "label": label, "n_waters": int(fragments.max()) + 1,
                "mode": mode_index,
                "frequency_cm1": float(frequency),
                "delta_frequency_cm1": float(frequency) - nu_0,
                "o_index": o_index, "h_index": h_index,
                "r_oh_angstrom": float(r_oh[best]),
                "delta_r_oh_angstrom": float(r_oh[best]) - r_0,
                "h_bond_o_distance_angstrom": float(descriptor["distance"][best]),
                "oh_character": character,
                "localization": localization,
                "delta_r_step_angstrom": float(delta_r[best]),
            })

        # Every O-H, localized by isotopic substitution. No new energy evaluations: the
        # Hessian above is mass-independent and only the diagonalization is repeated.
        for bond_index, (o_index, h_index) in enumerate(bonds):
            local = local_mode(
                matrix, frame.positions, frame.masses, bonds, bond_index,
                heavy_mass=args.heavy_mass, step=args.step,
            )
            local_rows.append({
                "structure": index, "label": label,
                "n_waters": int(fragments.max()) + 1,
                "o_index": int(o_index), "h_index": int(h_index),
                "is_donor": int(descriptor["is_donor"][bond_index]),
                "acceptor_o": int(descriptor["acceptor"][bond_index]),
                "h_bond_o_distance_angstrom": float(descriptor["distance"][bond_index]),
                "h_bond_angle_degrees": float(descriptor["angle"][bond_index]),
                "r_oh_angstrom": float(r_oh[bond_index]),
                "delta_r_oh_angstrom": float(r_oh[bond_index]) - r_0,
                "local_frequency_cm1": local.frequency,
                "delta_frequency_cm1": local.frequency - nu_local_0,
                "localization": local.localization,
                "oh_character": local.character,
                "mode": local.mode_index,
            })

        n_donors = int(descriptor["is_donor"].sum())
        imaginary = int((modes.frequencies < 0).sum())
        print(
            f"[{index + 1:3d}/{len(frames)}] {label:<38s} "
            f"{modes.frequencies.size:4d} modes  {n_stretch:3d} O-H stretches  "
            f"{n_donors:3d}/{len(bonds):3d} donors  imag {imaginary:2d}  "
            f"max|F| {np.abs(gradient).max():.1e}  {wall:6.1f} s"
        )

    for path, table in ((out_freq, freq_rows), (out_badger, badger_rows),
                        (out_local, local_rows)):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    # The correlations, on the rows just written.
    def columns(rows, mask=None):
        keep = [row for row in rows if mask is None or mask(row)]
        return (np.array([row["delta_r_oh_angstrom"] for row in keep]),
                np.array([row["delta_frequency_cm1"] for row in keep]))

    reference["badger_fit"] = {
        **linear_fit(*columns(badger_rows)),
        "modes_above_reference_excluded": n_above,
    }
    reference["local_mode_fit"] = {
        "all": linear_fit(*columns(local_rows)),
        "donors": linear_fit(*columns(local_rows, lambda row: row["is_donor"])),
        "free": linear_fit(*columns(local_rows, lambda row: not row["is_donor"])),
    }
    out_reference.write_text(json.dumps(reference, indent=2) + "\n")

    def report(name, fit):
        if fit.get("n_points", 0) < 3:
            print(f"  {name:<28s} too few points ({fit.get('n_points', 0)})")
            return
        print(
            f"  {name:<28s} n = {fit['n_points']:4d}   "
            f"d(nu) = {fit['slope_cm1_per_angstrom']:9.1f} * d(r_OH) "
            f"{fit['intercept_cm1']:+8.1f}    R^2 = {fit['r_squared']:.4f}   "
            f"RMS = {fit['rms_residual_cm1']:6.1f} cm^-1"
        )

    print("\nBadger correlation, normal modes assigned by displacement:")
    report("all assigned stretches", reference["badger_fit"])
    print(f"  ({n_above} O-H stretch modes at or above the reference frequency were "
          f"excluded; pass --all-stretch-modes to keep them)")
    print(f"\nBadger correlation, local modes (every other H at {args.heavy_mass:g} amu), "
          f"referenced to the HOD monomer at {nu_local_0:.2f} cm^-1:")
    for name in ("all", "donors", "free"):
        report(f"{name} O-H", reference["local_mode_fit"][name])
    worst = min(row["localization"] for row in local_rows)
    print(f"  (lowest local-mode localization over all O-H: {worst:.4f}; "
          f"far below 1 means the substitution did not decouple)")
    print(f"\nwrote {out_freq}\nwrote {out_badger}\nwrote {out_local}\n"
          f"wrote {out_reference}")


if __name__ == "__main__":
    main()
