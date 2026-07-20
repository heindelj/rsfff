"""Isolated-species reference data for anchoring E(q) at integer charges.

Computes b3lyp/def2-svpd single points (UKS for open shells; same method/basis as
the molecular labels) for the anchor systems of docs/charge_aware_embedding.md:

    H (doublet), H+ (bare proton, E = 0 exactly), O (triplet), O- (doublet),
    OH (doublet), OH- (singlet), H2O (singlet),
    and the vertical H2O+ / H2O- at the optimized neutral H2O geometry
    (vertical IP/EA anchors; optional -- failures are skipped with a warning).

Neutral H2O, OH, and OH- are geometry-optimized first; the charged verticals
reuse the neutral H2O geometry. Results go to ``data/isolated_species.extxyz``
with ``energy`` (Hartree, absolute), ``charge``, and ``multiplicity`` headers --
the format ``rsfff.train.data.load_isolated_species`` reads.

Usage:
    python scripts/isolated_species.py
"""

import os

import numpy as np
import psi4
from ase import Atoms
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_PATH = os.path.join(REPO, "data", "isolated_species.extxyz")

METHOD = "b3lyp"
BASIS = "def2-svpd"
BOHR = 0.52917721067  # Angstrom / bohr, matches psi4's constant closely enough


def _set_options(mult: int):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(
        {
            "basis": BASIS,
            "reference": "uks" if mult > 1 else "rks",
            "e_convergence": 1e-9,
            "d_convergence": 1e-9,
            "maxiter": 300,
        }
    )


def _geometry(symbols, coords, charge, mult):
    lines = [f"{charge} {mult}"]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append(f"{s} {x:.10f} {y:.10f} {z:.10f}")
    lines += ["units angstrom", "symmetry c1", "no_com", "no_reorient"]
    return psi4.geometry("\n".join(lines))


def single_point(symbols, coords, charge, mult) -> float:
    _set_options(mult)
    _geometry(symbols, coords, charge, mult)
    return float(psi4.energy(METHOD))


def optimize(symbols, coords, charge, mult):
    """Optimize; returns (energy, coords_angstrom)."""
    _set_options(mult)
    mol = _geometry(symbols, coords, charge, mult)
    e = float(psi4.optimize(METHOD, molecule=mol))
    xyz = np.asarray(mol.geometry()) * BOHR
    return e, xyz


def main():
    psi4.set_memory("8 GB")
    psi4.set_num_threads(os.cpu_count() or 1)
    psi4.set_output_file(os.path.join(REPO, "data", "isolated_species_psi4.out"), False)

    frames = []

    def add(name, symbols, coords, charge, mult, energy):
        atoms = Atoms(symbols=symbols, positions=np.asarray(coords, dtype=float))
        atoms.info.update(
            {
                "name": name, "energy": energy, "charge": charge,
                "multiplicity": mult, "method": METHOD, "basis": BASIS,
                "units": "atomic",
            }
        )
        frames.append(atoms)
        print(f"[{name:>5s}] charge={charge:+d} mult={mult}  E = {energy:.10f} Ha", flush=True)

    origin = [(0.0, 0.0, 0.0)]

    # --- atoms and atomic ions ---
    add("H", ["H"], origin, 0, 2, single_point(["H"], origin, 0, 2))
    add("H+", ["H"], origin, +1, 1, 0.0)  # bare proton: no electrons, E = 0 exactly
    add("O", ["O"], origin, 0, 3, single_point(["O"], origin, 0, 3))
    add("O-", ["O"], origin, -1, 2, single_point(["O"], origin, -1, 2))

    # --- OH radical and hydroxide, optimized ---
    oh_guess = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)]
    e_oh, xyz_oh = optimize(["O", "H"], oh_guess, 0, 2)
    add("OH", ["O", "H"], xyz_oh, 0, 2, e_oh)
    e_ohm, xyz_ohm = optimize(["O", "H"], oh_guess, -1, 1)
    add("OH-", ["O", "H"], xyz_ohm, -1, 1, e_ohm)

    # --- water, optimized; charged verticals at the neutral geometry ---
    h2o_guess = [
        (0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)
    ]
    e_h2o, xyz_h2o = optimize(["O", "H", "H"], h2o_guess, 0, 1)
    add("H2O", ["O", "H", "H"], xyz_h2o, 0, 1, e_h2o)
    for name, charge in (("H2O+", +1), ("H2O-", -1)):
        try:
            e = single_point(["O", "H", "H"], xyz_h2o, charge, 2)
            add(name, ["O", "H", "H"], xyz_h2o, charge, 2, e)
        except Exception as exc:  # vertical open-shell SCF can be stubborn; optional
            print(f"[{name}] SKIPPED: {exc}", flush=True)

    write(OUT_PATH, frames, format="extxyz")
    print(f"\nwrote {len(frames)} anchor systems -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
