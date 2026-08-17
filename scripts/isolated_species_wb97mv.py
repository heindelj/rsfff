"""Isolated-species reference data for the water/hydronium/hydroxide anchor set.

The wB97M-V/def2-tzvpd analogue of ``scripts/isolated_species.py`` (b3lyp/def2-svpd
via psi4), built on :mod:`rsfff.qcgen` so it shares the pyscf/gpu4pyscf backend with
the rest of the reference-data pipeline.

Species (all closed shell, so RKS throughout -- no UKS + NLC corner cases):

    H2O (singlet), H3O+ (singlet, +1), OH- (singlet, -1)

Every geometry is optimized at the same level of theory before the energy is taken.
Results go to ``data/isolated_species_wb97mv.extxyz`` with ``energy`` (Hartree,
absolute), ``charge``, and ``multiplicity`` headers -- the format
``rsfff.train.data.load_isolated_species`` reads.

Usage:
    python scripts/isolated_species_wb97mv.py
"""

import os

import numpy as np
from ase import Atoms
from ase.io import write

from rsfff.qcgen.compute import optimize_geometry

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_PATH = os.path.join(REPO, "data", "isolated_species_wb97mv.extxyz")

METHOD = "wB97M-V"
BASIS = "def2-tzvpd"

# (name, symbols, guess geometry in Angstrom, charge, multiplicity)
SPECIES = [
    (
        "H2O",
        ["O", "H", "H"],
        [(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)],
        0,
        1,
    ),
    (
        # C3v pyramidal guess: O apex, three H on a ring below it.
        "H3O+",
        ["O", "H", "H", "H"],
        [
            (0.0, 0.0, 0.0837),
            (0.0, 0.9376, -0.2231),
            (0.8120, -0.4688, -0.2231),
            (-0.8120, -0.4688, -0.2231),
        ],
        1,
        1,
    ),
    ("OH-", ["O", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.97)], -1, 1),
]


def main():
    frames = []
    for name, symbols, guess, charge, mult in SPECIES:
        spin = mult - 1  # pyscf convention: Nalpha - Nbeta
        coords, energy = optimize_geometry(
            symbols, np.asarray(guess, dtype=float), charge, spin, METHOD, BASIS
        )
        atoms = Atoms(symbols=symbols, positions=np.asarray(coords, dtype=float))
        atoms.info.update(
            {
                "name": name, "energy": float(energy), "charge": charge,
                "multiplicity": mult, "method": METHOD, "basis": BASIS,
                "units": "atomic",
            }
        )
        frames.append(atoms)
        print(
            f"[{name:>5s}] charge={charge:+d} mult={mult}  "
            f"E = {energy:.10f} Ha",
            flush=True,
        )

    write(OUT_PATH, frames, format="extxyz")
    print(f"\nwrote {len(frames)} anchor systems -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
