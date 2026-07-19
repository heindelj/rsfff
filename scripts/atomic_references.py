"""Compute isolated-atom reference energies at b3lyp/def2-svpd.

The MLIP is trained on *atomization-style* energies: each molecular total energy is
referenced against the sum of the ground-state energies of its constituent atoms. Those
atomic references are open-shell (most neutral atoms have unpaired electrons), so unlike
the closed-shell molecular labels in ``generate_dataset.py`` (rhf reference), the atoms
are computed with an **unrestricted** reference and their correct ground-state spin
multiplicity.

The method/basis match the molecular labels exactly (b3lyp / def2-svpd) so the
subtraction is consistent. Results are written to ``data/atomic_references.json`` as
element symbol -> energy (Hartree); the training pipeline loads this file.

Usage:
    python scripts/atomic_references.py [ELEMENT ...]

With no arguments it computes references for every element that appears in the
labeled molecules (``data/labels/*.extxyz``). Pass explicit symbols (e.g. ``H O``)
to compute only those.
"""

import json
import os
import sys

import numpy as np
import psi4

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LABELS_DIR = os.path.join(REPO, "data", "labels")
OUT_PATH = os.path.join(REPO, "data", "atomic_references.json")

METHOD = "b3lyp"
BASIS = "def2-svpd"

# Ground-state spin multiplicities (2S+1) of the neutral atoms, from Hund's rules /
# the experimental atomic ground terms. Only the elements relevant to this dataset are
# listed; extend as needed.
GROUND_STATE_MULTIPLICITY = {
    "H": 2,    # 2S     (1s^1)
    "He": 1,   # 1S
    "Li": 2,   # 2S
    "Be": 1,   # 1S
    "B": 2,    # 2P
    "C": 3,    # 3P
    "N": 4,    # 4S
    "O": 3,    # 3P
    "F": 2,    # 2P
    "Ne": 1,   # 1S
    "Na": 2,   # 2S
    "Mg": 1,   # 1S
    "Al": 2,   # 2P
    "Si": 3,   # 3P
    "P": 4,    # 4S
    "S": 3,    # 3P
    "Cl": 2,   # 2P
    "Ar": 1,   # 1S
}


def elements_in_labels():
    """Collect the set of element symbols appearing in data/labels/*.extxyz."""
    from ase.io import read

    symbols = set()
    for fname in sorted(os.listdir(LABELS_DIR)):
        if not fname.endswith(".extxyz"):
            continue
        atoms = read(os.path.join(LABELS_DIR, fname), index=0)
        symbols.update(atoms.get_chemical_symbols())
    return sorted(symbols)


def atom_energy(symbol, method=METHOD, basis=BASIS):
    """Open-shell ground-state energy (Hartree) of a single neutral atom."""
    mult = GROUND_STATE_MULTIPLICITY[symbol]
    psi4.core.clean()
    psi4.core.clean_options()
    # charge 0, ground-state multiplicity; c1 symmetry, no reorientation needed for one atom.
    mol = psi4.geometry(f"0 {mult}\n{symbol}\nunits angstrom\nsymmetry c1\n")  # noqa: F841
    psi4.set_options(
        {
            "basis": basis,
            "reference": "uks" if mult > 1 else "rks",
            "e_convergence": 1e-9,
            "d_convergence": 1e-9,
            "maxiter": 200,
        }
    )
    return float(psi4.energy(method))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    psi4.set_memory("8 GB")
    psi4.set_num_threads(os.cpu_count() or 1)
    psi4.set_output_file(os.path.join(LABELS_DIR, "atomic_references_psi4.out"), False)

    args = sys.argv[1:]
    symbols = args if args else elements_in_labels()

    # Merge with any previously computed references so incremental runs accumulate.
    energies, mults = {}, {}
    if os.path.isfile(OUT_PATH):
        prev = json.load(open(OUT_PATH))
        energies.update(prev.get("energies", {}))
        mults.update(prev.get("multiplicities", {}))

    for sym in symbols:
        if sym not in GROUND_STATE_MULTIPLICITY:
            print(f"[{sym}] SKIP: no ground-state multiplicity tabulated", flush=True)
            continue
        mult = GROUND_STATE_MULTIPLICITY[sym]
        print(f"[{sym}] mult={mult}  {METHOD}/{BASIS} (open-shell)...", flush=True)
        e = atom_energy(sym)
        energies[sym] = e
        mults[sym] = mult
        print(f"[{sym}] E = {e:.10f} Ha", flush=True)

    out = {
        "method": METHOD,
        "basis": BASIS,
        "reference": "uks/rks (unrestricted for open-shell atoms)",
        "units": "Hartree",
        "energies": energies,
        "multiplicities": mults,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nwrote {len(energies)} atomic references -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
