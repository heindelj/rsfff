"""Compute isolated-atom reference energies.

The MLIP is trained on *atomization-style* energies: each molecular total energy is
referenced against the sum of the ground-state energies of its constituent atoms. Those
atomic references are open-shell (most neutral atoms have unpaired electrons), so unlike
the closed-shell molecular labels in ``generate_dataset.py`` (rhf reference), the atoms
are computed with an **unrestricted** reference and their correct ground-state spin
multiplicity.

**The method and basis must match the labels being referenced**, or the difference is a
constant offset the model has to absorb. Two label sets now exist:

    b3lyp/def2-svpd     data/labels/*.extxyz     -> data/atomic_references.json
    wB97M-V/def2-TZVPD  data/wb97mv_tzvpd/*.xyz  -> data/atomic_references_wb97mv_tzvpd.json

Usage::

    python scripts/atomic_references.py [ELEMENT ...]
    python scripts/atomic_references.py --backend pyscf --method wB97M-V \
        --basis def2-TZVPD --out data/atomic_references_wb97mv_tzvpd.json H O

With no elements it computes references for every element in ``data/labels/*.extxyz``.

Backends: ``psi4`` (the original) and ``pyscf``, which is what ``rsfff.qcgen`` uses and
which carries wB97M-V's VV10 non-local correlation automatically -- psi4's wB97M-V is a
different implementation of the same functional and the two need not agree to the
microhartree.

**These are not the same numbers Q-Chem would produce.** Grid, VV10 treatment and
convergence thresholds all differ, and the fragment energies these are subtracted from come
from Q-Chem. The residual is a per-atom constant, which for water-only data is absorbed
entirely by the bond-energy head (``n_H = 2 n_O`` makes only ``E0_O + 2 E0_H``
identifiable anyway) -- but it stops being harmless the moment a second stoichiometry
arrives. ``qchem_roundtrip/atoms/`` holds the inputs for the Q-Chem-exact replacements.
"""

import argparse
import json
import os

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


def atom_energy_pyscf(symbol, method, basis):
    """Open-shell ground-state energy (Hartree) of one neutral atom, via pyscf.

    Uses the same ``make_mol``/``make_mf`` path as ``rsfff.qcgen``, so a non-local
    functional carries its VV10 correlation the same way the molecular labels do.
    """
    import sys as _sys

    _sys.path.insert(0, os.path.join(REPO, "src"))
    from qcgen.backend import make_mf, make_mol

    spin = GROUND_STATE_MULTIPLICITY[symbol] - 1     # pyscf spin = Nalpha - Nbeta = 2S
    mol = make_mol([symbol], [[0.0, 0.0, 0.0]], charge=0, spin=spin, basis=basis)
    mf = make_mf(mol, method, spin=spin)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"{symbol}: SCF did not converge")
    return energy


def atom_energy(symbol, method=METHOD, basis=BASIS):
    """Open-shell ground-state energy (Hartree) of a single neutral atom, via psi4."""
    import psi4

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
    ap = argparse.ArgumentParser(
        description="Compute isolated-atom reference energies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("elements", nargs="*", help="element symbols (default: those in data/labels)")
    ap.add_argument("--backend", choices=("psi4", "pyscf"), default="psi4")
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--basis", default=BASIS)
    ap.add_argument("--out", default=OUT_PATH, help="JSON to write (merged if it exists)")
    args = ap.parse_args()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.backend == "psi4":
        import psi4

        psi4.set_memory("8 GB")
        psi4.set_num_threads(os.cpu_count() or 1)
        psi4.set_output_file(os.path.join(LABELS_DIR, "atomic_references_psi4.out"), False)
        compute = lambda sym: atom_energy(sym, args.method, args.basis)  # noqa: E731
    else:
        compute = lambda sym: atom_energy_pyscf(sym, args.method, args.basis)  # noqa: E731

    symbols = args.elements if args.elements else elements_in_labels()

    # Merge with any previously computed references so incremental runs accumulate --
    # but only within one level of theory, since mixing them would be silently wrong.
    energies, mults = {}, {}
    if os.path.isfile(out_path):
        prev = json.load(open(out_path))
        if (prev.get("method"), prev.get("basis")) != (args.method, args.basis):
            raise SystemExit(
                f"{out_path} holds {prev.get('method')}/{prev.get('basis')} references but "
                f"this run is {args.method}/{args.basis}; write to a different --out rather "
                f"than mixing levels of theory in one file"
            )
        energies.update(prev.get("energies", {}))
        mults.update(prev.get("multiplicities", {}))

    for sym in symbols:
        if sym not in GROUND_STATE_MULTIPLICITY:
            print(f"[{sym}] SKIP: no ground-state multiplicity tabulated", flush=True)
            continue
        mult = GROUND_STATE_MULTIPLICITY[sym]
        print(
            f"[{sym}] mult={mult}  {args.method}/{args.basis} via {args.backend} "
            f"(open-shell)...",
            flush=True,
        )
        e = compute(sym)
        energies[sym] = e
        mults[sym] = mult
        print(f"[{sym}] E = {e:.10f} Ha", flush=True)

    out = {
        "method": args.method,
        "basis": args.basis,
        "backend": args.backend,
        "reference": "uks/rks (unrestricted for open-shell atoms)",
        "units": "Hartree",
        "energies": energies,
        "multiplicities": mults,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nwrote {len(energies)} atomic references -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
