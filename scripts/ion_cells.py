"""Periodic starting cells for ion MD (extxyz with Lattice and pbc, what ChemLab's ipi worker and ASE read).

    python scripts/ion_cells.py crystal --n 4 -o nacl_512.xyz                  # rocksalt 4x4x4, 512 ions, a = 5.64 A
    python scripts/ion_cells.py crystal --n 4 --schottky 1 -o nacl_schottky.xyz # one Na-Cl pair removed (neutral)
    python scripts/ion_cells.py crystal --n 4 --vacancy Cl -o nacl_vcl.xyz      # one Cl removed (charged cell:
                                                                               # the Ewald sum adds the background)
    python scripts/ion_cells.py crystal --n 4 --density 1.55 --jitter 0.3 -o nacl_melt0.xyz
                                                                               # expanded to the melt's density, a start
                                                                               # for heating (molten NaCl ~1.55 g/cm^3)

``--cation`` / ``--anion`` choose other rocksalt salts; ``--a`` the lattice constant. A Schottky pair is removed
nearest-neighbour (``--bound``) or as far apart as the cell allows (default).
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

AMU_G = 1.66053906660e-24


def rocksalt(cation, anion, a, n):
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    pos, sym = [], []
    for c in itertools.product(range(n), repeat=3):
        for b in base:
            o = (np.array(c) + b) * a
            pos += [o, o + [a / 2, 0, 0]]
            sym += [cation, anion]
    return Atoms(sym, positions=pos, cell=np.eye(3) * a * n, pbc=True)


def _mic(atoms, i, idx):
    d = atoms.get_distances(i, idx, mic=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["crystal"])
    ap.add_argument("--cation", default="Na")
    ap.add_argument("--anion", default="Cl")
    ap.add_argument("--a", type=float, default=5.64, help="lattice constant (A)")
    ap.add_argument("--n", type=int, default=4, help="conventional cells per side (8 n^3 ions)")
    ap.add_argument("--schottky", type=int, default=0, help="cation-anion pairs to remove")
    ap.add_argument("--bound", action="store_true", help="remove each Schottky pair as nearest neighbours")
    ap.add_argument("--vacancy", choices=["cation", "anion", "Na", "Cl"], help="remove one ion (charged cell)")
    ap.add_argument("--density", type=float, help="rescale the cell to this density (g/cm^3)")
    ap.add_argument("--jitter", type=float, default=0.0, help="Gaussian displacement (A) of every ion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    atoms = rocksalt(a.cation, a.anion, a.a, a.n)
    remove = []
    cats = [k for k, s in enumerate(atoms.get_chemical_symbols()) if s == a.cation]
    ans = [k for k, s in enumerate(atoms.get_chemical_symbols()) if s == a.anion]
    for _ in range(a.schottky):
        i = int(rng.choice([k for k in cats if k not in remove]))
        free = [k for k in ans if k not in remove]
        d = _mic(atoms, i, free)
        j = free[int(np.argmin(d))] if a.bound else free[int(np.argmax(d))]
        remove += [i, j]
    if a.vacancy:
        pool = cats if a.vacancy in ("cation", a.cation) else ans
        remove.append(int(rng.choice([k for k in pool if k not in remove])))
    if remove:
        del atoms[sorted(remove)]
    if a.density:
        mass = sum(atomic_masses[atomic_numbers[s]] for s in atoms.get_chemical_symbols()) * AMU_G
        v_target = mass / a.density * 1e24                       # A^3
        s = (v_target / atoms.get_volume()) ** (1 / 3)
        atoms.set_cell(atoms.cell * s, scale_atoms=True)
    if a.jitter:
        atoms.positions += rng.normal(0, a.jitter, atoms.positions.shape)
    atoms.wrap()
    write(a.out, atoms, format="extxyz")
    q = atoms.get_chemical_symbols().count(a.cation) - atoms.get_chemical_symbols().count(a.anion)
    rho = sum(atomic_masses[atomic_numbers[s]] for s in atoms.get_chemical_symbols()) * AMU_G / (atoms.get_volume() * 1e-24)
    print(f"{a.out}: {len(atoms)} ions, cell {np.round(atoms.cell.lengths(), 3).tolist()} A, {rho:.3f} g/cm^3, "
          f"net charge {q:+d}")


if __name__ == "__main__":
    main()
