#!/usr/bin/env python
"""Build a monomer-density database and sample reference electrostatics (rsfff.qcgen.elst_db).

    # monomers from xyz/extxyz files (all frames; charge/spin from atoms.info)
    python scripts/elst_db.py build db/ data/monomers/water.extxyz data/monomers/ions.xyz --xc wb97m-v --basis def2-tzvpd
    # potential/field grids on scaled vdW surfaces, every monomer
    python scripts/elst_db.py grids db/
    # pairs: random orientations, closest contact uniform in [1.4, 6.0] A
    python scripts/elst_db.py pairs db/ --n 5000 --out db/pairs_all.extxyz --seed 0
    python scripts/elst_db.py pairs db/ --n 2000 --a 'water_*' --b 'na' 'cl' --out db/pairs_ion_water.extxyz
"""
import argparse
import fnmatch

from rsfff.qcgen import elst_db as D


def _select(db, patterns):
    if not patterns:
        return None
    names = [n for n in db.index if any(fnmatch.fnmatch(n, p) for p in patterns)]
    if not names:
        raise SystemExit(f"no monomers match {patterns}")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("root")
    b.add_argument("files", nargs="+")
    b.add_argument("--xc", default="wb97m-v")
    b.add_argument("--basis", default="def2-tzvpd")
    b.add_argument("--auxbasis", default="def2-universal-jkfit")
    b.add_argument("--overwrite", action="store_true")
    g = sub.add_parser("grids")
    g.add_argument("root")
    g.add_argument("--names", nargs="*")
    g.add_argument("--density", type=float, default=2.0, help="points per bohr^2 of sphere area")
    g.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pairs")
    p.add_argument("root")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--a", nargs="*", help="name patterns for the first monomer (default: all)")
    p.add_argument("--b", nargs="*", help="name patterns for the second monomer (default: all)")
    p.add_argument("--contact", nargs=2, type=float, default=(1.4, 6.0), metavar=("LO", "HI"), help="Angstrom")
    p.add_argument("--exact-ee", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    a = ap.parse_args()

    if a.cmd == "build":
        structs = (s for f in a.files for s in D.structures_from_file(f))
        db = D.build_db(a.root, structs, xc=a.xc, basis=a.basis, auxbasis=a.auxbasis, overwrite=a.overwrite)
        print(f"{len(db)} monomers in {a.root}")
    elif a.cmd == "grids":
        db = D.MonomerDB(a.root)
        for k, n in enumerate(_select(db, a.names) or list(db.index)):
            r = D.monomer_grid(db, n, density=a.density, rng=a.seed + k)
            print(f"{n}: {len(r['points'])} points")
    else:
        db = D.MonomerDB(a.root)
        D.sample_pairs(db, a.n, a.out, tuple(a.contact), _select(db, a.a), _select(db, a.b), rng=a.seed,
                       exact_ee=a.exact_ee, tag=a.tag)


if __name__ == "__main__":
    main()
