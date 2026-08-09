"""Label monomer geometries with their quantum response to random probe charges.

    python scripts/probe_response.py geoms.xyz \
        --charge 0 --spin 0 --xc pbe0 --basis def2-svp \
        --cutoff 5.0 --n-charges 20 --total-charge 0.0 \
        --n-configs 8 --charge-dist normal --spread 0.3 \
        --out data/probe/monomer_response.npz

For each frame in a (multi-frame) ``.xyz`` -- e.g. Wigner samples of one monomer
-- this draws ``--n-configs`` probe-charge configurations on the ``--cutoff``
shell, records the classical per-atom potential/field/field-gradient descriptor
they induce (:mod:`rsfff.qcgen.probe_charges`), and labels the polarized monomer
by pyscf QM/MM (:mod:`rsfff.qcgen.probe_response`): energy, forces, dipole,
quadrupole, and polarizability. Everything is stacked and written to one ``.npz``
of paired (input features, QM labels) records.

Arrays in the output (G = #geoms, C = #configs, N = #atoms, M = #charges):
  symbols            (N,)                 element symbols
  coords             (G, N, 3)            geometries, Angstrom
  charge_positions   (G, C, M, 3)         probe positions, Angstrom
  charges            (G, C, M)            probe magnitudes, e
  features           (G, C, N, 10)        [V, E(3), field-grad(6)], a.u.
  energy             (G, C)               Hartree
  forces             (G, C, N, 3)         Hartree/Bohr
  dipole             (G, C, 3)            e*a0
  quadrupole         (G, C, 3, 3)         e*a0^2
  polarizability     (G, C, 3, 3)         a0^3
  converged          (G, C)               bool
"""

import argparse

import numpy as np

from rsfff.qcgen.extxyz import read_geoms
from rsfff.qcgen.probe_charges import probe_features
from rsfff.qcgen.probe_response import compute_response_under_charges


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("geometry", help="input .xyz (one or more frames), Angstrom")
    p.add_argument("--charge", type=int, default=0, help="QM charge")
    p.add_argument("--spin", type=int, default=0, help="pyscf spin (Na - Nb)")
    p.add_argument("--xc", default="pbe0", help="DFT functional (default pbe0)")
    # Diffuse functions are essential: without them (e.g. def2-svp) the
    # polarizability comes out ~half its true value, since alpha is dominated by
    # the soft outer density tail. def2-svpd is the cheapest sane choice.
    p.add_argument("--basis", default="def2-svpd",
                   help="basis (default def2-svpd; needs diffuse fns for alpha)")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="probe shell radius, Angstrom (default 5.0)")
    p.add_argument("--n-charges", type=int, default=20,
                   help="probe charges per configuration (default 20)")
    p.add_argument("--total-charge", type=float, default=0.0,
                   help="total probe charge per config, e (default 0.0)")
    p.add_argument("--n-configs", type=int, default=8,
                   help="probe configs per geometry (default 8)")
    p.add_argument("--charge-dist", choices=("equal", "uniform", "normal"),
                   default="normal", help="per-charge magnitude distribution")
    p.add_argument("--spread", type=float, default=0.3,
                   help="spread of per-charge magnitudes (uniform/normal)")
    p.add_argument("--response", choices=("cpscf", "finite-difference"),
                   default="cpscf", help="polarizability path")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--out", default=None, help="optional .npz output path")
    args = p.parse_args()

    symbols, frames = read_geoms(args.geometry)
    rng = np.random.default_rng(args.seed)
    natoms = len(symbols)
    G, C, M = len(frames), args.n_configs, args.n_charges

    out = {
        "charge_positions": np.empty((G, C, M, 3)),
        "charges": np.empty((G, C, M)),
        "features": np.empty((G, C, natoms, 10)),
        "energy": np.empty((G, C)),
        "forces": np.empty((G, C, natoms, 3)),
        "dipole": np.empty((G, C, 3)),
        "quadrupole": np.empty((G, C, 3, 3)),
        "polarizability": np.empty((G, C, 3, 3)),
        "converged": np.empty((G, C), dtype=bool),
    }

    for g, coords in enumerate(frames):
        for c in range(C):
            feat, pos, q = probe_features(
                coords, cutoff=args.cutoff, n_charges=M,
                total_charge=args.total_charge, rng=rng,
                charge_dist=args.charge_dist, spread=args.spread,
            )
            res = compute_response_under_charges(
                symbols, coords, pos, q, args.charge, args.spin,
                args.xc, args.basis, response=args.response,
            )
            out["charge_positions"][g, c] = pos
            out["charges"][g, c] = q
            out["features"][g, c] = feat
            out["energy"][g, c] = res["energy"]
            out["forces"][g, c] = res["forces"]
            out["dipole"][g, c] = res["dipole"]
            out["quadrupole"][g, c] = res["quadrupole"]
            out["polarizability"][g, c] = res["polarizability"]
            out["converged"][g, c] = res["converged"]
            print(f"geom {g:3d} config {c:3d}  E={res['energy']:+.6f}  "
                  f"|mu|={np.linalg.norm(res['dipole']):.4f}  "
                  f"tr(alpha)/3={np.trace(res['polarizability'])/3:.4f}  "
                  f"{'ok' if res['converged'] else 'NOT CONVERGED'}", flush=True)

    if args.out:
        np.savez_compressed(
            args.out, symbols=np.array(symbols),
            coords=np.array(frames), **out,
            meta_xc=args.xc, meta_basis=args.basis, meta_cutoff=args.cutoff,
            meta_charge=args.charge, meta_spin=args.spin,
        )
        print(f"\nwrote {G}x{C} records -> {args.out}")


if __name__ == "__main__":
    main()
