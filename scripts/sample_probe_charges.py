"""Place probe charges on a monomer's outer shell and dump the electrostatic
potential/field/field-gradient they induce at each atom.

    python scripts/sample_probe_charges.py geom.xyz \
        --cutoff 5.0 --n-charges 20 --total-charge 0.0 \
        --n-configs 100 --charge-dist normal --spread 0.5 \
        --out data/probe/geom_probe.npz

Reads a plain ``.xyz`` (Angstrom), draws ``--n-configs`` independent probe-charge
configurations on the ``--cutoff`` shell (each with ``--n-charges`` charges
summing to ``--total-charge``), and records the per-atom 10-vector
``[V, Ex, Ey, Ez, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]`` (atomic units) for every one.
See :mod:`rsfff.qcgen.probe_charges` for units and conventions. This is the
classical featurization stage; the quantum response to these same charges is
labeled separately with pyscf as background point charges.
"""

import argparse

import numpy as np

from rsfff.qcgen.extxyz import read_xyz
from rsfff.qcgen.probe_charges import probe_features


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("geometry", help="input .xyz geometry (Angstrom)")
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="shell radius / standoff in Angstrom (default 5.0)")
    p.add_argument("--n-charges", type=int, default=20,
                   help="probe charges per configuration (default 20)")
    p.add_argument("--total-charge", type=float, default=0.0,
                   help="total charge per configuration, in e (default 0.0)")
    p.add_argument("--n-configs", type=int, default=100,
                   help="independent charge configurations to sample (default 100)")
    p.add_argument("--charge-dist", choices=("equal", "uniform", "normal"),
                   default="equal", help="per-charge magnitude distribution")
    p.add_argument("--spread", type=float, default=1.0,
                   help="spread of per-charge magnitudes for uniform/normal")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--out", default=None,
                   help="optional .npz output; otherwise a summary is printed")
    args = p.parse_args()

    symbols, coords = read_xyz(args.geometry)
    rng = np.random.default_rng(args.seed)

    natoms = len(symbols)
    features = np.empty((args.n_configs, natoms, 10))
    positions = np.empty((args.n_configs, args.n_charges, 3))
    charges = np.empty((args.n_configs, args.n_charges))
    for c in range(args.n_configs):
        feat, pos, q = probe_features(
            coords, cutoff=args.cutoff, n_charges=args.n_charges,
            total_charge=args.total_charge, rng=rng,
            charge_dist=args.charge_dist, spread=args.spread,
        )
        features[c], positions[c], charges[c] = feat, pos, q

    if args.out:
        np.savez_compressed(
            args.out, symbols=np.array(symbols), coords=coords,
            features=features, charge_positions=positions, charges=charges,
            cutoff=args.cutoff, total_charge=args.total_charge,
        )
        print(f"wrote {args.n_configs} configs -> {args.out}  "
              f"(features {features.shape})")
    else:
        print(f"{args.n_configs} configs, {natoms} atoms, "
              f"{args.n_charges} charges each")
        print(f"feature tensor: {features.shape}  (config, atom, [V,E(3),G(6)])")
        print("per-atom potential V (a.u.)  mean +/- std over configs:")
        for a, sym in enumerate(symbols):
            v = features[:, a, 0]
            print(f"  {sym:<3} {a:2d}  {v.mean():+.6f} +/- {v.std():.6f}")


if __name__ == "__main__":
    main()
