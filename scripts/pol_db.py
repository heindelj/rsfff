"""Polarization reference data for an elst_db database (rsfff.qcgen.pol_ref).

  python scripts/pol_db.py field    DB [--names water_000 Clm ...] [--inner] [--grad] [--field-gradients]
  python scripts/pol_db.py probes   DB --n 200 --tag probes_v1 [--names ...] [--cutoff 2.5 5.0]
  python scripts/pol_db.py partners DB --n 200 --tag partners_v1 [--names-a ...] [--names-b ...]
                                     [--contact 1.6 5.0] [--mutual]

field:    uniform-field CPKS response per monomer -> DB/pol/<name>.npz (alpha, d dm/dF, d phi/dF, d E/dF on its grid);
          --field-gradients adds the uniform field-gradient responses (d phi/dG, dE/dG, dQ/dG) the quadrupole
          polarizabilities and the response-model training need
probes:   full SCF + linear response in random probe charges   -> DB/pol_env/<tag>/ (npz + index.jsonl)
partners: full SCF + linear response in a frozen partner density -> DB/pol_env/<tag>/
"""
import argparse

from rsfff.qcgen import elst_db as D
from rsfff.qcgen import elst_fit as F
from rsfff.qcgen import pol_ref as P

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("field", "probes", "partners"))
ap.add_argument("root")
ap.add_argument("--names", nargs="*")
ap.add_argument("--names-a", nargs="*")
ap.add_argument("--names-b", nargs="*")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--tag", default=None)
ap.add_argument("--cutoff", nargs=2, type=float, default=(2.5, 5.0))
ap.add_argument("--contact", nargs=2, type=float, default=(1.6, 5.0))
ap.add_argument("--mutual", action="store_true")
ap.add_argument("--inner", action="store_true", help="use the inner grids (0.2-3.0 x vdW, grids_inner)")
ap.add_argument("--grad", action="store_true", help="also the field gradient of the induced density on the grid")
ap.add_argument("--field-gradients", action="store_true", help="also responses to uniform field gradients")
ap.add_argument("--no-linear", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

db = D.MonomerDB(a.root)
gk = dict(scales=F.INNER_SCALES, cache="grids_inner") if a.inner else None
if a.mode == "field":
    for n in a.names or list(db.index):
        r = P.field_response(db, n, grid_kw=gk, gradients=a.grad, cache="pol_inner" if a.inner else "pol",
                             field_gradients=a.field_gradients)
        print(n, "alpha_iso", round(float(r["alpha"].trace() / 3), 4), flush=True)
elif a.mode == "probes":
    P.sample_probe_envs(db, a.n, a.tag or "probes", names=a.names, cutoff_ang=tuple(a.cutoff), rng=a.seed,
                        grid_kw=gk, gradients=a.grad, linear=not a.no_linear)
else:
    P.sample_partner_envs(db, a.n, a.tag or ("partners_mutual" if a.mutual else "partners"), names_a=a.names_a,
                          names_b=a.names_b, contact_ang=tuple(a.contact), mutual=a.mutual, rng=a.seed,
                          grid_kw=gk, gradients=a.grad, linear=not a.no_linear)
