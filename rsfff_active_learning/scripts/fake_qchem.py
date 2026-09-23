#!/usr/bin/env python3
"""Stand-in for Q-Chem: finish every pending job of a store with labels from a film model.

    python scripts/fake_qchem.py STORE [--committee committees/film_committee_100k] [--member 0]

For dry runs of the whole loop without spending Q-Chem time (``run_loop.py --quick`` uses it as
the label stage's sync hook). A ``force`` job gets the model's energy and forces; an ``eda2``
job gets the model's channel breakdown as its EDA terms (elst -> cls_elec, pauli ->
mod_pauli, disp, induction -> pol, ct = 0), zero multipoles, and the model's per-fragment
energies. Results are written in Q-Chem's units and marked ``program: fake_qchem`` in their
envelope and ``fake: true`` in the job's run.json, so they can never pass for real labels:
**never point this at a production store**, it refuses one without ``_fake_ok`` in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

KJMOL = 2625.4996394798254
BOHR = 0.529177210903


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store", type=Path)
    ap.add_argument("--committee", default=str(HERE.parent / "committees" / "film_committee_100k"))
    ap.add_argument("--member", type=int, default=0)
    a = ap.parse_args(argv)

    from cc_workers.common import store as st
    from cc_workers.common.protocol import Result
    from rsfff.md.film_driver import FilmPotential, load_film_model

    from rsfff_al.committee import member_checkpoints

    store = st.Store(a.store, create=False)
    if not (a.store / "_fake_ok").exists():
        print(f"{a.store} has no _fake_ok marker; refusing to write fake labels into it",
              file=sys.stderr)
        return 2
    pending = store.pending("qchem", include_delegated=True)
    if not pending:
        return 0
    model, _ = load_film_model(member_checkpoints(a.committee)[a.member])
    z_of = {"O": 8, "H": 1}
    n = 0
    for job in pending:
        if not job.claim():
            continue
        try:
            rec = job.record()
            mol = rec["spec"]["molecule"]
            sym = mol["symbols"]
            pos = np.asarray(mol["coords"], float)
            frag = mol.get("fragment_idx") or rec.get("tags", {}).get("fragment_idx") \
                or list(np.repeat(np.arange(len(sym) // 3), 3))
            pot = FilmPotential(model, [z_of[s] for s in sym], frag)
            e, g = pot.energy_and_gradient(pos)
            data = {"symbols": sym, "energy": float(e[0]), "scf_converged": True,
                    "completed": True}
            arrays = {"positions": pos, "mulliken_charges": np.zeros(len(sym)),
                      "dipole": np.zeros(3), "quadrupole": np.zeros(6),
                      "octopole": np.zeros(10), "hexadecapole": np.zeros(15)}
            if rec["calc"] == "force":
                arrays["forces"] = -g[0] * BOHR                    # Ha/A -> Ha/bohr
                schema = "qchem.force/1"
            else:
                inter = pot.interaction(pos)
                eda = {"cls_elec": inter.get("elst", 0.0), "mod_pauli": inter.get("pauli", 0.0),
                       "disp": inter.get("disp", 0.0), "pol": inter.get("induction", 0.0),
                       "ct": 0.0, "prp": 0.0, "frz": 0.0}
                eda["int"] = sum(eda[k] for k in ("cls_elec", "mod_pauli", "disp", "pol", "ct"))
                eda["elec"], eda["pauli"] = eda["cls_elec"], eda["mod_pauli"]
                data["eda"] = {k: v * KJMOL for k, v in eda.items()}
                nf = max(frag) + 1
                data.update(n_fragments=nf, has_fragment_blocks=True)
                arrays.update(fragment_idx=np.asarray(frag),
                              fragment_energies=np.full(nf, (float(e[0]) - eda["int"]) / nf),
                              fragment_mulliken_charges=np.zeros(len(sym)),
                              fragment_dipole=np.zeros((nf, 3)),
                              fragment_quadrupole=np.zeros((nf, 6)))
                schema = "qchem.eda2/1"
            attempt = job.new_attempt()
            (attempt / "run.json").write_text(json.dumps({"fake": True, "atom_order":
                                                          list(range(len(sym)))}))
            frame = {"positions": "spec", "origin": "spec", "atom_order": list(range(len(sym))),
                     "rotation": np.eye(3).tolist(), "rmsd": 0.0}
            job.write_result(Result(schema=schema, data=data, arrays=arrays, frame=frame,
                                    program={"name": "fake_qchem"}), attempt, "fake")
            job.set_state(st.DONE, attempt=attempt.name, fake=True)
            n += 1
        finally:
            job.release()
    print(f"fake_qchem: finished {n} jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
