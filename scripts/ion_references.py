"""Reference energies for monatomic-ion fragments, read off the training data itself.

A monatomic fragment (Na+, Cl-, ...) has no bonded term, so in the film model its fragment
energy is the reference energy ``E0[Z]`` and nothing else (``tests/film/test_new_species.py``).
For water the reference only has to be *roughly* right -- a per-atom constant mismatch is
absorbed by the Morse well depths -- but for a lone ion there is nothing to absorb it: any
offset between ``E0[Z]`` and the isolated-ion SCF energy is a permanent fragment-energy error,
and through the total-energy loss it leaks into the interaction channels.

So for an ionic dataset the "atomic" reference of Na and Cl is the **isolated ion** (Na+,
Cl-), and it has to be the same Q-Chem number the fragment energies come from. Every ALMO-EDA
frame already carries it, as the ``fragment_energies`` entry of each monatomic fragment; this
script collects those, checks that they agree (they are geometry-independent, so the spread
should be at the SCF convergence level), and writes the JSON ``data.reference_energies``
expects (element symbol -> Hartree, under ``energies``).

Elements present only in multi-atom fragments are not written: take those from
``scripts/atomic_references.py`` and merge with ``--merge``.

Usage::

    python scripts/ion_references.py data/nacl/clusters/*.xyz \
        --out data/nacl/ion_references_wb97mv_tzvpd.json [--merge other_refs.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def collect(paths) -> dict[tuple[int, int], torch.Tensor]:
    """``{(Z, charge): energies}`` over every monatomic fragment in ``paths``."""
    from rsfff.train.data import fragment_view, load_cluster_datasets

    dataset = load_cluster_datasets(paths, dtype=torch.float64, fragmentations=0)
    view = fragment_view(dataset)
    single = (view._counts == 1).nonzero().flatten()
    found: dict[tuple[int, int], list[float]] = {}
    for f in single.tolist():
        z = int(view._num[int(view._offsets[f])])
        q = int(round(float(view._fragment_charge[f])))
        found.setdefault((z, q), []).append(float(view._energy[f]))
    return {key: torch.tensor(v, dtype=torch.float64) for key, v in found.items()}


def main() -> None:
    from ase.data import chemical_symbols

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="cluster extxyz files with fragment_energies")
    ap.add_argument("--out", required=True, help="JSON to write")
    ap.add_argument("--merge", default=None, help="existing reference JSON to merge into")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max spread (Ha) of one ion's energies before refusing")
    args = ap.parse_args()

    found = collect(args.paths)
    if not found:
        raise SystemExit("no monatomic fragments in the given files")

    by_symbol: dict[str, tuple[int, torch.Tensor]] = {}
    for (z, q), e in sorted(found.items()):
        sym = chemical_symbols[z]
        if sym in by_symbol:
            raise SystemExit(
                f"{sym} appears as a monatomic fragment with charges {by_symbol[sym][0]} and "
                f"{q}; one reference per element is all the model has"
            )
        spread = float(e.max() - e.min())
        print(f"{sym:>2} charge {q:+d}: {e.numel()} fragments, E = {float(e.mean()):.10f} Ha, "
              f"spread {spread:.2e}")
        if spread > args.tol:
            raise SystemExit(f"{sym}{q:+d}: spread {spread:.2e} Ha exceeds --tol {args.tol}")
        by_symbol[sym] = (q, e)

    out = {"energies": {}, "charges": {}, "units": "Hartree"}
    if args.merge:
        out = json.loads(Path(args.merge).read_text())
        out.setdefault("charges", {})
    for sym, (q, e) in by_symbol.items():
        out["energies"][sym] = float(e.mean())
        out["charges"][sym] = q
    out["note"] = (
        "monatomic-ion references from the isolated-fragment SCF energies of "
        + ", ".join(str(p) for p in args.paths[:4])
        + (" ..." if len(args.paths) > 4 else "")
        + " (scripts/ion_references.py)"
    )
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
