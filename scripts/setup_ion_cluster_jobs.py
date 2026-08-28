#!/usr/bin/env python3
"""Stage EDA and force jobs for the hydronium/hydroxide cluster sets.

Writes into two new nested job dirs, which the existing worker discovers on its own -- any
directory under a calculation root that contains an ``inputs/`` is a job dir
(``qchem_roundtrip.claim_next_job``):

- ``qchem_roundtrip/eda/ion_clusters``
- ``qchem_roundtrip/force/ion_clusters``

    python scripts/setup_ion_cluster_jobs.py                  # 14 geometries, 28 inputs
    python scripts/setup_ion_cluster_jobs.py --max-waters 3 --overwrite

One isomer per size -- the lowest-energy one -- for N = 1..7 waters of each ion, with the
reference fragmentation only. Level of theory is wB97M-V/def2-TZVPD with the ``$rem`` blocks of
``qchem_roundtrip/templates/{eda,force}.in``, matching every other job in the corpus.

Why this exists instead of `benchmarks/scripts/setup_qchem_jobs.py`
------------------------------------------------------------------
That script is for **neutral** clusters: it hard-codes charge 0 and multiplicity 1 for the
system and every fragment, and it validates that the atoms arrive in strict ``O H H`` order.
Both assumptions fail here -- these systems carry a charge, one fragment has to hold it, and the
two sets order their atoms differently from each other and from the corpus. Fragments come from
:func:`rsfff.md.assign.rank_oh_fragment_assignments`, the same minimized-O-H-distance rule the
AIMD harvester used to build the training set, so the decomposition written here is the one the
model was fitted against.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ase.io import read                                                   # noqa: E402

from rsfff.md.assign import rank_oh_fragment_assignments                  # noqa: E402

HYDRONIUM = ROOT / "data" / "hydronium_clusters_ccdb"
HYDROXIDE = ROOT / "data" / "hydroxide_clusters" / "jp5b03893_si_002.xyz"
JOB_NAME = "ion_clusters"

#: Copied verbatim from `qchem_roundtrip/templates/eda.in` and `force.in`. Duplicated rather
#: than parsed out of the templates because the template `$rem` is what the *generator* would
#: emit for a geometry dropped in `geoms/`, and this script bypasses the generator (it needs a
#: nested job dir, which `generate_inputs` does not walk into).
REM_COMMON = [
    "   SCF_CONVERGENCE  =  8",
    "   THRESH           =  14",
    "   XC_GRID          =  000099000590",
    "   NL_GRID          =  1",
    "   SYMMETRY         =  false",
]


def rem_block(jobtype: str, method: str, basis: str, mem_total: int, mem_static: int) -> str:
    head = ["$rem", f"   JOBTYPE          =  {jobtype}"]
    if jobtype == "eda":
        head.append("   EDA2             =  1")
    head += [f"   METHOD           =  {method}", f"   BASIS            =  {basis}"]
    tail = []
    if jobtype == "eda":
        tail += ["   EDA_BSSE         =  false", "   FD_MAT_VEC_PROD  =  false"]
    tail += [f"   MEM_TOTAL        =  {mem_total}", f"   MEM_STATIC       =  {mem_static}"]
    if jobtype == "eda":
        tail.append("   SCF_PRINT_FRGM   =  true")
    return "\n".join(head + REM_COMMON + tail + ["$end"])


def atom_line(symbol: str, xyz) -> str:
    return f"  {symbol:<3} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}"


def molecule_block(symbols, coords, charge: int, fragment_idx=None, fragment_charges=None) -> str:
    """``$molecule``, fragmented when ``fragment_idx`` is given.

    Q-Chem wants the **total** charge and multiplicity first, then one ``--`` separated block
    per fragment carrying that fragment's own charge and multiplicity. Fragments are emitted in
    ascending ``fragment_idx``, which is what the roundtrip reader assumes.
    """
    lines = ["$molecule", f"{charge} 1"]
    if fragment_idx is None:
        lines += [atom_line(s, c) for s, c in zip(symbols, coords)]
    else:
        for f in range(int(max(fragment_idx)) + 1):
            lines += ["--", f"{int(fragment_charges[f])} 1"]
            lines += [atom_line(symbols[i], coords[i])
                      for i in range(len(symbols)) if fragment_idx[i] == f]
    lines.append("$end")
    return "\n".join(lines)


def extxyz(symbols, coords, charge: int, fragment_idx, fragment_charges) -> str:
    """The geometry as the corpus stores it, so the frame can be re-read later."""
    n_frag = int(max(fragment_idx)) + 1
    header = (
        "Properties=species:S:1:pos:R:3:fragment_idx:I:1 "
        f"charge={charge} multiplicity=1 n_fragments={n_frag} "
        f'fragment_charges="{" ".join(str(int(q)) for q in fragment_charges)}" '
        f'fragment_multiplicities="{" ".join(["1"] * n_frag)}"'
    )
    lines = [str(len(symbols)), header]
    for i, (s, c) in enumerate(zip(symbols, coords)):
        lines.append(f"{s:<2} {c[0]:16.8f} {c[1]:16.8f} {c[2]:16.8f} {int(fragment_idx[i]):d}")
    return "\n".join(lines) + "\n"


def ensure_job_dir(root: Path) -> None:
    for sub in ("geoms", "inputs", "outputs",
                "state/generated", "state/done", "state/failed", "state/locks"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def select_geometries(max_waters: int) -> list[dict]:
    """The lowest-energy isomer at each size, both ions.

    The hydronium files are already sorted by ascending ASP energy, so frame 0 is the global
    minimum of that size. The hydroxide file carries no energies at all, so the first frame of
    each size is taken -- the SI lists them ``a, b, c, ...`` in the paper's own order, which is
    by energy. Note the labels are not reliable as keys (``Isomer 5k`` appears twice, one of
    them a 4-water cluster), so the size is read off the atom count.
    """
    picked: list[dict] = []
    for path in sorted(HYDRONIUM.glob("asp-H2O_*--H3O+.xyz"),
                       key=lambda p: int(re.search(r"_(\d+)--", p.name).group(1))):
        n = int(re.search(r"_(\d+)--", path.name).group(1))
        if 1 <= n <= max_waters:
            picked.append(dict(atoms=read(str(path), index=0), charge=+1, waters=n,
                               stem=f"h3o+_w{n}_iso00", source=f"{path.name}[0]"))

    if HYDROXIDE.exists():
        seen: set[int] = set()
        for k, atoms in enumerate(read(str(HYDROXIDE), index=":")):
            n = (len(atoms) - 2) // 3
            if n < 1 or n > max_waters or n in seen:
                continue
            seen.add(n)
            picked.append(dict(atoms=atoms, charge=-1, waters=n,
                               stem=f"oh-_w{n}_iso00", source=f"{HYDROXIDE.name}[{k}]"))
    return picked


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-waters", type=int, default=7)
    p.add_argument("--roundtrip-root", default=str(ROOT / "qchem_roundtrip"))
    p.add_argument("--method", default="wB97M-V")
    p.add_argument("--basis", default="def2-TZVPD")
    p.add_argument("--mem-total", type=int, default=64000)
    p.add_argument("--mem-static", type=int, default=10000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    rt = Path(args.roundtrip_root)
    eda_dir, force_dir = rt / "eda" / JOB_NAME, rt / "force" / JOB_NAME
    ensure_job_dir(eda_dir)
    ensure_job_dir(force_dir)

    manifest, written = [], 0
    for spec in select_geometries(args.max_waters):
        atoms, charge, stem = spec["atoms"], spec["charge"], spec["stem"]
        symbols = atoms.get_chemical_symbols()
        coords = np.asarray(atoms.get_positions())

        best = rank_oh_fragment_assignments(symbols, coords, charge)[0]
        frag = np.asarray(best.fragment_idx)
        qfrag = np.asarray(best.fragment_charges)
        if int(qfrag.sum()) != charge:
            raise SystemExit(f"{stem}: fragment charges sum to {qfrag.sum()}, expected {charge}")

        # EDA needs the atoms grouped by fragment in the input; the force job does not care,
        # but writing the same ordering to both keeps the two comparable atom for atom.
        order = np.argsort(frag, kind="stable")
        sym_s = [symbols[i] for i in order]
        pos_s = coords[order]
        frag_s = frag[order]

        eda_in = (molecule_block(sym_s, pos_s, charge, frag_s, qfrag) + "\n\n"
                  + rem_block("eda", args.method, args.basis, args.mem_total, args.mem_static)
                  + "\n")
        force_in = (molecule_block(sym_s, pos_s, charge) + "\n\n"
                    + rem_block("force", args.method, args.basis,
                                args.mem_total, args.mem_static) + "\n")
        geom = extxyz(sym_s, pos_s, charge, frag_s, qfrag)

        record = dict(stem=stem, source=spec["source"], waters=spec["waters"], charge=charge,
                      n_atoms=len(symbols), n_fragments=int(frag.max()) + 1,
                      fragment_charges=[int(q) for q in qfrag])
        for path, text in ((eda_dir / "inputs" / f"{stem}.in", eda_in),
                           (force_dir / "inputs" / f"{stem}.in", force_in),
                           (eda_dir / "geoms" / f"{stem}.extxyz", geom),
                           (force_dir / "geoms" / f"{stem}.extxyz", geom)):
            if path.exists() and not args.overwrite:
                record.setdefault("skipped", []).append(str(path.relative_to(rt)))
                continue
            path.write_text(text)
            written += 1
        manifest.append(record)

    (rt / "ion_cluster_jobs_manifest.json").write_text(json.dumps(
        {"method": args.method, "basis": args.basis, "jobs": manifest}, indent=2))
    print(f"{len(manifest)} geometries, {written} files written")
    print(f"  eda   -> {eda_dir.relative_to(rt.parent)}/inputs")
    print(f"  force -> {force_dir.relative_to(rt.parent)}/inputs")
    for r in manifest:
        print(f"  {r['stem']:16s} {r['n_atoms']:3d} atoms  {r['n_fragments']} fragments  "
              f"q={r['charge']:+d}  {r['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
