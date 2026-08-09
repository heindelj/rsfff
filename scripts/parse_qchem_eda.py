"""Turn a glob of Q-Chem ALMO-EDA outputs into a single extended-XYZ file.

Usage
-----
    python scripts/parse_qchem_eda.py OUTPUT.xyz 'PATTERN' [PATTERN ...]

Patterns are globbed by the script itself, so quote them to avoid the shell
expanding thousands of paths (either way works)::

    python scripts/parse_qchem_eda.py w2_wb97xv_qzvppd.xyz \
        '~/dev/CMM_Data/dimers/water_water/water_dimer_cluster/*/eda.out'

Frames are ordered by the natural-sorted source path, so ``.../7/eda.out``
precedes ``.../10/eda.out``.

Output schema
-------------
Per-atom columns: ``species pos(3) mulliken_charges(1) fragment_idx(1)``.

Header keys (atomic units by default: Hartree, e*a0^n; positions in Angstrom)::

    energy                  total electronic energy of the supersystem
    eda_cls_elec            classical (frozen-density) electrostatics
    eda_mod_pauli           modified Pauli repulsion (CLS PAULI - DISP)
    eda_disp                dispersion
    eda_pol                 polarization
    eda_ct                  charge transfer (Q-Chem's E_vct)
    eda_prp eda_frz eda_int preparation / frozen / total interaction energy
    eda_elec eda_pauli eda_cls_pauli   remaining EDA2 terms, for reference
    fragment_energies       isolated-fragment SCF energies (n_fragments values)
    dipole quadrupole octopole hexadecapole
                            Cartesian multipoles as full symmetric tensors,
                            flattened row-major: 3, 9, 27, 81 values (see
                            --multipole-format unique for the compact form)
    multipole_format        "tensor" or "unique"
    n_fragments charge multiplicity fragment_charges fragment_multiplicities
    method basis config_type sample_id source units

``--units qchem`` keeps Q-Chem's native kJ/mol EDA terms and Debye-Ang^(n-1)
multipoles instead (total and fragment energies are Hartree either way).

Note on ASE: ``ase.io.read`` treats ``energy`` and ``dipole`` as calculator
properties, so they land in ``atoms.calc.results`` rather than ``atoms.info``
(and are labelled eV / e*Ang even though the values are Hartree / e*a0). That
matches the existing ``data/labels/*.extxyz`` schema, which ``src/train/data.py``
already reads that way; every other key here stays in ``atoms.info``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qcgen.qchem_eda import (  # noqa: E402
    KJMOL_PER_HARTREE,
    MULTIPOLE_LABELS,
    QChemEDAParseError,
    check_consistency,
    parse_eda_output,
    to_atomic_units,
    unique_components,
)

#: Order of EDA keys on the header line: the five requested components first,
#: then the bookkeeping terms.
EDA_KEY_ORDER = (
    "cls_elec", "mod_pauli", "disp", "pol", "ct",
    "prp", "frz", "int", "elec", "pauli", "cls_pauli", "cls_disp",
)


def natural_key(path: str):
    """Sort key that orders embedded integers numerically (``7`` before ``10``)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def collect_paths(patterns: list[str]) -> list[str]:
    """Expand and de-duplicate patterns (already-expanded literal paths pass through)."""
    paths: list[str] = []
    for pattern in patterns:
        expanded = glob.glob(os.path.expanduser(pattern), recursive=True)
        if not expanded and os.path.exists(os.path.expanduser(pattern)):
            expanded = [os.path.expanduser(pattern)]
        if not expanded:
            print(f"warning: pattern matched nothing: {pattern}", file=sys.stderr)
        paths.extend(expanded)
    return sorted(dict.fromkeys(os.path.abspath(p) for p in paths), key=natural_key)


def sample_id(path: str) -> str:
    """The numbered run directory containing ``eda.out`` (falls back to the stem)."""
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else os.path.splitext(os.path.basename(path))[0]


def fmt(arr) -> str:
    # 13 significant digits: enough to hold Q-Chem's 10-decimal fragment
    # energies (~-76.4366986719) without dropping the last digit.
    return " ".join(f"{v:.12e}" for v in np.asarray(arr).ravel())


def frame_header(
    rec, sid: str, source: str, config_type: str, multipole_format: str = "tensor"
) -> str:
    keys = [
        "Properties=species:S:1:pos:R:3:mulliken_charges:R:1:fragment_idx:I:1",
        f"energy={rec.energy:.12e}",
    ]
    for name in EDA_KEY_ORDER:
        if name in rec.eda:
            keys.append(f"eda_{name}={rec.eda[name]:.12e}")
    keys.append(f'fragment_energies="{fmt(rec.fragment_energies)}"')
    for name in MULTIPOLE_LABELS:
        tensor = rec.multipoles[name]
        values = tensor if multipole_format == "tensor" else unique_components(tensor, name)
        keys.append(f'{name}="{fmt(values)}"')
    keys += [
        f"multipole_format={multipole_format}",
        f"n_fragments={rec.n_fragments}",
        # `charge`/`multiplicity`/`config_type` keep the names the existing
        # loader in src/train/data.py already reads.
        f"charge={rec.total_charge}",
        f"multiplicity={rec.multiplicity}",
        f'fragment_charges="{" ".join(str(c) for c in rec.fragment_charges)}"',
        f'fragment_multiplicities="{" ".join(str(m) for m in rec.fragment_mults)}"',
        f"method={rec.method}",
        f"basis={rec.basis}",
        f"config_type={config_type}",
        f"sample_id={sid}",
        f"source={source}",
        f"units={'atomic' if rec.units == 'atomic' else 'qchem'}",
    ]
    return " ".join(keys)


def auto_config_type(rec) -> str:
    """``w<n>`` for a cluster of n waters, else the per-fragment formula list.

    Example: three waters and a hydroxide -> ``H2O_H2O_H2O_OH``.
    """
    formulas = []
    for f in range(rec.n_fragments):
        syms = [s for s, i in zip(rec.symbols, rec.fragment_idx) if i == f]
        counts = {s: syms.count(s) for s in dict.fromkeys(sorted(syms))}
        formulas.append("".join(s + (str(c) if c > 1 else "") for s, c in counts.items()))
    if all(f == "H2O" for f in formulas):
        return f"w{rec.n_fragments}"
    return "_".join(formulas)


def write_frame(
    fh, rec, sid: str, source: str, config_type: str, multipole_format: str = "tensor"
) -> None:
    fh.write(f"{rec.n_atoms}\n{frame_header(rec, sid, source, config_type, multipole_format)}\n")
    for sym, (x, y, z), q, f in zip(
        rec.symbols, rec.positions, rec.mulliken_charges, rec.fragment_idx
    ):
        fh.write(f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f} {q:16.10f} {int(f):4d}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse Q-Chem ALMO-EDA outputs into an extended-XYZ dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("output", help="path of the extxyz file to write")
    ap.add_argument("patterns", nargs="+", help="glob(s) matching Q-Chem eda.out files")
    ap.add_argument(
        "--units",
        choices=("atomic", "qchem"),
        default="atomic",
        help="atomic: Hartree + e*a0^n (default). qchem: kJ/mol + Debye-Ang^(n-1).",
    )
    ap.add_argument(
        "--multipole-format",
        choices=("tensor", "unique"),
        default="tensor",
        help=(
            "tensor: full symmetric tensors, 3/9/27/81 values (default). "
            "unique: independent components only, 3/6/10/15 values in Q-Chem's "
            "print order (XX XY YY XZ YZ ZZ, ...)."
        ),
    )
    ap.add_argument(
        "--config-type",
        default=None,
        help="config_type header value (default: w<n> for pure water clusters, "
        "otherwise the underscore-joined fragment formulas)",
    )
    ap.add_argument(
        "--relative-to",
        default=None,
        help="write the source= key relative to this directory (default: absolute)",
    )
    ap.add_argument(
        "--skip-errors",
        action="store_true",
        help="log and skip unparseable files instead of aborting",
    )
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="skip the EDA sum-rule / plausibility / SCF-convergence checks",
    )
    ap.add_argument(
        "--drop-inconsistent",
        action="store_true",
        help="omit frames that fail a consistency check instead of only warning",
    )
    args = ap.parse_args(argv)

    paths = collect_paths(args.patterns)
    if not paths:
        print("error: no input files matched", file=sys.stderr)
        return 1
    print(f"parsing {len(paths)} file(s) -> {args.output}", file=sys.stderr)

    # Absolute slack covers Q-Chem's 4-decimal kJ/mol printing of the six summed
    # terms; check_consistency adds its own relative term on top.
    atol = 1e-3 if args.units == "qchem" else 1e-3 / KJMOL_PER_HARTREE

    n_written, n_failed, n_warned, n_dropped = 0, 0, 0, 0
    with open(args.output, "w") as fh:
        for i, path in enumerate(paths):
            try:
                rec = parse_eda_output(path)
                if args.units == "atomic":
                    to_atomic_units(rec)
                if not args.no_check:
                    msgs = check_consistency(rec, atol=atol)
                    for msg in msgs:
                        print(f"warning: {path}: {msg}", file=sys.stderr)
                    n_warned += len(msgs)
                    if msgs and args.drop_inconsistent:
                        n_dropped += 1
                        continue
                source = (
                    os.path.relpath(path, os.path.expanduser(args.relative_to))
                    if args.relative_to
                    else path
                )
                write_frame(
                    fh,
                    rec,
                    sample_id(path),
                    source,
                    args.config_type or auto_config_type(rec),
                    args.multipole_format,
                )
                n_written += 1
            except (QChemEDAParseError, OSError, ValueError) as exc:
                n_failed += 1
                msg = f"{path}: {exc}"
                if not args.skip_errors:
                    print(f"error: {msg}", file=sys.stderr)
                    return 1
                print(f"skipped: {msg}", file=sys.stderr)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(paths)}", file=sys.stderr)

    print(
        f"wrote {n_written} frame(s) to {args.output}"
        f"{f'; skipped {n_failed} unparseable' if n_failed else ''}"
        f"{f'; dropped {n_dropped} inconsistent' if n_dropped else ''}"
        f"{f'; {n_warned} consistency warning(s)' if n_warned else ''}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
