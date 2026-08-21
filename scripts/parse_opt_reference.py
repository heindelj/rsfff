"""Turn the monomer opt+freq jobs into single-frame reference extxyz.

Usage
-----
    python scripts/parse_opt_reference.py [--root qchem_roundtrip]
                                          [--out-dir data/wb97mv_tzvpd]
                                          [--stems h3o+_wb97mv_tzvpd ...] [--strict]

``qchem_roundtrip/opt/outputs/{h2o,h3o+,oh-}_wb97mv_tzvpd.out`` are opt+freq jobs
on the three monomers this dataset is built out of. Because
``templates/opt_and_freq.in`` recomputes the Hessian at the final structure,
Q-Chem solves the CPSCF equations there and prints a **polarizability** -- and
for H3O+ and OH- that is the only one in this repo. Each output becomes one
frame carrying the optimized geometry, its energy, the residual gradient,
Mulliken charges, the full Cartesian multipoles, the polarizability and the
harmonic frequencies.

Everything is taken from the blocks *before* the CPSCF solve so that the tensor
and the coordinates are in the same frame, and the printed matrix is negated to
turn ``d^2E/dF^2`` back into ``alpha``. Both of those are
:mod:`rsfff.qcgen.qchem_opt`'s business and its docstring explains why each one
is load-bearing.

By default every ``opt/outputs`` file whose name has no ``_frameNNNN`` suffix is
read, which is exactly these three: the rest of that directory is the multi-frame
cluster optimizations, which are not monomer references and have no CPSCF block.

Not the same thing as ``scripts/parse_polarizability.py``, which folds a
*different* Q-Chem block -- ``JOBTYPE = polarizability``, printed with the
opposite sign -- into an existing sampled-geometry anchor file. That script and
its output, ``data/wb97mv_tzvpd/h2o_wb97mv_tzvpd_pol.xyz``, are untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qcgen.multifrag import (  # noqa: E402
    TrajectoryFrame,
    canonical_basis,
    fragment_formula,
    write_frames,
)
from qcgen.qchem_opt import (  # noqa: E402
    QChemParseError,
    check_consistency,
    parse_opt_output,
    to_atomic_units,
)


def slug(text: str) -> str:
    text = re.sub(r"^def2[-_]?", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def label_slug(text: str) -> str:
    """``H3O+`` -> ``h3o+``: keeps the charge, which here is the whole distinction."""
    return re.sub(r"[^a-z0-9+_-]", "", text.lower())


def monomer_stems(opt_dir: str) -> list[str]:
    """Outputs with no ``_frameNNNN`` suffix: the single-geometry monomer jobs."""
    return sorted(
        os.path.splitext(n)[0]
        for n in os.listdir(opt_dir)
        if n.endswith(".out") and not re.search(r"_frame\d+\.out$", n)
    )


def write_stem(stem: str, root: str, out_dir: str, strict: bool) -> str | None:
    path = os.path.join(root, "opt", "outputs", f"{stem}.out")
    try:
        rec = to_atomic_units(parse_opt_output(path))
    except (QChemParseError, OSError, ValueError) as exc:
        print(f"{stem}: {exc}", file=sys.stderr)
        return None

    msgs = check_consistency(rec)
    for m in msgs:
        print(f"{stem}: {m}", file=sys.stderr)
    if msgs and strict:
        raise SystemExit(f"{stem}: failed a consistency check")

    config_type = fragment_formula(rec.symbols, rec.total_charge)
    basis = canonical_basis(rec.basis)
    frame = TrajectoryFrame(
        symbols=list(rec.symbols),
        positions=rec.positions,
        forces=rec.forces,
        energy=rec.energy,
        total_charge=rec.total_charge,
        multiplicity=rec.multiplicity,
        method=rec.method,
        basis=basis,
        config_type=config_type,
        sample_id=0,
        source=os.path.relpath(path, root),
        polarizability=rec.polarizability,
        mulliken=rec.mulliken_charges,
        multipoles=rec.multipoles,
        frequencies=rec.frequencies,
        ir_intensities=rec.ir_intensities,
    )

    name = f"{label_slug(config_type)}_opt_{slug(rec.method)}_{slug(basis)}_pol.xyz"
    out_path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    write_frames(out_path, [frame])
    eig = np.linalg.eigvalsh(rec.polarizability)
    print(
        f"{stem}: {config_type}, E = {rec.energy:.8f} Ha, "
        f"alpha eigenvalues {np.array2string(eig, precision=3)} a0^3 "
        f"(isotropic {eig.mean():.3f})",
        file=sys.stderr,
    )
    print(f"  -> {out_path}", file=sys.stderr)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse monomer opt+freq outputs into reference extxyz frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default="qchem_roundtrip", help="the round-trip bundle")
    ap.add_argument("--out-dir", default="data/wb97mv_tzvpd", help="where to write extxyz")
    ap.add_argument(
        "--stems", nargs="*", default=None,
        help="opt output stems (default: every monomer job)",
    )
    ap.add_argument("--strict", action="store_true", help="abort on any warning")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.root))
    opt_dir = os.path.join(root, "opt", "outputs")
    if not os.path.isdir(opt_dir):
        print(f"error: no opt outputs under {opt_dir}", file=sys.stderr)
        return 1

    stems = args.stems if args.stems is not None else monomer_stems(opt_dir)
    if not stems:
        print(f"error: no monomer opt outputs under {opt_dir}", file=sys.stderr)
        return 1

    written = [write_stem(s, root, args.out_dir, args.strict) for s in stems]
    return 0 if any(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
