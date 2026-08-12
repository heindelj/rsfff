"""Turn a ``qchem_roundtrip`` output bundle into extended-XYZ training data.

Usage
-----
    python scripts/parse_roundtrip.py [--root qchem_roundtrip] [--out-dir data/wb97mv_tzvpd]
                                      [--stems w2_... h2o] [--limit N] [--strict]

For every geometry stem it finds, this merges the ``eda`` and ``force`` outputs
for the *same frame* into one extxyz frame carrying both the EDA components and
the analytic forces. Stems that exist only under ``force`` (the isolated ``h2o``
monomers) are written force-only, with the same schema minus the ``eda_*`` keys.

Why merging is legitimate: Q-Chem reorients each job into its own standard
nuclear orientation, but the eda and force jobs are given the identical input
geometry and produce **identical** oriented coordinates. That is checked per
frame to 1e-8 Angstrom rather than assumed, and any frame that disagrees is
dropped with a message. The supersystem energies are cross-checked too -- the
force job's SCF and the EDA job's CT-allowed SCF are the same wavefunction.

Output schema
-------------
Per-atom columns::

    species pos(3) mulliken_charges(1) fragment_idx(1) forces(3)

Forces are Hartree/bohr (``-dE/dR``), matching ``data/labels/*.extxyz`` and what
``rsfff.train.data.load_extxyz`` expects. Header keys are the ones
``scripts/parse_qchem_eda.py`` writes, plus:

    fragment_dipoles         n_fragments x 3 values, e*a0
    fragment_second_moments  n_fragments x 6 values, e*a0^2, the unique Cartesian
                             components in Q-Chem's print order (XX XY YY XZ YZ ZZ)

Both are the **isolated-fragment** (frozen monomer) moments, and both are
**primitive** (traced) moments **about the coordinate origin** -- deliberately
unshifted. The consumer shifts them to whatever center it uses and takes the
traceless part there, so that prediction and target travel through the same
algebra and a convention error cancels instead of biasing a fit. For the
force-only monomer files these are just the molecular moments, which is the same
thing for a one-fragment system.

`method` and `basis` come from the parsed ``$rem``, never from the file name:
the round-trip stems are inherited from the geometry files and name the previous
level of theory (``w2_wb97xv_qzvppd_frame0000.out`` is a wB97M-V/def2-TZVPD
job). Output file names are built from the parsed values, and a stem whose
frames disagree on either is refused.

``scripts/parse_qchem_eda.py`` remains the path for the flat CMM_Data tree,
which has no force jobs and one output per directory.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from qcgen.qchem_eda import (  # noqa: E402
    QChemEDAParseError,
    check_consistency as check_eda,
    parse_eda_output,
    to_atomic_units as eda_to_atomic_units,
    unique_components,
)
from qcgen.qchem_force import (  # noqa: E402
    check_consistency as check_force,
    parse_force_output,
    to_atomic_units as force_to_atomic_units,
)
from qcgen.qchem_out import KJMOL_PER_HARTREE  # noqa: E402

#: Order of EDA keys on the header line: the five requested components first,
#: then the bookkeeping terms. Matches scripts/parse_qchem_eda.py.
EDA_KEY_ORDER = (
    "cls_elec", "mod_pauli", "disp", "pol", "ct",
    "prp", "frz", "int", "elec", "pauli", "cls_pauli", "cls_disp",
)

MULTIPOLE_NAMES = ("dipole", "quadrupole", "octopole", "hexadecapole")

#: Positions must agree between the two jobs to this tolerance (Angstrom).
#: Q-Chem prints 10 decimals, so identical orientations agree exactly; anything
#: above round-off means the jobs saw different geometries.
GEOM_ATOL = 1e-8
#: Supersystem energies must agree to this (Hartree). Both come from a 10-decimal
#: SCF iteration row of the same wavefunction, but they are two *independent* SCF
#: runs converged to ``SCF_CONVERGENCE = 8``, so they legitimately differ at the
#: 1e-8 level -- measured across w4/w5, the disagreements cluster tightly at
#: 1.00e-8 to 1.02e-8 Ha. This threshold is loose enough to accept convergence
#: noise (1e-6 Ha is 2.6e-3 kJ/mol, far below anything trained on) and tight
#: enough that a genuinely mismatched frame, which would differ in the millihartree,
#: is still caught.
ENERGY_ATOL = 1e-6


def slug(text: str) -> str:
    """``wB97M-V`` -> ``wb97mv``; ``def2-TZVPD`` -> ``tzvpd``."""
    text = re.sub(r"^def2[-_]?", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def system_label(stem: str) -> str:
    """The system part of a geometry stem, dropping any level-of-theory suffix.

    ``w2_wb97xv_qzvppd`` -> ``w2`` (that suffix is stale -- see the module
    docstring); ``h2o`` -> ``h2o``.
    """
    head = stem.split("_")[0]
    return head if re.fullmatch(r"w\d+", head) else stem


def frame_index(path: str) -> int | None:
    """The ``NNNN`` of ``<stem>_frameNNNN.out``, or None for a single-frame stem."""
    m = re.search(r"_frame(\d+)\.out$", os.path.basename(path))
    return int(m.group(1)) if m else None


def outputs_by_frame(calc_dir: str, stem: str) -> dict[int, str]:
    """``{frame index: path}`` for one stem; a single-frame stem maps to key 0."""
    out_dir = os.path.join(calc_dir, "outputs")
    if not os.path.isdir(out_dir):
        return {}
    found: dict[int, str] = {}
    for name in os.listdir(out_dir):
        if not name.endswith(".out") or not name.startswith(stem):
            continue
        rest = name[len(stem) :]
        if rest == ".out":
            found[0] = os.path.join(out_dir, name)
        else:
            m = re.fullmatch(r"_frame(\d+)\.out", rest)
            if m:
                found[int(m.group(1))] = os.path.join(out_dir, name)
    return found


def geometry_stems(calc_dir: str) -> list[str]:
    geom_dir = os.path.join(calc_dir, "geoms")
    if not os.path.isdir(geom_dir):
        return []
    return sorted(
        os.path.splitext(n)[0]
        for n in os.listdir(geom_dir)
        if n.endswith((".xyz", ".extxyz"))
    )


def fmt(arr) -> str:
    # 13 significant digits: enough to hold Q-Chem's 10-decimal fragment
    # energies (~-76.4366986719) without dropping the last digit.
    return " ".join(f"{v:.12e}" for v in np.asarray(arr).ravel())


def config_type(symbols, fragment_idx, n_fragments: int) -> str:
    """``w<n>`` for a cluster of n waters, else the per-fragment formula list."""
    formulas = []
    for f in range(n_fragments):
        syms = [s for s, i in zip(symbols, fragment_idx) if i == f]
        counts = {s: syms.count(s) for s in dict.fromkeys(sorted(syms))}
        formulas.append("".join(s + (str(c) if c > 1 else "") for s, c in counts.items()))
    if all(f == "H2O" for f in formulas):
        return f"w{n_fragments}"
    return "_".join(formulas)


class Frame:
    """One merged frame, ready to be written."""

    def __init__(
        self,
        *,
        symbols,
        positions,
        forces,
        mulliken,
        fragment_idx,
        energy,
        eda,
        fragment_energies,
        fragment_charges,
        fragment_mults,
        fragment_dipoles,
        fragment_second_moments,
        multipoles,
        total_charge,
        multiplicity,
        method,
        basis,
        sample_id,
        source,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def n_fragments(self) -> int:
        return len(self.fragment_charges)

    def header(self) -> str:
        keys = [
            "Properties=species:S:1:pos:R:3:mulliken_charges:R:1:"
            "fragment_idx:I:1:forces:R:3",
            f"energy={self.energy:.12e}",
        ]
        for name in EDA_KEY_ORDER:
            if name in self.eda:
                keys.append(f"eda_{name}={self.eda[name]:.12e}")
        keys.append(f'fragment_energies="{fmt(self.fragment_energies)}"')
        keys.append(f'fragment_dipoles="{fmt(self.fragment_dipoles)}"')
        keys.append(f'fragment_second_moments="{fmt(self.fragment_second_moments)}"')
        for name in MULTIPOLE_NAMES:
            keys.append(f'{name}="{fmt(self.multipoles[name])}"')
        keys += [
            "multipole_format=tensor",
            f"n_fragments={self.n_fragments}",
            f"charge={self.total_charge}",
            f"multiplicity={self.multiplicity}",
            f'fragment_charges="{" ".join(str(c) for c in self.fragment_charges)}"',
            f'fragment_multiplicities="{" ".join(str(m) for m in self.fragment_mults)}"',
            f"method={self.method}",
            f"basis={self.basis}",
            f"config_type={config_type(self.symbols, self.fragment_idx, self.n_fragments)}",
            f"sample_id={self.sample_id}",
            f"source={self.source}",
            "units=atomic",
        ]
        return " ".join(keys)

    def write(self, fh) -> None:
        fh.write(f"{self.n_atoms}\n{self.header()}\n")
        rows = zip(
            self.symbols, self.positions, self.mulliken, self.fragment_idx, self.forces
        )
        for sym, (x, y, z), q, f, (fx, fy, fz) in rows:
            fh.write(
                f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f} {q:16.10f} {int(f):4d}"
                f" {fx:22.14e} {fy:22.14e} {fz:22.14e}\n"
            )


def fragment_moments(rec) -> tuple[np.ndarray, np.ndarray]:
    """``(dipoles (F,3), second moments (F,6))`` from the isolated-fragment blocks."""
    dipoles = np.array([m["dipole"] for m in rec.fragment_multipoles])
    seconds = np.array(
        [unique_components(m["quadrupole"], "quadrupole") for m in rec.fragment_multipoles]
    )
    return dipoles, seconds


def build_merged_frame(eda_path: str, force_path: str, relative_to: str) -> tuple[Frame, list[str]]:
    """Parse and merge one eda/force pair. Raises on anything that makes it unusable."""
    eda = eda_to_atomic_units(parse_eda_output(eda_path))
    force = force_to_atomic_units(parse_force_output(force_path))

    if eda.symbols != force.symbols:
        raise QChemEDAParseError("eda and force jobs disagree on the element list")
    delta = float(np.abs(eda.positions - force.positions).max())
    if delta > GEOM_ATOL:
        raise QChemEDAParseError(
            f"eda and force geometries differ by {delta:.3g} Angstrom; the two jobs were "
            "oriented differently, so their forces and multipoles are in different frames"
        )
    if abs(eda.energy - force.energy) > ENERGY_ATOL:
        raise QChemEDAParseError(
            f"eda CT-allowed energy {eda.energy:.10f} != force job energy "
            f"{force.energy:.10f}; these should be the same wavefunction"
        )
    if not eda.has_fragment_blocks:
        raise QChemEDAParseError(
            "no isolated-fragment multipole blocks; the job needs SCF_PRINT_FRGM = true"
        )

    msgs = [f"eda: {m}" for m in check_eda(eda, atol=1e-3 / KJMOL_PER_HARTREE)]
    msgs += [f"force: {m}" for m in check_force(force)]

    dipoles, seconds = fragment_moments(eda)
    frame = Frame(
        symbols=eda.symbols,
        positions=eda.positions,
        forces=force.forces,
        mulliken=eda.mulliken_charges,
        fragment_idx=eda.fragment_idx,
        energy=eda.energy,
        eda=eda.eda,
        fragment_energies=eda.fragment_energies,
        fragment_charges=eda.fragment_charges,
        fragment_mults=eda.fragment_mults,
        fragment_dipoles=dipoles,
        fragment_second_moments=seconds,
        multipoles=eda.multipoles,
        total_charge=eda.total_charge,
        multiplicity=eda.multiplicity,
        method=eda.method,
        basis=eda.basis,
        sample_id=frame_index(eda_path) or 0,
        source=os.path.relpath(eda_path, relative_to),
    )
    return frame, msgs


def build_force_frame(force_path: str, relative_to: str) -> tuple[Frame, list[str]]:
    """A force-only frame: one fragment, no EDA components."""
    force = force_to_atomic_units(parse_force_output(force_path))
    n = force.n_atoms
    frame = Frame(
        symbols=force.symbols,
        positions=force.positions,
        forces=force.forces,
        mulliken=force.mulliken_charges,
        fragment_idx=np.zeros(n, dtype=int),
        energy=force.energy,
        eda={},
        fragment_energies=np.array([force.energy]),
        fragment_charges=[force.total_charge],
        fragment_mults=[force.multiplicity],
        fragment_dipoles=force.multipoles["dipole"][None, :],
        fragment_second_moments=unique_components(
            force.multipoles["quadrupole"], "quadrupole"
        )[None, :],
        multipoles=force.multipoles,
        total_charge=force.total_charge,
        multiplicity=force.multiplicity,
        method=force.method,
        basis=force.basis,
        sample_id=frame_index(force_path) or 0,
        source=os.path.relpath(force_path, relative_to),
    )
    return frame, [f"force: {m}" for m in check_force(force)]


def write_stem(stem: str, root: str, out_dir: str, args) -> str | None:
    """Parse every frame of one stem and write its extxyz. Returns the path written."""
    eda_frames = outputs_by_frame(os.path.join(root, "eda"), stem)
    force_frames = outputs_by_frame(os.path.join(root, "force"), stem)
    if not force_frames:
        print(f"{stem}: no force outputs, skipping", file=sys.stderr)
        return None
    merged = bool(eda_frames)
    indices = sorted(set(eda_frames) & set(force_frames)) if merged else sorted(force_frames)
    if args.limit:
        indices = indices[: args.limit]
    if not indices:
        print(f"{stem}: no frames present in both eda and force, skipping", file=sys.stderr)
        return None

    kind = "eda+force" if merged else "force-only"
    print(f"{stem}: {len(indices)} {kind} frame(s)", file=sys.stderr)

    frames: list[Frame] = []
    n_dropped, n_warned = 0, 0
    for k, i in enumerate(indices):
        try:
            if merged:
                frame, msgs = build_merged_frame(eda_frames[i], force_frames[i], root)
            else:
                frame, msgs = build_force_frame(force_frames[i], root)
        except (QChemEDAParseError, OSError, ValueError) as exc:
            n_dropped += 1
            print(f"  dropped frame {i}: {exc}", file=sys.stderr)
            continue
        if msgs:
            n_warned += 1
            for m in msgs:
                print(f"  frame {i}: {m}", file=sys.stderr)
            if args.strict:
                raise SystemExit(f"{stem}: frame {i} failed a consistency check")
        frames.append(frame)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(indices)}", file=sys.stderr)

    if not frames:
        print(f"{stem}: nothing usable", file=sys.stderr)
        return None

    methods = {(f.method, f.basis) for f in frames}
    if len(methods) != 1:
        raise SystemExit(f"{stem}: frames disagree on method/basis: {sorted(methods)}")
    method, basis = methods.pop()

    name = f"{system_label(stem)}_{slug(method)}_{slug(basis)}.xyz"
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        for frame in frames:
            frame.write(fh)
    print(
        f"  -> {path}: {len(frames)} frame(s)"
        f"{f', {n_dropped} dropped' if n_dropped else ''}"
        f"{f', {n_warned} with warnings' if n_warned else ''}",
        file=sys.stderr,
    )
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge qchem_roundtrip eda/force outputs into extxyz datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default="qchem_roundtrip", help="the round-trip bundle")
    ap.add_argument("--out-dir", default="data/wb97mv_tzvpd", help="where to write extxyz")
    ap.add_argument("--stems", nargs="*", default=None, help="geometry stems (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames per stem")
    ap.add_argument(
        "--strict", action="store_true", help="abort on any consistency warning"
    )
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.root))
    stems = args.stems
    if stems is None:
        stems = sorted(
            set(geometry_stems(os.path.join(root, "eda")))
            | set(geometry_stems(os.path.join(root, "force")))
        )
    if not stems:
        print(f"error: no geometry stems under {root}", file=sys.stderr)
        return 1

    written = [write_stem(s, root, args.out_dir, args) for s in stems]
    return 0 if any(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
