"""Section readers shared by every Q-Chem output parser in this package.

A Q-Chem output is a sequence of labelled blocks, and the blocks a plain SCF job
prints -- geometry, Mulliken charges, Cartesian multipoles, the echoed ``$rem``
and ``$molecule`` -- are byte-for-byte the same ones an EDA2 job prints. They
were private to :mod:`rsfff.qcgen.qchem_eda` until a second job type
(:mod:`rsfff.qcgen.qchem_force`) needed them; this module is that extraction, so
there is exactly one implementation of each and one place a Q-Chem formatting
change has to be absorbed.

**Read method and basis from the echoed ``$rem``, never from a filename.** The
round-trip bundle in ``qchem_roundtrip/`` names its outputs after the geometry
files they came from (``w2_wb97xv_qzvppd_frame0000.out``), and those names
describe the *previous* level of theory -- every one of those jobs is actually
wB97M-V/def2-TZVPD. A filename is a label somebody typed; the ``$rem`` echo is
what the program ran.

Coordinates come from ``Standard Nuclear Orientation``, which is Q-Chem's own
frame: it translates the system so its **center of nuclear charge sits at the
origin** and may rotate it. That matters because the Cartesian multipole moments
are reported about that origin, so positions, gradients and multipoles are only
mutually consistent when all three are taken from the output rather than from
the input geometry.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

# ---------------------------------------------------------------------------
# Unit conversions (CODATA 2018)
# ---------------------------------------------------------------------------

KJMOL_PER_HARTREE = 2625.4996394798254
DEBYE_PER_AU_DIPOLE = 2.5417464519  # e*a0 -> Debye
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

#: Multipole ranks in print order, with their unique Cartesian component labels
#: exactly as Q-Chem writes them.
MULTIPOLE_LABELS = {
    "dipole": ["X", "Y", "Z"],
    "quadrupole": ["XX", "XY", "YY", "XZ", "YZ", "ZZ"],
    "octopole": ["XXX", "XXY", "XYY", "YYY", "XXZ", "XYZ", "YYZ", "XZZ", "YZZ", "ZZZ"],
    "hexadecapole": [
        "XXXX", "XXXY", "XXYY", "XYYY", "YYYY", "XXXZ", "XXYZ", "XYYZ",
        "YYYZ", "XXZZ", "XYZZ", "YYZZ", "XZZZ", "YZZZ", "ZZZZ",
    ],
}

_MULTIPOLE_HEADINGS = {
    "Dipole Moment": "dipole",
    "Quadrupole Moments": "quadrupole",
    "Octopole Moments": "octopole",
    "Hexadecapole Moments": "hexadecapole",
}

RANK = {"dipole": 1, "quadrupole": 2, "octopole": 3, "hexadecapole": 4}

#: Written at the end of every job that ran to completion. Absent from a file
#: that was truncated by a wall-clock kill, which otherwise looks parseable.
JOB_COMPLETE_MARKER = "Thank you very much for using Q-Chem"


class QChemParseError(RuntimeError):
    """Raised when a required section is missing or malformed."""


SNO_HEADER = "Standard Nuclear Orientation (Angstroms)"
_GEOM_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,3})\s+" + r"\s+".join([r"(-?\d+\.\d+)"] * 3))

#: One SCF iteration row: counter, energy, DIIS error. Q-Chem uses a fixed-width
#: energy field, so a pathological (very large negative) energy runs straight
#: into the counter with no separating space -- hence the ``(?=-)`` alternative.
#: Worth preferring over the ``Total energy =`` summary line, which is printed to
#: only 8 decimals against the iteration row's 10.
SCF_ITER = re.compile(r"^\s*\d+(?:\s+|(?=-))(-?\d+\.\d+)\s+\d\.\d{2}e[+-]\d{2}")


def find_all(lines: list[str], needle: str) -> list[int]:
    """Indices of every line containing ``needle``."""
    return [i for i, ln in enumerate(lines) if needle in ln]


def job_completed(text: str) -> bool:
    """Whether Q-Chem printed its sign-off, i.e. the job was not truncated."""
    return JOB_COMPLETE_MARKER in text


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def parse_geometry(lines: list[str], start: int) -> tuple[list[str], np.ndarray]:
    """Parse the ``Standard Nuclear Orientation`` table whose header is at ``start``."""
    symbols, coords = [], []
    for ln in lines[start + 1 :]:
        m = _GEOM_ROW.match(ln)
        if m:
            symbols.append(m.group(2))
            coords.append([float(m.group(3)), float(m.group(4)), float(m.group(5))])
        elif symbols:  # table ended
            break
    if not symbols:
        raise QChemParseError("empty Standard Nuclear Orientation table")
    return symbols, np.array(coords)


def parse_mulliken(lines: list[str], n_atoms: int, start: int | None = None) -> np.ndarray:
    """Parse a ``Ground-State Mulliken Net Atomic Charges`` table.

    ``start`` selects which table; the default is the last one in ``lines``,
    which for a supersystem job is the one belonging to the final wavefunction.
    """
    if start is None:
        idx = find_all(lines, "Mulliken Net Atomic Charges")
        if not idx:
            return np.full(n_atoms, np.nan)
        start = idx[-1]
    charges = []
    for ln in lines[start:]:
        toks = ln.split()
        if len(toks) == 3 and toks[0].isdigit() and toks[1].isalpha():
            charges.append(float(toks[2]))
        elif charges:
            break
    if len(charges) != n_atoms:
        raise QChemParseError(
            f"Mulliken table has {len(charges)} rows but the system has {n_atoms} atoms"
        )
    return np.array(charges)


def expand_multipole(unique: dict[str, float], rank: int) -> np.ndarray:
    """Scatter unique Cartesian components into a full symmetric rank-n tensor."""
    axis = {"X": 0, "Y": 1, "Z": 2}
    tensor = np.zeros((3,) * rank)
    for label, value in unique.items():
        base = tuple(axis[c] for c in label)
        for perm in set(itertools.permutations(base)):
            tensor[perm] = value
    return tensor


def unique_components(tensor: np.ndarray, name: str) -> np.ndarray:
    """Inverse of the symmetric expansion: pull the unique components back out.

    Returns the independent Cartesian components in the order Q-Chem prints them
    (``MULTIPOLE_LABELS[name]``): 3, 6, 10, 15 values for ranks 1--4.
    """
    axis = {"X": 0, "Y": 1, "Z": 2}
    return np.array(
        [tensor[tuple(axis[c] for c in label)] for label in MULTIPOLE_LABELS[name]]
    )


def parse_multipoles(lines: list[str], start: int | None = None) -> dict[str, np.ndarray]:
    """Parse a ``Cartesian Multipole Moments`` block into full symmetric tensors.

    ``start`` selects which block; the default is the last one, which for a
    supersystem job is the final wavefunction's. Components are read as
    ``LABEL value`` token pairs inside each rank's subsection, which is
    layout-independent (Q-Chem wraps them across lines in a width-dependent
    way). The ``Tot`` entry on the dipole line is skipped.

    The moments are about the origin of the coordinate system the block belongs
    to -- see this module's docstring; they are *not* referenced to any
    molecular center.
    """
    if start is None:
        idx = find_all(lines, "Cartesian Multipole Moments")
        if not idx:
            raise QChemParseError("no 'Cartesian Multipole Moments' block")
        start = idx[-1]

    out: dict[str, np.ndarray] = {}
    current, unique, started = None, {}, False

    def flush():
        if current is None:
            return
        expected = MULTIPOLE_LABELS[current]
        if set(unique) != set(expected):
            raise QChemParseError(
                f"{current}: expected components {expected}, got {sorted(unique)}"
            )
        out[current] = expand_multipole(unique, RANK[current])

    for ln in lines[start + 1 :]:
        stripped = ln.strip()
        heading = next((v for k, v in _MULTIPOLE_HEADINGS.items() if stripped.startswith(k)), None)
        if heading is not None:
            flush()
            current, unique, started = heading, {}, True
            continue
        if stripped.startswith("Charge ("):
            flush()
            current, unique, started = None, {}, True
            continue
        # The block opens and closes with a dashed rule; only the closing one
        # ends the scan.
        if started and stripped and set(stripped) <= set("- "):
            break
        if current is None:
            continue
        toks = stripped.split()
        for label, value in zip(toks[0::2], toks[1::2]):
            if label in MULTIPOLE_LABELS[current]:
                unique[label] = float(value)
    flush()

    missing = [k for k in MULTIPOLE_LABELS if k not in out]
    if missing:
        raise QChemParseError(f"multipole block missing: {', '.join(missing)}")
    return out


def multipoles_to_atomic_units(multipoles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Debye-Angstrom^(n-1) -> e*a0^n, per rank."""
    return {
        name: tensor * (BOHR_PER_ANGSTROM ** (RANK[name] - 1) / DEBYE_PER_AU_DIPOLE)
        for name, tensor in multipoles.items()
    }


def parse_rem(lines: list[str]) -> dict[str, str]:
    """Return the echoed ``$rem`` section as a lowercase-keyed dict.

    Q-Chem accepts both ``KEY VALUE`` and ``KEY = VALUE`` and echoes whichever
    the input used -- the CMM_Data jobs wrote the first, the ``qchem_roundtrip``
    templates write the second. Reading only the second token would have made
    ``method`` come back as the literal string ``"="``, which is exactly the kind
    of failure that survives into a dataset header unnoticed.
    """
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "$rem")
        end = next(i for i in range(start, len(lines)) if lines[i].strip().lower() == "$end")
    except StopIteration:
        return {}
    rem = {}
    for ln in lines[start + 1 : end]:
        toks = [t for t in ln.split() if t != "="]
        if len(toks) >= 2:
            rem[toks[0].lower()] = toks[1]
    return rem


def method_and_basis(rem: dict[str, str]) -> tuple[str, str]:
    """``(method, basis)`` from a parsed ``$rem``, tolerating the ``exchange`` spelling."""
    return rem.get("method") or rem.get("exchange", ""), rem.get("basis", "")


def parse_molecule_block(
    lines: list[str],
) -> tuple[list[int], list[int], list[int], int, int]:
    """Return ``(frag_charges, frag_mults, frag_natoms, total_charge, multiplicity)``.

    Reads the ``$molecule`` block echoed under ``User input:``. A fragmented
    block opens with the supersystem charge/multiplicity, then one ``--``
    separated section per fragment. Unfragmented blocks are handled too (a
    single implicit fragment).
    """
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "$molecule")
        end = next(i for i in range(start, len(lines)) if lines[i].strip().lower() == "$end")
    except StopIteration:
        raise QChemParseError("no $molecule block in the User input echo")

    body = [ln.strip() for ln in lines[start + 1 : end] if ln.strip()]
    # Split on the "--" fragment separators.
    sections: list[list[str]] = [[]]
    for ln in body:
        if ln.startswith("--"):
            sections.append([])
        else:
            sections[-1].append(ln)

    def charge_mult(tokens: list[str]) -> tuple[int, int]:
        return int(tokens[0]), int(tokens[1])

    if len(sections) == 1:  # unfragmented: one charge/mult line + atoms
        head = sections[0]
        chg, mult = charge_mult(head[0].split())
        return [chg], [mult], [len(head) - 1], chg, mult

    total_charge, multiplicity = charge_mult(sections[0][0].split())
    frag_charges, frag_mults, frag_natoms = [], [], []
    for sec in sections[1:]:
        if not sec:
            continue
        chg, mult = charge_mult(sec[0].split())
        frag_charges.append(chg)
        frag_mults.append(mult)
        frag_natoms.append(len(sec) - 1)
    if not frag_charges:
        raise QChemParseError("$molecule block declares no fragments")
    return frag_charges, frag_mults, frag_natoms, total_charge, multiplicity


__all__ = [
    "BOHR_PER_ANGSTROM",
    "DEBYE_PER_AU_DIPOLE",
    "JOB_COMPLETE_MARKER",
    "KJMOL_PER_HARTREE",
    "MULTIPOLE_LABELS",
    "RANK",
    "SNO_HEADER",
    "QChemParseError",
    "expand_multipole",
    "find_all",
    "job_completed",
    "method_and_basis",
    "multipoles_to_atomic_units",
    "parse_geometry",
    "parse_molecule_block",
    "parse_mulliken",
    "parse_multipoles",
    "parse_rem",
    "unique_components",
]
