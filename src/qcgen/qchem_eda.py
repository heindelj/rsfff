"""Parser for Q-Chem ALMO-EDA (EDA2) output files.

Extracts, from a single ``eda.out``:

  * the fragment definition (charge/multiplicity/atom counts) echoed in the
    ``$molecule`` block of the ``User input:`` section,
  * the supersystem geometry (the first ``Standard Nuclear Orientation`` block,
    which is the frame the multipoles are reported in),
  * the total electronic energy of the supersystem (the converged SCF energy of
    the final, CT-allowed wavefunction),
  * the isolated-fragment energies (``Fragment Energies (Ha)``),
  * the EDA terms -- ``cls_elec``, ``mod_pauli``, ``disp``, ``pol``, ``ct`` --
    plus ``prp``/``frz``/``int`` for cross-checking,
  * the ground-state Mulliken charges and the Cartesian multipole moments
    (charge through hexadecapole) printed at the end of the file.

Unit handling: Q-Chem prints EDA terms in kJ/mol and multipoles in
Debye-Angstrom^(n-1). :func:`to_atomic_units` converts a parsed record in place
to the atomic units used everywhere else in this repo (Hartree, e*a0^n), which
is what the writer emits by default. Positions stay in Angstrom.

The five requested EDA components decompose the interaction energy exactly::

    E_int = E_prp + E_cls_elec + E_mod_pauli + E_disp + E_pol + E_ct

and ``E_int = E_total - sum(E_fragment)`` -- the latter only to ~1.3e-5 relative,
because Q-Chem converts the EDA terms to kJ/mol with an internal constant of
~2625.5323 rather than CODATA's 2625.4996. For a -50 kJ/mol interaction that is
a ~7e-4 kJ/mol (~3e-7 Hartree) discrepancy; :func:`check_consistency` absorbs it
with a relative tolerance rather than pretending it away.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

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

_RANK = {"dipole": 1, "quadrupole": 2, "octopole": 3, "hexadecapole": 4}


class QChemEDAParseError(RuntimeError):
    """Raised when a required section is missing or malformed."""


@dataclass
class EDARecord:
    """Everything pulled out of one Q-Chem EDA output file."""

    path: str = ""
    symbols: list[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: 0-based fragment index per atom, in file order.
    fragment_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    fragment_charges: list[int] = field(default_factory=list)
    fragment_mults: list[int] = field(default_factory=list)
    total_charge: int = 0
    multiplicity: int = 1
    #: Supersystem SCF energy of the CT-allowed wavefunction (Hartree).
    energy: float = float("nan")
    #: Isolated-fragment SCF energies (Hartree).
    fragment_energies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: EDA terms, keyed by short name (cls_elec, mod_pauli, disp, pol, ct, ...).
    eda: dict[str, float] = field(default_factory=dict)
    mulliken_charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Cartesian multipoles as full symmetric tensors: (3,), (3,3), (3,3,3), (3,)*4.
    multipoles: dict[str, np.ndarray] = field(default_factory=dict)
    method: str = ""
    basis: str = ""
    #: True when the final CT-allowed SCF printed "Convergence criterion met".
    converged: bool = False
    units: str = "qchem"

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def n_fragments(self) -> int:
        return len(self.fragment_charges)

    def interaction_energy(self) -> float:
        """``E_total - sum(E_fragment)``, in whatever units the record holds."""
        return self.energy - float(np.sum(self.fragment_energies))


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

_SNO_HEADER = "Standard Nuclear Orientation (Angstroms)"
_GEOM_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,3})\s+" + r"\s+".join([r"(-?\d+\.\d+)"] * 3))
#: One SCF iteration row: counter, energy, DIIS error. Q-Chem uses a fixed-width
#: energy field, so a pathological (very large negative) energy runs straight
#: into the counter with no separating space -- hence the ``(?=-)`` alternative.
_SCF_ITER = re.compile(r"^\s*\d+(?:\s+|(?=-))(-?\d+\.\d+)\s+\d\.\d{2}e[+-]\d{2}")

#: EDA terms and the substring that identifies their line. Order matters only for
#: readability; ``E_cls_elec``/``E_cls_pauli`` must be matched before ``E_elec``
#: would be, which the explicit tokens below guarantee.
_EDA_PATTERNS = {
    "prp": r"E_prp\s*\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "frz": r"E_frz\s*\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "pol": r"E_pol\s*\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "ct": r"E_vct\s*\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "int": r"E_int\s*\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "elec": r"E_elec\s+\(ELEC\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "pauli": r"E_pauli\s+\(PAULI\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "disp": r"E_disp\s+\(DISP\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "cls_elec": r"E_cls_elec\s+\(CLS ELEC\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "cls_pauli": r"E_cls_pauli\s+\(CLS PAULI\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    "mod_pauli": r"E_mod_pauli\s+\(MOD PAULI\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
    # Present only when the job ran with cls_disp = 1.
    "cls_disp": r"E_cls_disp\s+\(CLS DISP\)\s+\(kJ/mol\)\s*=\s*(-?\d+\.\d+)",
}

#: The subset that must be present for a record to be usable.
REQUIRED_EDA_TERMS = ("cls_elec", "mod_pauli", "disp", "pol", "ct", "int")


def _parse_molecule_block(lines: list[str]) -> tuple[list[int], list[int], list[int], int, int]:
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
        raise QChemEDAParseError("no $molecule block in the User input echo")

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
        raise QChemEDAParseError("$molecule block declares no fragments")
    return frag_charges, frag_mults, frag_natoms, total_charge, multiplicity


def _parse_geometry(lines: list[str], start: int) -> tuple[list[str], np.ndarray]:
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
        raise QChemEDAParseError("empty Standard Nuclear Orientation table")
    return symbols, np.array(coords)


def _parse_ct_energy(lines: list[str]) -> tuple[float, bool]:
    """Return ``(energy, converged)`` for the final CT-allowed supersystem SCF.

    Q-Chem's EDA2 driver prints no "Total energy" summary line, so the total
    electronic energy is the last SCF iteration energy of the CT-allowed
    wavefunction -- the block between the "CT-Allowed wavefunction" banner and
    the "Results of EDA2" header.
    """
    try:
        lo = max(i for i, ln in enumerate(lines) if "CT-Allowed wavefunction" in ln)
    except ValueError:
        raise QChemEDAParseError("no CT-allowed SCF section (incomplete EDA job?)")
    his = [i for i, ln in enumerate(lines) if "Results of EDA2" in ln and i > lo]
    hi = his[0] if his else len(lines)

    energy, converged = None, False
    for ln in lines[lo:hi]:
        m = _SCF_ITER.match(ln)
        if m:
            energy = float(m.group(1))
            converged = "Convergence criterion met" in ln
    if energy is None:
        raise QChemEDAParseError("no SCF iterations found in the CT-allowed section")
    return energy, converged


def _parse_fragment_energies(lines: list[str]) -> np.ndarray:
    try:
        start = max(i for i, ln in enumerate(lines) if "Fragment Energies (Ha)" in ln)
    except ValueError:
        raise QChemEDAParseError("no 'Fragment Energies (Ha)' block")
    energies = []
    for ln in lines[start + 1 :]:
        toks = ln.split()
        if len(toks) == 2 and toks[0].isdigit():
            energies.append(float(toks[1]))
        elif energies:
            break
    if not energies:
        raise QChemEDAParseError("empty fragment-energy block")
    return np.array(energies)


def _parse_eda_terms(text: str) -> dict[str, float]:
    terms = {}
    for name, pattern in _EDA_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            terms[name] = float(m.group(1))
    missing = [t for t in REQUIRED_EDA_TERMS if t not in terms]
    if missing:
        raise QChemEDAParseError(f"missing EDA terms: {', '.join(missing)}")
    return terms


def _parse_mulliken(lines: list[str], n_atoms: int) -> np.ndarray:
    """Parse the last ``Ground-State Mulliken Net Atomic Charges`` table."""
    idx = [i for i, ln in enumerate(lines) if "Mulliken Net Atomic Charges" in ln]
    if not idx:
        return np.full(n_atoms, np.nan)
    charges = []
    for ln in lines[idx[-1] :]:
        toks = ln.split()
        if len(toks) == 3 and toks[0].isdigit() and toks[1].isalpha():
            charges.append(float(toks[2]))
        elif charges:
            break
    if len(charges) != n_atoms:
        raise QChemEDAParseError(
            f"Mulliken table has {len(charges)} rows but the system has {n_atoms} atoms"
        )
    return np.array(charges)


def _expand_multipole(unique: dict[str, float], rank: int) -> np.ndarray:
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


def _parse_multipoles(lines: list[str]) -> dict[str, np.ndarray]:
    """Parse the last ``Cartesian Multipole Moments`` block into full tensors.

    Components are read as ``LABEL value`` token pairs inside each rank's
    subsection, which is layout-independent (Q-Chem wraps them across lines in a
    width-dependent way). The ``Tot`` entry on the dipole line is skipped.
    """
    idx = [i for i, ln in enumerate(lines) if "Cartesian Multipole Moments" in ln]
    if not idx:
        raise QChemEDAParseError("no 'Cartesian Multipole Moments' block")

    out: dict[str, np.ndarray] = {}
    current, unique, started = None, {}, False

    def flush():
        if current is None:
            return
        expected = MULTIPOLE_LABELS[current]
        if set(unique) != set(expected):
            raise QChemEDAParseError(
                f"{current}: expected components {expected}, got {sorted(unique)}"
            )
        out[current] = _expand_multipole(unique, _RANK[current])

    for ln in lines[idx[-1] + 1 :]:
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
        raise QChemEDAParseError(f"multipole block missing: {', '.join(missing)}")
    return out


def _parse_rem(lines: list[str]) -> dict[str, str]:
    """Return the ``$rem`` section as a lowercase-keyed dict."""
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "$rem")
        end = next(i for i in range(start, len(lines)) if lines[i].strip().lower() == "$end")
    except StopIteration:
        return {}
    rem = {}
    for ln in lines[start + 1 : end]:
        toks = ln.split()
        if len(toks) >= 2:
            rem[toks[0].lower()] = toks[1]
    return rem


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def parse_eda_output(path: str) -> EDARecord:
    """Parse one Q-Chem EDA2 output file into an :class:`EDARecord` (native units)."""
    with open(path, errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()

    if "Results of EDA2" not in text:
        raise QChemEDAParseError("no 'Results of EDA2' section (job failed or was truncated?)")

    frag_charges, frag_mults, frag_natoms, total_charge, multiplicity = _parse_molecule_block(lines)

    sno = [i for i, ln in enumerate(lines) if _SNO_HEADER in ln]
    if not sno:
        raise QChemEDAParseError("no Standard Nuclear Orientation block")
    # The first table is the supersystem; later ones (printed only with
    # SCF_PRINT_FRGM) are the isolated fragments.
    symbols, positions = _parse_geometry(lines, sno[0])

    n_expected = sum(frag_natoms)
    if len(symbols) != n_expected:
        raise QChemEDAParseError(
            f"geometry has {len(symbols)} atoms but $molecule declares {n_expected}"
        )

    energy, converged = _parse_ct_energy(lines)
    fragment_energies = _parse_fragment_energies(lines)
    if len(fragment_energies) != len(frag_charges):
        raise QChemEDAParseError(
            f"{len(fragment_energies)} fragment energies for {len(frag_charges)} fragments"
        )

    rem = _parse_rem(lines)
    return EDARecord(
        path=path,
        symbols=symbols,
        positions=positions,
        fragment_idx=np.repeat(np.arange(len(frag_natoms)), frag_natoms),
        fragment_charges=frag_charges,
        fragment_mults=frag_mults,
        total_charge=total_charge,
        multiplicity=multiplicity,
        energy=energy,
        fragment_energies=fragment_energies,
        eda=_parse_eda_terms(text),
        mulliken_charges=_parse_mulliken(lines, len(symbols)),
        multipoles=_parse_multipoles(lines),
        method=rem.get("method") or rem.get("exchange", ""),
        basis=rem.get("basis", ""),
        converged=converged,
        units="qchem",
    )


def to_atomic_units(rec: EDARecord) -> EDARecord:
    """Convert EDA terms (kJ/mol -> Hartree) and multipoles (Debye-Ang^n -> e*a0^n).

    Mutates and returns ``rec``. Energies already in Hartree (total, fragment)
    and positions (Angstrom) are untouched. Idempotent guard: raises if the
    record has already been converted.
    """
    if rec.units == "atomic":
        raise ValueError("record is already in atomic units")
    rec.eda = {k: v / KJMOL_PER_HARTREE for k, v in rec.eda.items()}
    for name, tensor in rec.multipoles.items():
        rank = _RANK[name]
        rec.multipoles[name] = tensor * (
            BOHR_PER_ANGSTROM ** (rank - 1) / DEBYE_PER_AU_DIPOLE
        )
    rec.units = "atomic"
    return rec


def check_consistency(
    rec: EDARecord,
    atol: float = 1e-3,
    rtol: float = 1e-4,
    max_int_energy: float = 1000.0,
) -> list[str]:
    """Return a list of human-readable warnings about internal inconsistencies.

    Checks that (a) the five EDA components plus preparation reproduce ``E_int``,
    (b) ``E_int`` matches ``E_total - sum(E_fragment)``, (c) ``E_int`` is
    physically plausible, and (d) the final SCF converged.

    ``atol`` is in whatever units ``rec`` holds (kJ/mol natively, Hartree after
    :func:`to_atomic_units`); it needs to cover Q-Chem's 4-decimal kJ/mol
    printing of six summed terms. ``rtol`` covers check (b): Q-Chem's internal
    Hartree->kJ/mol constant is ~2625.5323 against CODATA's 2625.4996, so the
    two sides of that identity differ by ~1.3e-5 relative regardless of which
    constant we convert with.

    Check (c) is what catches a variationally collapsed CT-allowed SCF: the
    fragment energies stay sane while the supersystem falls to a nonsensical
    minimum, so (a) and (b) both still hold internally. ``max_int_energy`` is in
    kJ/mol regardless of ``rec.units``.
    """
    msgs = []
    parts = ("prp", "cls_elec", "mod_pauli", "disp", "pol", "ct")
    e_int = rec.eda["int"]
    tol = atol + rtol * abs(e_int)

    summed = sum(rec.eda.get(p, 0.0) for p in parts)
    if abs(summed - e_int) > tol:
        msgs.append(f"EDA components sum to {summed:.6g} but E_int is {e_int:.6g}")

    scale = 1.0 if rec.units == "atomic" else KJMOL_PER_HARTREE
    direct = rec.interaction_energy() * scale
    if abs(direct - e_int) > tol:
        msgs.append(f"E_total - sum(E_frag) = {direct:.6g} but E_int is {e_int:.6g}")

    e_int_kj = e_int * (KJMOL_PER_HARTREE if rec.units == "atomic" else 1.0)
    if abs(e_int_kj) > max_int_energy:
        msgs.append(
            f"implausible interaction energy {e_int_kj:.6g} kJ/mol "
            "(SCF likely collapsed to a spurious solution)"
        )

    if not rec.converged:
        msgs.append("final CT-allowed SCF did not report convergence")
    return msgs
