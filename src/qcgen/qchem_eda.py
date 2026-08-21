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
    (charge through hexadecapole) printed at the end of the file,
  * and, when the job ran with ``SCF_PRINT_FRGM = true``, the same two
    quantities for each **isolated fragment** -- see below.

The per-fragment blocks are worth more than they look. Each frozen-fragment
sub-job inherits the supersystem's coordinates verbatim (its
``Standard Nuclear Orientation`` table is byte-identical to the corresponding
rows of the supersystem's), so its multipoles are reported in the *same frame
about the same origin*. That makes them directly comparable, and directly
summable: for a water dimer the fragment dipoles sum to 88% of the relaxed
supersystem dipole, the remainder being exactly the polarization and charge
transfer the EDA quantifies. They are the frozen-monomer reference a
distributed-multipole model wants, at every cluster geometry.

Their origin is the *supersystem's* center of nuclear charge, not each
fragment's, so the second moments are translation-dependent as written and have
to be shifted before use. That shift is deliberately left to the consumer: this
module reports what Q-Chem printed.

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

import re
from dataclasses import dataclass, field

import numpy as np

from .qchem_out import (
    BOHR_PER_ANGSTROM,
    DEBYE_PER_AU_DIPOLE,
    KJMOL_PER_HARTREE,
    MULTIPOLE_LABELS,
    SCF_ITER as _SCF_ITER,
    SNO_HEADER as _SNO_HEADER,
    QChemParseError,
    find_all,
    method_and_basis,
    multipoles_to_atomic_units,
    parse_geometry as _parse_geometry,
    parse_molecule_block as _parse_molecule_block,
    parse_mulliken as _parse_mulliken,
    parse_multipoles as _parse_multipoles,
    parse_rem as _parse_rem,
    unique_components,
)

#: The section readers live in :mod:`rsfff.qcgen.qchem_out` so a plain SCF job
#: and an EDA job share one implementation. They are re-exported here because
#: this module's public surface predates that split.
__all__ = [
    "BOHR_PER_ANGSTROM",
    "DEBYE_PER_AU_DIPOLE",
    "KJMOL_PER_HARTREE",
    "MULTIPOLE_LABELS",
    "REQUIRED_EDA_TERMS",
    "EDARecord",
    "QChemEDAParseError",
    "check_consistency",
    "parse_eda_output",
    "to_atomic_units",
    "unique_components",
]

#: Alias rather than a subclass: callers (``scripts/parse_qchem_eda.py``,
#: ``tests/test_qchem_eda.py``) catch this name, and the shared readers raise
#: the base, so the two must be the same class or ``--skip-errors`` would stop
#: catching half the failures.
QChemEDAParseError = QChemParseError


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
    #: Isolated-fragment Mulliken charges, one array per fragment. Empty unless
    #: the job ran with ``SCF_PRINT_FRGM = true``.
    fragment_mulliken: list[np.ndarray] = field(default_factory=list)
    #: Isolated-fragment Cartesian multipoles, one dict per fragment, **about the
    #: supersystem's origin** (see the module docstring). Empty unless
    #: ``SCF_PRINT_FRGM = true``.
    fragment_multipoles: list[dict[str, np.ndarray]] = field(default_factory=list)
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

    @property
    def has_fragment_blocks(self) -> bool:
        """Whether the isolated-fragment Mulliken/multipole blocks were printed."""
        return len(self.fragment_multipoles) == self.n_fragments > 0

    def interaction_energy(self) -> float:
        """``E_total - sum(E_fragment)``, in whatever units the record holds."""
        return self.energy - float(np.sum(self.fragment_energies))


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


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


def _parse_fragment_blocks(
    lines: list[str], sno: list[int], frag_natoms: list[int]
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Isolated-fragment Mulliken charges and multipoles, one entry per fragment.

    With ``SCF_PRINT_FRGM = true`` the output holds ``1 + n_fragments``
    ``Standard Nuclear Orientation`` tables: the supersystem first, then one per
    frozen-fragment sub-job in fragment order. Each sub-job prints exactly one
    Mulliken table and one multipole block before the next table appears, so
    "the first of each after fragment *k*'s geometry" identifies them without
    depending on Q-Chem's ``Spawning Job For Fragment`` wording.

    Returns two empty lists when the fragment blocks are absent, which is not an
    error -- it just means the job ran without ``SCF_PRINT_FRGM``.

    Each fragment's geometry is checked against the corresponding slice of the
    supersystem's. That check is what licenses treating the fragment multipoles
    as living in the supersystem's frame; without it a future Q-Chem that
    re-orients sub-jobs would silently produce moments in per-fragment frames
    that still look entirely plausible.
    """
    n_frag = len(frag_natoms)
    if len(sno) != n_frag + 1:
        return [], []

    _, super_positions = _parse_geometry(lines, sno[0])
    mulliken_idx = find_all(lines, "Mulliken Net Atomic Charges")
    multipole_idx = find_all(lines, "Cartesian Multipole Moments")

    charges: list[np.ndarray] = []
    multipoles: list[dict[str, np.ndarray]] = []
    offset = 0
    for k, n_at in enumerate(frag_natoms):
        head = sno[k + 1]
        _, positions = _parse_geometry(lines, head)
        expected = super_positions[offset : offset + n_at]
        if positions.shape != expected.shape or not np.allclose(positions, expected, atol=1e-8):
            raise QChemEDAParseError(
                f"fragment {k} geometry does not match the supersystem rows "
                f"{offset}:{offset + n_at}; the fragment sub-job was re-oriented, so its "
                "multipoles are not in the supersystem frame"
            )
        try:
            m_at = next(i for i in mulliken_idx if i > head)
            p_at = next(i for i in multipole_idx if i > head)
        except StopIteration:
            raise QChemEDAParseError(f"fragment {k} printed no Mulliken/multipole block")
        charges.append(_parse_mulliken(lines, n_at, start=m_at))
        multipoles.append(_parse_multipoles(lines, start=p_at))
        offset += n_at
    return charges, multipoles


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

    frag_mulliken, frag_multipoles = _parse_fragment_blocks(lines, sno, frag_natoms)
    method, basis = method_and_basis(_parse_rem(lines))
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
        fragment_mulliken=frag_mulliken,
        fragment_multipoles=frag_multipoles,
        method=method,
        basis=basis,
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
    rec.multipoles = multipoles_to_atomic_units(rec.multipoles)
    rec.fragment_multipoles = [multipoles_to_atomic_units(m) for m in rec.fragment_multipoles]
    rec.units = "atomic"
    return rec


def check_consistency(
    rec: EDARecord,
    atol: float = 1e-3,
    rtol: float = 1e-4,
    max_int_energy: float = 1000.0,
    dipole_rtol: float = 0.5,
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

    # The frozen fragment dipoles must sum to roughly the relaxed supersystem
    # dipole -- they differ by exactly the polarization and charge transfer, which
    # for water clusters runs 10-25% of each monomer's own dipole. A gross mismatch
    # means the per-fragment blocks were paired with the wrong sub-jobs, which
    # nothing else here would notice because each block is individually well-formed.
    #
    # The tolerance is scaled by ``sum |mu_f|``, not by the supersystem dipole:
    # in a near-symmetric cluster the fragment dipoles largely cancel, so a
    # perfectly good frame can have a supersystem dipole *smaller* than the
    # polarization correction to it. Scaling by the total dipole present is what
    # makes this measure "are these the right monomers" rather than "is this
    # cluster symmetric".
    if rec.has_fragment_blocks:
        per_fragment = np.array([m["dipole"] for m in rec.fragment_multipoles])
        frozen = per_fragment.sum(axis=0)
        relaxed = rec.multipoles["dipole"]
        scale = float(np.linalg.norm(per_fragment, axis=-1).sum())
        if scale > 1e-6 and float(np.linalg.norm(frozen - relaxed)) > dipole_rtol * scale:
            msgs.append(
                f"sum of fragment dipoles {frozen} is far from the supersystem "
                f"dipole {relaxed}; the fragment blocks may be misaligned"
            )
    return msgs
