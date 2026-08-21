"""Parser for Q-Chem ``JOBTYPE = force`` output files.

Extracts the geometry, total energy, analytic nuclear gradient, Mulliken
charges and Cartesian multipole moments from one output. The section readers it
does not own live in :mod:`rsfff.qcgen.qchem_out`, shared with the EDA parser.

Two details carry the whole file's correctness:

**The gradient is printed transposed.** Q-Chem writes

    Gradient of SCF Energy
                1           2           3           4           5           6
        1   0.0125798  -0.0145080  -0.0021226   0.0017178   0.0030132  -0.0006803
        2  -0.0366652   0.0178740   0.0215039  -0.0112284   0.0105225  -0.0020069
        3   0.0128961  -0.0057128  -0.0060089  -0.0057086   0.0084048  -0.0038707
                7           8   ...

i.e. **rows are x/y/z and columns are atoms**, in blocks of six atoms. Reading
it as ``(n_atoms, 3)`` gives a garbage array of exactly the right shape for
systems with 6 atoms and silently wrong values for every other size, so the
block structure is parsed explicitly and the atom count is checked.

**Forces are the negative gradient**, and this module returns forces, in
Hartree/bohr -- the unit ``data/labels/*.extxyz`` uses and the one
``rsfff.train.data.load_extxyz`` assumes when it divides by ``ase.units.Bohr``.

Energies come from the last SCF iteration row rather than the ``Total energy =``
summary line, which Q-Chem prints to only 8 decimals against the row's 10.

Everything is reported in the output's own ``Standard Nuclear Orientation``
frame: Q-Chem translates the system so its center of nuclear charge is at the
origin and may rotate it, and the gradient and multipoles are expressed in that
frame. Positions, forces and moments are therefore mutually consistent only if
all three are taken from the output, never from the input geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .qchem_out import (
    KJMOL_PER_HARTREE,
    SCF_ITER,
    SNO_HEADER,
    QChemParseError,
    find_all,
    job_completed,
    method_and_basis,
    multipoles_to_atomic_units,
    parse_geometry,
    parse_molecule_block,
    parse_mulliken,
    parse_multipoles,
    parse_rem,
)

_GRADIENT_HEADER = "Gradient of SCF Energy"


@dataclass
class ForceRecord:
    """Everything pulled out of one Q-Chem force output."""

    path: str = ""
    symbols: list[str] = field(default_factory=list)
    #: Angstrom, in the output's standard nuclear orientation.
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Hartree/bohr, ``-dE/dR``, in the same frame as ``positions``.
    forces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Total electronic energy, Hartree.
    energy: float = float("nan")
    total_charge: int = 0
    multiplicity: int = 1
    mulliken_charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Cartesian multipoles as full symmetric tensors, about the frame origin.
    multipoles: dict[str, np.ndarray] = field(default_factory=dict)
    method: str = ""
    basis: str = ""
    #: The final SCF printed "Convergence criterion met".
    converged: bool = False
    #: Q-Chem printed its sign-off, i.e. the file is not truncated.
    completed: bool = False
    units: str = "qchem"

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def parse_gradient(lines: list[str], n_atoms: int) -> np.ndarray:
    """Parse the blocked, transposed ``Gradient of SCF Energy`` table: ``(n_atoms, 3)``.

    A header line inside the table is all integers (the atom column indices); a
    data line is a component index 1/2/3 followed by floats. Q-Chem prints
    decimals, so "all tokens parse as int" separates the two unambiguously.

    The *last* block in ``lines`` is the one read, so an AIMD parser hands in one
    time step's slice rather than the whole file (see
    :mod:`rsfff.qcgen.qchem_aimd`).
    """
    idx = find_all(lines, _GRADIENT_HEADER)
    if not idx:
        raise QChemParseError(f"no '{_GRADIENT_HEADER}' block")

    grad = np.full((n_atoms, 3), np.nan)
    columns: list[int] = []
    seen_any = False
    for ln in lines[idx[-1] + 1 :]:
        toks = ln.split()
        if not toks:
            if seen_any:
                break
            continue
        if all(t.isdigit() for t in toks):        # new block of atom columns
            columns = [int(t) - 1 for t in toks]
            seen_any = True
            continue
        if columns and toks[0] in ("1", "2", "3") and len(toks) == len(columns) + 1:
            comp = int(toks[0]) - 1
            try:
                values = [float(t) for t in toks[1:]]
            except ValueError:
                break
            for col, v in zip(columns, values):
                if not 0 <= col < n_atoms:
                    raise QChemParseError(
                        f"gradient names atom {col + 1} but the system has {n_atoms}"
                    )
                grad[col, comp] = v
            continue
        if seen_any:                              # "Max gradient component = ..."
            break

    if np.isnan(grad).any():
        missing = int(np.isnan(grad).any(axis=1).sum())
        raise QChemParseError(f"gradient table is missing {missing} of {n_atoms} atoms")
    return grad


def _parse_scf_energy(lines: list[str]) -> tuple[float, bool]:
    """``(energy, converged)`` from the last SCF iteration row."""
    energy, converged = None, False
    for ln in lines:
        m = SCF_ITER.match(ln)
        if m:
            energy = float(m.group(1))
            converged = "Convergence criterion met" in ln
    if energy is None:
        raise QChemParseError("no SCF iterations found")
    return energy, converged


def parse_force_output(path: str) -> ForceRecord:
    """Parse one Q-Chem force output into a :class:`ForceRecord` (native units)."""
    with open(path, errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()

    sno = find_all(lines, SNO_HEADER)
    if not sno:
        raise QChemParseError("no Standard Nuclear Orientation block")
    symbols, positions = parse_geometry(lines, sno[0])

    _, _, frag_natoms, total_charge, multiplicity = parse_molecule_block(lines)
    if sum(frag_natoms) != len(symbols):
        raise QChemParseError(
            f"geometry has {len(symbols)} atoms but $molecule declares {sum(frag_natoms)}"
        )

    energy, converged = _parse_scf_energy(lines)
    method, basis = method_and_basis(parse_rem(lines))
    return ForceRecord(
        path=path,
        symbols=symbols,
        positions=positions,
        forces=-parse_gradient(lines, len(symbols)),
        energy=energy,
        total_charge=total_charge,
        multiplicity=multiplicity,
        mulliken_charges=parse_mulliken(lines, len(symbols)),
        multipoles=parse_multipoles(lines),
        method=method,
        basis=basis,
        converged=converged,
        completed=job_completed(text),
        units="qchem",
    )


def to_atomic_units(rec: ForceRecord) -> ForceRecord:
    """Convert the multipoles to ``e*a0^n``; everything else is already atomic.

    Mutates and returns ``rec``. Energies are Hartree, forces Hartree/bohr and
    positions Angstrom as parsed. Idempotent guard: raises if already converted.
    """
    if rec.units == "atomic":
        raise ValueError("record is already in atomic units")
    rec.multipoles = multipoles_to_atomic_units(rec.multipoles)
    rec.units = "atomic"
    return rec


def check_consistency(rec: ForceRecord, max_force: float = 1.0) -> list[str]:
    """Human-readable warnings about a parsed force record.

    ``max_force`` is in Hartree/bohr. A converged water cluster near equilibrium
    has components well under 0.1; anything approaching 1 means an atom is on
    top of another and the frame should not be trained on.
    """
    msgs = []
    if not rec.completed:
        msgs.append("no Q-Chem sign-off: the output is truncated")
    if not rec.converged:
        msgs.append("final SCF did not report convergence")
    peak = float(np.abs(rec.forces).max()) if rec.forces.size else 0.0
    if peak > max_force:
        msgs.append(f"implausible force component {peak:.4g} Ha/bohr")
    return msgs


__all__ = [
    "KJMOL_PER_HARTREE",
    "ForceRecord",
    "QChemParseError",
    "check_consistency",
    "parse_force_output",
    "parse_gradient",
    "to_atomic_units",
]
