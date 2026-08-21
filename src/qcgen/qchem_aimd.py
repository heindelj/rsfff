"""Parser for Q-Chem ``JOBTYPE = aimd`` output files.

An AIMD output is a preamble (the echoed ``$molecule`` and ``$rem``) followed by
one self-contained block per time step, each of which prints exactly what a
``JOBTYPE = force`` job prints: a ``Standard Nuclear Orientation`` table, an SCF
iteration history and a ``Gradient of SCF Energy`` table. So the section readers
are the ones :mod:`rsfff.qcgen.qchem_out` and :mod:`rsfff.qcgen.qchem_force`
already own -- this module is the step splitter around them, plus the one thing
that is genuinely different about a trajectory.

**The frame is the trajectory's, not Q-Chem's.** For a single-point job the
``Standard Nuclear Orientation`` heading means what it says: the system has been
translated to put its center of nuclear charge at the origin and rotated to its
principal axes. During AIMD the heading is reused for the *propagated*
coordinates, which are continuous from step to step and drift away from that
origin as the cluster picks up momentum. Anything that has to be compared with a
separate Q-Chem job on the same frame -- an EDA job harvested from it, say --
must therefore be aligned rather than assumed to share a frame; see
:mod:`rsfff.qcgen.multifrag`.

Two smaller things:

* ``ordinal`` is the 0-based index of a step among the parsed steps, and
  ``step`` is Q-Chem's own 1-based ``TIME STEP #`` counter. They differ by one,
  and both are kept because ``qchem_roundtrip``'s AIMD harvester names its
  derived jobs with *both* (``..._step00051_frame00050_...``).
* AIMD applies a ~1e-11 a.u. Cartesian multipole field to break the symmetry of
  a symmetric starting geometry. It shifts the energy by ~1e-11 Hartree, which
  is two orders of magnitude below SCF convergence noise, and is ignored here.

Energies are Hartree, forces Hartree/bohr (``-dE/dR``) and positions Angstrom,
matching :class:`rsfff.qcgen.qchem_force.ForceRecord`. Parsing is streamed one
step at a time: these outputs run to tens of megabytes and hundreds of thousands
of lines, and a strided read should not have to hold all of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from .qchem_force import parse_gradient
from .qchem_out import (
    SCF_ITER,
    SNO_HEADER,
    JOB_COMPLETE_MARKER,
    QChemParseError,
    find_all,
    method_and_basis,
    parse_geometry,
    parse_molecule_block,
    parse_rem,
)

_TIME_STEP = re.compile(r"^TIME STEP #(\d+)")


@dataclass
class AIMDStep:
    """One time step of an AIMD trajectory."""

    #: Q-Chem's 1-based ``TIME STEP #`` counter.
    step: int
    #: 0-based index among the steps in the file. This is the ``frameNNNNN`` that
    #: ``qchem_roundtrip``'s harvester uses.
    ordinal: int
    symbols: list[str]
    #: Angstrom, in the trajectory's own frame (see the module docstring).
    positions: np.ndarray
    #: Hartree, from the last SCF iteration row (10 decimals, against the
    #: ``Total energy =`` line's 8).
    energy: float
    #: Hartree/bohr, ``-dE/dR``, in the same frame as ``positions``.
    forces: np.ndarray
    #: The step's SCF printed "Convergence criterion met".
    converged: bool = True

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


@dataclass
class AIMDRecord:
    """A parsed AIMD output: its metadata plus the steps that were asked for."""

    path: str = ""
    steps: list[AIMDStep] = field(default_factory=list)
    #: Steps seen in the file, whether or not they were parsed into ``steps``.
    n_steps: int = 0
    total_charge: int = 0
    multiplicity: int = 1
    method: str = ""
    basis: str = ""
    #: Q-Chem printed its sign-off, i.e. the trajectory is not truncated.
    completed: bool = False
    units: str = "atomic"


def _iter_step_blocks(path: str) -> Iterator[tuple[int, int, list[str]]]:
    """Yield ``(step, ordinal, lines)`` per ``TIME STEP`` block, and the preamble first.

    The preamble is yielded as ordinal ``-1`` with step ``0`` so a caller can read
    the echoed ``$molecule``/``$rem`` without a second pass over the file.
    """
    buf: list[str] = []
    step, ordinal = 0, -1
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = _TIME_STEP.match(line)
            if m:
                yield step, ordinal, buf
                buf = []
                step, ordinal = int(m.group(1)), ordinal + 1
            buf.append(line)
    yield step, ordinal, buf


def _parse_step(step: int, ordinal: int, lines: list[str], path: str) -> AIMDStep:
    sno = find_all(lines, SNO_HEADER)
    if not sno:
        raise QChemParseError(f"{path}: step {step} has no Standard Nuclear Orientation")
    symbols, positions = parse_geometry(lines, sno[0])

    energy, converged = None, False
    for ln in lines:
        m = SCF_ITER.match(ln)
        if m:
            energy = float(m.group(1))
            converged = "Convergence criterion met" in ln
    if energy is None:
        raise QChemParseError(f"{path}: step {step} has no SCF iterations")

    return AIMDStep(
        step=step,
        ordinal=ordinal,
        symbols=symbols,
        positions=positions,
        energy=energy,
        forces=-parse_gradient(lines, len(symbols)),
        converged=converged,
    )


def parse_aimd_output(
    path: str, ordinals: Iterable[int] | None = None, stride: int = 1
) -> AIMDRecord:
    """Parse an AIMD output, optionally only some of its steps.

    ``ordinals`` names the wanted steps in the 0-based ``AIMDStep.ordinal``
    numbering; ``stride`` keeps every Nth one, and is the way to subsample a
    trajectory whose length you do not know yet without reading it twice. They
    are mutually exclusive. Restricting either is what makes a strided read of a
    5000-step trajectory cheap: every step is still *scanned* (the ordinals are
    only knowable in order), but only the wanted ones are turned into arrays.
    ``n_steps`` reports the full length either way.

    A trailing step truncated by a wall-clock kill is dropped rather than raising,
    since the steps before it are perfectly good; ``completed`` reports whether
    the file has Q-Chem's sign-off.
    """
    if ordinals is not None and stride != 1:
        raise ValueError("pass ordinals or stride, not both")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    wanted = None if ordinals is None else set(int(o) for o in ordinals)
    rec = AIMDRecord(path=path)
    tail = ""

    for step, ordinal, lines in _iter_step_blocks(path):
        if ordinal < 0:  # the preamble
            frag_charges, _, _, total_charge, multiplicity = parse_molecule_block(lines)
            del frag_charges
            rec.total_charge = total_charge
            rec.multiplicity = multiplicity
            rec.method, rec.basis = method_and_basis(parse_rem(lines))
            continue
        rec.n_steps = ordinal + 1
        tail = "\n".join(lines[-40:])
        if wanted is not None and ordinal not in wanted:
            continue
        if wanted is None and ordinal % stride:
            continue
        try:
            rec.steps.append(_parse_step(step, ordinal, lines, path))
        except QChemParseError:
            # Only a truncated *final* step is tolerable; anything earlier means
            # the file is malformed in a way a silent skip would hide.
            if ordinal + 1 == rec.n_steps and JOB_COMPLETE_MARKER not in tail:
                continue
            raise

    rec.completed = JOB_COMPLETE_MARKER in tail
    if not rec.steps:
        raise QChemParseError(f"{path}: no usable AIMD steps")
    return rec


def check_consistency(rec: AIMDRecord, max_force: float = 1.0) -> list[str]:
    """Human-readable warnings about a parsed trajectory.

    ``max_force`` is in Hartree/bohr, the same threshold
    :func:`rsfff.qcgen.qchem_force.check_consistency` uses.
    """
    msgs = []
    if not rec.completed:
        msgs.append("no Q-Chem sign-off: the trajectory is truncated")
    unconverged = [s.step for s in rec.steps if not s.converged]
    if unconverged:
        head = ", ".join(str(s) for s in unconverged[:5])
        more = f" (+{len(unconverged) - 5} more)" if len(unconverged) > 5 else ""
        msgs.append(f"SCF did not converge at step(s) {head}{more}")
    for s in rec.steps:
        peak = float(np.abs(s.forces).max()) if s.forces.size else 0.0
        if peak > max_force:
            msgs.append(f"step {s.step}: implausible force component {peak:.4g} Ha/bohr")
    return msgs


__all__ = [
    "AIMDRecord",
    "AIMDStep",
    "QChemParseError",
    "check_consistency",
    "parse_aimd_output",
]
