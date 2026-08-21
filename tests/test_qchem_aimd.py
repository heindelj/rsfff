"""Parsing an AIMD trajectory: step numbering, the gradient table, subsampling.

The fixture is the preamble plus the first three ``TIME STEP`` blocks of
``h3o_w1``, copied byte for byte. Trimming *between* blocks rather than inside
one keeps the two things that actually break here intact: the layout of a step
block, and the fact that a trajectory is a sequence of them.

Seven atoms is the point of picking this system. Q-Chem prints the gradient
transposed in six-atom column blocks, so a 7-atom gradient spans two blocks with
a one-column ragged tail -- and reading it as ``(n_atoms, 3)`` would give an
array of the right shape and the wrong contents.
"""

import os

import numpy as np
import pytest

from rsfff.qcgen.qchem_aimd import (
    QChemParseError,
    check_consistency,
    parse_aimd_output,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "qchem_aimd_h3o_water_3steps.out")

#: The three steps' SCF energies, from the last iteration row of each block.
ENERGIES = (-153.1917779180, -153.1902690207, -153.1884892169)

#: Step 1's gradient, read straight off the printed table. Atom 0 is column 1 of
#: the first block; atom 6 is the lone column of the ragged second block, which is
#: exactly where a naive row-major read goes wrong.
GRAD_ATOM0 = (-0.0186348, 0.0103575, 0.0040646)
GRAD_ATOM6 = (-0.0037049, -0.0039586, -0.0078977)


def test_steps_and_ordinals():
    rec = parse_aimd_output(FIXTURE)
    assert rec.n_steps == 3
    assert [s.step for s in rec.steps] == [1, 2, 3]
    # Q-Chem counts time steps from 1; the harvester names frames from 0.
    assert [s.ordinal for s in rec.steps] == [0, 1, 2]
    assert all(s.step == s.ordinal + 1 for s in rec.steps)


def test_metadata_from_the_echoed_input():
    rec = parse_aimd_output(FIXTURE)
    assert rec.total_charge == 1
    assert rec.multiplicity == 1
    assert rec.method == "wB97M-V"
    # Whatever case the template used; normalizing is multifrag.canonical_basis's job.
    assert rec.basis.lower() == "def2-tzvpd"


def test_energies_come_from_the_scf_row():
    rec = parse_aimd_output(FIXTURE)
    for step, expected in zip(rec.steps, ENERGIES):
        # The SCF row carries 10 decimals; the "Total energy =" line only 8.
        assert step.energy == pytest.approx(expected, abs=1e-10)
        assert step.converged


def test_gradient_is_read_transposed_across_ragged_blocks():
    step = parse_aimd_output(FIXTURE).steps[0]
    assert step.forces.shape == (7, 3)
    # Forces are the negative gradient.
    assert step.forces[0] == pytest.approx(np.array(GRAD_ATOM0) * -1.0, abs=1e-9)
    assert step.forces[6] == pytest.approx(np.array(GRAD_ATOM6) * -1.0, abs=1e-9)


def test_geometry_is_the_trajectory_frame_not_a_recentered_one():
    """AIMD reuses the "Standard Nuclear Orientation" heading for propagated coordinates.

    So consecutive frames must be *continuous* -- a per-step reorientation would
    show up as a jump far larger than one 0.48 fs step of motion.
    """
    steps = parse_aimd_output(FIXTURE).steps
    assert steps[0].symbols == ["O", "H", "H", "H", "O", "H", "H"]
    shift = np.abs(steps[1].positions - steps[0].positions).max()
    assert shift < 0.05


def test_ordinals_filter_selects_without_renumbering():
    rec = parse_aimd_output(FIXTURE, ordinals=[0, 2])
    assert [s.ordinal for s in rec.steps] == [0, 2]
    assert [s.step for s in rec.steps] == [1, 3]
    # The full length is still reported, so a caller can stride without a second pass.
    assert rec.n_steps == 3


def test_stride_subsamples():
    rec = parse_aimd_output(FIXTURE, stride=2)
    assert [s.ordinal for s in rec.steps] == [0, 2]
    assert rec.n_steps == 3


def test_ordinals_and_stride_are_mutually_exclusive():
    with pytest.raises(ValueError):
        parse_aimd_output(FIXTURE, ordinals=[0], stride=2)


def test_truncated_file_raises_rather_than_returning_nothing(tmp_path):
    head = open(FIXTURE, errors="replace").read().splitlines()[:50]
    path = tmp_path / "truncated.out"
    path.write_text("\n".join(head))
    with pytest.raises(QChemParseError):
        parse_aimd_output(str(path))


def test_check_consistency_flags_a_truncated_trajectory():
    rec = parse_aimd_output(FIXTURE)
    # The fixture stops mid-trajectory, so it has no Q-Chem sign-off by construction.
    assert not rec.completed
    assert any("truncated" in m for m in check_consistency(rec))
