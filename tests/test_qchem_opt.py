"""Parsing an opt+freq output: the polarizability's sign, and which frame it is in.

The fixture is ``opt/outputs/h2o_wb97mv_tzvpd.out``, untrimmed. Both of the
things this parser has to get right are properties of *where* a block sits in
the file, so trimming would be trimming away the test.
"""

import os

import numpy as np
import pytest

from rsfff.qcgen.qchem_opt import (
    QChemParseError,
    check_consistency,
    parse_opt_output,
    to_atomic_units,
)
from rsfff.qcgen.qchem_out import SNO_HEADER, find_all, parse_geometry

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "qchem_opt_freq_water.out")

#: What the CPSCF block prints, verbatim -- note every diagonal is negative.
PRINTED_DIAGONAL = (-10.1648939, -9.5980793, -9.7067983)


def test_polarizability_is_negated_and_symmetric():
    rec = parse_opt_output(FIXTURE)
    assert np.diag(rec.polarizability) == pytest.approx(
        [-v for v in PRINTED_DIAGONAL], abs=1e-7
    )
    assert rec.polarizability == pytest.approx(rec.polarizability.T, abs=1e-12)
    # The physical object is positive definite; the printed d^2E/dF^2 is not.
    assert (np.linalg.eigvalsh(rec.polarizability) > 0).all()


def test_polarizability_magnitude_is_waters():
    """~9.8 a0^3 isotropic. A missed unit conversion or a Debye slip is not subtle."""
    rec = parse_opt_output(FIXTURE)
    assert np.trace(rec.polarizability) / 3 == pytest.approx(9.82, abs=0.05)


def test_geometry_is_the_one_before_the_cpscf_not_the_one_after():
    """The tensor belongs to the geometry that precedes the CPSCF solve.

    The vibrational analysis prints another ``Standard Nuclear Orientation``
    afterwards, and Q-Chem may reorient onto the symmetry axes there. In *this*
    file the two agree to 1.7e-5 Angstrom, so the check is not that the numbers
    differ -- it is that the parser selected the pre-CPSCF block on purpose, and
    is not merely taking the last block in the file and getting away with it.
    """
    rec = parse_opt_output(FIXTURE)
    lines = open(FIXTURE, errors="replace").read().splitlines()
    blocks = find_all(lines, SNO_HEADER)
    pol = next(i for i, ln in enumerate(lines) if "Polarizability Matrix" in ln)
    chosen = max(b for b in blocks if b < pol)

    assert chosen != blocks[-1], "the fixture must have a block after the CPSCF solve"
    _, last_before_pol = parse_geometry(lines, chosen)
    _, last_in_file = parse_geometry(lines, blocks[-1])
    assert rec.positions == pytest.approx(last_before_pol, abs=1e-12)
    assert not np.array_equal(last_before_pol, last_in_file)


def test_converged_geometry_carries_a_vanishing_gradient():
    rec = parse_opt_output(FIXTURE)
    assert rec.forces.shape == (3, 3)
    assert np.abs(rec.forces).max() < 1e-4


def test_frequencies_and_intensities_line_up():
    rec = parse_opt_output(FIXTURE)
    # 3N - 6 = 3 for a bent triatomic, all real at a minimum.
    assert rec.frequencies.shape == (3,)
    assert rec.ir_intensities.shape == (3,)
    assert (rec.frequencies > 0).all()
    assert rec.frequencies[0] == pytest.approx(1623.12, abs=0.01)


def test_energy_agrees_with_the_final_energy_line():
    rec = parse_opt_output(FIXTURE)
    printed = float(
        next(ln for ln in open(FIXTURE, errors="replace") if "Final energy is" in ln).split()[-1]
    )
    assert rec.energy == pytest.approx(printed, abs=1e-8)


def test_units_conversion_leaves_the_polarizability_alone():
    """Multipoles are Debye-Angstrom^(n-1); the CPSCF tensor is already a.u."""
    rec = parse_opt_output(FIXTURE)
    before_dipole = rec.multipoles["dipole"].copy()
    before_alpha = rec.polarizability.copy()
    to_atomic_units(rec)
    assert rec.polarizability == pytest.approx(before_alpha, abs=0)
    assert np.abs(rec.multipoles["dipole"] - before_dipole).max() > 1e-3
    with pytest.raises(ValueError):
        to_atomic_units(rec)


def test_consistency_checks_are_clean_on_a_good_job():
    assert check_consistency(parse_opt_output(FIXTURE)) == []


def test_a_job_without_a_cpscf_block_is_refused(tmp_path):
    text = open(FIXTURE, errors="replace").read().replace("Polarizability Matrix", "xxx")
    path = tmp_path / "no_pol.out"
    path.write_text(text)
    with pytest.raises(QChemParseError):
        parse_opt_output(str(path))
