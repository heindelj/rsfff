"""Q-Chem ALMO-EDA parsing: sections, sum rules, unit conversion, extxyz output.

The fixture in ``tests/data`` is a real ``eda.out`` (water dimer, wB97X-V/
def2-QZVPPD) trimmed to the sections the parser reads; it reproduces the full
47 kB file field-for-field.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from rsfff.qcgen.qchem_eda import (
    KJMOL_PER_HARTREE,
    MULTIPOLE_LABELS,
    QChemEDAParseError,
    check_consistency,
    parse_eda_output,
    to_atomic_units,
    unique_components,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "qchem_eda_water_dimer.out")
SCRIPT = os.path.join(HERE, "..", "scripts", "parse_qchem_eda.py")


@pytest.fixture
def rec():
    return parse_eda_output(FIXTURE)


def test_geometry_and_fragments(rec):
    assert rec.symbols == ["O", "H", "H", "O", "H", "H"]
    assert rec.positions.shape == (6, 3)
    assert rec.positions[0] == pytest.approx([1.669733, -0.369294, 0.212006])
    assert rec.n_fragments == 2
    assert rec.fragment_idx.tolist() == [0, 0, 0, 1, 1, 1]
    assert rec.total_charge == 0 and rec.multiplicity == 1
    assert rec.fragment_charges == [0, 0] and rec.fragment_mults == [1, 1]
    assert (rec.method, rec.basis) == ("wB97X-V", "def2-qzvppd")


def test_energies(rec):
    # Total electronic energy is the converged CT-allowed supersystem SCF.
    assert rec.energy == pytest.approx(-152.8823453414, abs=1e-10)
    assert rec.converged
    assert rec.fragment_energies == pytest.approx([-76.4366986719, -76.4405410968], abs=1e-10)
    # E_int == E_total - sum(E_frag), to Q-Chem's kJ/mol print precision.
    assert rec.interaction_energy() * KJMOL_PER_HARTREE == pytest.approx(-13.4048, abs=1e-3)


def test_eda_terms_and_sum_rule(rec):
    assert rec.eda["cls_elec"] == pytest.approx(-32.9260)
    assert rec.eda["mod_pauli"] == pytest.approx(42.3259)
    assert rec.eda["disp"] == pytest.approx(-8.6698)
    assert rec.eda["pol"] == pytest.approx(-5.3262)
    assert rec.eda["ct"] == pytest.approx(-8.8088)
    assert rec.eda["int"] == pytest.approx(-13.4048)
    parts = ("prp", "cls_elec", "mod_pauli", "disp", "pol", "ct")
    assert sum(rec.eda[p] for p in parts) == pytest.approx(rec.eda["int"], abs=1e-3)
    assert check_consistency(rec) == []


def test_mulliken_charges(rec):
    assert rec.mulliken_charges == pytest.approx(
        [-0.418610, 0.180612, 0.179949, -0.395047, 0.230777, 0.222320]
    )
    # Q-Chem prints charges to 6 decimals, so the sum is zero only to that.
    assert rec.mulliken_charges.sum() == pytest.approx(0.0, abs=1e-5)


def test_multipole_tensors_are_symmetric(rec):
    shapes = {"dipole": (3,), "quadrupole": (3, 3), "octopole": (3,) * 3,
              "hexadecapole": (3,) * 4}
    for name, shape in shapes.items():
        t = rec.multipoles[name]
        assert t.shape == shape
        for ax in range(1, t.ndim):  # symmetric under every pair swap
            perm = list(range(t.ndim))
            perm[0], perm[ax] = perm[ax], perm[0]
            assert np.allclose(t, np.transpose(t, perm))
    # Printed values land in the right slots, and round-trip back out.
    assert rec.multipoles["dipole"] == pytest.approx([-3.1412, 2.0955, -1.9652])
    assert rec.multipoles["quadrupole"][0, 1] == pytest.approx(2.4975)  # XY
    assert rec.multipoles["octopole"][0, 1, 2] == pytest.approx(-1.9159)  # XYZ
    assert rec.multipoles["hexadecapole"][0, 0, 2, 2] == pytest.approx(-35.2721)  # XXZZ
    for name, labels in MULTIPOLE_LABELS.items():
        assert len(unique_components(rec.multipoles[name], name)) == len(labels)


def test_dipole_magnitude_matches_printed_total(rec):
    assert np.linalg.norm(rec.multipoles["dipole"]) == pytest.approx(4.2568, abs=1e-4)


def test_to_atomic_units(rec):
    e_int_kj, mu_debye = rec.eda["int"], rec.multipoles["dipole"].copy()
    to_atomic_units(rec)
    assert rec.units == "atomic"
    assert rec.eda["int"] * KJMOL_PER_HARTREE == pytest.approx(e_int_kj)
    assert rec.multipoles["dipole"] * 2.5417464519 == pytest.approx(mu_debye, rel=1e-9)
    # Rank-n scaling: Debye-Ang^(n-1) -> e*a0^n picks up one extra Bohr per rank.
    assert np.linalg.norm(rec.multipoles["dipole"]) == pytest.approx(4.2568 / 2.5417464519, abs=1e-5)
    assert check_consistency(rec, atol=1e-3 / KJMOL_PER_HARTREE) == []
    with pytest.raises(ValueError):
        to_atomic_units(rec)  # not idempotent by design


def test_implausible_interaction_energy_is_flagged(rec):
    # A collapsed CT-allowed SCF keeps the internal sum rules but lands nowhere
    # near a physical interaction energy.
    rec.eda["int"] = -7.09e9
    rec.eda["ct"] = rec.eda["int"] - sum(
        rec.eda[p] for p in ("prp", "cls_elec", "mod_pauli", "disp", "pol")
    )
    rec.energy = float(rec.fragment_energies.sum()) + rec.eda["int"] / KJMOL_PER_HARTREE
    msgs = check_consistency(rec)
    assert len(msgs) == 1 and "implausible interaction energy" in msgs[0]


def test_truncated_file_raises(tmp_path):
    head = open(FIXTURE).read().split("Results of EDA2")[0]
    path = tmp_path / "truncated.out"
    path.write_text(head)
    with pytest.raises(QChemEDAParseError):
        parse_eda_output(str(path))


@pytest.mark.parametrize("multipole_format,sizes", [("tensor", (3, 9, 27, 81)),
                                                    ("unique", (3, 6, 10, 15))])
def test_cli_writes_readable_extxyz(tmp_path, multipole_format, sizes):
    ase_io = pytest.importorskip("ase.io")
    out = tmp_path / "frames.xyz"
    subprocess.run(
        [sys.executable, SCRIPT, str(out), FIXTURE,
         "--multipole-format", multipole_format],
        check=True, capture_output=True,
    )
    (atoms,) = ase_io.read(str(out), ":")
    assert atoms.get_chemical_symbols() == ["O", "H", "H", "O", "H", "H"]
    assert atoms.arrays["fragment_idx"].tolist() == [0, 0, 0, 1, 1, 1]
    assert atoms.arrays["mulliken_charges"] == pytest.approx(
        [-0.418610, 0.180612, 0.179949, -0.395047, 0.230777, 0.222320]
    )
    # ASE routes `energy` and `dipole` to the calculator, everything else to info.
    assert atoms.calc.results["energy"] == pytest.approx(-152.8823453414, abs=1e-10)
    for name, size in zip(MULTIPOLE_LABELS, sizes):
        value = atoms.calc.results["dipole"] if name == "dipole" else atoms.info[name]
        assert np.asarray(value).size == size
    info = atoms.info
    assert info["config_type"] == "w2" and info["n_fragments"] == 2
    assert info["charge"] == 0 and info["units"] == "atomic"
    assert info["multipole_format"] == multipole_format
    assert info["fragment_energies"] == pytest.approx(
        [-76.4366986719, -76.4405410968], abs=1e-10
    )
    # EDA components are written in Hartree and still satisfy the sum rule.
    total = sum(info[f"eda_{p}"] for p in
                ("prp", "cls_elec", "mod_pauli", "disp", "pol", "ct"))
    assert total * KJMOL_PER_HARTREE == pytest.approx(-13.4048, abs=1e-3)
