"""Parsing the ``qchem_roundtrip`` bundle: force outputs, fragment blocks, the merge.

Both fixtures are **untrimmed** real outputs, unlike the older
``qchem_eda_water_dimer.out``. That is deliberate: the two things most likely to
break here -- the blocked, transposed gradient table and the interleaving of
per-fragment sub-jobs with the supersystem -- are properties of the *layout* of a
whole file, and a hand-trimmed fixture is exactly where that layout would get
quietly regularized.

- ``qchem_force_water_pentamer.out``: 15 atoms, so its gradient spans three
  6-wide column blocks, including a ragged last one.
- ``qchem_eda_water_dimer_frgm.out``: ran with ``SCF_PRINT_FRGM = true``, so it
  carries two isolated-monomer SCF blocks ahead of the supersystem's.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from rsfff.qcgen.qchem_eda import parse_eda_output
from rsfff.qcgen.qchem_eda import to_atomic_units as eda_to_atomic_units
from rsfff.qcgen.qchem_force import (
    QChemParseError,
    check_consistency,
    parse_force_output,
)
from rsfff.qcgen.qchem_force import to_atomic_units as force_to_atomic_units
from rsfff.qcgen.qchem_out import parse_rem

HERE = os.path.dirname(os.path.abspath(__file__))
FORCE_FIXTURE = os.path.join(HERE, "data", "qchem_force_water_pentamer.out")
EDA_FRGM_FIXTURE = os.path.join(HERE, "data", "qchem_eda_water_dimer_frgm.out")
SCRIPT = os.path.join(HERE, "..", "scripts", "parse_roundtrip.py")

#: Hartree/bohr, read straight off the printed gradient table. Column 1 of the
#: first block, column 9 of the second, column 15 of the third -- one per block,
#: with the last one in the ragged tail.
GRADIENT_SPOT_CHECKS = {
    0: [0.0087255, -0.0218661, 0.0166606],
    8: [0.0025227, -0.0060434, 0.0172704],
    14: [0.0044658, 0.0079319, -0.0047989],
}


@pytest.fixture
def force():
    return parse_force_output(FORCE_FIXTURE)


@pytest.fixture
def eda():
    return parse_eda_output(EDA_FRGM_FIXTURE)


# ---------------------------------------------------------------------------
# The force parser
# ---------------------------------------------------------------------------


def test_force_geometry_and_metadata(force):
    assert force.n_atoms == 15
    assert force.symbols == ["O", "H", "H"] * 5
    # Method and basis come from the echoed $rem, never the file name: this
    # fixture was copied from w5_wb97xv_qzvppd_frame0000.out, whose stem names a
    # level of theory the job did not use.
    assert (force.method, force.basis) == ("wB97M-V", "def2-TZVPD")
    assert force.total_charge == 0 and force.multiplicity == 1
    assert force.converged and force.completed
    assert check_consistency(force) == []


def test_force_energy_uses_the_full_precision_scf_row(force):
    # The "Total energy =" summary line carries 8 decimals; the SCF iteration row
    # carries 10, and the difference matters when this is differenced against an
    # EDA job's energy at the 1e-8 level.
    assert force.energy == pytest.approx(-382.1814699343, abs=1e-10)


def test_gradient_is_read_transposed_and_blocked(force):
    """Rows are x/y/z and columns are atoms, six atoms per block."""
    for atom, gradient in GRADIENT_SPOT_CHECKS.items():
        assert (-force.forces[atom]).tolist() == pytest.approx(gradient, abs=1e-9)


def test_forces_are_the_negative_gradient(force):
    # Sign convention, stated independently of the layout: a bound cluster's
    # gradient and force must be opposite and the same magnitude.
    assert force.forces.shape == (15, 3)
    assert np.abs(force.forces).max() == pytest.approx(0.03504527, abs=1e-6)


def test_translational_invariance_of_the_gradient(force):
    # A correctly transposed table sums to (near) zero over atoms; a table read
    # as (n_atoms, 3) would scramble components and break this.
    assert np.abs(force.forces.sum(axis=0)).max() < 5e-5


def test_force_multipoles_and_mulliken(force):
    assert force.mulliken_charges.shape == (15,)
    assert force.mulliken_charges.sum() == pytest.approx(0.0, abs=1e-6)
    assert force.multipoles["dipole"].shape == (3,)
    assert force.multipoles["quadrupole"].shape == (3, 3)
    q = force.multipoles["quadrupole"]
    assert q == pytest.approx(q.T)


def test_force_unit_conversion_is_idempotent_guarded(force):
    dipole_debye = force.multipoles["dipole"].copy()
    force_to_atomic_units(force)
    assert force.units == "atomic"
    assert force.multipoles["dipole"] == pytest.approx(dipole_debye / 2.5417464519)
    with pytest.raises(ValueError):
        force_to_atomic_units(force)


def test_rem_accepts_both_spellings():
    # "KEY = VALUE" (roundtrip templates) and "KEY VALUE" (CMM_Data inputs).
    assert parse_rem(["$rem", "METHOD = wB97M-V", "$end"])["method"] == "wB97M-V"
    assert parse_rem(["$rem", "METHOD wB97M-V", "$end"])["method"] == "wB97M-V"


def test_a_truncated_gradient_raises_rather_than_returning_nan(tmp_path):
    text = open(FORCE_FIXTURE).read()
    cut = text.index("Gradient of SCF Energy")
    stub = tmp_path / "truncated.out"
    # Keep one block of six atoms out of three.
    stub.write_text(text[: cut + 400])
    with pytest.raises(QChemParseError, match="missing"):
        parse_force_output(str(stub))


# ---------------------------------------------------------------------------
# Per-fragment blocks in the EDA output
# ---------------------------------------------------------------------------


def test_fragment_blocks_are_parsed(eda):
    assert eda.has_fragment_blocks
    assert len(eda.fragment_mulliken) == 2
    assert len(eda.fragment_multipoles) == 2
    for charges in eda.fragment_mulliken:
        assert charges.shape == (3,)
        assert charges.sum() == pytest.approx(0.0, abs=1e-6)


def test_fragment_blocks_are_the_isolated_monomers_not_the_supersystem(eda):
    # The frozen monomer's Mulliken charges differ from its in-cluster ones by
    # exactly the polarization and charge transfer, so they must not be equal.
    frozen = np.concatenate(eda.fragment_mulliken)
    assert frozen.shape == eda.mulliken_charges.shape
    assert not np.allclose(frozen, eda.mulliken_charges, atol=1e-3)


def test_fragment_dipoles_sum_to_the_supersystem_dipole_minus_polarization(eda):
    per_fragment = np.array([m["dipole"] for m in eda.fragment_multipoles])
    frozen = per_fragment.sum(axis=0)
    relaxed = eda.multipoles["dipole"]
    scale = np.linalg.norm(per_fragment, axis=-1).sum()
    ratio = np.linalg.norm(frozen - relaxed) / scale
    # Measured across w2-w5 this sits at 0.13-0.24: the frozen density is
    # systematically *less* polar than the relaxed one, and by a real amount.
    assert 0.05 < ratio < 0.35
    assert np.linalg.norm(frozen) < np.linalg.norm(relaxed)


def test_fragment_second_moments_shift_to_a_physical_monomer_value(eda):
    """The printed trace is meaningless; the trace about the monomer's own center is not."""
    eda_to_atomic_units(eda)
    bohr = 0.529177210903
    positions = eda.positions / bohr
    z = np.array([8.0 if s == "O" else 1.0 for s in eda.symbols])

    printed, shifted = [], []
    for f in range(eda.n_fragments):
        mask = eda.fragment_idx == f
        center = (z[mask, None] * positions[mask]).sum(0) / z[mask].sum()
        dipole = eda.fragment_multipoles[f]["dipole"]
        moment = eda.fragment_multipoles[f]["quadrupole"]
        # Neutral fragment, so the Q C x C term drops.
        about_center = moment - np.outer(center, dipole) - np.outer(dipole, center)
        printed.append(np.trace(moment))
        shifted.append(np.trace(about_center))

    # A water monomer's second-moment trace is ~-18.9 D*Ang = ~-14.05 e*a0^2,
    # and both monomers must land there once referenced to their own centers.
    assert shifted == pytest.approx([-14.10, -14.00], abs=0.1)
    # The printed values are about the *dimer's* origin and are nowhere near it,
    # which is the whole reason the shift is mandatory.
    assert min(abs(p - s) for p, s in zip(printed, shifted)) > 1.0


# ---------------------------------------------------------------------------
# The merge script
# ---------------------------------------------------------------------------


def _run_merge(tmp_path, root, extra=()):
    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", str(root), "--out-dir", str(tmp_path), *extra],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_merge_writes_forces_and_fragment_labels(tmp_path):
    """End to end on the real bundle, if it is present in this checkout."""
    root = os.path.join(HERE, "..", "qchem_roundtrip")
    if not os.path.isdir(os.path.join(root, "eda", "outputs")):
        pytest.skip("qchem_roundtrip outputs are not in this checkout")
    _run_merge(tmp_path, root, extra=("--stems", "w2_wb97xv_qzvppd", "--limit", "3"))

    # The file is named for the level of theory that actually ran, not the stem.
    out = tmp_path / "w2_wb97mv_tzvpd.xyz"
    assert out.exists()

    torch = pytest.importorskip("torch")
    from rsfff.train.data import load_extxyz

    ds = load_extxyz(str(out), dtype=torch.float64)
    assert len(ds) == 3
    batch = ds.flat_batch(range(3))
    assert batch.n_fragments == 6
    assert batch.forces is not None and batch.forces.shape == (18, 3)
    assert batch.fragment_energy.shape == (6,)
    assert batch.fragment_dipole.shape == (6, 3)
    assert batch.fragment_second_moment.shape == (6, 3, 3)
    assert batch.fragment_to_batch.tolist() == [0, 0, 1, 1, 2, 2]
    # Every fragment is a water monomer, so every fragment energy is near -76.43.
    assert batch.fragment_energy.max() < -76.4
    # Second moments come back symmetric after the unique-component round trip.
    m = batch.fragment_second_moment
    assert torch.allclose(m, m.transpose(-1, -2))


def test_merged_forces_land_in_hartree_per_angstrom(tmp_path):
    root = os.path.join(HERE, "..", "qchem_roundtrip")
    if not os.path.isdir(os.path.join(root, "force", "outputs")):
        pytest.skip("qchem_roundtrip outputs are not in this checkout")
    _run_merge(tmp_path, root, extra=("--stems", "h2o", "--limit", "2"))

    torch = pytest.importorskip("torch")
    from rsfff.train.data import load_extxyz

    ds = load_extxyz(str(tmp_path / "h2o_wb97mv_tzvpd.xyz"), dtype=torch.float64)
    batch = ds.flat_batch([0])
    # The file stores Hartree/bohr and the loader divides by Bohr, so the
    # in-memory forces are larger by 1/0.529 than the printed gradient.
    reference = parse_force_output(
        os.path.join(root, "force", "outputs", "h2o_frame0000.out")
    )
    assert batch.forces.numpy() == pytest.approx(reference.forces / 0.529177210903, abs=1e-10)
    # One fragment, so its "fragment" labels are the molecular ones.
    assert batch.n_fragments == 1
    assert batch.fragment_energy.item() == pytest.approx(reference.energy, abs=1e-10)
    assert batch.eda is None


def test_an_unusable_pair_is_dropped_rather_than_written(tmp_path):
    """A frame the merge cannot validate must be reported, not silently degraded."""
    root = tmp_path / "bundle"
    for calc in ("eda", "force"):
        (root / calc / "geoms").mkdir(parents=True)
        (root / calc / "outputs").mkdir(parents=True)
        (root / calc / "geoms" / "w2_stale.xyz").write_text("")
    # Both sides get the EDA output. The "force" side therefore has no gradient
    # table at all, which must surface as a dropped frame rather than as a frame
    # of zero forces -- the failure mode that would poison a force fit silently.
    for calc in ("eda", "force"):
        (root / calc / "outputs" / "w2_stale_frame0000.out").write_text(
            open(EDA_FRGM_FIXTURE).read()
        )

    result = subprocess.run(
        [sys.executable, SCRIPT, "--root", str(root), "--out-dir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert "dropped frame 0" in result.stderr
    assert "Gradient of SCF Energy" in result.stderr
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
