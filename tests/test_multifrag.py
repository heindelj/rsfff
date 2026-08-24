"""The multi-fragmentation schema: the geometry algebra, and the extxyz round trip.

Two separable concerns, tested separately:

* :func:`rotation_between` and :func:`rotate_multipole` are what make labels from
  differently-oriented Q-Chem jobs commensurable, and a wrong rotation is not
  detectable downstream -- every rotation invariant of a rotated tensor is right.
  So they are checked against an independent ``einsum`` and against a case that
  must be *refused*.
* The writer and the two readers must agree on what each header means. A
  transposed reshape of ``fragment_dipoles`` would survive any single-fragmentation
  test, so the round trip is checked with fragmentations that differ.
"""

import numpy as np
import pytest
import torch

from rsfff.qcgen.multifrag import (
    Fragmentation,
    MultiFragFrame,
    QChemParseError,
    canonical_basis,
    center_of_nuclear_charge,
    fragment_formula,
    fragmentation_config_type,
    read_multifrag_extxyz,
    recenter,
    rotate_multipole,
    rotate_second_moments,
    rotation_between,
    same_level_of_theory,
    write_frames,
)
from rsfff.qcgen.qchem_out import expand_multipole, unique_components
from rsfff.train.data import load_extxyz


def random_rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def symmetric_tensor(rank, seed=0):
    """A fully symmetric rank-n tensor, which is what a Cartesian multipole is."""
    import itertools
    import math

    t = np.random.default_rng(seed).normal(size=(3,) * rank)
    perms = list(itertools.permutations(range(rank)))
    return sum(t.transpose(p) for p in perms) / math.factorial(rank)


# --- geometry algebra -------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_rotate_multipole_matches_an_explicit_contraction(rank):
    rot = random_rotation(1)
    tensor = symmetric_tensor(rank, seed=rank)
    letters = "abcd"[:rank]
    out = "ijkl"[:rank]
    subs = ",".join(f"{o}{a}" for o, a in zip(out, letters)) + f",{letters}->{out}"
    expected = np.einsum(subs, *([rot] * rank), tensor)
    assert rotate_multipole(tensor, rot) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_rotate_multipole_round_trips(rank):
    rot = random_rotation(2)
    tensor = symmetric_tensor(rank, seed=rank)
    back = rotate_multipole(rotate_multipole(tensor, rot), rot.T)
    assert back == pytest.approx(tensor, abs=1e-12)


def test_rank_two_rotation_is_the_familiar_congruence():
    rot = random_rotation(3)
    tensor = symmetric_tensor(2, seed=7)
    assert rotate_multipole(tensor, rot) == pytest.approx(rot @ tensor @ rot.T, abs=1e-12)


def test_rotate_second_moments_goes_through_the_full_tensor():
    rot = random_rotation(4)
    full = symmetric_tensor(2, seed=9)
    unique = unique_components(full, "quadrupole")[None, :]
    rotated = rotate_second_moments(unique, rot)[0]
    expected = unique_components(rot @ full @ rot.T, "quadrupole")
    assert rotated == pytest.approx(expected, abs=1e-12)
    # And the print order really is XX XY YY XZ YZ ZZ, not diagonal-first.
    assert expand_multipole(
        dict(zip(["XX", "XY", "YY", "XZ", "YZ", "ZZ"], rotated)), 2
    ) == pytest.approx(rot @ full @ rot.T, abs=1e-12)


def test_rotation_between_recovers_a_known_rotation():
    rot = random_rotation(5)
    source = np.random.default_rng(11).normal(size=(6, 3))
    found, rmsd = rotation_between(source, source @ rot.T)
    assert found == pytest.approx(rot, abs=1e-10)
    assert rmsd < 1e-12
    assert np.linalg.det(found) == pytest.approx(1.0, abs=1e-12)


def test_rotation_between_refuses_a_reflection():
    """A mirrored copy is not a frame change, and must not be fitted as one."""
    source = np.random.default_rng(12).normal(size=(6, 3))
    reflected = source * np.array([1.0, 1.0, -1.0])
    with pytest.raises(QChemParseError):
        rotation_between(source, reflected)


def test_rotation_between_refuses_a_translated_copy():
    """The uncentered Procrustes fit is what proves the two frames share an origin."""
    source = np.random.default_rng(13).normal(size=(6, 3))
    with pytest.raises(QChemParseError):
        rotation_between(source, source + np.array([0.5, 0.0, 0.0]))


def test_recenter_puts_the_nuclear_charge_centroid_at_the_origin():
    symbols = ["O", "H", "H"]
    positions = np.array([[1.0, 2.0, 3.0], [1.9, 2.0, 3.0], [0.6, 2.8, 3.0]])
    moved = recenter(symbols, positions)
    assert center_of_nuclear_charge(symbols, moved) == pytest.approx(np.zeros(3), abs=1e-12)
    # A pure translation: internal geometry is untouched.
    assert np.linalg.norm(moved[0] - moved[1]) == pytest.approx(
        np.linalg.norm(positions[0] - positions[1]), abs=1e-12
    )


# --- naming -----------------------------------------------------------------


def test_fragment_formulas_use_conventional_names():
    assert fragment_formula(["O", "H", "H"], 0) == "H2O"
    assert fragment_formula(["O", "H", "H", "H"], 1) == "H3O+"
    # Hill order would say "HO-"; everything else in this repo says "OH-".
    assert fragment_formula(["O", "H"], -1) == "OH-"


def test_fragmentation_config_type_lists_fragments_in_order():
    symbols = ["O", "H", "H", "H", "O", "H", "H"]
    assert fragmentation_config_type(symbols, [0, 0, 0, 0, 1, 1, 1], [1, 0]) == "H3O+_H2O"
    assert fragmentation_config_type(symbols, [0, 0, 0, 1, 1, 1, 1], [0, 1]) == "H2O_H3O+"


def test_canonical_basis_normalizes_the_case_the_templates_disagree_on():
    assert canonical_basis("def2-tzvpd") == "def2-TZVPD"
    assert canonical_basis("def2-TZVPD") == "def2-TZVPD"
    assert canonical_basis("cc-pVTZ") == "cc-pVTZ"


def test_same_level_of_theory_ignores_case_but_not_content():
    """The aimd and eda templates spell one basis two ways; that must not be a mismatch."""
    assert same_level_of_theory(("wB97M-V", "def2-tzvpd"), ("wB97M-V", "def2-TZVPD"))
    assert not same_level_of_theory(("wB97M-V", "def2-TZVPD"), ("wB97X-V", "def2-TZVPD"))
    assert not same_level_of_theory(("wB97M-V", "def2-TZVPD"), ("wB97M-V", "def2-SVPD"))


# --- the extxyz round trip --------------------------------------------------


def synthetic_frame():
    """An H3O+(H2O) frame with the two charge placements, all values distinct.

    Every array is filled with values that differ per fragmentation and per
    fragment, so a transposed or off-by-one reshape cannot pass.
    """
    symbols = ["O", "H", "H", "H", "O", "H", "H"]
    positions = np.array(
        [
            [-1.20, -0.03, -0.04], [0.01, -0.00, 0.03], [-1.62, 0.71, -0.44],
            [-1.67, -0.33, 0.74], [1.19, 0.04, -0.04], [1.64, -0.72, -0.45],
            [1.70, 0.30, 0.74],
        ]
    )
    positions = recenter(symbols, positions)
    rng = np.random.default_rng(21)

    frags = []
    for k, (idx, charges) in enumerate(
        [
            (np.array([0, 0, 0, 0, 1, 1, 1]), [1, 0]),
            (np.array([0, 1, 0, 0, 1, 1, 1]), [0, 1]),
        ]
    ):
        frags.append(
            Fragmentation(
                fragment_idx=idx,
                fragment_charges=charges,
                fragment_mults=[1, 1],
                fragment_energies=np.array([-76.7 - k, -76.4 - k]),
                fragment_dipoles=rng.normal(size=(2, 3)),
                fragment_second_moments=rng.normal(size=(2, 6)),
                fragment_mulliken=rng.normal(size=7),
                eda={
                    name: float(k + 1) * (i + 1) * 1e-3
                    for i, name in enumerate(
                        ["cls_elec", "mod_pauli", "disp", "pol", "ct", "prp", "frz", "int"]
                    )
                },
                rank=k,
                charge_fragment=k,
                excess_distance=0.25 * k,
                source=f"eda/fake/state{k:02d}.out",
            )
        )

    return MultiFragFrame(
        symbols=symbols,
        positions=positions,
        forces=rng.normal(size=(7, 3)) * 0.01,
        energy=-153.19,
        mulliken=rng.normal(size=7),
        multipoles={
            "dipole": symmetric_tensor(1, 31),
            "quadrupole": symmetric_tensor(2, 32),
            "octopole": symmetric_tensor(3, 33),
            "hexadecapole": symmetric_tensor(4, 34),
        },
        fragmentations=frags,
        total_charge=1,
        multiplicity=1,
        method="wB97M-V",
        basis="def2-TZVPD",
        config_type="w1_H3O+",
        sample_id=50,
        aimd_step=51,
        source="aimd/outputs/h3o_w1.out",
    )


@pytest.fixture
def written(tmp_path):
    frame = synthetic_frame()
    path = tmp_path / "two_fragmentations.xyz"
    write_frames(path, [frame, frame])
    return frame, path


def test_round_trip_preserves_every_per_fragmentation_array(written):
    frame, path = written
    frames = read_multifrag_extxyz(path)
    assert len(frames) == 2

    got = frames[0]
    assert got["n_fragmentations"] == 2
    assert got["symbols"] == frame.symbols
    assert got["positions"] == pytest.approx(frame.positions, abs=1e-9)
    assert got["forces"] == pytest.approx(frame.forces, abs=1e-12)
    assert got["energy"] == pytest.approx(frame.energy, abs=1e-12)
    assert got["mulliken"] == pytest.approx(frame.mulliken, abs=1e-9)

    for k, ref in enumerate(frame.fragmentations):
        assert np.array_equal(got["fragment_idx"][k], ref.fragment_idx)
        assert got["fragment_charges"][k] == pytest.approx(ref.fragment_charges)
        assert got["fragment_energies"][k] == pytest.approx(ref.fragment_energies, abs=1e-9)
        assert got["fragment_dipoles"][k] == pytest.approx(ref.fragment_dipoles, abs=1e-12)
        assert got["fragment_second_moments"][k] == pytest.approx(
            ref.fragment_second_moments, abs=1e-12
        )
        assert got["fragment_mulliken"][k] == pytest.approx(ref.fragment_mulliken, abs=1e-9)
        for name, value in ref.eda.items():
            assert got["eda"][name][k] == pytest.approx(value, abs=1e-15)

    assert list(got["ranks"]) == [0, 1]
    assert got["excess_distance"] == pytest.approx([0.0, 0.25], abs=1e-12)
    assert got["config_types"] == ["H3O+_H2O", "H2O_H3O+"]
    for name, tensor in frame.multipoles.items():
        assert got["multipoles"][name] == pytest.approx(tensor, abs=1e-12)


def test_the_fragmentations_are_actually_different(written):
    """Otherwise the round-trip test above would pass on a broken slice."""
    got = read_multifrag_extxyz(written[1])[0]
    assert not np.array_equal(got["fragment_idx"][0], got["fragment_idx"][1])
    assert got["eda"]["ct"][0] != got["eda"]["ct"][1]


def test_load_extxyz_selects_one_fragmentation(written):
    """Each fragmentation loads as an ordinary dataset, with its atoms re-sorted.

    ``load_extxyz`` groups a frame's atoms by the selected partition, because
    ``union_pairs`` refuses an interleaved one. So the per-atom rows come back
    *permuted* relative to the file, and the check is that the permutation carried
    everything with it -- not that the order was preserved, which it deliberately is not.
    """
    import ase.units

    frame, path = written
    for k, ref in enumerate(frame.fragmentations):
        ds = load_extxyz(path, dtype=torch.float64, fragmentation=k)
        assert len(ds) == 2
        got = ds._fragment_idx[:7].numpy()
        assert (np.diff(got) >= 0).all(), "atoms must come back grouped by fragment"
        # A stable sort of the partition labels is exactly the labels, sorted.
        assert got.tolist() == sorted(ref.fragment_idx)

        order = np.argsort(ref.fragment_idx, kind="stable")
        assert ds._pos[:7].numpy() == pytest.approx(frame.positions[order], abs=1e-9)
        assert ds._forces[:7].numpy() == pytest.approx(
            frame.forces[order] / ase.units.Bohr, abs=1e-9
        )

        assert ds._fragment_charge[:2].tolist() == [float(c) for c in ref.fragment_charges]
        assert ds._fragment_energy[:2].numpy() == pytest.approx(ref.fragment_energies, abs=1e-9)
        assert ds._fragment_dipole[:2].numpy() == pytest.approx(ref.fragment_dipoles, abs=1e-9)
        for name, value in ref.eda.items():
            assert ds._eda[name][0].item() == pytest.approx(value, abs=1e-15)


def test_the_synthetic_frame_actually_exercises_the_sort(written):
    """Otherwise the test above would pass on a loader that never re-sorted anything."""
    frame, _ = written
    interleaved = [
        k for k, f in enumerate(frame.fragmentations)
        if (np.diff(f.fragment_idx) < 0).any()
    ]
    assert interleaved, "no fragmentation in the fixture is interleaved"


def test_an_interleaved_partition_survives_union_pairs(written):
    """The reason the sort exists: the pair builder refuses an interleaved partition."""
    from rsfff.ff.pairs import union_pairs

    _, path = written
    for k in (0, 1):
        ds = load_extxyz(path, dtype=torch.float64, fragmentation=k)
        batch = ds.flat_batch(range(len(ds)))
        union_pairs(batch.positions, batch.batch_idx, batch.fragment_idx, 6.0)


def test_load_extxyz_rejects_an_out_of_range_fragmentation(written):
    with pytest.raises(ValueError, match="out of range"):
        load_extxyz(written[1], dtype=torch.float64, fragmentation=2)


def test_load_extxyz_rejects_a_fragmentation_on_a_single_fragmentation_file():
    with pytest.raises(ValueError, match="one fragmentation per frame"):
        load_extxyz(
            "data/wb97mv_tzvpd/h2o_wb97mv_tzvpd.xyz",
            dtype=torch.float64,
            fragmentation=1,
        )
