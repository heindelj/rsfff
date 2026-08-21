"""End-to-end: the AIMD/EDA driver against the real ``qchem_roundtrip`` bundle.

Skipped when the bundle is absent, since its outputs are ~130 MB of Q-Chem and
are not committed as fixtures. What it pins is the part no unit test can: that
the harvest markers, the AIMD outputs and the EDA outputs on disk still describe
each other, and that the writer's output survives both readers.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from rsfff.qcgen.multifrag import read_multifrag_extxyz, recenter
from rsfff.train.data import load_extxyz

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
ROOT = os.path.join(REPO, "qchem_roundtrip")
SCRIPT = os.path.join(REPO, "scripts", "parse_aimd_eda.py")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, "aimd", "outputs")),
    reason="the qchem_roundtrip bundle is not present",
)


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    """Three frames of ``h3o_w2``: a 3-fragmentation system, run through the CLI."""
    out = tmp_path_factory.mktemp("aimd_eda")
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--root", ROOT, "--out-dir", str(out),
         "--stems", "h3o_w2", "--limit", "3", "--strict"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    path = out / "w2_h3o+_wb97mv_tzvpd.xyz"
    assert path.exists(), proc.stderr
    return path, proc.stderr


def test_strict_mode_produces_every_frame_with_every_fragmentation(parsed):
    path, _ = parsed
    frames = read_multifrag_extxyz(path)
    assert len(frames) == 3
    for f in frames:
        assert f["n_fragmentations"] == 3
        assert list(f["ranks"]) == [0, 1, 2]
        assert list(f["n_fragments"]) == [3, 3, 3]


def test_alignment_residual_is_reported_and_tiny(parsed):
    _, stderr = parsed
    line = next(ln for ln in stderr.splitlines() if "worst alignment RMSD" in ln)
    rmsd = float(line.split("worst alignment RMSD")[1].split()[0])
    # Q-Chem prints 10 decimals, so a genuine rigid-body match lands near 1e-9.
    assert rmsd < 1e-7, line


def test_every_fragmentation_partitions_the_same_atoms_differently(parsed):
    path, _ = parsed
    for f in read_multifrag_extxyz(path):
        idx = f["fragment_idx"]
        assert idx.shape == (3, len(f["symbols"]))
        for k in range(3):
            assert sorted(set(idx[k])) == [0, 1, 2]
        # The whole point of the format: they are not all the same partition.
        assert not np.array_equal(idx[0], idx[1]) or not np.array_equal(idx[1], idx[2])


def test_the_charge_sits_on_a_different_fragment_in_each(parsed):
    path, _ = parsed
    for f in read_multifrag_extxyz(path):
        placements = [int(np.argmax(np.abs(q))) for q in f["fragment_charges"]]
        assert sorted(placements) == [0, 1, 2]
        for q in f["fragment_charges"]:
            assert q.sum() == pytest.approx(1.0)


def test_frozen_fragment_charges_are_integral(parsed):
    """Each frozen fragment is its own isolated SCF, so its Mulliken sums to its charge."""
    path, _ = parsed
    for f in read_multifrag_extxyz(path):
        for k in range(f["n_fragmentations"]):
            idx = f["fragment_idx"][k]
            for frag, formal in enumerate(f["fragment_charges"][k]):
                got = f["fragment_mulliken"][k][idx == frag].sum()
                assert got == pytest.approx(formal, abs=1e-4)


def test_energy_decomposition_closes_for_every_fragmentation(parsed):
    """``E_total = sum(E_frag) + E_int``, to Q-Chem's kJ/mol round-off."""
    path, _ = parsed
    for f in read_multifrag_extxyz(path):
        for k in range(f["n_fragmentations"]):
            total = f["fragment_energies"][k].sum() + f["eda"]["int"][k]
            # Q-Chem converts the EDA terms with ~2625.5323 rather than CODATA's
            # 2625.4996, which is ~1.3e-5 relative on E_int.
            assert total == pytest.approx(f["energy"], abs=2e-5)


def test_the_geometry_is_recentered_on_the_nuclear_charge_centroid(parsed):
    path, _ = parsed
    for f in read_multifrag_extxyz(path):
        assert f["positions"] == pytest.approx(
            recenter(f["symbols"], f["positions"]), abs=1e-10
        )


def test_each_fragmentation_loads_as_an_ordinary_dataset(parsed):
    path, _ = parsed
    raw = read_multifrag_extxyz(path)
    seen = []
    for k in range(3):
        ds = load_extxyz(path, dtype=torch.float64, fragmentation=k)
        assert len(ds) == 3
        assert ds.has_fragments and ds.has_forces
        n = len(raw[0]["symbols"])
        assert ds._fragment_idx[:n].tolist() == list(raw[0]["fragment_idx"][k])
        seen.append(ds._eda["ct"][0].item())
        assert ds._eda["ct"][0].item() == pytest.approx(raw[0]["eda"]["ct"][k], abs=1e-15)
    assert len(set(seen)) == 3, "the three decompositions should not share a CT energy"
