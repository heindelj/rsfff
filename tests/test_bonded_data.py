"""scripts/bonded_data.py: the finite-field labels are the derivatives they claim to be, the sampler has the right
widths, and the files it writes are the ones rsfff.ff.bonded reads. Hartree-Fock in a small basis, for speed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscf")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bonded_data", ROOT / "scripts" / "bonded_data.py")
B = importlib.util.module_from_spec(spec)
sys.modules["bonded_data"] = B
spec.loader.exec_module(B)

SYM = ["O", "H", "H"]
POS = np.array([[0.02, 0.01, 0.12], [-0.78, 0.03, -0.45], [0.75, -0.02, -0.49]])
LEVEL = dict(xc="hf", basis="sto-3g")


def test_forces_and_moments_in_a_field_are_energy_derivatives():
    lv = B.Level(**LEVEL)
    F = np.array([0.004, -0.003, 0.005])
    G = np.array([[0.002, 0.001, -0.0015], [0.001, -0.001, 0.0005], [-0.0015, 0.0005, 0.003]])
    f = B.State(lv, lv.mol(SYM, POS, 0, 1), F=F, G=G).forces()
    h = 1e-4
    for a, i in ((0, 0), (1, 2), (2, 1)):
        e = []
        for s in (1, -1):
            p = POS.copy()
            p[a, i] += s * h * B.BOHR
            e.append(B.State(lv, lv.mol(SYM, p, 0, 1), F=F, G=G).energy)
        assert abs(-(e[0] - e[1]) / (2 * h) - f[a, i]) < 2e-7
    mol = lv.mol(SYM, POS, 0, 1)
    mu, m2 = B.State(lv, mol).moments()
    d, eye = 1e-4, np.eye(3)
    for k in range(3):
        assert abs(-(B.State(lv, mol, F=d * eye[k]).energy - B.State(lv, mol, F=-d * eye[k]).energy) / (2 * d) - mu[k]) < 1e-6
    S = np.zeros((3, 3))
    S[0, 2] = S[2, 0] = 1.0
    assert abs(-(B.State(lv, mol, G=d * S).energy - B.State(lv, mol, G=-d * S).energy) / (2 * d) - m2[0, 2]) < 1e-6


def test_response_labels_against_geometry_differences():
    lv = B.Level(**LEVEL)
    lab = B.label_frame(lv, SYM, POS, 0, 1, raman=True, field_gradient=True, hessian=True, esp=True)
    assert np.allclose(lab["polarizability"], lab["polarizability"].T) and np.linalg.eigvalsh(lab["polarizability"]).min() > 0
    c = lab["quadrupole_polarizability"].reshape(3, 3, 3, 3)
    assert np.abs(c - c.transpose(2, 3, 0, 1)).max() < 1e-3 * np.abs(c).max()
    assert np.abs(lab["dipole_derivatives"].sum(0)).max() < 1e-4                 # neutral: the tensors sum to zero
    a, i, h = 1, 0, 1e-3
    r = []
    for s in (1, -1):
        p = POS.copy()
        p[a, i] += s * h * B.BOHR
        r.append(B.label_frame(lv, SYM, p, 0, 1))
    diff = lambda k: (r[0][k] - r[1][k]) / (2 * h)
    assert np.abs(diff("dipole") - lab["dipole_derivatives"][a, i]).max() < 1e-3
    assert np.abs(diff("polarizability").reshape(9) - lab["polarizability_derivatives"][a, i]).max() < 5e-3
    assert np.abs(diff("quadrupole").reshape(9) - lab["quadrupole_derivatives"][a, i]).max() < 1e-3
    assert np.abs(-diff("forces").reshape(-1) - lab["hessian"][3 * a + i]).max() < 1e-4
    assert lab["frequencies"].shape == (3,) and lab["esp_response"].shape == (3, len(lab["esp_points"]))
    far = np.array([[30.0, 10.0, -20.0]])
    v = B.State(lv, lv.mol(SYM, POS, 0, 1)).esp(far)[0]
    assert abs(v - lab["dipole"] @ far[0] / np.linalg.norm(far) ** 3) < 5e-5     # a neutral molecule's far potential


def test_wigner_widths_and_scans():
    sym = ["O", "H"]
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    k = 0.5                                                   # Hartree / bohr^2 along the bond
    h = np.zeros((6, 6))
    h[np.ix_([2, 5], [2, 5])] = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    mu = B.MASS["O"] * B.MASS["H"] / (B.MASS["O"] + B.MASS["H"]) * B.AMU
    w = np.sqrt(k / mu)
    for classical, T in ((False, 300.0), (True, 300.0)):
        x = B.wigner_samples(pos, h, sym, 20000, T, classical, seed=1)
        r = np.linalg.norm(x[:, 1] - x[:, 0], axis=-1) / B.BOHR
        var = B.KB * T / k if classical else 0.5 / (mu * w) / np.tanh(0.5 * w / (B.KB * T))
        assert abs(r.var() / var - 1) < 0.05
    with pytest.raises(ValueError, match="imaginary"):
        B.wigner_samples(pos, -h, sym, 2)
    h3o = np.array(B.SPECIES["h3o+"][1])
    sc = B.scans(B.SPECIES["h3o+"][0], h3o, [(0, 1), (0, 2), (0, 3)], points=5)
    kinds = [t.rsplit("_", 1)[0] if t.startswith("scan_umbrella") else "_".join(t.split("_")[:2]) for t, _ in sc]
    assert kinds.count("scan_stretch") == 15 and kinds.count("scan_bend") == 15 and kinds.count("scan_umbrella") == 9
    stretch = [p for t, p in sc if t == "scan_stretch_0_1"]
    r = [np.linalg.norm(p[1] - p[0]) for p in stretch]
    assert np.allclose(np.array(r) / np.linalg.norm(h3o[1] - h3o[0]), np.linspace(0.75, 1.6, 5))
    umb = [p for t, p in sc if t.startswith("scan_umbrella")]
    height = [(p[0] - p[1:].mean(0))[2] for p in umb]
    assert height[0] > 0 > height[-1] and abs(height[4]) < 1e-12              # through planar to inverted


def test_pipeline_writes_what_the_model_reads(tmp_path):
    B.main(["all", "--species", "oh-", "--out", str(tmp_path), "--xc", "hf", "--basis", "sto-3g", "--wigner", "2",
            "--points", "3", "--esp"])
    from rsfff.ff.bonded import collate, read_extxyz
    ref = read_extxyz(tmp_path / "oh-_opt.xyz")[0]
    assert ref.frequencies.shape == (1,) and ref.charge == -1 and np.abs(ref.forces).max() < 1e-4
    mols = read_extxyz(tmp_path / "oh-.xyz")
    b = collate(mols)
    assert len(mols) == 5 and b.labelled("polarizability").all() and b.forces.shape == (10, 3)
    esp = np.load(tmp_path / "oh-.esp.npz")
    assert esp["esp_response_4"].shape == (3, len(esp["points_4"]))
    before = sorted(p.stat().st_mtime_ns for p in (tmp_path / "oh-.xyz.parts").iterdir())
    B.main(["label", str(tmp_path / "oh-_geoms.xyz"), "--out", str(tmp_path / "oh-.xyz"), "--xc", "hf", "--basis", "sto-3g", "--esp"])
    assert before == sorted(p.stat().st_mtime_ns for p in (tmp_path / "oh-.xyz.parts").iterdir())   # resumed, not redone
