"""Monomer database, grids and pair sampling (rsfff.qcgen.elst_db): water + Na+, PBE/def2-SVP."""

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("ase")

import pyscf.lib

pyscf.lib.num_threads(1)

from ase.io import read

from rsfff.qcgen import elst_db as D
from rsfff.qcgen import elst_ref as E

_W = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    root = tmp_path_factory.mktemp("edb")
    structs = [{"name": "water", "symbols": ["O", "H", "H"], "coords": _W},
               {"name": "na", "symbols": ["Na"], "coords": np.zeros((1, 3)), "charge": 1}]
    return D.build_db(root, structs, xc="pbe", basis="def2-svp", log=None)


def test_build_is_resumable(db):
    assert len(db) == 2 and db.index["na"]["charge"] == 1
    assert not db.add("water", ["O", "H", "H"], _W, xc="pbe", basis="def2-svp")     # already there
    again = D.MonomerDB(db.root)
    assert set(again.index) == {"water", "na"} and np.allclose(again.get("water").dm, db.get("water").dm)


def test_grid_points_respect_scaled_surfaces(db):
    g = D.monomer_grid(db, "water", scales=(0.8, 1.4), density=1.0, rng=0)
    R = D.vdw_radii(["O", "H", "H"])
    ratio = np.linalg.norm(g["points"][:, None] - g["coords"][None], axis=2) / R[None]
    for s in (0.8, 1.4):
        assert np.all(ratio[g["scale"] == s].min(1) >= s - 1e-6)
    assert np.allclose(g["phi"], E.potential(db.get("water").place(), g["points"]))


def test_contact_placement(db):
    A, B = db.get("water"), db.get("na")
    for c in (2.0, 3.0, 6.0):
        pa, pb = D.place_at_contact(A, B, c, rng=1)
        assert abs(E.min_contact(pa, pb) - c) < 1e-3


def test_pairs_roundtrip_and_accuracy(db, tmp_path):
    out = tmp_path / "pairs.extxyz"
    D.sample_pairs(db, 6, out, contact_range_ang=(1.8, 4.0), rng=3, log=None)
    frames = read(str(out), ":")
    assert len(frames) == 6
    for f in frames:
        pa, pb = D.reassemble(db, f)
        assert abs(f.info["e_elst"] - E.pair_energy(pa, pb, exact_ee=True)["total"]) < 3e-6
        assert np.allclose(f.arrays["field_inter"][: len(pa.coords)], E.field(pb, pa.coords), atol=1e-8)
        assert abs(f.info["contact_ang"] - E.min_contact(pa, pb) * E.BOHR) < 1e-3
