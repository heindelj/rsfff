"""Store round trip without Q-Chem: adopt a finished eda/force pair, ask the label stage for
the same geometry, and check it reuses those jobs, produces a training frame identical to the
legacy pipeline's, and that the export reads back through rsfff's loader."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "data"

cc = pytest.importorskip("cc_workers")
from cc_workers.common.store import Store  # noqa: E402
from cc_workers.workers.qchem.adopt import adopt_output  # noqa: E402
from easyal import new_frame, read_extxyz  # noqa: E402

from rsfff_al.dataset import export_training  # noqa: E402
from rsfff_al.label import QChemStoreLabel  # noqa: E402
from rsfff_al.store_io import specs_for  # noqa: E402


class FakeCtx:
    """The parts of easyal.Context a stage touches."""

    def __init__(self, tmp, frames, params):
        self.dir = tmp / "label"
        self.dir.mkdir()
        self.root = tmp / "loop_test"
        self.iteration = 0
        self.params = params
        self._frames = frames
        self.record = {}

    @property
    def scratch(self):
        p = self.dir / "scratch"
        p.mkdir(exist_ok=True)
        return p

    def read(self):
        return self._frames

    def log(self, **kw):
        self.record.setdefault("metrics", {}).update(kw)

    def note(self, text):
        self.record.setdefault("notes", []).append(text)

    def track(self, *a):
        pass


@pytest.fixture()
def store_with_pair(tmp_path):
    store = Store(tmp_path / "store")
    ids = {}
    for kind in ("eda", "force"):
        jid, what = adopt_output(store, (DATA / f"{kind}_w2.out").read_text(),
                                 tags={"bundle": "legacy"})
        assert what == "done"
        ids[kind] = jid
    return store, ids


def selected_frame(store, ids, labels="force,eda"):
    rec = store.get("qchem", ids["eda"]).record()["spec"]["molecule"]
    n = len(rec["symbols"]) // 3
    return new_frame(rec["symbols"], np.array(rec["coords"]), {
        "charge": 0, "multiplicity": 1, "n_waters": n, "n_fragments": n,
        "fragment_charges": [0] * n, "fragment_multiplicities": [1] * n,
        "traj_id": "i00_w002_t000", "time_fs": 120.0, "labels": labels,
        "sigma_energy": 1e-3, "sigma_forces": 1e-2},
        fragment_idx=rec["fragment_idx"])


def test_same_geometry_same_jobs(store_with_pair):
    store, ids = store_with_pair
    specs = specs_for(selected_frame(store, ids))
    assert specs["force"].id == ids["force"] and specs["eda2"].id == ids["eda"]


def test_label_stage_and_export(tmp_path, store_with_pair):
    store, ids = store_with_pair
    ctx = FakeCtx(tmp_path, [selected_frame(store, ids)], {"store": str(store.root)})
    out = QChemStoreLabel().run(ctx)
    assert len(out) == 1 and "eda_int" in out[0]["info"]
    assert out[0]["info"]["split_group"] == "i00_w002_t000"
    assert ctx.record["metrics"]["n_new_jobs"] == 0          # reused, nothing recomputed
    assert (store.root / "_code" / "cc_workers").is_dir()     # batch runners can import it
    # the legacy pipeline gives bitwise the same numbers (checked in detail elsewhere); here
    # just that the job now also answers to the AL request's tags
    rec = store.get("qchem", ids["force"]).record()
    assert any(a.get("traj_id") == "i00_w002_t000" for a in rec["aliases"])

    manifest = export_training(store.root, tmp_path / "export", log=lambda s: None)
    assert manifest["totals"] == {"eda": 1}
    frames = read_extxyz(tmp_path / "export" / "eda" / "w2.xyz")
    assert frames[0]["info"]["method"] == "wB97M-V"

    rsfff = pytest.importorskip("rsfff.train.data")
    ds = rsfff.load_cluster_datasets([str(tmp_path / "export" / "eda" / "w2.xyz")])
    assert len(ds) == 1 and ds._eda is not None and ds._forces is not None


def test_force_only_export(tmp_path, store_with_pair):
    store, ids = store_with_pair
    import shutil

    shutil.rmtree(store.job_dir("qchem", ids["eda"]))       # as if no EDA was ever run
    manifest = export_training(store.root, tmp_path / "export", log=lambda s: None)
    assert manifest["totals"] == {"force": 1}
    f = read_extxyz(tmp_path / "export" / "force" / "w2.xyz")[0]
    assert list(f["arrays"]["fragment_idx"]) == [0, 0, 0, 1, 1, 1]
    assert not any(k.startswith(("eda_", "fragment_energies")) for k in f["info"])


def test_label_force_only_request_reuses_legacy_force(tmp_path, store_with_pair):
    store, ids = store_with_pair
    ctx = FakeCtx(tmp_path, [selected_frame(store, ids, labels="force")], {"store": str(store.root)})
    out = QChemStoreLabel().run(ctx)
    assert "eda_int" not in out[0]["info"]
    assert list(out[0]["arrays"]["fragment_idx"]) == [0, 0, 0, 1, 1, 1]
