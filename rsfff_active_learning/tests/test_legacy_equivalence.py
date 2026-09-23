"""The store path gives bitwise the frame the old qchem_roundtrip parser gave.

Needs the rsfff checkout (for scripts/parse_roundtrip.py and rsfff/src/qcgen):

    RSFFF_REPO=~/dev/rsfff python -m pytest tests/test_legacy_equivalence.py
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
REPO = Path(os.environ.get("RSFFF_REPO", ROOT.parent)).expanduser()
PARSER = REPO / "scripts" / "parse_roundtrip.py"
pytestmark = pytest.mark.skipif(not PARSER.exists(), reason=f"no {PARSER} (set RSFFF_REPO)")


def test_same_frame_as_parse_roundtrip(tmp_path):
    sys.path.insert(0, str(ROOT))
    qc = types.ModuleType("qcgen")                 # the parsers, without qcgen/__init__ (pyscf)
    qc.__path__ = [str(REPO / "src" / "qcgen")]
    sys.modules.setdefault("qcgen", qc)
    sys.path.insert(0, str(PARSER.parent))
    import parse_roundtrip as pr
    from cc_workers.common.store import Store
    from cc_workers.workers.qchem.adopt import adopt_output

    from rsfff_al.store_io import training_frame

    store = Store(tmp_path / "store")
    ids = {k: adopt_output(store, (DATA / f"{k}_w2.out").read_text())[0] for k in ("eda", "force")}
    fj, ej = store.get("qchem", ids["force"]), store.get("qchem", ids["eda"])
    new = training_frame(fj.load_result(), ej.load_result(), spec_record=fj.record())
    old, _ = pr.build_merged_frame(str(DATA / "eda_w2.out"), str(DATA / "force_w2.out"), str(DATA))
    pairs = {"pos": (new["arrays"]["pos"], old.positions),
             "forces": (new["arrays"]["forces"], old.forces),
             "mulliken": (new["arrays"]["mulliken_charges"], old.mulliken),
             "energy": (new["info"]["energy"], old.energy),
             "fragment_energies": (new["info"]["fragment_energies"], old.fragment_energies),
             "fragment_dipoles": (new["info"]["fragment_dipoles"], old.fragment_dipoles),
             "fragment_second_moments": (new["info"]["fragment_second_moments"],
                                         old.fragment_second_moments)}
    pairs.update({f"eda_{k}": (new["info"][f"eda_{k}"], v) for k, v in old.eda.items()})
    pairs.update({k: (new["info"][k], old.multipoles[k]) for k in old.multipoles})
    for name, (a, b) in pairs.items():
        assert np.array_equal(np.ravel(a), np.ravel(b)), name
    assert (new["info"]["method"], new["info"]["basis"]) == (old.method, old.basis)
