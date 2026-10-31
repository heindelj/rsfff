"""The frozen fit (rsfff.ff.complete.frozen_fit): reading labelled pair scans, and recovering known Pauli parameters."""
import json
from pathlib import Path

import torch

from rsfff.ff.complete import frozen_fit as FF
from rsfff.ff.complete.elst import permanent_electrostatics
from rsfff.ff.ionct.params import IonTable

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2.json"


def _xyz(a, b, r, extra=""):
    return f"2\nscan=pair eda=1{extra}\n{a} 0 0 0 0\n{b} 0 0 {r} 1\n"


def test_read_labels(tmp_path):
    lines = [{"xyz": _xyz("Na", "Cl", 2.5), "job": "j1", "es.eda": {"cls_elec": -500.0, "mod_pauli": 40.0}},
             {"xyz": _xyz("Na", "Cl", 3.0), "job": "j2", "report": {"data": {"es.eda": {"cls_elec": -450.0, "mod_pauli": 9.0}}}},
             {"xyz": _xyz("Na", "Cl", 3.0, " field=x:0.002"), "job": "j3", "es.eda": {"cls_elec": -1.0, "mod_pauli": 1.0}},
             {"xyz": "1\nmonomer=1\nNa 0 0 0 0\n", "job": "j4", "es.eda": {"cls_elec": 0.0}},
             {"xyz": _xyz("Cl", "Na", 2.5), "job": "j5"}]
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = FF.read_labels(p)
    assert [r["job"] for r in rows] == ["j1", "j2"]
    assert abs(rows[0]["r"] - 2.5) < 1e-12 and rows[1]["mod_pauli"] == 9.0
    pts = FF.pair_points(rows + [{"symbols": ["Cl", "Na"], "r": 2.5, "cls_elec": 0.0, "mod_pauli": 0.0}], ["Na", "Cl"])
    assert len(pts) == 2 and pts.key == ["Na-Cl", "Na-Cl"]          # the reversed duplicate is the same point


def test_recovers_known_pauli():
    """Labels made by the model itself (the table's shells; known w and s): with the shells held by the prior, the fit
    finds w and s again."""
    doc = json.load(open(PARAMS))
    sym = ["Na", "Cl"]
    tab = IonTable.load(PARAMS, sym)
    S, T = tab.shell_n.shape
    live = tab.shell_n > 0
    w_true = torch.where(live, torch.tensor([[6.0, 0.0], [7.0, 2.5]], dtype=torch.float64)[:, :T], torch.zeros_like(tab.shell_b))
    s_true = torch.tensor([0.8, 1.1], dtype=torch.float64)
    a, b, r = [], [], []
    for i, j in ((0, 1), (0, 0), (1, 1)):
        for x in torch.arange(2.2, 6.01, 0.2):
            a.append(i); b.append(j); r.append(float(x))
    a, b, r = torch.tensor(a), torch.tensor(b), torch.tensor(r, dtype=torch.float64)
    pos = torch.stack([torch.zeros_like(r)] * 5 + [r], 1).reshape(-1, 3)
    elec = permanent_electrostatics(pos, torch.stack([a, b], 1).reshape(-1), torch.arange(len(r)).repeat_interleave(2), tab) * FF.KJ
    pauli = FF.pauli_energy(w_true, tab.shell_b * s_true[:, None], a, b, r) * FF.KJ
    rows = [{"symbols": [sym[i], sym[j]], "r": float(x), "cls_elec": float(e), "mod_pauli": float(p)}
            for i, j, x, e, p in zip(a.tolist(), b.tolist(), r, elec, pauli)]
    pts = FF.pair_points(rows, sym, window=0.0)
    out, m = FF.fit(doc, sym, pts, mu=1.0, lam=100.0, shared=(), log=lambda *_: None)
    assert m["weighted_rms"]["pauli"] < 1e-3 and m["weighted_rms"]["elec"] < 1e-3
    for k, s in enumerate(sym):
        p = out["ions"][s]["pauli"]
        assert abs(p["s"] - float(s_true[k])) < 1e-3
        assert all(abs(x - float(y)) < 1e-2 * float(y) for x, y in zip(p["w"], w_true[k][live[k]]))
        assert out["ions"][s]["shells_esp"] == doc["ions"][s]["shells"]
    # the table the training reads: pauli_w / pauli_s per ion
    tab_out = IonTable.load(_write(out), sym)
    assert torch.allclose(tab_out.pauli_s, torch.tensor([out["ions"][s]["pauli"]["s"] for s in sym], dtype=torch.float64))


def _write(doc):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(doc, fh)
    return fh.name
