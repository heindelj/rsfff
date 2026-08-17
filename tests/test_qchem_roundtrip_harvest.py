import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "qchem_roundtrip" / "scripts" / "qchem_roundtrip.py"
SPEC = importlib.util.spec_from_file_location("qchem_roundtrip_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qchem_roundtrip = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qchem_roundtrip
SPEC.loader.exec_module(qchem_roundtrip)


def h3o_w2_geometry():
    symbols = ["O", "H", "H", "H", "O", "H", "H", "O", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.95, 0.0, 0.0),
        (0.0, 0.95, 0.0),
        (1.20, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 0.95, 0.0),
        (3.0, -0.95, 0.0),
        (6.0, 0.0, 0.0),
        (6.0, 0.95, 0.0),
        (6.0, -0.95, 0.0),
    ]
    return symbols, coords


def test_rank_oh_fragment_assignments_orders_charge_states_by_excess_distance():
    symbols, coords = h3o_w2_geometry()
    assignments = qchem_roundtrip.rank_oh_fragment_assignments(symbols, coords, total_charge=1)

    assert [assignment.rank for assignment in assignments] == [0, 1, 2]
    assert [assignment.charge_fragment for assignment in assignments] == [0, 1, 2]
    assert assignments[0].excess_distance == pytest.approx(0.0)
    assert assignments[0].excess_distance <= assignments[1].excess_distance
    assert assignments[1].excess_distance <= assignments[2].excess_distance
    assert assignments[0].fragment_charges == [1, 0, 0]
    assert assignments[0].fragment_idx == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_harvest_aimd_outputs_writes_ranked_states_under_eda_structure_folder(tmp_path):
    root = tmp_path
    template = root / "templates" / "eda.in"
    template.parent.mkdir()
    template.write_text("$molecule\n0 1\n$end\n\n$rem\nJOBTYPE eda\n$end\n")
    cfg = {
        "_config_path": str(root / "config.json"),
        "calculations": {
            "aimd": {"template": "templates/aimd.in", "molecule": {"mode": "plain"}},
            "eda": {"template": "templates/eda.in", "molecule": {"mode": "fragments"}},
        },
    }
    (root / "config.json").write_text("{}")
    (root / "aimd" / "geoms").mkdir(parents=True)
    (root / "aimd" / "outputs").mkdir(parents=True)
    symbols, coords = h3o_w2_geometry()
    geom_lines = [
        str(len(symbols)),
        (
            'Properties=species:S:1:pos:R:3:fragment_idx:I:1 charge=1 multiplicity=1 '
            'n_fragments=3 fragment_charges="1 0 0" fragment_multiplicities="1 1 1"'
        ),
    ]
    fragment_idx = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    for sym, coord, frag in zip(symbols, coords, fragment_idx):
        geom_lines.append(f"{sym} {coord[0]} {coord[1]} {coord[2]} {frag}")
    (root / "aimd" / "geoms" / "h3o_w2.extxyz").write_text("\n".join(geom_lines) + "\n")

    out_lines = ["TIME STEP #0", "Standard Nuclear Orientation (Angstroms)"]
    for idx, (sym, coord) in enumerate(zip(symbols, coords), start=1):
        out_lines.append(f"{idx:5d} {sym:2s} {coord[0]:14.8f} {coord[1]:14.8f} {coord[2]:14.8f}")
    out_lines.append("--------------")
    (root / "aimd" / "outputs" / "h3o_w2.out").write_text("\n".join(out_lines))

    n_frames, n_inputs = qchem_roundtrip.harvest_aimd_outputs(root, cfg, stride=1)

    assert (n_frames, n_inputs) == (1, 3)
    job_dir = root / "eda" / "w2_H3O+"
    assert sorted(path.name for path in (job_dir / "inputs").glob("*.in")) == [
        "w2_H3O+_step00000_frame00000_state00_qfrag0.in",
        "w2_H3O+_step00000_frame00000_state01_qfrag1.in",
        "w2_H3O+_step00000_frame00000_state02_qfrag2.in",
    ]
    summary = json.loads((job_dir / "state" / "harvest_summary.json").read_text())
    assert summary["0"]["n"] == 1
    assert summary["1"]["excess_distance_min"] > 0.0


def test_claim_next_job_scans_nested_eda_job_dirs(tmp_path):
    root = tmp_path
    cfg = {"calculations": {"eda": {"priority": 20, "template": "templates/eda.in"}}}
    job_dir = root / "eda" / "w2_H3O+"
    qchem_roundtrip.ensure_job_layout(job_dir)
    input_path = job_dir / "inputs" / "sample.in"
    input_path.write_text("$molecule\n0 1\n$end\n")

    job = qchem_roundtrip.claim_next_job(root, cfg)

    assert job is not None
    assert job.calculation == "eda"
    assert job.calc_dir == job_dir
    assert job.input_path == input_path
    assert job.output_path == job_dir / "outputs" / "sample.out"
