"""active_learning: stage bookkeeping, provenance, resume, and the Q-Chem label flow.

The campaign tests use toy stages and need only the standard library. The label-flow test
reuses the ion-cluster EDA/force outputs already in qchem_roundtrip/ as stand-in results and
needs what scripts/parse_roundtrip.py imports (numpy, and pyscf through rsfff.qcgen).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from active_learning import (  # noqa: E402
    Campaign,
    LabelFrames,
    ProvenanceError,
    Stage,
    StagePending,
)
from active_learning import qchem  # noqa: E402

ION = REPO / "qchem_roundtrip"


def _xyz(n_atoms: int, sid: str, charge: int = 0) -> str:
    rows = "".join(f"H {i:.1f} 0.0 0.0\n" for i in range(n_atoms))
    return (f"{n_atoms}\nProperties=species:S:1:pos:R:3 charge={charge} multiplicity="
            f"{1 if n_atoms % 2 == 0 else 2} structure_id={sid}\n{rows}")


class ToyBuild(Stage):
    name = "build"
    outputs = {"structures": "structures.extxyz"}

    def run(self, ctx):
        n = self.params["n"]
        ctx.output("structures").write_text("".join(_xyz(2 + ctx.iteration, f"s{i}") for i in range(n)))
        ctx.log_metrics(n_structures=n)


class ToySample(Stage):
    name = "sample"
    outputs = {"candidates": "candidates.extxyz"}

    def run(self, ctx):
        shutil.copyfile(ctx.input("structures"), ctx.output("candidates"))


class ToyLabel(Stage):
    """Pends until a flag file appears, like a Q-Chem wait."""

    name = "label"
    outputs = {"dataset": "dataset/"}

    def run(self, ctx):
        flag = ctx.root / f"released_{ctx.iteration}"
        if self.params.get("gate") and not flag.exists():
            raise StagePending("waiting for flag")
        shutil.copyfile(ctx.input("candidates"), ctx.output("dataset") / f"it{ctx.iteration}.xyz")


class ToyTrain(Stage):
    name = "train"
    outputs = {"checkpoint": "best.pt"}

    def run(self, ctx):
        data = ctx.training_data()
        ctx.output("checkpoint").write_text(json.dumps({
            "from": str(ctx.checkpoint), "data": [p.name for p in data]}))
        ctx.log_metrics(n_files=len(data))


def _campaign(tmp_path, gate=False, n=3):
    init = tmp_path / "init.pt"
    if not init.exists():
        init.write_text("initial")
    base = tmp_path / "base.xyz"
    if not base.exists():
        base.write_text(_xyz(3, "b"))
    return Campaign(tmp_path / "camp",
                    [ToyBuild(n=n), ToySample(), ToyLabel(gate=gate), ToyTrain()],
                    initial_checkpoint=init, base_data=[base], repo=REPO)


def test_iterations_chain_checkpoints_and_data(tmp_path):
    camp = _campaign(tmp_path)
    assert camp.run(2) == "complete"
    ck0 = json.loads((camp.stage_dir(0, "train") / "best.pt").read_text())
    ck1 = json.loads((camp.stage_dir(1, "train") / "best.pt").read_text())
    assert ck0 == {"from": str(tmp_path / "init.pt"), "data": ["base.xyz", "it0.xyz"]}
    assert ck1["from"] == str(camp.stage_dir(0, "train") / "best.pt")
    assert ck1["data"] == ["base.xyz", "it0.xyz", "it1.xyz"]

    rec = camp.read_record(1, "train")
    assert rec["status"] == "complete"
    assert rec["inputs"]["checkpoint"]["sha256"] == camp.read_record(0, "train")["outputs"]["checkpoint"]["sha256"]
    assert [d["path"] for d in rec["training_data"]][-1].endswith("it1.xyz")
    assert rec["outputs"]["checkpoint"]["relpath"] == "best.pt"
    assert "environment" in rec and "code" in rec
    sample = camp.read_record(0, "sample")
    assert sample["inputs"]["build.structures"]["n_frames"] == 3
    rows = json.loads((camp.root / "summary.json").read_text())
    assert len(rows) == 8 and rows[0]["metrics"] == {"n_structures": 3}


def test_rerun_is_a_no_op_when_up_to_date(tmp_path):
    camp = _campaign(tmp_path)
    camp.run(1)
    before = camp.read_record(0, "train")["started"]
    events = (camp.root / "events.jsonl").read_text().count("\n")
    _campaign(tmp_path).run(1)
    assert camp.read_record(0, "train")["started"] == before
    assert (camp.root / "events.jsonl").read_text().count("\n") == events


def test_changed_output_supersedes_and_cascades(tmp_path):
    camp = _campaign(tmp_path)
    camp.run(1)
    (camp.stage_dir(0, "build") / "structures.extxyz").write_text(_xyz(5, "edited"))
    camp.run(1)
    superseded = sorted(p.name.split("_")[0] for p in (camp.iteration_dir(0) / "_superseded").iterdir())
    # build is re-run and regenerates the original file, so downstream inputs are unchanged
    assert superseded == ["build"]
    assert camp.read_record(0, "build")["metrics"] == {"n_structures": 3}


def test_param_change_cascades_downstream(tmp_path):
    _campaign(tmp_path, n=3).run(1)
    camp = _campaign(tmp_path, n=4)
    camp.run(1)
    superseded = sorted(p.name.split("_")[0] for p in (camp.iteration_dir(0) / "_superseded").iterdir())
    assert superseded == ["build", "label", "sample", "train"]
    assert camp.read_record(0, "sample")["inputs"]["build.structures"]["n_frames"] == 4
    manifest = json.loads((camp.root / "campaign.json").read_text())
    assert len(manifest["pipeline_history"]) == 1


def test_pending_stops_then_resumes(tmp_path):
    camp = _campaign(tmp_path, gate=True)
    assert camp.run(2) == "pending"
    assert camp.read_record(0, "label")["status"] == "pending"
    assert camp.read_record(0, "train") is None
    (camp.root / "released_0").write_text("")
    assert camp.run(2) == "pending"          # iteration 1 now waits
    rec = camp.read_record(0, "label")
    assert rec["status"] == "complete" and rec["attempts"][0]["status"] == "pending"
    (camp.root / "released_1").write_text("")
    assert camp.run(2) == "complete"
    assert "pending" not in camp.status()


def test_failure_is_recorded_and_reraised(tmp_path):
    class Boom(ToyTrain):
        def run(self, ctx):
            raise ValueError("bad fit")

    camp = Campaign(tmp_path / "c", [ToyBuild(n=1), ToySample(), ToyLabel(), Boom()], repo=REPO)
    with pytest.raises(ValueError):
        camp.run(1)
    rec = camp.read_record(0, "train")
    assert rec["status"] == "failed" and "bad fit" in rec["error"]
    assert not (camp.root / ".lock").exists()


def test_missing_output_fails_validation(tmp_path):
    class Lazy(ToySample):
        def run(self, ctx):
            pass

    camp = Campaign(tmp_path / "c", [ToyBuild(n=1), Lazy()], repo=REPO)
    with pytest.raises(FileNotFoundError):
        camp.run(1)


def test_changed_start_is_refused(tmp_path):
    _campaign(tmp_path).run(1)
    (tmp_path / "init.pt").write_text("a different model")
    with pytest.raises(ProvenanceError):
        _campaign(tmp_path)


def test_duplicate_output_keys_rejected(tmp_path):
    class Other(ToySample):
        name = "other"

    with pytest.raises(ValueError):
        Campaign(tmp_path / "c", [ToySample(), Other()], repo=REPO)


# ------------------------------------------------------------------------------------------
# Q-Chem label flow
# ------------------------------------------------------------------------------------------

def _roundtrip_copy(tmp_path) -> Path:
    root = tmp_path / "qchem_roundtrip"
    root.mkdir()
    shutil.copy(ION / "config.json", root)
    shutil.copytree(ION / "templates", root / "templates")
    return root


def test_write_jobs_matches_existing_ion_cluster_inputs(tmp_path):
    root = _roundtrip_copy(tmp_path)
    geom = ION / "eda/ion_clusters/geoms/oh-_w3_iso00.extxyz"
    jobs = qchem.write_jobs(geom, "al_t/iter_000", stem="al_t_it000", roundtrip_root=root)
    for kind in ("eda", "force"):
        made = (jobs[kind] / "inputs/al_t_it000_frame0000.in").read_text()
        assert made == (ION / kind / "ion_clusters/inputs/oh-_w3_iso00.in").read_text()
    # idempotent, and refuses a different selection under the same name
    qchem.write_jobs(geom, "al_t/iter_000", stem="al_t_it000", roundtrip_root=root)
    with pytest.raises(FileExistsError):
        qchem.write_jobs(ION / "eda/ion_clusters/geoms/oh-_w4_iso00.extxyz", "al_t/iter_000",
                         stem="al_t_it000", roundtrip_root=root)
    with pytest.raises(StagePending):
        qchem.wait_for_jobs(jobs)


needs_parser = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("numpy", "pyscf")),
    reason="scripts/parse_roundtrip.py needs numpy and pyscf",
)


@needs_parser
def test_label_stage_end_to_end(tmp_path, monkeypatch):
    root = _roundtrip_copy(tmp_path)
    stems = ["oh-_w2_iso00", "h3o+_w3_iso00"]

    class Label(LabelFrames):
        def select(self, ctx, path):
            ctx.input("candidates")
            path.write_text("".join(
                (ION / "eda/ion_clusters/geoms" / f"{s}.extxyz").read_text() for s in stems))

        def evaluate(self, ctx, files):
            return {"n_files": len(files)}

    orig = qchem.write_jobs
    monkeypatch.setattr(qchem, "write_jobs",
                        lambda *a, **k: orig(*a, **{**k, "roundtrip_root": root}))
    monkeypatch.setattr(qchem, "ROUNDTRIP_ROOT", root)
    orig_parse = qchem.parse_jobs
    monkeypatch.setattr(qchem, "parse_jobs",
                        lambda *a, **k: orig_parse(*a, **{**k, "roundtrip_root": root}))

    camp = Campaign(tmp_path / "camp", [ToyBuild(n=1), ToySample(), Label()], repo=REPO)
    assert camp.run(1) == "pending"
    rec = camp.read_record(0, "label")
    assert rec["metrics"]["qchem_eda_inputs"] == 2 and rec["metrics"]["qchem_eda_done"] == 0

    # "Perlmutter": drop the real outputs in under the generated names.
    for kind in ("eda", "force"):
        jd = root / kind / "al_camp/iter_000"
        for i, s in enumerate(stems):
            shutil.copy(ION / kind / "ion_clusters/outputs" / f"{s}.out",
                        jd / "outputs" / f"al_camp_it000_frame{i:04d}.out")

    assert camp.run(1) == "complete"
    rec = camp.read_record(0, "label")
    m = rec["metrics"]
    assert (m["n_selected"], m["n_labeled"], m["n_dropped"], m["pre_n_files"]) == (2, 2, 0, 1)
    assert "qchem_template.eda" in rec["inputs"] and "qchem_outputs.force" in rec["inputs"]
    assert (camp.stage_dir(0, "label") / "dataset/al_camp_it000_wb97mv_tzvpd.xyz").exists()
