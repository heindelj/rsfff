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

def test_bundle_inputs_match_existing_ion_cluster_inputs(tmp_path):
    bundle = qchem.JobBundle(tmp_path / "anywhere" / "bundle")
    geom = ION / "eda/ion_clusters/geoms/oh-_w3_iso00.extxyz"
    assert bundle.add(geom, "al_t_it000") == 2
    for kind in ("eda", "force"):
        made = (bundle.kind_dir(kind) / "inputs/al_t_it000_frame0000.in").read_text()
        assert made == (ION / kind / "ion_clusters/inputs/oh-_w3_iso00.in").read_text()
    for name in ("config.json", "worker.py", "worker.slurm", "templates/eda.in", "templates/force.in"):
        assert (bundle.root / name).exists()
    # idempotent, and refuses a different selection under the same stem
    assert bundle.add(geom, "al_t_it000") == 0
    with pytest.raises(FileExistsError):
        bundle.add(ION / "eda/ion_clusters/geoms/oh-_w4_iso00.extxyz", "al_t_it000")
    with pytest.raises(StagePending):
        bundle.wait()


def test_bundle_refuses_template_swap(tmp_path):
    bundle = qchem.JobBundle(tmp_path / "b").create()
    other = tmp_path / "eda.in"
    other.write_text(bundle.templates["eda"].read_text().replace("def2-TZVPD", "def2-SVPD"))
    with pytest.raises(FileExistsError):
        qchem.JobBundle(tmp_path / "b", templates={"eda": other}).create()


def test_bad_fragment_charges_rejected(tmp_path):
    src = (ION / "eda/ion_clusters/geoms/oh-_w3_iso00.extxyz").read_text()
    bad = tmp_path / "bad.extxyz"
    bad.write_text(src.replace('fragment_charges="-1 0 0 0"', 'fragment_charges="0 0 0 0"'))
    with pytest.raises(ValueError, match="sum"):
        qchem.check_frames(bad, fragments=True)


def test_worker_runs_bundle_with_fake_qchem(tmp_path):
    bundle = qchem.JobBundle(tmp_path / "b")
    bundle.add(ION / "eda/ion_clusters/geoms/oh-_w2_iso00.extxyz", "s")
    fake = tmp_path / "fake_qchem"
    # args: -save -nt N input output ; write a finished-looking output for force, a crash for eda
    fake.write_text('#!/bin/sh\n'
                    'case "$(pwd)" in */eda) echo partial > "$5"; exit 1;; esac\n'
                    'echo "Thank you very much for using Q-Chem." > "$5"\n')
    fake.chmod(0o755)
    import subprocess
    subprocess.run([sys.executable, str(bundle.root / "worker.py"), "--qchem", str(fake),
                    "--threads", "1", "--idle-timeout", "0", "--poll-seconds", "0"], check=True,
                   capture_output=True)
    st = bundle.status()
    assert st["force"]["done"] == ["s_frame0000"]
    assert st["eda"]["failed"] == ["s_frame0000"]
    bundle.wait()  # nothing missing or running


needs_parser = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("numpy", "pyscf")),
    reason="scripts/parse_roundtrip.py needs numpy and pyscf",
)


@needs_parser
@pytest.mark.parametrize("external", [False, True])
def test_label_stage_end_to_end(tmp_path, external):
    stems = ["oh-_w2_iso00", "h3o+_w3_iso00"]

    class Label(LabelFrames):
        def select(self, ctx, path):
            ctx.input("candidates")
            path.write_text("".join(
                (ION / "eda/ion_clusters/geoms" / f"{s}.extxyz").read_text() for s in stems))

        def evaluate(self, ctx, files):
            return {"n_files": len(files)}

    params = {"jobs_root": str(tmp_path / "scratch/{campaign}/it{iteration}")} if external else {}
    camp = Campaign(tmp_path / "camp", [ToyBuild(n=1), ToySample(), Label(**params)], repo=REPO)
    assert camp.run(1) == "pending"
    rec = camp.read_record(0, "label")
    assert rec["metrics"]["qchem_eda_missing"] == 2
    root = (tmp_path / "scratch/camp/it0") if external else camp.stage_dir(0, "label") / "qchem"
    assert rec["extra_outputs"]["qchem_bundle"] == str(root.resolve())

    # "the cluster": drop the real outputs in under the generated names.
    for kind in ("eda", "force"):
        for i, s in enumerate(stems):
            shutil.copy(ION / kind / "ion_clusters/outputs" / f"{s}.out",
                        root / kind / "outputs" / f"al_camp_it000_frame{i:04d}.out")

    assert camp.run(1) == "complete"
    rec = camp.read_record(0, "label")
    m = rec["metrics"]
    assert (m["n_selected"], m["n_labeled"], m["n_dropped"], m["pre_n_files"]) == (2, 2, 0, 1)
    assert m["qchem_eda_done"] == 2
    assert "qchem_bundle.template.eda" in rec["inputs"]
    assert "qchem_bundle.outputs.force" in rec["inputs"]
    assert (camp.stage_dir(0, "label") / "dataset/al_camp_it000_wb97mv_tzvpd.xyz").exists()
    assert camp.run(1) == "complete"  # up to date
