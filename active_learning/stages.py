"""The four stage templates. Subclass one, implement ``run``, and hand instances to a Campaign.

Each template fixes the stage's ``name``, its standard output locations, and what ``validate``
checks after ``run`` returns. The frame format that flows between stages is extxyz with the
header keys the Q-Chem generator needs::

    Properties=species:S:1:pos:R:3:fragment_idx:I:1 charge=1 multiplicity=1
        fragment_charges="0 0 1" fragment_multiplicities="1 1 1" structure_id=h3o+_w6_iso03

``fragment_idx`` / ``fragment_charges`` / ``fragment_multiplicities`` are required wherever an
EDA job may be written from the frame (the label stage's ``selected.extxyz``); ``charge`` and
``multiplicity`` are required everywhere. ``ase.io.write(path, atoms_list, format="extxyz")``
produces this when ``atoms.info`` and ``atoms.arrays["fragment_idx"]`` are set.
"""

from __future__ import annotations

from .core import Stage, StageContext
from . import qchem

__all__ = ["BuildStructures", "SampleDynamics", "LabelFrames", "TrainModel", "require_header"]


def require_header(path, keys=("charge", "multiplicity")) -> int:
    """Raise if any frame of ``path`` lacks one of ``keys``. Returns the frame count."""
    frames = qchem.read_frames(path)
    for f in frames:
        missing = [k for k in keys if k not in f.info]
        if missing:
            raise ValueError(f"{path} frame {f.index}: missing header key(s) {missing}")
    return len(frames)


class BuildStructures(Stage):
    """Stage 1 -- make the starting structures for this iteration's dynamics.

    Write ``ctx.output("structures")``: one frame per starting structure, each with ``charge``,
    ``multiplicity`` and a unique ``structure_id``. Fragment columns are optional here (the
    sampler may reassign them) but cheap to carry.

    Useful context: ``ctx.checkpoint`` (e.g. to pre-optimize with the current model),
    ``ctx.history("build")`` / ``ctx.history("label")`` (grow from last iteration's clusters,
    or seed from the frames that were most wrong). Log e.g. ``n_structures``, size range.
    """

    name = "build"
    outputs = {"structures": "structures.extxyz"}

    def validate(self, ctx: StageContext) -> None:
        super().validate(ctx)
        path = ctx.output("structures")
        frames = qchem.read_frames(path)
        ids = [f.info.get("structure_id") for f in frames]
        if None in ids:
            raise ValueError(f"{path}: every frame needs a structure_id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: structure_id values must be unique")
        require_header(path)
        sizes = [len(f.symbols) for f in frames]
        ctx.log_metrics(n_structures=len(frames), n_atoms_min=min(sizes), n_atoms_max=max(sizes))


class SampleDynamics(Stage):
    """Stage 2 -- run the current model from each structure and collect candidate frames.

    Inputs: ``ctx.input("structures")``, ``ctx.checkpoint``.

    Write one trajectory per structure to ``ctx.output("trajectories") / f"{structure_id}.extxyz"``
    and pool every frame that is eligible for labeling into ``ctx.output("candidates")``. Each
    candidate frame carries ``charge``, ``multiplicity``, ``structure_id``, ``step``, and whatever
    the selector will rank on (a committee spread, a feature-space novelty score, the
    applicability/routing weight, an energy-drift flag, ...).

    Log what says how the model behaved: trajectories that blew up, max temperature, fraction
    of steps with a failed CG solve, score percentiles.
    """

    name = "sample"
    outputs = {"trajectories": "trajectories/", "candidates": "candidates.extxyz"}

    def validate(self, ctx: StageContext) -> None:
        super().validate(ctx)
        n = require_header(ctx.output("candidates"), ("charge", "multiplicity", "structure_id"))
        n_traj = sum(1 for p in ctx.output("trajectories").iterdir()
                     if p.suffix in (".xyz", ".extxyz"))
        ctx.log_metrics(n_candidates=n, n_trajectories=n_traj)


class LabelFrames(Stage):
    """Stage 3 -- select frames, write and run Q-Chem, and parse the results into training data.

    ``run`` is implemented here and is resumable; fill in ``select`` (and ideally ``evaluate``)::

        select(ctx, path)      write the frames to label to ``path`` (extxyz with fragment
                               columns). Called once; the file is kept across resumes.
        write_jobs             EDA + force inputs into qchem_roundtrip/<kind>/al_<campaign>/iter_NNN
        wait                   raises StagePending until every output is back
        parse                  scripts/parse_roundtrip.py -> dataset/
        evaluate(ctx, files)   the current model (ctx.checkpoint) scored on the new labels

    ``evaluate`` is the number active learning is about -- the model's error on data it has
    never seen, picked because it looked uncertain -- and it has to be measured here, before
    ``train`` fits it away. Override ``run`` entirely if the flow does not fit.

    Parameters understood by the default ``run`` (pass them to the constructor):
    ``allow_unfinished`` -- accept crashed Q-Chem outputs (the parser drops them).
    """

    name = "label"
    outputs = {"selected": "selected.extxyz", "dataset": "dataset/"}

    #: Job kinds written for every selected frame.
    kinds = qchem.JOB_KINDS

    def select(self, ctx: StageContext, path) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}.select: write the frames to label to {path}. Typical inputs: "
            f"ctx.input('candidates'), ctx.history('label') to avoid re-selecting."
        )

    def evaluate(self, ctx: StageContext, files) -> dict:
        """Metrics of ``ctx.checkpoint`` on the freshly labeled ``files``. Optional."""
        return {}

    def job_name(self, ctx: StageContext) -> str:
        """Nested job directory under ``qchem_roundtrip/<kind>/``."""
        return f"al_{ctx.root.name}/iter_{ctx.iteration:03d}"

    def stem(self, ctx: StageContext) -> str:
        """File stem of the staged geometry, the inputs, and the parsed dataset."""
        return f"al_{ctx.root.name}_it{ctx.iteration:03d}"

    def run(self, ctx: StageContext) -> None:
        selected = ctx.output("selected")
        if not selected.exists():
            tmp = selected.with_name(f".{selected.name}.tmp")
            self.select(ctx, tmp)
            qchem.check_frames(tmp, fragments="eda" in self.kinds)
            tmp.replace(selected)
            ctx.note(f"selected {len(qchem.read_frames(selected))} frame(s)")
        stem = self.stem(ctx)
        jobs = qchem.write_jobs(selected, self.job_name(ctx), stem=stem, kinds=self.kinds, ctx=ctx)
        qchem.wait_for_jobs(jobs, allow_unfinished=self.params.get("allow_unfinished", False),
                            ctx=ctx)
        files = qchem.parse_jobs(jobs, ctx.output("dataset"), stem=stem,
                                 log_path=ctx.scratch / "parse_roundtrip.log", ctx=ctx)
        metrics = self.evaluate(ctx, files)
        if metrics:
            ctx.log_metrics(**{f"pre_{k}": v for k, v in metrics.items()})

    def validate(self, ctx: StageContext) -> None:
        super().validate(ctx)
        n_sel = qchem.check_frames(ctx.output("selected"), fragments="eda" in self.kinds)
        files = sorted(p for p in ctx.output("dataset").iterdir() if p.suffix in (".xyz", ".extxyz"))
        if not files:
            raise FileNotFoundError(f"{ctx.output('dataset')} holds no .xyz/.extxyz files")
        n_lab = 0
        for p in files:
            s = qchem.dataset_summary(p)
            n_lab += s["n_frames"]
            if "eda" in self.kinds and not s["has_eda"]:
                raise ValueError(f"{p}: expected EDA labels but found none")
        ctx.log_metrics(n_selected=n_sel, n_labeled=n_lab, n_dropped=n_sel - n_lab)


class TrainModel(Stage):
    """Stage 4 -- fit on the cumulative training set.

    Inputs: ``ctx.training_data()`` (base data + every labeled dataset so far),
    ``ctx.checkpoint`` (warm start).

    Write ``ctx.output("checkpoint")`` and the exact config that produced it to
    ``ctx.output("config")``, and log validation metrics -- ideally both on the random holdout
    and on a fixed benchmark set that does not change between iterations, so the numbers are
    comparable across the campaign.
    """

    name = "train"
    outputs = {"checkpoint": "best.pt", "config": "config.yaml"}
