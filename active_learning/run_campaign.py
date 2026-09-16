"""Driver for an active-learning campaign on larger water / ion clusters.

    conda activate rsfff
    python -m active_learning.run_campaign active_learning/campaigns/water_large --iterations 3
    python -m active_learning.run_campaign active_learning/campaigns/water_large --status
    python -m active_learning.run_campaign active_learning/campaigns/water_large --iterations 3 \\
        --rerun 1:label                     # force one stage to run again

Run it again whenever something external finishes (Q-Chem outputs synced down): completed
stages are skipped, the pending one resumes, and the loop carries on until the next wait.

``BuildClusters``, ``SampleClusterMD`` and ``select`` below are the parts to fill in.
``TrainFilm`` is a working sketch around ``rsfff.train.train_film``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from active_learning import (  # noqa: E402
    BuildStructures,
    Campaign,
    LabelFrames,
    SampleDynamics,
    StageContext,
    TrainModel,
)
from active_learning.core import REPO_ROOT  # noqa: E402

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# ----------------------------------------------------------------------------------------------
# stages to fill in
# ----------------------------------------------------------------------------------------------

class BuildClusters(BuildStructures):
    """Params: ``sizes`` (water counts), ``n_per_size``, ``seed``, ..."""

    def run(self, ctx: StageContext) -> None:
        out = ctx.output("structures")
        # e.g. sample isomers from data/benchmark sets, carve spheres out of a water box, or
        # grow last iteration's clusters (ctx.history("build")). Pre-relax with ctx.checkpoint.
        # Every frame: charge, multiplicity, structure_id (+ fragment columns if known).
        raise NotImplementedError(f"write starting structures to {out}")


class SampleClusterMD(SampleDynamics):
    """Params: ``temperature``, ``n_steps``, ``timestep_fs``, ``stride``, ``seed``, ..."""

    def run(self, ctx: StageContext) -> None:
        structures = ctx.input("structures")
        model = ctx.checkpoint
        traj_dir = ctx.output("trajectories")
        candidates = ctx.output("candidates")
        # e.g. chemlab/scripts/rsfff_jobs.load_backend(model) + RSFFFCalculator, or
        # rsfff.md.calculator for the mediated ion model; ASE Langevin inside a confining
        # sphere (rsfff.md.confine). Write traj_dir/<structure_id>.extxyz and pool eligible
        # frames (with structure_id, step, and a selection score) into `candidates`.
        # ctx.log_metrics(n_exploded=..., cg_fail_frac=..., score_p95=...)
        raise NotImplementedError(f"sample {structures} with {model} into {traj_dir}, {candidates}")


class LabelClusters(LabelFrames):
    """Params: ``n_select``, ``allow_unfinished``, ..."""

    def select(self, ctx: StageContext, path: Path) -> None:
        candidates = ctx.input("candidates")
        # e.g. rank by score, farthest-point in feature space (scripts/select_diverse.py),
        # skip near-duplicates of ctx.history("label") selections, assign fragments with
        # rsfff.md.assign.rank_oh_fragment_assignments, write extxyz with fragment columns.
        raise NotImplementedError(f"select from {candidates} into {path}")

    def evaluate(self, ctx: StageContext, files: list[Path]) -> dict:
        # Score ctx.checkpoint on the new labels, e.g. {"ind_mae": ..., "elst_mae": ...,
        # "e_int_mae": ..., "f_mae": ...} in kJ/mol. Logged as pre_<key>.
        return {}


class TrainFilm(TrainModel):
    """Warm-started ``train_film`` fit with the campaign's data appended to one data stream.

    Params
    ------
    base_config : YAML the fit starts from (repo-relative), e.g. configs/water_film_large.yaml
    stream      : ``data`` key that receives ``ctx.training_data()``; ``large_path`` by default,
                  since the big clusters are a separate stream there (see that config's notes)
    overrides   : ``{block: {field: value}}`` applied on top, e.g. ``{"train": {"epochs": 30}}``
    """

    def run(self, ctx: StageContext) -> None:
        import yaml

        from rsfff.train.config import load_config
        from rsfff.train.train_film import train

        base = REPO_ROOT / self.params["base_config"]
        stream = self.params.get("stream", "large_path")
        raw = yaml.safe_load(ctx.add_input("base_config", base).read_text())
        for key in ("path", "large_path", "monomer_path", "reference_energies"):
            if key == stream:
                continue
            value = raw["data"].get(key) or []
            for p in [value] if isinstance(value, str) else value:
                ctx.add_input(f"data.{key}:{p}", REPO_ROOT / p)
        raw["data"][stream] = [str(p) for p in ctx.training_data()]
        raw.setdefault("train", {})["init_from"] = str(ctx.checkpoint)
        raw.pop("stages", None)
        raw["run_name"] = "fit"
        raw["checkpoint_root"] = str(ctx.scratch)
        for block, values in self.params.get("overrides", {}).items():
            raw.setdefault(block, {}).update(values)
        cfg_path = ctx.output("config")
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))

        with _cwd(REPO_ROOT):  # the base config's data paths are repo-relative
            best_val = train(load_config(cfg_path))
        shutil.copyfile(ctx.scratch / "fit" / "best.pt", ctx.output("checkpoint"))
        if isinstance(best_val, (int, float)):
            ctx.log_metrics(best_val=float(best_val))
        # Also score a fixed benchmark here so iterations are comparable, e.g.
        # ctx.log_metrics(**benchmark(ctx.output("checkpoint"), "data/wb97mv_tzvpd_large"))


@contextlib.contextmanager
def _cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------------------------------------------------------------------------------------
# wiring
# ----------------------------------------------------------------------------------------------

def build_campaign(root: Path) -> Campaign:
    base_config = "configs/water_film_large.yaml"
    import yaml

    large = yaml.safe_load((REPO_ROOT / base_config).read_text())["data"]["large_path"]
    return Campaign(
        root,
        stages=[
            BuildClusters(sizes=list(range(8, 31, 2)), n_per_size=4, seed=0),
            SampleClusterMD(temperature=300.0, n_steps=20000, timestep_fs=0.5, stride=20, seed=0),
            LabelClusters(n_select=40),
            TrainFilm(base_config=base_config, stream="large_path",
                      overrides={"train": {"epochs": 30}}),
        ],
        initial_checkpoint=REPO_ROOT / "checkpoints/water_film_large/best.pt",
        base_data=[REPO_ROOT / p for p in large],
        description="Converge the film water model on larger clusters.",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--rerun", action="append", default=[], metavar="ITER:STAGE")
    ap.add_argument("--stop-after", metavar="ITER:STAGE")
    ap.add_argument("--on-stale", choices=("rerun", "keep", "error"), default="rerun")
    ap.add_argument("--status", action="store_true", help="print the campaign table and exit")
    args = ap.parse_args(argv)

    def pair(text):
        it, stage = text.split(":", 1)
        return int(it), stage

    campaign = build_campaign(args.campaign)
    if args.status:
        print(campaign.status())
        return 0
    result = campaign.run(
        args.iterations,
        rerun=[pair(r) for r in args.rerun],
        stop_after=pair(args.stop_after) if args.stop_after else None,
        on_stale=args.on_stale,
    )
    print(campaign.status())
    print(f"\ncampaign {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
