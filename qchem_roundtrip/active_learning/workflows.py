"""The water-cluster active-learning workflow, and how it is driven.

    build      packmol packs (H2O)n into a spherical cavity        structures.extxyz
    optimize   relax every packing on the model's own surface      samples.extxyz
    dynamics   Langevin NVT from each minimum, inside a soft wall  samples.extxyz
    label      Q-Chem EDA + force from the round-trip job pool     labeled.extxyz
    train      refit the film model on everything labeled so far   model          (TODO)
    assess     measure the new model on its holdout                metrics.json   (TODO)

``run`` returns ``"pending"`` whenever it is waiting on the cluster -- which today means
whenever the Q-Chem jobs are not finished. Call it again (by hand, from cron, or from a job
that re-runs it) until it converges; completed stages are skipped, so re-running is free.

    python workflows.py water --root $SCRATCH/water_al \\
        --checkpoint /path/to/checkpoints/water_film_full/best.pt \\
        --submit "bash scripts/submit_workers.sh --target 16"

On an interactive node, where the workers are in the same allocation, ``--wait`` turns that
into one command that runs to the end instead of stopping at ``pending``; ``scripts/``
next to this file has a preflight check and a smoke test built on it.

    python workflows.py water --root $SCRATCH/water_al --status

or from python, overriding any stage's parameters:

    from workflows import water_loop
    loop = water_loop("runs/water", initial_model="checkpoints/water_film_full/best.pt",
                      dynamics=dict(steps=4000, keep=...), label=dict(submit=[...]))
    print(loop.run(max_iterations=10))     # "converged" | "pending" | "max_iterations"

Where the hooks run
-------------------
``label`` shells out in the **round-trip directory** (``qchem_roundtrip/``, or
``$RSFFF_QCHEM_ROOT``). On Perlmutter itself that is all the submission there is::

    submit=["bash scripts/submit_workers.sh --target 16"]

Driving it from a laptop instead, it is an rsync up, an ssh sbatch, and an rsync back::

    submit=["bash scripts/sync_inputs_up.sh",
            "ssh perlmutter 'cd $REMOTE_DIR && bash scripts/submit_workers.sh --target 16'"],
    sync=["bash scripts/sync_outputs_down.sh"],

With no hooks at all, do those steps by hand and call ``run`` again.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __package__ in (None, ""):        # run as a script: make the sibling modules importable
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from easyal import ActiveLearning  # noqa: E402

from assess_stage import ValidationAssess  # noqa: E402
from build import PackmolWaterClusters  # noqa: E402
from label_stage import QChemLabel  # noqa: E402
from sampling import DynamicsSample, MinimizeSample, SelectByCommittee  # noqa: E402
from train_stage import CommitteeTrain  # noqa: E402

__all__ = ["SIZE_SCHEDULE", "WORKFLOWS", "water_loop", "loop_from_spec", "last_model",
           "schedules"]

#: The walk. One entry per iteration: which cluster sizes to sample, and how many structures
#: of **each** size to label. Small clusters first and most of them -- they are where the
#: reference is cheap and where the model's short-range behaviour is set -- then fewer of each
#: as the clusters grow and each label costs more. The totals are 1000, 800, 600, 400 and 200
#: labeled structures.
SIZE_SCHEDULE = (
    {"sizes": (2, 3, 4, 5), "labels_per_size": 250},            # 1000
    {"sizes": (6, 7, 8, 9, 10), "labels_per_size": 160},        #  800
    {"sizes": (11, 12, 13, 14, 15), "labels_per_size": 120},    #  600
    {"sizes": (16, 17, 18, 19, 20), "labels_per_size": 80},     #  400
    {"sizes": (21, 22, 23, 24, 25), "labels_per_size": 40},     #  200
)

#: Candidates sampled per label. Selecting the most uncertain 250 of 1000 is a real choice;
#: selecting 250 of 250 is not a choice at all, and the committee would be along for the ride.
POOL_MULTIPLIER = 4


def schedules(pool_multiplier: int = POOL_MULTIPLIER, frames_per_trajectory: int = 20,
              schedule=SIZE_SCHEDULE) -> tuple[list[dict], list[dict]]:
    """``(what to build, what to keep)`` per iteration, from the size schedule.

    The build stage makes *structures* and the dynamics stage turns each into
    ``frames_per_trajectory`` frames, so the number of structures to pack is the candidate
    count divided by that -- keep the two in step or the pool comes out the wrong size.
    """
    build, select = [], []
    for entry in schedule:
        labels = int(entry["labels_per_size"])
        candidates = labels * int(pool_multiplier)
        build.append({"sizes": list(entry["sizes"]),
                      "per_size": max(1, -(-candidates // max(frames_per_trajectory, 1)))})
        select.append({"sizes": list(entry["sizes"]), "per_size": labels})
    return build, select


def _merged(defaults: dict, overrides: dict | None) -> dict:
    return {**defaults, **(overrides or {})}


def water_loop(
    root, *, initial_model=None, initial_data: Sequence[str] = (), train_config=None,
    pool_multiplier: int = POOL_MULTIPLIER, size_schedule=SIZE_SCHEDULE,
    build: dict | None = None, optimize: dict | None = None, dynamics: dict | None = None,
    select: dict | None = None, label: dict | None = None, train: dict | None = None,
    assess: dict | None = None,
) -> ActiveLearning:
    """Neutral water clusters: pack, minimize, heat, choose, label with Q-Chem, refit.

    The loop walks up in cluster size on a fixed schedule -- one entry per iteration, so it
    runs ``len(size_schedule)`` iterations and stops because the walk is finished, not because
    a threshold was met. Every stage's parameters can still be overridden by passing that
    stage's dict, e.g. ``water_loop(root, dynamics=dict(steps=4000), train=dict(n_members=8))``.
    """
    dynamics = dict(dynamics or {})
    steps = int(dynamics.get("steps", 2000))
    stride = int(dynamics.get("stride", 100))
    build_schedule, select_schedule = schedules(
        pool_multiplier, max(steps // max(stride, 1), 1), size_schedule)
    return ActiveLearning(
        root,
        build=PackmolWaterClusters(**_merged(dict(schedule=build_schedule), build)),
        # Three samplers in sequence: where are this model's minima, what does it do at
        # temperature around them, and which of those frames is worth a reference calculation.
        # easyal runs a list of Sample stages in order, each reading the previous one's
        # output, each with its own directory, contract and record.
        sample=[
            MinimizeSample(**_merged(dict(gtol=1e-3, keep="converged"), optimize)),
            DynamicsSample(**_merged(
                dict(temperature_K=300.0, timestep_fs=0.5, equilibrate_steps=1000,
                     steps=steps, stride=stride), dynamics)),
            SelectByCommittee(**_merged(
                dict(schedule=select_schedule, select_on="both"), select)),
        ],
        # Every selected frame is two Q-Chem jobs, so this is the expensive stage and the one
        # that makes the loop wait.
        label=QChemLabel(**_merged(dict(max_failed_fraction=0.05), label)),
        # config= is left out when nothing asked for one, so the stage records the training
        # YAML it actually resolved rather than a null
        train=CommitteeTrain(**_merged(
            dict(n_members=4, warm_start=True,
                 **({"config": str(train_config)} if train_config else {})), train)),
        assess=ValidationAssess(**_merged(dict(patience=2), assess)),
        initial_model=initial_model,
        initial_data=list(initial_data),
    )


def last_model(root):
    """The committee of the last completed iteration of a loop."""
    loop_root = Path(root)
    for iteration in sorted(
        (int(p.name[5:]) for p in loop_root.glob("iter_*") if p.name[5:].isdigit()),
        reverse=True,
    ):
        candidate = loop_root / f"iter_{iteration:03d}" / "train" / "committee"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no finished model under {loop_root}")


WORKFLOWS = {"water": water_loop}


def loop_from_spec(spec: dict, root) -> ActiveLearning:
    """Build a loop from a task spec -- a JSON file a driver can be pointed at::

        {
          "workflow": "water",
          "max_iterations": 10,
          "initial_model": "/path/checkpoints/water_film_full/best.pt",
          "stages": {
            "build":  {"sizes": [2, 30], "per_size": 2},
            "label":  {"submit": ["bash scripts/submit_workers.sh --target 16"]}
          }
        }

    Everything except ``workflow`` is optional; ``stages`` overrides each stage's parameters
    one by one, so the defaults above stay in force for the rest. ``initial_model`` may also be
    ``{"from_loop": "<another loop root>"}``, meaning that loop's last trained model.
    """
    spec = dict(spec)
    name = spec.get("workflow", "water")
    if name not in WORKFLOWS:
        raise ValueError(f"unknown workflow {name!r}; use one of {sorted(WORKFLOWS)}")
    stages = dict(spec.get("stages") or {})
    known = {"build", "optimize", "dynamics", "select", "label", "train", "assess"}
    unknown = set(stages) - known
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)} in the task spec; "
                         f"use {sorted(known)}")
    kwargs = {stage: dict(params) for stage, params in stages.items()}
    if spec.get("initial_data") is not None:
        kwargs["initial_data"] = [str(p) for p in spec["initial_data"]]
    model = spec.get("initial_model")
    if isinstance(model, dict):
        model = last_model(model["from_loop"])
    if model is not None:
        kwargs["initial_model"] = str(model)
    return WORKFLOWS[name](root, **kwargs)


# --- CLI ----------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workflow", choices=sorted(WORKFLOWS), nargs="?", default="water")
    p.add_argument("--root", type=Path, required=True, help="loop directory")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="model iteration 0 samples with (easyal's initial_model)")
    p.add_argument("--spec", type=Path, default=None,
                   help="task spec JSON; --root still says where the loop lives")
    p.add_argument("--iterations", type=int, default=len(SIZE_SCHEDULE),
                   help=f"the size schedule has {len(SIZE_SCHEDULE)} entries")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--device", default="cpu")
    p.add_argument("--status", action="store_true", help="print the status and exit")
    p.add_argument("--outcome-file", type=Path, default=None,
                   help="write the run's outcome (converged | pending | max_iterations) here, "
                        "for a batch driver deciding whether to requeue itself")

    g = p.add_argument_group("build (packmol)")
    g.add_argument("--sizes", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="override the schedule with one fixed size range (for a smoke test)")
    g.add_argument("--per-size", type=int, default=None)
    g.add_argument("--pool-multiplier", type=int, default=None,
                   help=f"candidates sampled per label (default {POOL_MULTIPLIER})")
    g.add_argument("--packmol", default=None, help="packmol executable")

    g = p.add_argument_group("train (committee)")
    g.add_argument("--train-config", type=Path, default=None,
                   help="training YAML, e.g. configs/water_film.yaml")
    g.add_argument("--members", type=int, default=None, help="committee size (default 4)")
    g.add_argument("--train-parallel", type=int, default=None,
                   help="members fitted at once (default: one per GPU, else 1)")
    g.add_argument("--train-threads", type=int, default=None,
                   help="CPU threads per member process (default: the node's cores split "
                        "evenly between the members running at once)")
    g.add_argument("--no-warm-start", action="store_true",
                   help="fit every member from scratch instead of continuing the last one")
    g.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    g.add_argument("--no-stages", action="store_true",
                   help="clear the config's staged fitting and do one fit. --epochs only "
                        "touches train.epochs, which a staged config overrides per stage, so "
                        "a genuinely short fit wants both")
    g.add_argument("--initial-data", action="append", default=None, metavar="EXTXYZ",
                   help="labeled data the first committee is fitted on alongside the loop's "
                        "own; repeat for several")

    g = p.add_argument_group("sampling")
    g.add_argument("--gtol", type=float, default=None, help="max |dE/dR|, Hartree/A")
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--steps", type=int, default=None, help="sampled MD steps per structure")
    g.add_argument("--stride", type=int, default=None, help="steps between kept frames")
    g.add_argument("--equilibrate", type=int, default=None)

    g = p.add_argument_group("label (Q-Chem)")
    g.add_argument("--submit", action="append", default=None, metavar="CMD",
                   help="shell command run once in the round-trip directory after the inputs "
                        "are written; repeat for several")
    g.add_argument("--sync", action="append", default=None, metavar="CMD",
                   help="shell command run there before every completion check")
    g.add_argument("--roundtrip-root", type=Path, default=None,
                   help="the Q-Chem job pool (default $RSFFF_QCHEM_ROOT or <repo>/qchem_roundtrip)")
    g.add_argument("--max-failed-fraction", type=float, default=None)
    g.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                   help="keep polling for Q-Chem results for this long before reporting "
                        "pending; use it when the workers run in the same allocation")
    g.add_argument("--poll", type=float, default=None, metavar="SECONDS",
                   help="seconds between checks while waiting (default 30)")

    args = p.parse_args(argv)

    if args.spec is not None:
        import json
        spec = json.loads(args.spec.read_text())
        loop = loop_from_spec(spec, args.root)
        iterations = int(spec.get("max_iterations", args.iterations))
    else:
        build = _only(per_size=args.per_size, packmol=args.packmol, seed=args.seed)
        select = _only(device=args.device)
        if args.sizes:   # a fixed range instead of the walk, for a smoke test
            build["schedule"] = None
            build["sizes"] = tuple(args.sizes)
            select["schedule"] = None
            select["per_size"] = args.per_size or 2
        optimize = _only(gtol=args.gtol, device=args.device)
        dynamics = _only(temperature_K=args.temperature, steps=args.steps, stride=args.stride,
                         equilibrate_steps=args.equilibrate, seed=args.seed,
                         device=args.device)
        overrides = {}
        if args.epochs:
            overrides["train.epochs"] = args.epochs
        if args.no_stages:
            overrides["stages"] = []
        train = _only(config=str(args.train_config) if args.train_config else None,
                      n_members=args.members, overrides=overrides or None,
                      parallel=args.train_parallel, threads_per_member=args.train_threads,
                      warm_start=False if args.no_warm_start else None)
        label = _only(submit=args.submit, sync=args.sync,
                      max_failed_fraction=args.max_failed_fraction,
                      wait_seconds=args.wait, poll_seconds=args.poll,
                      roundtrip_root=str(args.roundtrip_root) if args.roundtrip_root else None)
        loop = water_loop(args.root, initial_model=args.checkpoint,
                          initial_data=args.initial_data or (), build=build,
                          optimize=optimize, dynamics=dynamics, select=select, label=label,
                          train=train, pool_multiplier=args.pool_multiplier or POOL_MULTIPLIER)
        iterations = args.iterations

    if args.status:
        print(loop.status())
        return 0
    outcome = loop.run(max_iterations=iterations)
    if args.outcome_file is not None:
        args.outcome_file.parent.mkdir(parents=True, exist_ok=True)
        args.outcome_file.write_text(f"{outcome}\n")
    print(f"\n[{outcome}]\n{loop.status()}")
    if outcome == "pending":
        print("\nwaiting on the cluster; run the same command again when the jobs are done")
    return 0


def _only(**kwargs) -> dict:
    """The keyword arguments that were actually given, so unset flags keep a stage's default."""
    return {k: v for k, v in kwargs.items() if v is not None}


if __name__ == "__main__":
    raise SystemExit(main())
