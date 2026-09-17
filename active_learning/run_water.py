"""Drive the water-cluster active-learning loop.

    python active_learning/run_water.py --root runs/water_al \
        --checkpoint checkpoints/water_film_full/best.pt --sizes 2 20 --per-size 2

The loop stops ``pending`` after ``dynamics`` until the label stage below is implemented;
everything it has done is on disk and a later call picks up where it stopped. Run it again
after implementing ``QChemLabel`` and it resumes at ``label`` without rebuilding or
re-sampling anything.

Rerunning a stage on purpose: ``ActiveLearning.reset(iteration, stage)`` moves that stage and
everything after it to ``<root>/_reset/<timestamp>/``; e.g. to redo the dynamics with a longer
trajectory, ``loop.reset(0, "dynamics")``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyal import (ActiveLearning, Assess, Contract, Label, Pending, Train,  # noqa: E402
                    count_frames)

from active_learning import (DynamicsSample, MinimizeSample,  # noqa: E402
                             PackmolWaterClusters)
from active_learning.sampling import CARRIED  # noqa: E402

#: What the label stage will produce and the trainer will consume. Stated here because the
#: loop checks it against ``initial_data`` and against ``Label.produces`` when it is built,
#: so getting it wrong is a startup error rather than a surprise three hours in.
TRAINING = Contract(info=(*CARRIED, "energy"), arrays=("forces",))


# ---------------------------------------------------------------------------------------
# not implemented yet -- the three stages after sampling
# ---------------------------------------------------------------------------------------

class QChemLabel(Label):
    """TODO: write Q-Chem EDA + force jobs for ``ctx.input`` and collect them.

    The shape this wants (``Pending`` is exactly the mechanism for it): write the job inputs
    that are missing into ``ctx.scratch``, submit what has not been submitted, collect what
    has finished, and raise :class:`Pending` while anything is outstanding. The loop calls
    ``run`` again on the next ``loop.run(...)``, in the same directory.
    """

    requires = Contract(info=CARRIED)
    produces = TRAINING

    def run(self, ctx):
        raise Pending(f"label is not implemented yet: {count_frames(ctx.input)} candidate "
                      f"frames in {ctx.input} are waiting for Q-Chem EDA + force jobs")


class FilmTrain(Train):
    """TODO: refit the film model on ``ctx.training_data``, starting from ``ctx.model``."""

    requires = TRAINING
    output = "model"

    def run(self, ctx):
        raise NotImplementedError("train stage not implemented yet")


class HeldOutAssess(Assess):
    """TODO: score the new model on a held-out set and decide convergence."""

    def run(self, ctx):
        raise NotImplementedError("assess stage not implemented yet")


# ---------------------------------------------------------------------------------------


def build_loop(args) -> ActiveLearning:
    return ActiveLearning(
        args.root,
        build=PackmolWaterClusters(
            sizes=tuple(args.sizes), per_size=args.per_size, density=args.density,
            padding=args.padding, tolerance=args.tolerance, seed=args.seed,
            packmol=args.packmol,
        ),
        sample=[
            MinimizeSample(gtol=args.gtol, keep=args.keep, device=args.device),
            DynamicsSample(
                temperature_K=args.temperature, timestep_fs=args.timestep,
                equilibrate_steps=args.equilibrate, steps=args.steps, stride=args.stride,
                wall_k=args.wall_k, wall_margin=args.wall_margin, seed=args.seed,
                device=args.device,
            ),
        ],
        label=QChemLabel(),
        train=FilmTrain(),
        assess=HeldOutAssess(),
        initial_model=args.checkpoint,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/water_al", help="loop directory")
    p.add_argument("--checkpoint", default="checkpoints/water_film_full/best.pt")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--device", default="cpu")

    g = p.add_argument_group("build (packmol)")
    g.add_argument("--sizes", type=int, nargs=2, default=(2, 20), metavar=("LO", "HI"))
    g.add_argument("--per-size", type=int, default=2)
    g.add_argument("--density", type=float, default=1.0, help="g/cm^3, sets the cavity radius")
    g.add_argument("--padding", type=float, default=1.5, help="A added to the cavity radius")
    g.add_argument("--tolerance", type=float, default=2.0, help="packmol min distance, A")
    g.add_argument("--packmol", default="packmol", help="packmol executable")

    g = p.add_argument_group("optimize")
    g.add_argument("--gtol", type=float, default=1e-3, help="max |dE/dR|, Hartree/A")
    g.add_argument("--keep", choices=("converged", "all"), default="converged")

    g = p.add_argument_group("dynamics")
    g.add_argument("--temperature", type=float, default=300.0)
    g.add_argument("--timestep", type=float, default=0.5, help="fs")
    g.add_argument("--equilibrate", type=int, default=1000, help="steps, discarded")
    g.add_argument("--steps", type=int, default=2000, help="sampled steps per structure")
    g.add_argument("--stride", type=int, default=100, help="steps between kept frames")
    g.add_argument("--wall-k", type=float, default=0.5, help="Hartree/A^2; 0 disables")
    g.add_argument("--wall-margin", type=float, default=2.0, help="A")

    args = p.parse_args(argv)
    loop = build_loop(args)
    state = loop.run(max_iterations=args.iterations)
    print(f"\n[{state}]\n{loop.status()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
