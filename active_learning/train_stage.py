"""Train stage -- **not implemented yet**.

The shape it wants, following ``job_runner/active_learning/train_stage.py`` in the HBQ
project: concatenate ``ctx.training_data`` (the initial data plus every ``labeled.extxyz`` so
far) into one file, then fit with ``rsfff.train.train_film``, writing the result to
``ctx.output``. Two things there are worth copying when this is written:

* **a committee, not a model.** N members differing only in their seed, sharing one
  train/holdout split, each in its own ``member_NN/`` under ``ctx.output``. The spread across
  members is what the sampler selects on and what the assess stage measures, so it has to
  exist before either of those can be more than a placeholder.
* **members are resumable.** A member that finished writes a marker and is skipped on a
  re-run, so a driver killed by the wall clock only loses what was in flight.

Until then this raises, and a loop stops at ``label`` anyway.
"""

from __future__ import annotations

from easyal import Contract, Train

from label_stage import QCHEM_TRAINING

__all__ = ["FilmTrain"]


class FilmTrain(Train):
    """TODO: refit the film model on ``ctx.training_data``, starting from ``ctx.model``."""

    requires = QCHEM_TRAINING
    output = "model"

    def run(self, ctx):
        raise NotImplementedError(
            f"train stage not implemented: {len(ctx.training_data)} labeled file(s) are "
            f"waiting to be fitted, starting from {ctx.model}"
        )
