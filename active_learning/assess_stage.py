"""Assess stage -- **not implemented yet**.

The HBQ project's ``ValidationAssess`` is the model to follow: judge the committee on the
structures it was *held out from* (the train stage records its exact split, so the assessment
never recomputes one), measure the 95% prediction interval built from the committee spread
rather than a bare sigma, pass an iteration when at most ``max_fraction`` of structures are
outside the threshold, and stop only after ``patience`` such iterations -- one good iteration
is luck of the sampling. Because the holdout carries reference labels it can also report
coverage, which is the check on whether the committee's own uncertainty means anything.

None of that is available until the train stage produces a committee, so this raises.
"""

from __future__ import annotations

from easyal import Assess

__all__ = ["ValidationAssess"]


class ValidationAssess(Assess):
    """TODO: score the new committee on its holdout and decide convergence."""

    def run(self, ctx):
        raise NotImplementedError("assess stage not implemented yet")

    def converged(self, history) -> bool:
        return False
