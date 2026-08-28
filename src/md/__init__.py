"""Molecular dynamics on the mediated model, for generating reactive geometries.

The mediator of ``docs/fff_v2.md`` §8 is fitted on 399 contested frames, and that corpus is
what limits any further work on its architecture. This package exists to grow it: run the
trained model under a weak spherical confinement with a seed ion, bias on how split the
membership is, and harvest the geometries where the model is being asked a question it cannot
answer from one fragmentation.

Three pieces, in dependency order:

* :mod:`~rsfff.md.assign` -- geometry to candidate decompositions. The one genuinely new thing
  here, because the training path never needed it: decompositions arrived from a file.
* :mod:`~rsfff.md.bias`, :mod:`~rsfff.md.confine` -- the two added energy terms.
* :mod:`~rsfff.md.similarity` -- distances between structures in the model's own
  descriptor, for deduplicating a harvest and finding what is off-manifold.
* :mod:`~rsfff.md.calculator` -- an ASE calculator that sums the three and takes one backward.

``scripts/run_reactive_md.py`` drives it.
"""

from .assign import (
    DEFAULT_BUMP,
    FragmentAssignment,
    base_assignment,
    enumerate_group,
    one_hop_candidates,
    rank_oh_fragment_assignments,
)
from .bias import HarmonicBias, ambiguity, logit, transfer_delta
from .calculator import MediatedCalculator, load_mediated_model, snapshot_info
from .confine import flat_bottom_sphere
from .similarity import FeatureMetric, FragmentDescriptors, Match, atom_features

__all__ = [
    "DEFAULT_BUMP",
    "FeatureMetric",
    "FragmentAssignment",
    "FragmentDescriptors",
    "Match",
    "base_assignment",
    "HarmonicBias",
    "MediatedCalculator",
    "ambiguity",
    "atom_features",
    "enumerate_group",
    "flat_bottom_sphere",
    "one_hop_candidates",
    "load_mediated_model",
    "logit",
    "rank_oh_fragment_assignments",
    "snapshot_info",
    "transfer_delta",
]
