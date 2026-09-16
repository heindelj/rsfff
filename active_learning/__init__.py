"""Active-learning campaigns for rsfff: build -> sample -> label -> train, with provenance.

See ``active_learning/README.md``.
"""

from .core import Campaign, Stage, StageContext, StagePending, ProvenanceError, fingerprint
from .stages import BuildStructures, SampleDynamics, LabelFrames, TrainModel
from . import qchem

__all__ = [
    "Campaign", "Stage", "StageContext", "StagePending", "ProvenanceError", "fingerprint",
    "BuildStructures", "SampleDynamics", "LabelFrames", "TrainModel", "qchem",
]
