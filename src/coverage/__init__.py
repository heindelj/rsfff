"""Coverage of the environment-invariant space: convex hull of the training set, extrapolation flags and greedy
hull extension for active learning (project doc claude/hull_sampling_plan.md).

CLI: ``python -m rsfff.coverage {score,select} --help``."""
from .hull import (INSIDE_TOL, FeatureHull, Whitener, greedy_feature_selection, hull_distance,  # noqa: F401
                   hull_support, information_imbalance, knn_distance, maxvol, maxvol_grade, twonn_dimension)
from .select import Guard, featurizer_for_params, frame_features, greedy_extend, score_frames  # noqa: F401
