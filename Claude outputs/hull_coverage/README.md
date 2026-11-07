# Convex-hull coverage of the complete-model invariants (2026-11-02)

Plan and findings: project doc `claude/hull_sampling_plan.md`.

Run in Claude's cloud container (CPU torch, rsfff src with a stub for bb; nothing here needs bb).
Inputs: research/rsfff train/halides/cache/cache_22c2040c53f3ffe9.pt (m26 five_salts_polarfilt),
its predictions.csv and model.pt (feat_scale), params_frozen_halides_45pairs.json, sample/nacl/round{1,2,3}/pool.xyz.
Paths in the scripts point at the container's staging folder; change them at the top of each script.

- `hullkit.py` — the library part: squared-norm invariants from cached blocks, PCA whitener, hull distance by NNLS
  (simplex as a stiff extra row), extreme points, MaxVol + grade, kNN distance, TwoNN, information imbalance,
  greedy information-imbalance feature selection.
- `prep.py` — per-ion table (I = feats / feat_scale), per-frame errors from predictions.csv.
- `redundancy.py`, `select_ii.py` — correlations, pivoted QR, intrinsic dimension, information imbalance by probe
  group, greedy selection.
- `hull_error.py K` — hull vertices of the training ions in K whitened PCs, held-out distances, error inside/outside.
- `pool_feats.py`, `pool_hull.py` — featurize the NaCl MD pool (every 5th frame), distances by coordination number.
- `greedy_demo.py` — greedy hull extension vs random vs committee sigma on held-out trajectories.
- `salt_overlap.py` — fraction of salt B's ions inside salt A's training hull.
- `figures.py` — hull_analysis.png.
