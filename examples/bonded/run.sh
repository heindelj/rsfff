#!/usr/bin/env bash
# The whole example, from the rsfff checkout: data, fit, figures. See README.md.
set -euo pipefail
WORKERS=${WORKERS:-4}

# 1. Data: optimize, sample (40 normal-mode samples + rigid scans per species) and label at wB97X/def2-SVP.
python scripts/bonded_data.py all --species h2o h3o+ oh- h2o2 --out examples/bonded/data \
    --xc wb97x --basis def2-svp --grid 75 302 --wigner 40 --points 7 --raman --analytic-hessian --workers "$WORKERS"

# 2. Fit: one model for the four species; the scan frames are held out.
python -m rsfff.train.bonded examples/bonded/config.toml

# 3. Figures: held-out scan curves and a summary table.
python examples/bonded/analysis.py examples/bonded/run
