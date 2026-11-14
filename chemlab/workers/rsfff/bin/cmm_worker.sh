#!/bin/bash
# train.cmm behind ChemLab's worker protocol: python -m rsfff.train.cmm_worker run WORKDIR.
# rsfff and bb come on PYTHONPATH from the staged packages; torch / e3nn / ase from the interpreter:
# CHEMLAB_PYTHON (the project's context), else python3 on PATH (the site's environment or conda env).
# RSFFF_CACHE_DIR (the project's context) keeps the precompute cache and the pair tables between jobs.
py="${CHEMLAB_PYTHON:-python3}"
if ! command -v "$py" > /dev/null 2>&1; then
  echo "cmm_worker.sh: $py not found; using python3 from PATH" >&2
  py=python3
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CHEMLAB_THREADS:-8}}"
exec "$py" -m rsfff.train.cmm_worker "$@"
