#!/bin/bash
# train.bonding behind ChemLab's worker protocol: python -m rsfff.train.bonding_worker run WORKDIR.
# rsfff and bb come on PYTHONPATH from the staged packages; torch / e3nn / ase from the interpreter:
# CHEMLAB_PYTHON (the project's context), else python3 on PATH (the site's environment or conda env).
py="${CHEMLAB_PYTHON:-python3}"
if ! command -v "$py" > /dev/null 2>&1; then
  echo "bonding_worker.sh: $py not found; using python3 from PATH" >&2
  py=python3
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CHEMLAB_THREADS:-4}}"
exec "$py" -m rsfff.train.bonding_worker "$@"
