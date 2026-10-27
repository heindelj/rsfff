#!/bin/bash
# The GPU programs behind ChemLab's worker protocol: proposing (search.select.uncertainty), training the film
# committee (train.fit) and the ion / complete models (train.ionct) -- `python -m rsfff_al.worker gpu`, from the
# project dir RSFFF_AL (the context's `project`). RSFFF_SAMPLE_DEVICE / RSFFF_TRAIN_DEVICE override the devices.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${RSFFF_AL:-}" ]; then echo "gpu_worker.sh: RSFFF_AL (the project dir) is not set" >&2; exit 2; fi
env_script="${RSFFF_WORKER_ENV-$RSFFF_AL/python/scripts/env_perlmutter.sh}"
if [ -n "$env_script" ] && [ -f "$env_script" ]; then source "$env_script" >/dev/null 2>&1 || true; fi
export PYTHONPATH="$RSFFF_AL/python${PYTHONPATH:+:$PYTHONPATH}"
exec "${CHEMLAB_PYTHON:-python3}" -m rsfff_al.worker gpu "$@"
