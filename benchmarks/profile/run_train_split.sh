#!/bin/bash
# Post-M5 training-step split. Run from the rsfff repo root on a GPU node (interactive or
# batch); results land in benchmarks/profile/results/*.md as usual, each .md with the
# per-phase / per-region training-step tables below the forward table.
#
#   salloc -A m3196 -C gpu -q interactive -N 1 --gpus-per-node=1 -c 32 -t 01:00:00
#   conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
#   bash benchmarks/profile/run_train_split.sh            # all four variants
#   bash benchmarks/profile/run_train_split.sh main       # just the torchff E+F run
#
# Variants (argument = which to run; default all):
#   main     torchff, E+F loss        -- the real step, backward attributed to regions (+ trace)
#   eonly    torchff, energy-only     -- ablation: no create_graph, no double backward
#   noind    torchff, no induction    -- ablation: bounds the coupled solve's share, all phases
#   torch    torch,   E+F loss        -- same split on the old path, for reference
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p benchmarks/profile/results
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

STRUCTS="${STRUCTS:-benchmarks/structures/w6_mp2_avtz.xyz benchmarks/structures/w21_mp2_avtz.xyz}"
FRAMES="${FRAMES:-128}"
COMMON="--device cuda --repeats ${REPEATS:-5} --frames $FRAMES --structures $STRUCTS"
PY="python benchmarks/profile/profile_film.py"

run() {
  case "$1" in
    main)  RSFFF_FF_BACKEND=torchff $PY $COMMON --trace ;;
    eonly) RSFFF_FF_BACKEND=torchff $PY $COMMON --loss energy ;;
    noind) RSFFF_FF_BACKEND=torchff $PY $COMMON --no-induction ;;
    torch) RSFFF_FF_BACKEND=torch   $PY $COMMON ;;
    *) echo "unknown variant: $1 (main|eonly|noind|torch)" >&2; exit 2 ;;
  esac
}

if [ $# -eq 0 ]; then set -- main eonly noind torch; fi
for v in "$@"; do echo; echo "### variant: $v"; run "$v"; done
