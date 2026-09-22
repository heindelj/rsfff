#!/bin/bash
# Post-M5: where does the *training step* go? Training-shaped batches (128 frames) of w6 and
# w21; each variant writes its own results/*.md with the per-phase / per-region training-step
# tables under the usual forward table.
#
#   salloc -A m3196 -C gpu -q interactive -N 1 --gpus-per-node=1 -c 32 -t 01:00:00
#   conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
#   bash benchmarks/profile/run_train_split.sh            # the four original variants
#   bash benchmarks/profile/run_train_split.sh main       # just the torchff E+F run
#   bash benchmarks/profile/run_train_split.sh main cg2 ccg ccg2 cinf   # the compile / sync A/B
#
# Variants (argument = which to run; default: main eonly noind torch):
#   main     torchff, E+F loss        -- the real step, backward attributed to regions (+ trace)
#   eonly    torchff, energy-only     -- ablation: no create_graph, no double backward
#   noind    torchff, no induction    -- ablation: bounds the coupled solve's share, all phases
#   torch    torch,   E+F loss        -- same split on the old path, for reference
#   cg2/cg4  main with the CG convergence check every 2 / 4 iterations (fewer host syncs)
#   ccg      main with the PCG iteration under torch.compile (training-safe)
#   ccg2     ccg + check every 2
#   cinf     inference: forward / forward+forces only, network + CG compiled (no train step)
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
    cg2)   RSFFF_FF_BACKEND=torchff $PY $COMMON --cg-check-every 2 ;;
    cg4)   RSFFF_FF_BACKEND=torchff $PY $COMMON --cg-check-every 4 ;;
    ccg)   RSFFF_FF_BACKEND=torchff $PY $COMMON --compile cg ;;
    ccg2)  RSFFF_FF_BACKEND=torchff $PY $COMMON --compile cg --cg-check-every 2 ;;
    cinf)  RSFFF_FF_BACKEND=torchff $PY $COMMON --compile all --skip-train-step --no-train-split ;;
    *) echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}

if [ $# -eq 0 ]; then set -- main eonly noind torch; fi
for v in "$@"; do echo; echo "### variant: $v"; run "$v"; done
