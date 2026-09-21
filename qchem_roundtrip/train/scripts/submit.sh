#!/bin/bash
# Queue a committee fit on a Perlmutter GPU node.
#
#   bash scripts/submit.sh --name film_committee                       # the production fit
#   bash scripts/submit.sh --name film_c8 -- --members 8 --seed 100    # extra args -> train_committee.py
#   bash scripts/submit.sh --name smoke -q debug -t 00:30:00 -- --quick --subset 64
#
# Output: $TRAIN_DIR/runs/<name>/ (committee.json, member_NN/...) and the Slurm log next to it.
# Account/QOS/time default from the environment: RSFFF_GPU_ACCOUNT (default
# $RSFFF_NERSC_ACCOUNT, i.e. m3196; if sbatch rejects it, try the m3196_g form),
# RSFFF_GPU_QOS (regular), RSFFF_GPU_TIME (12:00:00).
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="" QOS="${RSFFF_GPU_QOS:-regular}" TIME="${RSFFF_GPU_TIME:-12:00:00}"
ACCOUNT="${RSFFF_GPU_ACCOUNT:-${RSFFF_NERSC_ACCOUNT:-m3196}}"
while [ $# -gt 0 ]; do
    case "$1" in
        --name) NAME="$2"; shift 2;;
        -q|--qos) QOS="$2"; shift 2;;
        -t|--time) TIME="$2"; shift 2;;
        -A|--account) ACCOUNT="$2"; shift 2;;
        --) shift; break;;
        *) echo "unknown option $1 (train_committee.py arguments go after --)" >&2; exit 2;;
    esac
done
[ -n "$NAME" ] || { echo "--name is required" >&2; exit 2; }
OUT="$TRAIN_DIR/runs/$NAME"
mkdir -p "$OUT"
export TRAIN_DIR
sbatch --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
    --job-name "rsfff_committee_$NAME" --output "$OUT/slurm-%j.out" \
    --export ALL \
    "$TRAIN_DIR/scripts/committee.slurm" --out "$OUT" "$@"
echo "[submit] $OUT   (squeue -u \$USER -n rsfff_committee_$NAME; tail -f $OUT/member_00/train.log)"
