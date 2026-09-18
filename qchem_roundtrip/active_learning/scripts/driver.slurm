#!/bin/bash
# One driver of a production active-learning run. Submitted by launch_production.sh, which
# supplies account, QOS, wall clock, job name (--dependency=singleton), and AL_RUN_ROOT; don't
# sbatch this by hand -- use launch_production.sh --resume <root>.
#
# It runs the loop until it finishes or the label stage is waiting on Q-Chem, then:
#   pending                    requeue in $RESUBMIT_DELAY minutes
#   converged/max_iterations   write driver_state/DONE and stop
#   the loop raised            write driver_state/FAILED and stop (fix, then --resume)
# and 10 minutes before the wall clock (USR1) it requeues immediately and stops the loop, so a
# stage that outlasts one driver is picked up by the next.
#SBATCH --constraint cpu
#SBATCH --nodes 1

set -uo pipefail

ROOT="${AL_RUN_ROOT:?launch this through launch_production.sh}"
# shellcheck disable=SC1091
source "$ROOT/driver.env"
STATE="$ROOT/driver_state"
mkdir -p "$STATE"
LAUNCH="$POOL/active_learning/scripts/launch_production.sh"

say() { echo "[driver $(date +%H:%M:%S)] $*"; }

N=$(( $(cat "$STATE/resubmits" 2>/dev/null || echo 0) + 1 ))
echo "$N" > "$STATE/resubmits"
echo "${SLURM_JOB_ID:-} $(date -Is) start" >> "$STATE/history"
say "driver #$N (job ${SLURM_JOB_ID:-?}) for $ROOT"

requeue() {   # $1: sbatch --begin
    if [ "$N" -ge "$MAX_RESUBMITS" ]; then
        say "reached $MAX_RESUBMITS drivers; not requeueing (raise MAX_RESUBMITS in driver.env)"
        echo "max_resubmits $(date -Is)" > "$STATE/FAILED"
        return
    fi
    bash "$LAUNCH" --resume "$ROOT" --begin "$1" \
        || say "requeue failed -- resume by hand: bash $LAUNCH --resume $ROOT"
}

LOOP_PID=""
on_walltime() {
    say "wall clock nearly up; requeueing now and stopping the loop"
    echo "${SLURM_JOB_ID:-} $(date -Is) walltime" >> "$STATE/history"
    requeue now
    [ -z "$LOOP_PID" ] || { kill -TERM "$LOOP_PID" 2>/dev/null; wait "$LOOP_PID" 2>/dev/null; }
    exit 0
}
trap on_walltime USR1

module load python
# RSFFF_REPO as the first launch saw it; nersc_env.sh keeps it and sets pool + conda
if [ -n "${RSFFF_REPO:-}" ]; then export RSFFF_REPO; fi
# shellcheck disable=SC1091
source "$POOL/active_learning/scripts/nersc_env.sh"
export RSFFF_QCHEM_ROOT="$POOL"

# The hooks read these from the environment rather than carrying them on the command line,
# so the label stage's recorded parameters stay the same however driver.env is tuned.
export AL_WORKER_TARGET AL_WORKER_TIME AL_WORKER_QOS
HOOK="bash active_learning/scripts/al_workers.sh"

OUTCOME="$STATE/outcome"
rm -f "$OUTCOME"
python3 "$POOL/active_learning/workflows.py" water \
    --root "$ROOT" \
    --checkpoint "$CHECKPOINT" \
    --roundtrip-root "$POOL" \
    --submit "$HOOK" --sync "$HOOK" \
    --wait "$WAIT_SECONDS" --poll 60 \
    --train-parallel "$TRAIN_PARALLEL" --train-threads "$TRAIN_THREADS" \
    --outcome-file "$OUTCOME" \
    ${AL_EXTRA_ARGS[@]+"${AL_EXTRA_ARGS[@]}"} &
LOOP_PID=$!
wait "$LOOP_PID"
STATUS=$?
LOOP_PID=""

OUT="$(cat "$OUTCOME" 2>/dev/null || true)"
echo "${SLURM_JOB_ID:-} $(date -Is) exit=$STATUS outcome=${OUT:-none}" >> "$STATE/history"
case "$STATUS:$OUT" in
    0:pending)
        say "waiting on Q-Chem; next driver in $RESUBMIT_DELAY minutes"
        requeue "now+${RESUBMIT_DELAY}minutes" ;;
    0:converged|0:max_iterations)
        say "loop finished: $OUT"
        echo "$OUT $(date -Is)" > "$STATE/DONE" ;;
    *)
        say "loop exited $STATUS (outcome '${OUT:-none}'); not requeueing"
        say "fix it, then: bash $LAUNCH --resume $ROOT"
        echo "exit=$STATUS $(date -Is)" > "$STATE/FAILED"
        exit 1 ;;
esac
