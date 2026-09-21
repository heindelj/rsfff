#!/bin/bash
# The whole job, small, on an interactive GPU node -- minutes instead of a queue wait.
#
#   salloc -A m3196 -C gpu -q interactive -N 1 --gpus-per-node 4 -c 128 -t 00:30:00
#   source <bundle>/train/scripts/env.sh
#   bash $TRAIN_DIR/scripts/smoke.sh [--out DIR] [--subset 64] [--members 4]
#
# 1. data pins        scripts/stage_data.sh --check
# 2. GPU check        scripts/gpu_check.py (env, neighbor list, CPU/GPU parity, descent, timing)
# 3. tiny committee   train_committee.py --quick --subset N: every stage for one epoch, one
#                     member per GPU, through exactly the code path the batch job uses
# 4. load it back     scripts/check_committee.py on the CPU, as the active-learning loop will
#
# Output under $SCRATCH/rsfff_train_smoke (removed and rebuilt each time), never under runs/.
set -euo pipefail
TRAIN_DIR="${TRAIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${SCRATCH:-/tmp}/rsfff_train_smoke" SUBSET=64 MEMBERS=4
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2;; --subset) SUBSET="$2"; shift 2;;
        --members) MEMBERS="$2"; shift 2;; *) echo "unknown option $1" >&2; exit 2;;
    esac
done
step() { echo; echo "=== $* ==="; }

step "1/4 data pins";       bash "$TRAIN_DIR/scripts/stage_data.sh" --check
step "2/4 GPU check";       python3 "$TRAIN_DIR/scripts/gpu_check.py"
step "3/4 tiny committee";  rm -rf "$OUT"
t0=$(date +%s)
python3 "$TRAIN_DIR/train_committee.py" --out "$OUT" --quick --subset "$SUBSET" --members "$MEMBERS"
echo "member log (member_00):"; grep -E "^(===|epoch|done|warm)" "$OUT/member_00/train.log" | cut -c1-160
echo "committee trained in $(( $(date +%s) - t0 )) s"
step "4/4 load on CPU";     python3 "$TRAIN_DIR/scripts/check_committee.py" "$OUT" --frames 8
echo; echo "smoke test passed: $OUT"
