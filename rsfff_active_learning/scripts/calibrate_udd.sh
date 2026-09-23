#!/bin/bash
# AL1 calibration on one interactive GPU node (4 GPUs, runs spread over them).
#
#   salloc -A m3196 -C gpu -q interactive -N 1 --gpus-per-node 4 -t 04:00:00
#   source /global/cfs/cdirs/m3196/heindelj/rsfff_active_learning/scripts/env_perlmutter.sh
#   bash $RSFFF_AL/scripts/calibrate_udd.sh            # ~2-3 h; resumable (rerun the same line)
#   python $RSFFF_AL/scripts/summarize_udd.py $RSFFF_AL/runs/calibration
#
# Part 1, throughput: 200 steps of w8/w16/w32/w64 at 1, 4, 16 replicas -> s/step, which sets
# max_replicas and says whether 415 x 10 ps fits.
# Part 2, the bias: w8, w16, w32; 8 replicas x 2 ps each; unbiased vs matched (0.2) vs raw
# kappa 10 and 30. Compare pooled sigma distributions, failures, temperature.
set -uo pipefail
: "${RSFFF_AL:?source scripts/env_perlmutter.sh}"
OUT=${1:-$RSFFF_AL/runs/calibration}
mkdir -p "$OUT"
cd "$RSFFF_AL"
run() {  # gpu name args...
    local gpu=$1 name=$2; shift 2
    [ -e "$OUT/$name/summary.json" ] && { echo "done: $name"; return; }
    CUDA_VISIBLE_DEVICES=$gpu python scripts/udd.py --device cuda --out "$OUT/$name" "$@" \
        > "$OUT/$name.log" 2>&1 || echo "FAILED: $name (see $OUT/$name.log)"
}

echo "== throughput"
for n in 8 16 32 64; do
    for r in 1 4 16; do
        run 0 "tp_w${n}_r${r}" --waters $n --replicas $r --time-ps 0.05 --warmup-fs 25 --stride 50
    done
done

echo "== bias calibration (4 GPUs in parallel)"
for n in 8 16 32; do
    run 0 "cal_w${n}_none"    --waters $n --replicas 8 --time-ps 2 --bias-mode none &
    run 1 "cal_w${n}_matched" --waters $n --replicas 8 --time-ps 2 --bias-mode matched &
    run 2 "cal_w${n}_raw10"   --waters $n --replicas 8 --time-ps 2 --bias-mode raw --bias-kappa 10 &
    run 3 "cal_w${n}_raw30"   --waters $n --replicas 8 --time-ps 2 --bias-mode raw --bias-kappa 30 &
    wait
done
python scripts/summarize_udd.py "$OUT"
