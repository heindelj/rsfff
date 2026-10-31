#!/bin/bash
# usage: eval_all.sh <run>
r=$1; E=/home/claude/rsfff/scripts/eval_bonding.py; export OMP_NUM_THREADS=1
python $E ckpt/$r/best.pt data/train.xyz --split val --json eval_${r}_val.json > eval_${r}_val.txt 2>/dev/null
python $E ckpt/$r/best.pt data/probe.xyz --split val --no-hessian --vacuum data/train.xyz data/stretch.xyz --json eval_${r}_probe.json > eval_${r}_probe.txt 2>/dev/null
python $E ckpt/$r/best.pt data/stretch.xyz --json eval_${r}_stretch.json > eval_${r}_stretch.txt 2>/dev/null
echo done > eval_${r}.done
