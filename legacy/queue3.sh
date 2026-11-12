#!/bin/bash
# Waits for the ablation and noise queue, then runs PySR on the 630 exported tasks (8 at a time) and scores them
cd "$(dirname "$0")/.."
while pgrep -f "[q]ueue2.sh" >/dev/null; do sleep 60; done
for e in $(ls baselines/tasks/*.json | xargs -n1 basename | sed 's/.json//'); do for s in $(seq 0 9); do
  [ -f baselines/pysr_out/${e}_${s}.json ] || echo "$e $s"; done; done | \
  xargs -P 8 -L 1 sh -c '.venv-pysr/bin/python baselines/run_pysr.py $0 $1 >> results/compare_selectors/pysr.log 2>&1'
python baselines/score_pysr.py >> results/compare_selectors/pysr.log 2>&1
