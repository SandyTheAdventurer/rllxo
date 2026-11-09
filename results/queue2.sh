#!/bin/bash
# Waits for the protocol-check queue (../rllxo-rev/results/queue.sh), then runs the ablations and the noise runs
cd "$(dirname "$0")/.."
while pgrep -f "[q]ueue.sh" >/dev/null || pgrep -f "[t]ag extra" >/dev/null; do sleep 60; done
python compare_selectors.py --equations 1,2,3,4,5,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,115 --arms a2c_t03,a2c_nocritic,a2c_noconst,random_noconst,a2c_m01,random_m01,sizefair --seeds 30 --seed0 0 --jobs 8 --tag ablations > results/compare_selectors/ablations.log 2>&1
python compare_selectors.py --equations 1,2,3,4,5,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,115 --arms random,a2c_t03,ssc --seeds 30 --seed0 0 --jobs 8 --tag noise01 --noise 0.01 > results/compare_selectors/noise01.log 2>&1
python compare_selectors.py --equations 1,2,3,4,5,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,115 --arms random,a2c_t03,ssc --seeds 30 --seed0 0 --jobs 8 --tag noise10 --noise 0.1 > results/compare_selectors/noise10.log 2>&1
