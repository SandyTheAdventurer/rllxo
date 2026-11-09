# Ablations and noise for the revision (written before any of these runs)

Reviewer 5 asks for more ablations (comment 4), a second informed-crossover baseline (comment 1) and noisy
data (comment 2). These runs add comparisons only. a2c_t03 is unchanged and nothing is tuned.

**Equations:** the 23 held-out Nguyen and Keijzer equations (1-5, 8-11, 101-113, 115), paper protocol
(as in the held-out test), seeds 0-29, on the development laptop (Linux).

## Run C: ablations
| Arm | Question |
|---|---|
| a2c_t03 | reference, rerun on this machine |
| a2c_nocritic | does the critic matter (use_critic=False) |
| a2c_noconst, random_noconst | does the method still help without L-BFGS-B constant optimization |
| a2c_m01, random_m01 | does it help at mutation rate 0.1 instead of 0.5, where crossover dominates the search |
| sizefair | size-fair crossover (Langdon, 2000), a second informed-crossover baseline |

Independent vs joint sampling needs no arm: without H, the softmax over l_i + l'_j factorizes exactly into
two independent node choices. So a2c_noH (results/EXTRA_ARMS_PLAN.md) is the independent version, and
a2c_t03 vs a2c_noH is the comparison.
```
python compare_selectors.py --equations 1,2,3,4,5,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,115 \
    --arms a2c_t03,a2c_nocritic,a2c_noconst,random_noconst,a2c_m01,random_m01,sizefair --seeds 30 --seed0 0 --jobs 8 --tag ablations
```

## Run D: noise
- Gaussian noise on the training targets only, with sd = level x RMS(y) as in SRBench. The test set stays
  clean.
- Levels 0.01 and 0.1.
- Arms: random, a2c_t03, ssc.
```
python compare_selectors.py --equations <the same 23> --arms random,a2c_t03,ssc --seeds 30 --seed0 0 --jobs 8 --tag noise01 --noise 0.01
python compare_selectors.py --equations <the same 23> --arms random,a2c_t03,ssc --seeds 30 --seed0 0 --jobs 8 --tag noise10 --noise 0.1
```

## Analysis
Scoring is as in the held-out plan: per-equation median test R² over 30 seeds, with negative, non-finite or
error counted as 0. Comparisons are two-sided Wilcoxon over the 23 equations, ties dropped.
- **Run C, one Holm-corrected family:**
  - a2c_t03 vs a2c_nocritic;
  - a2c_noconst vs random_noconst;
  - a2c_m01 vs random_m01;
  - a2c_t03 vs sizefair;
  - sizefair vs random.
  - The random rows are from the extra-arms run (same machine, same equations).
- **Run D, one Holm-corrected family:** at each noise level, a2c_t03 vs random and a2c_t03 vs ssc.
- **Descriptive:** recovery, solved fraction, size and time for every arm; a2c_t03 vs a2c_noH on these 23
  equations.
- With 23 equations, a non-significant result is reported as inconclusive, not as evidence of no effect.
