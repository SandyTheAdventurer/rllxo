# Benchmark-protocol checks for the revision (written before any of these runs)

Reviewer 4 asks for the conventional Keijzer test sets (R4.4/4.5) and questions the departure from the
original Nguyen protocol (R4.6). These runs check whether the held-out result holds under each benchmark's
original protocol. a2c_t03 is unchanged and nothing is tuned.

## Run A: Keijzer's original test sets
- The 14 held-out Keijzer equations (101-113, 115), with the same training data and config as the held-out
  test.
- Each run also scores its exact final model on Keijzer's (2003) test set (`keijzer_test_set`,
  column `test_r2_orig`).
- Most of these test sets are dense grids over the training range. K4 (shifted) and K6 (x up to 120 instead
  of 50) extrapolate.
```
python compare_selectors.py --equations 101,102,103,104,105,106,107,108,109,110,111,112,113,115 \
    --arms random,a2c_t03,ssc --seeds 30 --seed0 0 --jobs 8 --tag keijzer_orig
```

## Run B: the original Nguyen protocol
- The 9 held-out Nguyen equations (1-5, 8-11), trained on 20 points from Uy et al.'s (2011) ranges
  (`NGUYEN_ORIG_RANGES`) instead of 512 points from the wider ranges in nguyen.py.
- Each run is tested on 512 fresh points from the same ranges.
```
python compare_selectors.py --equations 1,2,3,4,5,8,9,10,11 --arms random,a2c_t03,ssc --seeds 30 \
    --seed0 0 --jobs 8 --tag nguyen_orig --protocol orig
```

## Analysis (analyze_protocol_checks.py)
Scoring is as in the held-out plan: per-equation median over 30 seeds, with negative, non-finite or error
counted as 0. Comparisons are two-sided Wilcoxon over the equations, ties dropped.
- **Run A:**
  - a2c_t03 vs random on median `test_r2_orig`;
  - the same on K4 and K6 alone, as descriptive per-equation Mann-Whitney U tests;
  - a2c_t03 vs ssc.
- **Run B:**
  - a2c_t03 vs random on median test R², and on the per-equation recovery rate;
  - a2c_t03 vs ssc.
- With 14 and 9 equations these tests have little power. A non-significant result is reported as
  inconclusive, not as evidence of no effect.
- These runs are on the Linux laptop. The held-out runs were on the Windows PC.
