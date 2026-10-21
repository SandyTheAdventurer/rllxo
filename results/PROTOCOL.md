# Selector experiments: dev / confirmation protocol

Written 2026-10-21, before any of the variants below were run on the confirmation set.

- **Dev equations:** N6, N7, N12 (seeds 0-29). Variants and their settings may be tuned here freely.
  These have already been used for the six-arm study, the median- and improvement-trained
  pretrained scorers, so they are no longer an unbiased test.
- **Confirmation equations:** N3, N4, N9, N10 (random-crossover recovery in the paper's original
  runs: 17%, 37%, 53%, 37%). They are excluded from all offline pretraining data.
- **Confirmation run:** only the single best dev variant, with its settings frozen, run once on the
  confirmation equations with seeds 0-29 against random, mlp (paper method) and ssc (semantic
  similarity-based crossover baseline).
- **Primary outcome:** exact symbolic recovery pooled over the four equations (Fisher exact test vs
  random and vs mlp). Secondary: per-equation held-out test R² (Mann-Whitney U, Holm-corrected),
  generations to R² >= 0.999, expression size, wall time.
- Whatever the confirmation run shows is what gets reported.

## SRSD-Feynman medium (written 2026-10-21, before any SRSD run)

None of the selector settings were tuned on these equations, so this is a fresh test set.

- **Data:** the 40 SRSD-Feynman medium equations (Matsubara et al., 2024; spec ids 2000-2039).
  Each run trains on 512 rows of the SRSD train split (drawn per seed) and is scored on the full
  SRSD test split. y is divided by its training standard deviation for the GA and predictions are
  rescaled; the unary set is the functions in the true equation, plus sqrt for half-integer powers.
  GA settings are the paper's (nguyen.ga_config + learned_overrides), unchanged.
- **Arms:** random, mlp (paper method), ssc; seeds 0-29.
- **Primary outcome:** test R² compared per equation. Non-finite or negative test R² counts as 0. Per
  equation, take each arm's median over seeds; compare mlp with random by a two-sided Wilcoxon
  signed-rank test across the 40 equations (ties dropped, zero_method='wilcox').
- **Secondary:** the same paired test on per-equation recovery rate and on the share of runs with
  test R² >= 0.999; pooled exact recovery (Fisher); ssc vs random and vs mlp on the primary
  outcome; wall time. Recovery treats constants within 0.1% of a constant in the true equation
  (times a power of pi, either sign) as that constant.
- Whatever this shows is what gets reported.

## Revision-1 rerun, v2 (written 2026-10-21, before any v2 run; supersedes the SRSD section above)

The SRSD run above was stopped after 837 runs (seeds 0-6 seen) and is kept as srsd_v1.csv. v2 changes
the setup for every benchmark and starts again from seed 0; its results are in v2.csv.

- **Changes from the paper's config, for all arms and benchmarks:**
  - each benchmark's standard function set for every equation (previously each equation got only
    the functions in its true answer): Nguyen {+, -, *, /, sin, cos, exp, log} (Uy et al., 2011, as
    listed by Petersen et al., 2021); Keijzer and SRSD the same plus sqrt (Keijzer's original set not
    verified); no pow;
  - selector warmup 30 -> 10 generations (chosen after seeing SRSD seeds 0-6; mlp only);
  - elite constant optimization every 10 -> 5 generations;
  - the cached numpy implementation of the mlp selector's crossover sampling (same distribution).
  - fixes found in the first minutes of v2 (13 runs, discarded): invalid trees (NaN/inf output)
    scored 1e6 + size, which beat every valid tree when the target's variance is large (N4: MSE of
    x is 2.4e7), so the GA could end on an invalid tree; they now score 1e18 + size. run_ga now
    reports the final best's own train R². Recovery also snaps constants within 0.01% of a small
    rational times a power of e or pi (e.g. exp(-1)), and within 0.1% of the target's own float
    constants (Keijzer-6), and treats variables as real (so log(exp(x)) = x).
- **Equations:** Nguyen 1-12, Keijzer 1-15, SRSD-Feynman medium (40). **Arms:** random, mlp, ssc.
  **Seeds:** 0-29.
- **New metric:** tree edit distance to the true equation (Zhang-Shasha, both simplified, constants
  snapped as for recovery) and NED = min(1, TED / true tree size) as in SRSD; recovered runs count 0.
- **Primary outcome**, analysed separately for Nguyen+Keijzer and for SRSD: per-equation median test R²
  (non-finite or negative counts as 0), mlp vs random, two-sided Wilcoxon signed-rank across equations
  (ties dropped). **Secondary:** the same paired test on recovery rate, share of runs with test
  R² >= 0.999 and median NED; pooled recovery (Fisher); ssc vs random and vs mlp; generations to
  R² >= 0.999; expression size; wall time.
- Whatever this shows is what gets reported.
