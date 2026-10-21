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
