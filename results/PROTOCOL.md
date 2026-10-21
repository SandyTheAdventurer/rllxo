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
