# Results

All numbers come from the per-run files in `results/` and are printed by the analysis scripts (see
[README.md](README.md), step 6). Unless stated otherwise:

- **Score:** each method's median test R² over the seeds of an equation; a negative, non-finite or failed run
  counts as 0. Means in the tables are means of these per-equation medians.
- **Solved:** test R² >= 0.999. **Recovered:** the final model simplifies to the true equation.
- **Tests:** two-sided Wilcoxon signed-rank over equations, ties dropped; Holm correction within each family
  of planned comparisons; r is the matched-pairs rank-biserial correlation. Wins / losses / ties count
  equations on which the first method's median is higher / lower / equal.

## Methods

| Name in the code | Method |
|---|---|
| `random` | random subtree crossover |
| `a2c_t03` | **GP-SR-RLC**: A2C policy over node pairs plus the semantic bias H (temperature 0.03) |
| `ex64_t03` | H alone, no policy: the pair is sampled from H only |
| `dir_t03` | directed variant of H: each child's insertion point is chosen separately |
| `a2c_noH` | the policy without H (equivalent to sampling the two nodes independently) |
| `mlp` | the original RL-LXO policy with its structural heuristic |
| `random_x4` | random crossover with 4x the generations (about the same run time as GP-SR-RLC) |
| `ssc` | semantic similarity-based crossover (Uy et al., 2011) |
| `sizefair` | size-fair crossover (Langdon, 2000) |
| `a2c_nocritic`, `a2c_noconst` | GP-SR-RLC without the critic / without constant optimization |
| `random_noconst` | random crossover without constant optimization |
| `a2c_m01`, `random_m01` | GP-SR-RLC / random crossover at mutation rate 0.1 instead of 0.5 |

## 1. Held-out test

63 equations (9 Nguyen, 14 Keijzer, 40 SRSD-Feynman medium) x 30 seeds x 6 methods = 11,340 runs, under
`results/HELDOUT_PLAN.md`. 15 runs failed (memory or time limit) and count as 0.

**Primary comparison, GP-SR-RLC vs random crossover:** 47 wins, 5 losses, 11 ties, p = 2.3 x 10⁻⁸.

| Family | Wins | Losses | Ties | p |
|---|---|---|---|---|
| Nguyen (9) | 7 | 1 | 1 | 0.023 |
| Keijzer (14) | 10 | 1 | 3 | 0.014 |
| SRSD (40) | 30 | 3 | 7 | 1.3 x 10⁻⁷ |

**Secondary comparisons (one Holm family):**

| Comparison | Wins | Losses | Ties | p | Holm p |
|---|---|---|---|---|---|
| GP-SR-RLC vs H alone (does the policy add anything?) | 23 | 28 | 12 | 0.21 | 0.42 |
| GP-SR-RLC vs random with 4x generations (equal time) | 24 | 27 | 12 | 0.80 | 0.80 |
| GP-SR-RLC vs random, fraction solved | 37 | 4 | 22 | 2.5 x 10⁻⁶ | 9.8 x 10⁻⁶ |
| GP-SR-RLC vs random, exact recovery | 19 | 5 | 39 | 0.0027 | 0.0081 |

**Per method:**

| Method | Mean median R² | Solved | Recovered | Seconds per run | Mean size |
|---|---|---|---|---|---|
| random | 0.665 | 0.195 | 0.103 | 5.3 | 22.3 |
| GP-SR-RLC | 0.759 | 0.275 | 0.139 | 27.0 | 31.1 |
| H alone | 0.775 | 0.278 | 0.136 | 17.7 | 31.3 |
| directed H | 0.785 | 0.289 | 0.141 | 21.4 | 31.7 |
| random, 4x generations | 0.770 | 0.301 | 0.165 | 29.0 | 30.3 |
| original RL-LXO policy | 0.661 | 0.200 | 0.103 | 10.5 | 21.8 |

GP-SR-RLC beats random crossover clearly, but it performs no better than H alone, and random crossover given
the same run time does as well. The gain comes from the semantic bias, not the learned policy.

## 2. Other crossover operators and the policy alone (63 equations)

One Holm family, under `results/EXTRA_ARMS_PLAN.md`:

| Comparison | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|
| GP-SR-RLC vs SSC | 50 | 1 | 12 | 1.7 x 10⁻⁹ | 6.7 x 10⁻⁹ | 0.97 |
| SSC vs random | 28 | 24 | 11 | 0.42 | 0.46 | 0.13 |
| Policy without H vs random | 30 | 22 | 11 | 0.23 | 0.46 | 0.19 |
| GP-SR-RLC vs policy without H | 47 | 4 | 12 | 2.7 x 10⁻⁸ | 8.2 x 10⁻⁸ | 0.89 |

| Method | Mean median R² | Solved | Recovered | Seconds per run |
|---|---|---|---|---|
| random | 0.665 | 0.195 | 0.103 | 5.1 |
| SSC | 0.654 | 0.196 | 0.111 | 7.3 |
| policy without H | 0.670 | 0.196 | 0.107 | 10.0 |
| GP-SR-RLC | 0.759 | 0.275 | 0.139 | 27.0 |

These runs reproduce the held-out random runs exactly: all 1,890 final models are identical.

## 3. Ablations (23 Nguyen and Keijzer equations)

One Holm family, under `results/ABLATION_NOISE_PLAN.md`:

| Comparison | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|
| GP-SR-RLC vs no critic | 7 | 10 | 6 | 0.64 | 1 | -0.14 |
| No constant optimization: GP-SR-RLC vs random | 18 | 2 | 3 | 1.3 x 10⁻⁴ | 6.7 x 10⁻⁴ | 0.89 |
| Mutation 0.1: GP-SR-RLC vs random | 17 | 3 | 3 | 2.1 x 10⁻⁴ | 8.4 x 10⁻⁴ | 0.87 |
| GP-SR-RLC vs size-fair crossover | 16 | 2 | 5 | 3.3 x 10⁻⁴ | 9.8 x 10⁻⁴ | 0.88 |
| Size-fair crossover vs random | 10 | 9 | 4 | 0.74 | 1 | 0.09 |

| Method | Mean median R² | Solved | Recovered | Seconds per run |
|---|---|---|---|---|
| random | 0.957 | 0.399 | 0.241 | 3.7 |
| size-fair | 0.958 | 0.387 | 0.242 | 3.9 |
| policy without H | 0.959 | 0.401 | 0.258 | 7.0 |
| GP-SR-RLC | 0.978 | 0.478 | 0.291 | 21.2 |
| GP-SR-RLC, no critic | 0.979 | 0.494 | 0.310 | 20.9 |
| GP-SR-RLC, no constant optimization | 0.979 | 0.468 | 0.254 | 23.6 |
| random, no constant optimization | 0.953 | 0.386 | 0.196 | 3.9 |
| GP-SR-RLC, mutation 0.1 | 0.953 | 0.397 | 0.248 | 18.4 |
| random, mutation 0.1 | 0.909 | 0.309 | 0.187 | 3.0 |

Three runs (K3, seed 13: GP-SR-RLC, no critic, no constant optimization) stopped with an internal error in
the policy update and count as 0; see `results/pc_rev1/DEVIATION.md`.

## 4. Noise (23 equations, noise-free test sets)

Gaussian noise on the training targets with standard deviation 1% or 10% of their RMS. One Holm family:

| Comparison | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|
| 1%: GP-SR-RLC vs random | 19 | 2 | 2 | 2.4 x 10⁻⁴ | 7.2 x 10⁻⁴ | 0.84 |
| 1%: GP-SR-RLC vs SSC | 19 | 2 | 2 | 1.8 x 10⁻⁵ | 7.3 x 10⁻⁵ | 0.94 |
| 10%: GP-SR-RLC vs random | 18 | 5 | 0 | 0.0017 | 0.0033 | 0.72 |
| 10%: GP-SR-RLC vs SSC | 17 | 6 | 0 | 0.0060 | 0.0060 | 0.64 |

| Noise | Method | Mean median R² | Solved | Recovered |
|---|---|---|---|---|
| 1% | random | 0.957 | 0.396 | 0.154 |
| 1% | SSC | 0.957 | 0.391 | 0.136 |
| 1% | GP-SR-RLC | 0.977 | 0.478 | 0.167 |
| 10% | random | 0.956 | 0.274 | 0.058 |
| 10% | SSC | 0.960 | 0.261 | 0.039 |
| 10% | GP-SR-RLC | 0.977 | 0.322 | 0.046 |

## 5. Original benchmark protocols

Under `results/PROTOCOL_CHECKS_PLAN.md`; each test on its own (14 and 9 equations).

| Comparison | Wins | Losses | Ties | p | r |
|---|---|---|---|---|---|
| Keijzer's original test sets: GP-SR-RLC vs random | 9 | 1 | 4 | 0.020 | 0.82 |
| Keijzer's original test sets: GP-SR-RLC vs SSC | 8 | 2 | 4 | 0.049 | 0.71 |
| Original Nguyen protocol (20 points): GP-SR-RLC vs random | 6 | 3 | 0 | 0.73 | 0.16 |
| Original Nguyen protocol: GP-SR-RLC vs random, recovery | 3 | 2 | 4 | 0.63 | 0.33 |
| Original Nguyen protocol: GP-SR-RLC vs SSC | 4 | 5 | 0 | 0.43 | -0.33 |

On the two Keijzer test sets that extrapolate, K4 favours GP-SR-RLC (median R² 0.966 vs 0.912,
Mann-Whitney p = 0.0006) and K6 is a tie (1.000 for all methods). Under the original Nguyen protocol all
methods reach a mean median R² of about 0.994, so that comparison is inconclusive.

## 6. PySR, for context only (63 equations, seeds 0-9)

PySR differs from this GP in nearly every component, so this does not isolate a crossover operator
(`results/PYSR_CONTEXT_PLAN.md`).

| Method | Mean median R² | Solved | Recovered | Mean size | Seconds per run |
|---|---|---|---|---|---|
| random | 0.669 | 0.206 | 0.105 | 21.7 | 6.0 |
| GP-SR-RLC | 0.766 | 0.265 | 0.141 | 31.3 | 32.0 |
| PySR 2.7.2 | 0.872 | 0.635 | 0.367 | 15.6 | 15.9 |

| Family | random | GP-SR-RLC | PySR |
|---|---|---|---|
| Nguyen (9) | 0.987 | 0.994 | 1.000 |
| Keijzer (14) | 0.948 | 0.969 | 0.999 |
| SRSD (40) | 0.500 | 0.644 | 0.799 |

PySR is clearly stronger than this GP (50 wins, 2 losses against GP-SR-RLC).

## 7. Resistor networks

30 target resistances between 5 Ω and 5 kΩ x 10 seeds x 5 methods, under `results/CIRCUIT_PLAN.md`. The
methods differ only in which resistor pair crossover exchanges. Score: each target's median log10 relative
error (lower is better).

**Primary, GP-SR-RLC vs random pair:** lower error on 23 targets, higher on 5, 2 ties; p = 2.1 x 10⁻⁵, r = 0.84.

| Comparison (wins = lower error) | Wins | Losses | Ties | p | Holm p |
|---|---|---|---|---|---|
| H alone vs random pair | 25 | 3 | 2 | 6.6 x 10⁻⁷ | 3.3 x 10⁻⁶ |
| GP-SR-RLC vs H alone | 12 | 15 | 3 | 0.61 | 0.61 |
| Original policy vs random pair | 17 | 11 | 2 | 0.19 | 0.39 |
| GP-SR-RLC vs uniform crossover | 24 | 5 | 1 | 4.1 x 10⁻⁵ | 1.7 x 10⁻⁴ |
| GP-SR-RLC vs random pair, solved | 15 | 5 | 10 | 0.0033 | 0.0099 |

| Pair choice | Median log10 error | Solved | Resistors | Circuits scored per run | Seconds per run |
|---|---|---|---|---|---|
| random pair | -3.02 | 0.173 | 4.25 | 4,987 | 0.5 |
| uniform crossover | -3.00 | 0.163 | 4.28 | 4,983 | 0.6 |
| original policy | -3.06 | 0.197 | 4.24 | 9,008 | 1.8 |
| H alone | -3.35 | 0.257 | 4.30 | 136,043 | 7.9 |
| GP-SR-RLC | -3.37 | 0.263 | 4.23 | 135,955 | 9.2 |

As in symbolic regression, the gain comes from H, and it costs about 27 times as many circuit evaluations.

## 8. Cost in evaluations

One seed of GP-SR-RLC on each of the 63 held-out equations (`results/eval_count/`). These runs were made on a
different machine, so they are statistically equivalent to the held-out runs rather than identical.

- The semantic bias scores a median of 3.9 million candidate children per run on the 64-point batch (about
  23,000 per generation; from 0 to 14 million).
- A run also makes 60,000 full fitness evaluations on 512 points (300 per generation for 200 generations), as
  every method does.
- In points evaluated, the batch scoring is therefore about 8 times the cost of the full evaluations.
