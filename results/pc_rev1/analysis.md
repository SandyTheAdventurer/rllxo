# Revision-1 analysis (runs on the Windows PC, 20 workers)

Held-out reference rows: results/heldout/heldout.csv. Run folder: results/pc_rev1.

# 1. Extra arms (EXTRA_ARMS_PLAN.md)

Rows: 5670 of 5670.

## Extra arms: one Holm family, median test R², 63 equations (random = this run's rows)

| Comparison | n | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|---|
| a2c_t03 vs ssc | 63 | 50 | 1 | 12 | 1.67e-09 | 6.68e-09 | 0.97 |
| ssc vs random | 63 | 28 | 24 | 11 | 0.423 | 0.459 | 0.13 |
| a2c_noH vs random | 63 | 30 | 22 | 11 | 0.229 | 0.459 | 0.19 |
| a2c_t03 vs a2c_noH | 63 | 47 | 4 | 12 | 2.72e-08 | 8.16e-08 | 0.89 |

### Extra arms: descriptive

| Comparison | n | Wins | Losses | Ties | p (uncorrected) | r |
|---|---|---|---|---|---|---|
| a2c_noH vs ex64_t03 | 63 | 6 | 46 | 11 | 2.63e-08 | -0.89 |
| ex64_t03 vs ssc | 63 | 49 | 3 | 11 | 5.3e-09 | 0.93 |
| a2c_t03 vs ssc, solved fraction | 63 | 33 | 7 | 23 | 7.64e-06 | 0.81 |
| a2c_t03 vs ssc, recovery | 63 | 15 | 6 | 42 | 0.00883 | 0.65 |
| a2c_noH vs random, solved fraction | 63 | 17 | 19 | 27 | 0.711 | 0.07 |
- a2c_t03 vs ssc, Nguyen (9): 7 wins, 0 losses, 2 ties, p = 0.0156
- a2c_t03 vs ssc, Keijzer (14): 10 wins, 1 losses, 3 ties, p = 0.0244
- a2c_t03 vs ssc, SRSD (40): 33 wins, 0 losses, 7 ties, p = 2.33e-10

### Extra arms: per arm (63 equations)

| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |
|---|---|---|---|---|---|---|---|
| random | 1890 | 4 | 0.6652 | 0.195 | 0.103 | 5.1 | 22.3 |
| ssc | 1890 | 1 | 0.6544 | 0.196 | 0.111 | 7.3 | 22.6 |
| a2c_noH | 1890 | 0 | 0.6700 | 0.196 | 0.107 | 10.0 | 22.9 |
| ex64_t03 | 1890 | 2 | 0.7754 | 0.278 | 0.136 | 17.7 | 31.3 |
| a2c_t03 | 1890 | 2 | 0.7592 | 0.275 | 0.139 | 27.0 | 31.1 |

Machine check (the plan compared laptop and PC random; both random sets are now from the PC):
- held-out random vs this run's random: 1890 of 1890 random runs give the identical final expression.
- Mean seconds per random run: held-out 5.27 (18 workers), this run 5.09 (20 workers).

Error rows (extra):

- random, eq 2039, seed 14: `TimeoutError()`
- random, eq 2033, seed 14: `MemoryError()`
- ssc, eq 2019, seed 18: `MemoryError()`
- random, eq 2033, seed 19: `MemoryError()`
- random, eq 2019, seed 25: `MemoryError()`

# 2. Protocol checks (PROTOCOL_CHECKS_PLAN.md)

Rows: keijzer_orig 1260 of 1260, nguyen_orig 810 of 810.

### Protocol checks (each test on its own, as the plan lists them; 14 and 9 equations)

| Comparison | n | Wins | Losses | Ties | p (uncorrected) | r |
|---|---|---|---|---|---|---|
| Run A: a2c_t03 vs random, Keijzer's test sets | 14 | 9 | 1 | 4 | 0.0195 | 0.82 |
| Run A: a2c_t03 vs ssc, Keijzer's test sets | 14 | 8 | 2 | 4 | 0.0488 | 0.71 |
| Run A (reference): a2c_t03 vs random, our test sets | 14 | 10 | 1 | 3 | 0.0137 | 0.82 |
| Run B: a2c_t03 vs random, median R² | 9 | 6 | 3 | 0 | 0.734 | 0.16 |
| Run B: a2c_t03 vs random, recovery rate | 9 | 3 | 2 | 4 | 0.625 | 0.33 |
| Run B: a2c_t03 vs ssc, median R² | 9 | 4 | 5 | 0 | 0.426 | -0.33 |

K4 and K6 on Keijzer's test sets (extrapolation), per-equation Mann-Whitney U, a2c_t03 vs random:

| Eq | Median a2c_t03 | Median random | Median ssc | p |
|---|---|---|---|---|
| K4 | 0.9658 | 0.9124 | 0.9525 | 0.00062 |
| K6 | 1.0000 | 1.0000 | 1.0000 | 0.127 |

### Run A per arm (score on Keijzer's test sets)

| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |
|---|---|---|---|---|---|---|---|
| random | 420 | 0 | 0.9299 | 0.362 | 0.250 | 3.7 | 13.3 |
| ssc | 420 | 0 | 0.9319 | 0.357 | 0.260 | 5.5 | 13.9 |
| a2c_t03 | 420 | 0 | 0.9602 | 0.386 | 0.255 | 18.6 | 19.6 |

### Run B per arm (original Nguyen protocol)

| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |
|---|---|---|---|---|---|---|---|
| random | 270 | 0 | 0.9939 | 0.237 | 0.148 | 3.2 | 10.4 |
| ssc | 270 | 0 | 0.9944 | 0.307 | 0.189 | 4.0 | 10.9 |
| a2c_t03 | 270 | 0 | 0.9936 | 0.304 | 0.170 | 15.3 | 11.2 |
- Run A vs held-out (same training data): 419 of 420 a2c_t03 runs give the identical final expression.

# 3. Ablations (ABLATION_NOISE_PLAN.md, run C)

Rows: 4830 of 4830.

## Ablations: one Holm family, median test R², 23 Nguyen and Keijzer equations

| Comparison | n | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|---|
| a2c_t03 vs a2c_nocritic | 23 | 7 | 10 | 6 | 0.644 | 1 | -0.14 |
| a2c_noconst vs random_noconst | 23 | 18 | 2 | 3 | 0.000134 | 0.000668 | 0.89 |
| a2c_m01 vs random_m01 | 23 | 17 | 3 | 3 | 0.00021 | 0.000839 | 0.87 |
| a2c_t03 vs sizefair | 23 | 16 | 2 | 5 | 0.000328 | 0.000984 | 0.88 |
| sizefair vs random | 23 | 10 | 9 | 4 | 0.738 | 1 | 0.09 |

### Ablations: descriptive

| Comparison | n | Wins | Losses | Ties | p (uncorrected) | r |
|---|---|---|---|---|---|---|
| a2c_t03 vs a2c_noH (joint vs independent sampling) | 23 | 15 | 3 | 5 | 0.00896 | 0.68 |
| a2c_t03 vs random | 23 | 17 | 2 | 4 | 0.00042 | 0.85 |
| a2c_noconst vs a2c_t03 | 23 | 8 | 11 | 4 | 0.86 | 0.05 |
| random_noconst vs random | 23 | 8 | 12 | 3 | 0.784 | -0.08 |

### Ablations: per arm (23 equations)

| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |
|---|---|---|---|---|---|---|---|
| random | 690 | 0 | 0.9569 | 0.399 | 0.241 | 3.7 | 14.5 |
| sizefair | 690 | 0 | 0.9578 | 0.387 | 0.242 | 3.9 | 13.7 |
| random_noconst | 690 | 0 | 0.9527 | 0.386 | 0.196 | 3.9 | 15.7 |
| random_m01 | 690 | 0 | 0.9089 | 0.309 | 0.187 | 3.0 | 14.3 |
| a2c_noH | 690 | 0 | 0.9589 | 0.401 | 0.258 | 7.0 | 14.7 |
| a2c_nocritic | 690 | 2 | 0.9790 | 0.494 | 0.310 | 20.9 | 22.3 |
| a2c_noconst | 690 | 1 | 0.9792 | 0.468 | 0.254 | 23.6 | 24.9 |
| a2c_m01 | 690 | 0 | 0.9525 | 0.397 | 0.248 | 18.4 | 22.3 |
| a2c_t03 | 690 | 2 | 0.9779 | 0.478 | 0.291 | 21.2 | 21.3 |
- ablation a2c_t03 vs held-out a2c_t03: 688 of 690 a2c_t03 runs give the identical final expression.

Error rows (ablations):

- a2c_nocritic, eq 111, seed 10: `TimeoutError()`
- a2c_t03, eq 103, seed 13: `RuntimeError('The size of tensor a (19) must match the size of tensor b (18) at non-singleton dimension 1')`
- a2c_nocritic, eq 103, seed 13: `RuntimeError('The size of tensor a (19) must match the size of tensor b (18) at non-singleton dimension 1')`
- a2c_noconst, eq 103, seed 13: `RuntimeError('The size of tensor a (19) must match the size of tensor b (18) at non-singleton dimension 1')`
- a2c_t03, eq 111, seed 25: `TimeoutError()`

# 4. Noise (ABLATION_NOISE_PLAN.md, run D)

Rows: noise01 2070, noise10 2070 (of 2070 each). Test sets are noise-free.

## Noise: one Holm family, median clean-test R², 23 equations

| Comparison | n | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|---|
| 1% noise: a2c_t03 vs random | 23 | 19 | 2 | 2 | 0.000241 | 0.000724 | 0.84 |
| 1% noise: a2c_t03 vs ssc | 23 | 19 | 2 | 2 | 1.81e-05 | 7.25e-05 | 0.94 |
| 10% noise: a2c_t03 vs random | 23 | 18 | 5 | 0 | 0.00166 | 0.00332 | 0.72 |
| 10% noise: a2c_t03 vs ssc | 23 | 17 | 6 | 0 | 0.00603 | 0.00603 | 0.64 |

| Noise | random | ssc | a2c_t03 (mean of per-equation median R²) |
|---|---|---|---|
| 0 (held-out / extra rows) | 0.9569 | 0.9609 | 0.9785 |
| 1% | 0.9567 | 0.9566 | 0.9769 |
| 10% | 0.9559 | 0.9600 | 0.9768 |

### 1% noise per arm

| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |
|---|---|---|---|---|---|---|---|
| random | 690 | 0 | 0.9567 | 0.396 | 0.154 | 4.4 | 14.4 |
| ssc | 690 | 0 | 0.9566 | 0.391 | 0.136 | 7.0 | 15.1 |
| a2c_t03 | 690 | 0 | 0.9769 | 0.478 | 0.167 | 23.3 | 22.0 |

### 10% noise per arm

| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |
|---|---|---|---|---|---|---|---|
| random | 690 | 0 | 0.9559 | 0.274 | 0.058 | 4.6 | 17.9 |
| ssc | 690 | 0 | 0.9600 | 0.261 | 0.039 | 7.5 | 19.0 |
| a2c_t03 | 690 | 0 | 0.9768 | 0.322 | 0.046 | 24.1 | 30.6 |

# 5. Resistor networks (CIRCUIT_PLAN.md)

Rows: 1500 of 1500 (5 arms x 30 targets x 10 seeds).
## Primary: a2c_sem vs swap, median log10 relative error per target (wins = lower error)

30 targets: 23 wins, 5 losses, 2 ties; two-sided Wilcoxon p = 2.05e-05, r = 0.84 (significant at 0.05)

## Circuits secondary: one Holm family (log-error comparisons: wins = lower error)

| Comparison | n | Wins | Losses | Ties | p | Holm p | r |
|---|---|---|---|---|---|---|---|
| sem vs swap | 30 | 25 | 3 | 2 | 6.56e-07 | 3.28e-06 | 0.94 |
| a2c_sem vs sem | 30 | 12 | 15 | 3 | 0.611 | 0.611 | -0.12 |
| learned vs swap | 30 | 17 | 11 | 2 | 0.194 | 0.387 | 0.29 |
| a2c_sem vs uniform | 30 | 24 | 5 | 1 | 4.13e-05 | 0.000165 | 0.81 |
| a2c_sem vs swap, solved fraction | 30 | 15 | 5 | 10 | 0.00329 | 0.00988 | 0.74 |

| Arm | Median of target medians, log10 error | Solved | Mean resistors | Mean evaluations | Mean s/run |
|---|---|---|---|---|---|
| swap | -3.02 | 0.173 | 4.25 | 4987 | 0.5 |
| uniform | -3.00 | 0.163 | 4.28 | 4983 | 0.6 |
| learned | -3.06 | 0.197 | 4.24 | 9008 | 1.8 |
| sem | -3.35 | 0.257 | 4.30 | 136043 | 7.9 |
| a2c_sem | -3.37 | 0.263 | 4.23 | 135955 | 9.2 |
