# Held-out analysis (results/heldout/heldout.csv)

Rows: 11340 of 11340 expected; error rows: 15

## Primary: a2c_t03 vs random, median test R² per equation
63 equations: 47 wins, 5 losses, 11 ties; two-sided Wilcoxon p = 2.25e-08 (significant at 0.05)
Mean of per-equation medians: a2c_t03 0.7592, random 0.6649

## Secondary (Holm-corrected as one family)
| Comparison | Wins | Losses | Ties | p | Holm p |
|---|---|---|---|---|---|
| a2c_t03 vs ex64_t03 (does the policy add anything), median R² | 23 | 28 | 12 | 0.209 | 0.418 |
| a2c_t03 vs random_x4 (equal compute), median R² | 24 | 27 | 12 | 0.8 | 0.8 |
| a2c_t03 vs random, fraction of seeds solved (R² >= 0.999) | 37 | 4 | 22 | 2.46e-06 | 9.83e-06 |
| a2c_t03 vs random, exact recovery rate | 19 | 5 | 39 | 0.00271 | 0.00812 |

## Descriptive (no correction, not part of the claim)
| Comparison | Wins | Losses | Ties | p |
|---|---|---|---|---|
| dir_t03 vs random, median R² | 51 | 2 | 10 | 1.29e-09 |
| mlp vs random, median R² | 29 | 23 | 11 | 0.899 |
| ex64_t03 vs random, median R² | 50 | 3 | 10 | 7.83e-10 |
| random_x4 vs random, median R² | 50 | 1 | 12 | 1.77e-09 |

| Arm | Mean median R² | Solved | Recovered | Mean seconds per run | Mean size |
|---|---|---|---|---|---|
| random | 0.6649 | 0.195 | 0.103 | 5.3 | 22.3 |
| a2c_t03 | 0.7592 | 0.275 | 0.139 | 27.0 | 31.1 |
| ex64_t03 | 0.7754 | 0.278 | 0.136 | 17.7 | 31.3 |
| dir_t03 | 0.7850 | 0.289 | 0.141 | 21.4 | 31.7 |
| random_x4 | 0.7699 | 0.301 | 0.165 | 29.0 | 30.3 |
| mlp | 0.6613 | 0.200 | 0.103 | 10.5 | 21.8 |

### a2c_t03 vs random by benchmark family (median R²)
- Nguyen (9 equations): 7 wins, 1 losses, 1 ties, p = 0.0234
- Keijzer (14 equations): 10 wins, 1 losses, 3 ties, p = 0.0137
- SRSD (40 equations): 30 wins, 3 losses, 7 ties, p = 1.25e-07

### Error rows
arm        error                                      
a2c_t03    MemoryError((41, 54, 64), dtype('float64'))    1
           MemoryError()                                  1
dir_t03    MemoryError((45, 48, 64), dtype('float64'))    1
           MemoryError()                                  2
ex64_t03   TimeoutError()                                 2
mlp        MemoryError()                                  1
random     MemoryError()                                  4
random_x4  MemoryError()                                  3
