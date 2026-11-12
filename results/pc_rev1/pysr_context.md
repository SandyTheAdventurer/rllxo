# PySR as context (seeds 0-9, 63 held-out equations)

PySR rows: 630 of 630; errors 0.

| Arm | Mean median R² | Solved | Recovered | Mean size | Mean s/run |
|---|---|---|---|---|---|
| random | 0.6690 | 0.206 | 0.105 | 21.7 | 6.0 |
| a2c_t03 | 0.7661 | 0.265 | 0.141 | 31.3 | 32.0 |
| pysr | 0.8718 | 0.635 | 0.367 | 15.6 | 15.9 |

By family (mean of per-equation median R²):

| Family | n | random | a2c_t03 | pysr |
|---|---|---|---|---|
| Nguyen | 9 | 0.9873 | 0.9940 | 0.9997 |
| Keijzer | 14 | 0.9475 | 0.9690 | 0.9988 |
| SRSD | 40 | 0.4999 | 0.6439 | 0.7986 |

For information only (two-sided Wilcoxon over the 63 equations, ties dropped):

- pysr vs a2c_t03, median R²: 50 wins, 2 losses, 11 ties, p = 3.41e-09, r = 0.94
- pysr vs a2c_t03, solved fraction: 44 wins, 1 losses, 18 ties, p = 6.75e-09, r = 0.99
- pysr vs random, median R²: 53 wins, 0 losses, 10 ties, p = 2.39e-10, r = 1.00
- pysr vs random, solved fraction: 45 wins, 1 losses, 17 ties, p = 4.84e-09, r = 0.99
