# PySR as context (written before the PySR runs)

Reviewers 4 and 5 ask where the method stands against established symbolic-regression systems. A full SR
system differs from our GP in many ways at once, including its search, operators, constant fitting and
simplification. So a comparison against one says little about a crossover operator. That is why
crossover operators are compared inside one GP engine, as in the held-out test. PySR (Cranmer, 2023) is
reported as a reference point next to our GP, not as a competitor.

## Setup (baselines/)
- `export_tasks.py`: writes the exact training and test data of each held-out task (63 equations, seeds
  0-9), as run_task builds it, including SRSD's scaling.
- `run_pysr.py`: PySR 2.7.2 in its own environment, with the same training data and the benchmark's
  function set ({+, -, *, /} plus sin, cos, exp, log, and sqrt for Keijzer and SRSD). Otherwise PySR's
  defaults: 40 iterations, serial, deterministic, random_state = seed. PySR's own model selection
  ("best") picks the final expression.
- `score_pysr.py`: computes test R² on the same test set, and recovery and tree edit distance with the GP
  runs' own checks.

## Reporting
- A context table: per-equation median test R², solved fraction, recovery and size, for PySR, random and
  a2c_t03, over seeds 0-9 for all three.
- A Wilcoxon over the 63 equations is given for information. No claim depends on it.
