# Resistor-network application, rerun (written before these runs)

Reviewers 2 and 5 found the resistor example qualitative and too easy: a single run, no baseline, four
standard values, and a dangling resistor in Fig. 3. This rerun replaces it.

## Fixes
- **Pruning:** `prune_unused` now keeps only resistors that carry current between V+ and GND. The old
  reachability test kept a dangling branch off V+, because GND reaches it through V+. The Fig. 3 circuit
  prunes to three resistors with R_eq unchanged (53.87 ohm).
- **Like-for-like arms:** every arm swaps one resistor pair, and the arms differ only in how the pair is
  chosen. circuit_evolve's uniform crossover is kept as a reference.

## Setup (circuit_experiment.py)
- 6 nodes, 6 resistors, E6 values from 10 ohm to 6.8 kohm.
- Population 100, 50 generations, crossover 0.85, mutation 0.3, NSGA-II on (relative error, resistor
  count) as in circuit_evolve.
- 30 targets, log-uniform between 5 ohm and 5 kohm (seed 123), and 10 seeds per target.

| Arm | Pair choice |
|---|---|
| swap | uniform pair (random-crossover baseline) |
| sem | softmax of H = -rank/0.03 over the exact child scores of all pairs |
| a2c_sem | selector pair logits plus H (the method) |
| learned | the paper's original selector |
| uniform | circuit_evolve's uniform crossover (reference) |

The outcome of a run is the smallest relative error on its final Pareto front. "Solved" means a relative
error below 1e-4.

## Analysis
- **Primary:** a2c_sem vs swap, two-sided Wilcoxon over the 30 targets, on each target's median log10
  error over the 10 seeds.
- **Secondary, one Holm-corrected family:**
  - sem vs swap;
  - a2c_sem vs sem;
  - learned vs swap;
  - a2c_sem vs uniform;
  - a2c_sem vs swap on the solved fraction.
- **Reported:** resistor count, and evaluations per run (unique circuits scored). sem and a2c_sem score
  every candidate swap, so their evaluation counts are far higher, and the paper says so.
