# Held-out test: semantic-bias A2C crossover vs random crossover

This plan was fixed before any held-out run was analysed, and the user approved it. The git tag
`heldout-frozen`, made just before the run starts, freezes the code and the analysis. Nothing in either changes after the run starts, and
the result is reported whichever way it goes.

## Method under test
`a2c_t03` uses the paper's joint pair selection, L_ij = l_i + l'_j + H, with the paper's network, critic
and A2C update. The structural heuristic is replaced by H(i,j) = -rank(i,j)/0.03, where:
- rank(i,j) is the child's normalized rank in [0,1] (0 = best), by exact MSE on 64 training points;
- a pair's child loss is the better of its two children;
- pairs that leave a parent unchanged get -inf;
- if |A|*|B| > 2500, H = 0 and the policy alone chooses.

Other settings:
- Reward: logratio = clip(log10(best parent MSE / best child MSE), -1, 1).
- Warmup: 1 generation.
- Everything else is the paper's config: nguyen.ga_config + learned_overrides (population 300, 200
  generations).

Code: semx.py, main.py and compare_selectors.py.

## Arms
| Arm | Role |
|---|---|
| random | uniform crossover points (the baseline) |
| a2c_t03 | method under test |
| ex64_t03 | ablation: the same semantic ranking without the network (sampled straight from the softmax of H) |
| dir_t03 | non-RL reference: directed exact selection, a separate pair per child |
| random_x4 | equal-compute reference: random with 800 generations |
| mlp | the paper's current method (A2C with the structural heuristic) |

## Equations and seeds
- 63 held-out equations, none used in development (`--equations heldout`):
  - Nguyen 1-5 and 8-11 (9);
  - Keijzer 101-113 and 115 (14);
  - SRSD-Feynman medium 2000-2039 (40).
- Seeds 0-29.
- Development used only seeds 1000-1004 on these equations: Jin 201-206, Neat 301-309 (not 306),
  SRSD-easy 401-412, N6, N7, N12 and K14.

## Run
```
python compare_selectors.py --equations heldout --arms random,a2c_t03,ex64_t03,dir_t03,random_x4,mlp \
    --seeds 30 --seed0 0 --jobs <cores> --tag heldout
python analyze_heldout.py results/compare_selectors/heldout.csv
```
All runs come from one machine: a Windows 11 PC with Python 3.14 and the versions in
requirements.txt. Numpy's sin, cos, exp and log differ in the last bits between Windows and Linux, so a
rerun on another platform matches statistically, not bit for bit (REPRODUCE.md). All arms share the
machine, so the comparison is unaffected.

Two earlier starts on the development laptop were stopped early, by the user's choice of machine: one
with 126 rows, one with 331. Their rows were set aside unanalysed and are not part of this test. One row
of the second start (a single ex64_t03 run's R²) was printed by accident while checking that the run had
stopped.

## Analysis (analyze_heldout.py)
- Per equation, score each arm by its median test R² over the 30 seeds, with non-finite or negative R²
  and error rows set to 0.
- **Primary test:** a2c_t03 vs random, two-sided Wilcoxon signed-rank over the 63 per-equation medians.
  Zero differences are dropped (scipy zero_method="wilcox"). Alpha = 0.05.
- **Secondary tests, Holm-corrected as one family:**
  - a2c_t03 vs ex64_t03 (does the policy add anything);
  - a2c_t03 vs random_x4 (equal compute);
  - a2c_t03 vs random on the fraction of seeds solved (test R² >= 0.999), per equation;
  - a2c_t03 vs random on the exact recovery rate, per equation.
- **Descriptive only:** dir_t03, mlp, ex64_t03 and random_x4 vs random; mean runtime and size per arm;
  a2c_t03 vs random by benchmark family (Nguyen, Keijzer, SRSD).
- Any error rows are listed.

## Development evidence (dev equations only; not part of the claim)
30 dev equations x 5 seeds, laptop:
- a2c_t03 vs random: 24W/5L, p = .0002
- a2c_t03 vs ex64_t03: 11W/17L, p = .23
- dir_t03 vs random: 24W/4L, p < .0001
- mlp vs random: 12W/14L, p = .90
