# Extra arms for the revision (written before any of these runs)

The held-out test (results/HELDOUT_PLAN.md) is finished and analysed. Reviewer 4 asked for two things
it lacks: a semantic-crossover baseline (R4.1) and an ablation separating the learned policy from H
(R4.2). These arms add comparisons only. Nothing about a2c_t03 or the other held-out arms changes, and
none of their settings is tuned.

## Arms
| Arm | Role |
|---|---|
| ssc | semantic similarity-based crossover (Uy et al., 2011), the baseline reviewer 4 names |
| a2c_noH | a2c_t03 without H: the same policy, warmup 1 and logratio reward, no semantic bias |
| random | the same baseline rerun on this machine, as an anchor between the two machines |

With ex64_t03 (H alone, no learning) and a2c_t03 (policy plus H), a2c_noH completes the three-way
ablation reviewer 4 asks for.

## Run
Same equations, seeds, config and code as the held-out test, plus the a2c_noH entry in ARMS.
This runs on the development laptop (Linux, 8 workers), not the Windows PC:
```
python compare_selectors.py --equations heldout --arms ssc,a2c_noH,random --seeds 30 --seed0 0 --jobs 8 --tag extra
```

## Analysis
Scoring is as in the held-out plan: per-equation median test R² over the 30 seeds, two-sided Wilcoxon
over the 63 equations, ties dropped. The a2c_t03 and ex64_t03 rows come from the held-out run.
- **One Holm-corrected family:**
  - a2c_t03 vs ssc;
  - ssc vs random;
  - a2c_noH vs random;
  - a2c_t03 vs a2c_noH (does H add anything to the policy).
- **Machine check (descriptive):** laptop random vs PC random, as two unpaired sets of runs. They should
  not differ systematically. The tests above use the PC random rows.
- Recovery, solved fraction, time and size are reported descriptively for every arm.
