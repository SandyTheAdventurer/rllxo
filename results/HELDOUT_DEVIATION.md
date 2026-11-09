# Held-out run: what happened after the freeze

The held-out test ran from the tag `heldout-frozen` on a Windows 11 PC (11,340 runs, 18 workers). Its
analysis ran unchanged from that tag. This file records the one departure from the frozen run, and the
code change made afterwards because of it.

## The last 9 runs
11,331 runs finished normally. The last 9 never finished:
- **7 hung** in SymPy during the exact-recovery check, after the GA had finished and test R² was computed.
  The hang is one long C-level integer power (`Integer._eval_power`). Python cannot interrupt a C call, so
  the 30 s alarm in `is_recovered` never fired. On Linux, SIGALRM would not fire either.
- **2 were lost** when a worker process died while the PC ran short of memory.

These 9 were rerun with the frozen `run_task` unchanged, except that `is_recovered` and
`tree_edit_distance` ran in a child process killed after 60 s / 150 s of compute:
- A killed recovery check counts as not recovered. That is what the frozen code returns when its alarm
  fires.
- A killed tree edit distance is NaN; the analysis does not use it.
- The GA and every R² field are as the frozen code computes them. A check run of an already finished run
  (ex64_t03, equation 2039, seed 29) reproduced its CSV row exactly, apart from `elapsed_sec`.

Recovery was killed in 7 runs: a2c_t03 x3, ex64_t03 x2, mlp x1, random_x4 x1. Each of these models is
11-75 edits from the true equation, or NaN, so none is a recovery. The change affects only the recovery
rate, and it can only lower a2c_t03's, since no random run was killed. The primary test (median test R²)
is unaffected.

## Code change after the run
`compare_selectors.py` now runs these two checks the same way, in a helper process with the same hard
limits (`_Checker`). This means no run can hang. The helper uses PYTHONHASHSEED=0, because SymPy's
simplification can depend on the order of hashed arguments. That makes the recovery check repeatable from
one process to the next. Otherwise the change has no effect: the Linux and Windows selftests match their
reference output from the tag.
