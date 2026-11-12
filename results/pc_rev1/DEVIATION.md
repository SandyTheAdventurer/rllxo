# Deviations from the revision-1 plans (written after the runs, 2026-11-10)

Plans: EXTRA_ARMS_PLAN.md, PROTOCOL_CHECKS_PLAN.md, ABLATION_NOISE_PLAN.md, PYSR_CONTEXT_PLAN.md and
CIRCUIT_PLAN.md. The arms, equations, seeds, settings and analyses are as planned. What changed:

1. **Machine.** The plans name the Linux laptop with 8 workers. All runs were done instead on the Windows PC
   that ran the frozen held-out test (Intel Core Ultra 9 275HX, 24 cores), with 20 workers, from
   `revision1-ablations` at eb614f0. That commit contains the code of all three revision branches. The
   1,512 partial laptop rows of the extra-arms run and the 150 partial circuit rows were not used, so every
   result comes from the PC.
   - Before the runs, the PC selftest reproduced `selftest_windows.jsonl` exactly.
   - The extra run's 1,890 random runs reproduce the held-out random runs exactly (all final expressions
     identical), so the plan's laptop-vs-PC machine check is replaced by this exact match.
   - The repeated a2c_t03 runs match the held-out runs in 688 of 690 (ablations) and 419 of 420
     (keijzer_orig).
   - Run times come from 20 concurrent workers; the held-out test used 18. Mean random-run time was 5.09 s
     here against 5.27 s in the held-out test.
2. **PySR worker layout.** The plan runs one process per task. Instead, 14 long-lived worker processes
   (baselines/pysr_queue.py) each fit a share of the 630 tasks by calling run_pysr.main(eq, seed), so Julia
   starts once per worker, not once per task. Every fit keeps the plan's settings: PySR 2.7.2, 40
   iterations, serial, deterministic, random_state = seed.
   - After its last task, 10 workers hung at Julia shutdown. All 630 fits were already saved, and those
     workers were stopped.
   - Scoring used baselines/score_pysr_par.py, a parallel copy of score_pysr.py with the same checks and
     limits. The serial scorer was stopped before it finished.
3. **Crashed runs.** Three ablation runs on K3 (spec 103) with seed 13 stopped with
   `RuntimeError('The size of tensor a (19) must match the size of tensor b (18) at non-singleton dimension 1')`
   in the policy update. The affected arms are a2c_t03, a2c_nocritic and a2c_noconst.
   - The same a2c_t03 task ran without error in the held-out test and in keijzer_orig, with an identical
     result.
   - As the plans fix for error rows, these runs count as test R² = 0.
   - The cause is not yet found. A shape mismatch between the stored node features and the semantic-bias
     grid is suspected, triggered by state that a worker process carries over between tasks.
4. **Circuit log error.** The plan does not say how an error of exactly 0 enters the log10. The analysis
   clips errors at 1e-12, but no run reached 0.

Analysis: analysis/analyze_revision.py (output analysis.md) and analysis/pysr_context.py (output
pysr_context.md). Queue and timings: queue_status.txt.

## Later checks (2026-11-12)
- **Crashed runs, not reproduced.** On the laptop, (a2c_nocritic, K3, 13) and (a2c_t03, K3, 13) run without error
  in a fresh process, and also after short K7 and K8 runs in the same process. The cause is still unknown; one
  candidate is that `main._NODE_SCORES` is not cleared between runs in a worker. The rows stay counted as 0.
- **Resistor figure.** `analysis/circuit_figure.py` re-ran (a2c_sem, 1484.89, seed 4) on the laptop; it reproduced
  the circuits.csv row exactly (error, resistors and evaluations) and drew the pruned circuit for the paper.
- **Batch evaluation count.** See results/eval_count/README.md.
