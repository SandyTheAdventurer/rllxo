# GP-SR-RLC: reinforcement-learning-guided crossover for genetic programming

Code, test plans, per-run results and analysis scripts for the paper *Genetic Programming for Symbolic
Regression with Reinforcement Learning-Guided Crossover* (under review). The method was called RL-LXO in
earlier versions; some file and variable names still use that name.

GP-SR-RLC is a subtree crossover operator for tree-based GP. It samples the pair of subtrees to exchange from
a joint distribution over all node pairs of the two parents. The distribution combines an online A2C policy
over node features with a semantic bias H that ranks every candidate child by its exact loss on a 64-point
batch of training data. In the paper's tests the gain over random crossover comes from H: the bias alone
performs as well as the full operator, and the policy alone is no better than random crossover.

## Layout

| Path | Contents |
|---|---|
| `main.py` | the GP and the crossover selectors (A2C policy, pair sampling, selector update) |
| `semx.py` | exact batch losses of every candidate child, and the semantic bias H built on them |
| `compare_selectors.py` | benchmark equations, experiment arms, the runner, exact-recovery and tree-edit-distance checks |
| `nguyen.py` | Nguyen and Keijzer definitions and the GP configuration |
| `circuit_evolve.py`, `circuit_experiment.py` | the resistor-network application (Section 9) |
| `run_queue.py` | the queue that ran all revision experiments on the test PC |
| `analyze_heldout.py`, `analysis/` | the pre-registered analyses, the paper's figures and tables, and later checks |
| `baselines/` | PySR context runs: task export, fitting, scoring |
| `download_data.py`, `dataset_sha256.txt` | SRSD-Feynman download at pinned revisions, with checksums |
| `results/*_PLAN.md` | the test plans, each written and tagged before its runs |
| `results/heldout/` | the frozen held-out test: 11,340 runs (63 equations x 30 seeds x 6 arms) |
| `results/pc_rev1/` | the revision experiments: extra baselines, ablations, noise, original protocols, PySR, resistor networks |
| `results/eval_count/` | the count of batch child evaluations per run (Section 8.4) |
| `climate/` | a separate microclimate application of the same GP, not part of this paper |
| `legacy/` | earlier exploration, not part of the paper's pipeline |

## Where each result comes from

| Paper | Plan | Data | Analysis |
|---|---|---|---|
| Held-out test (Sections 8.1-8.4, Tables 3-5, Figures 3-5) | `results/HELDOUT_PLAN.md` | `results/heldout/heldout.csv`, `heldout_curves.jsonl` | `analyze_heldout.py`, `analysis/per_equation_tests.py`, `analysis/make_figures.py` |
| SSC and policy-only arms (8.2, 8.3) | `results/EXTRA_ARMS_PLAN.md` | `results/pc_rev1/compare_selectors/extra.csv` | `analysis/analyze_revision.py` |
| Ablations and size-fair crossover (8.2, 8.3) | `results/ABLATION_NOISE_PLAN.md` | `.../ablations.csv` | same |
| Noise 1% and 10% (8.5) | same | `.../noise01.csv`, `.../noise10.csv` | same |
| Original protocols (8.6) | `results/PROTOCOL_CHECKS_PLAN.md` | `.../keijzer_orig.csv`, `.../nguyen_orig.csv` | same |
| PySR as context (8.7) | `results/PYSR_CONTEXT_PLAN.md` | `results/pc_rev1/pysr.csv`, `pysr_out/` | `analysis/pysr_context.py` |
| Resistor networks (9) | `results/CIRCUIT_PLAN.md` | `results/pc_rev1/circuits/circuits.csv` | `analysis/analyze_revision.py`, `analysis/circuit_figure.py` |
| Evaluation count (8.4) | none (a measurement) | `results/eval_count/evals_seed0.csv` | `analysis/count_evals.py` |

`results/pc_rev1/analysis.md` and `results/heldout/analysis.md` are the analysis outputs. What changed after
each freeze is recorded in `results/HELDOUT_DEVIATION.md` and `results/pc_rev1/DEVIATION.md`.

## Frozen tags

Each plan was committed and tagged before its runs started, and the runs used the tagged code:
`heldout-frozen`, `extra-arms-frozen`, `protocol-checks-frozen`, `ablation-noise-frozen` and `circuit-frozen`.
`git diff <tag> -- <file>` shows anything that changed afterwards.

## Reproducing

See [REPRODUCE.md](REPRODUCE.md). In short, with Python 3.14:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python download_data.py
python compare_selectors.py --selftest
```
Every run is fixed by its seed. On the same platform and library versions the results are identical; across
platforms, NumPy's transcendental functions can differ in the last bits, so reruns agree statistically rather
than bit for bit. The published runs were made on a Windows 11 PC (Intel Core Ultra 9 275HX).
