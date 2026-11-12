# GP-SR-RLC: running the experiments

This repository recreates every result in *Genetic Programming for Symbolic Regression with Reinforcement
Learning-Guided Crossover* (under review). The method was called RL-LXO in earlier versions, and some names
in the code still use that. The results themselves, with the tables, are in [RESULTS.md](RESULTS.md).

Everything runs on the CPU, one thread per run, on Linux, macOS or Windows.

## 1. Set up
Python 3.14:
```
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python download_data.py
```
`download_data.py` fetches the SRSD-Feynman datasets from Hugging Face at pinned revisions and checks them
against `dataset_sha256.txt`. Nguyen, Keijzer and the development equations are generated in code.

## 2. Check the installation
```
python compare_selectors.py --selftest
```
This makes six short runs and prints one JSON line each. On Linux the lines should match
`results/selftest_linux.jsonl`, and on Windows `results/selftest_windows.jsonl`, apart from `elapsed_sec`.

Every run is fixed by its seed, so on the same platform and library versions a rerun gives identical results.
On another platform, NumPy's `sin`, `cos`, `exp` and `log` can differ in the last bits, so some runs take a
different path and the results agree statistically rather than bit for bit. The published runs were made on a
Windows 11 PC (Intel Core Ultra 9 275HX).

## 3. Held-out test (Sections 8.1-8.4)
The main test: 63 equations (Nguyen, Keijzer and SRSD-Feynman medium) x 30 seeds x 6 methods = 11,340 runs.
Plan: `results/HELDOUT_PLAN.md`, frozen at the tag `heldout-frozen`.
```
python compare_selectors.py --equations heldout --arms random,a2c_t03,ex64_t03,dir_t03,random_x4,mlp \
    --seeds 30 --seed0 0 --jobs <cores> --tag heldout
python analyze_heldout.py results/compare_selectors/heldout.csv
python analysis/per_equation_tests.py results/compare_selectors/heldout.csv
```
- Rows are appended to `results/compare_selectors/heldout.csv` as runs finish. Running the same command
  again resumes an interrupted run.
- The full test takes about 45-125 core-hours. Plan on 1 GB of RAM per worker; a run that passes 3 GB is
  stopped and recorded as an error (on Windows this needs psutil).

## 4. Revision experiments (Sections 8.2-8.7 and 9)
```
python run_queue.py
python analysis/analyze_revision.py results/compare_selectors/heldout.csv results
```
`run_queue.py` runs each revision plan in turn with the exact commands used for the paper (set the worker
count `J` at its top; the paper used 20). On 20 workers it took about 3.7 hours:

| Run | Plan | Time on 20 workers |
|---|---|---|
| Extra baselines: SSC, policy without H | `results/EXTRA_ARMS_PLAN.md` | 60 min |
| Keijzer's original test sets, original Nguyen protocol | `results/PROTOCOL_CHECKS_PLAN.md` | 22 min |
| Ablations and size-fair crossover | `results/ABLATION_NOISE_PLAN.md` | 76 min |
| 1% and 10% noise | `results/ABLATION_NOISE_PLAN.md` | 59 min |
| Resistor networks | `results/CIRCUIT_PLAN.md` | 5 min |

**PySR as context (Section 8.7):** PySR runs in its own environment, which `pysr_setup.py` creates.
```
python baselines/pysr_setup.py          # export the 630 tasks, install PySR 2.7.2 and Julia in .venv-pysr
python baselines/pysr_queue.py 14       # fit them with 14 workers, then score (baselines/score_pysr_par.py)
python analysis/pysr_context.py results/compare_selectors/heldout.csv baselines/pysr.csv
```

**Evaluation count (Section 8.4):**
```
python analysis/count_evals.py results/compare_selectors/heldout.csv evals.csv 0 <jobs>
```

## 5. Figures and tables
```
python analysis/make_figures.py                                  # held-out figures and numbers -> figures/
python analysis/circuit_figure.py a2c_sem 1484.89 4 circuit.pdf  # the resistor circuit (Figure 6)
```
`make_figures.py` reads `results/heldout/`; `circuit_figure.py` reruns that one circuit run and checks it
against `results/pc_rev1/circuits/circuits.csv` before drawing it.

## 6. Analysing the published runs instead
The per-run results behind the paper are in the repository, so every analysis runs without rerunning anything:
```
python analyze_heldout.py results/heldout/heldout.csv
python analysis/analyze_revision.py results/heldout/heldout.csv results/pc_rev1
python analysis/pysr_context.py results/heldout/heldout.csv results/pc_rev1/pysr.csv
```
These print `results/heldout/analysis.md`, `results/pc_rev1/analysis.md` and `results/pc_rev1/pysr_context.md`.
What changed after each plan was frozen is recorded in `results/HELDOUT_DEVIATION.md` and
`results/pc_rev1/DEVIATION.md`.

## Code
| File | Contents |
|---|---|
| `main.py` | the GP and the crossover selectors |
| `semx.py` | exact batch losses of every candidate child, and the semantic bias H |
| `compare_selectors.py` | equations, methods (`ARMS`), the experiment runner and the recovery checks |
| `nguyen.py` | Nguyen and Keijzer definitions and the GP settings |
| `circuit_evolve.py`, `circuit_experiment.py` | the resistor-network application |
| `climate/` | a separate microclimate application of the same GP, not part of this paper |
| `legacy/` | earlier exploration, not used for the paper |

Released under the MIT License (see `LICENSE`).
