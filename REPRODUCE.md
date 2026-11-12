# Reproducing the crossover-selector experiments

These steps work on Linux, macOS and Windows. Everything runs on the CPU, one thread per run.

## 1. Environment
Use Python 3.14:
```
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Data
The SRSD-Feynman datasets (Matsubara et al., 2024) are downloaded from Hugging Face at pinned
revisions, then checked against dataset_sha256.txt:
```
python download_data.py
```
Nguyen, Keijzer and the development equations are generated in code (compare_selectors.py).

## 3. Check the installation
```
python compare_selectors.py --selftest
```
This runs six short development runs and prints one JSON line each.
- **Same platform:** the output should match `results/selftest_windows.jsonl` or
  `results/selftest_linux.jsonl`, apart from `elapsed_sec`.
- **Other platforms:** Numpy's sin, cos, exp and log can differ in the last bits between operating
  systems, so some runs can take a different path there. Results then agree statistically, not bit
  for bit.

## 4. Held-out test (results/HELDOUT_PLAN.md)
```
python compare_selectors.py --equations heldout --arms random,a2c_t03,ex64_t03,dir_t03,random_x4,mlp \
    --seeds 30 --seed0 0 --jobs <number of cores> --tag heldout
python analyze_heldout.py results/compare_selectors/heldout.csv
```
The published rows are in `results/heldout/` (`python analyze_heldout.py results/heldout/heldout.csv`
reproduces `results/heldout/analysis.md`), and `python analysis/make_figures.py` writes the paper's held-out
figures and numbers to `figures/`.
- **Output:** rows are appended to `results/compare_selectors/heldout.csv` as runs finish. An
  interrupted run resumes where it stopped when the same command is run again.
- **Run time:** 11,340 runs take roughly 45-125 core-hours, depending on the CPU.
- **Recovery check:** exact recovery and tree edit distance run in a helper process with a hard time limit,
  so a SymPy simplification that runs away cannot stall a run (results/HELDOUT_DEVIATION.md).
- **Memory:** plan on about 1 GB of RAM per worker. A run that goes over 3 GB is stopped and recorded
  with status `error`. On Windows, enforcing that limit needs psutil.

## 5. Revision experiments
All of them ran on the same PC with 20 workers through `run_queue.py`, which lists the exact commands (one
`compare_selectors.py` call per plan, then `circuit_experiment.py`). Each call resumes if interrupted.
```
python run_queue.py
python analysis/analyze_revision.py results/heldout/heldout.csv results
```
- **PySR context:** `python baselines/pysr_setup.py` exports the tasks and installs PySR 2.7.2 in `.venv-pysr`;
  `python baselines/pysr_queue.py 14` fits the 630 tasks and scores them (`baselines/score_pysr_par.py`);
  `python analysis/pysr_context.py results/heldout/heldout.csv baselines/pysr.csv` gives the context table.
- **Evaluation count:** `python analysis/count_evals.py results/heldout/heldout.csv evals.csv 0 <jobs>`.
- **Resistor figure:** `python analysis/circuit_figure.py a2c_sem 1484.89 4 circuit.pdf` reruns that run, checks
  it against `results/pc_rev1/circuits/circuits.csv` and draws the pruned circuit.
- The published outputs are in `results/pc_rev1/` and `results/eval_count/`, with the deviations from the plans
  in `results/pc_rev1/DEVIATION.md`.

## 6. Development runs
The arms explored during development are in `ARMS` in compare_selectors.py. They ran on
`--equations dev --seed0 1000 --seeds 5`. For example:
```
python compare_selectors.py --equations dev --arms random,a2c_t03,ex64_t03,dir_t03 --seeds 5 --seed0 1000 --tag dev
```

## Files
| File | Contents |
|---|---|
| semx.py | exact child losses for every crossover pair, and the selection and bias built on them |
| main.py | the GA and the A2C crossover selector (`selector_type="sem"` and `sem_bias`) |
| compare_selectors.py | equations, arms, the experiment runner, recovery and tree-edit-distance checks |
| analyze_heldout.py | the pre-registered analysis of the held-out test |
| results/HELDOUT_PLAN.md | the frozen plan |
| results/HELDOUT_DEVIATION.md | what happened after the freeze: the last 9 runs and the recovery-check fix |
| results/*_PLAN.md | the revision plans, each tagged before its runs |
| run_queue.py | the revision run queue |
| analysis/ | revision analysis, figures, evaluation count, PySR context table |
