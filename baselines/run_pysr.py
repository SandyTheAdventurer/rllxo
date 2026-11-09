"""Fit PySR (Cranmer, 2023) to one exported task, as context for the GP results (not a head-to-head
competitor). Runs in its own environment (.venv-pysr). Same training data, the benchmark's function set,
PySR's defaults otherwise, a fixed iteration budget, serial and deterministic.
Usage: .venv-pysr/bin/python baselines/run_pysr.py EQ SEED -> baselines/pysr_out/<eq>_<seed>.json"""
import sys, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NITERATIONS = 40


def main(eq, seed):
    from pysr import PySRRegressor
    d = np.load(HERE / "tasks" / f"{eq}_{seed}.npz")
    meta = json.loads((HERE / "tasks" / f"{eq}.json").read_text())
    model = PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=meta["unary"],
                          niterations=NITERATIONS, parallelism="serial", deterministic=True, random_state=seed,
                          progress=False, verbosity=0, temp_equation_file=True)
    t0 = time.time()
    model.fit(d["X"], d["y"], variable_names=meta["names"])
    elapsed = time.time() - t0
    best = model.get_best()
    with np.errstate(all="ignore"):
        pred = float(d["scale"]) * model.predict(d["X_test"])
    yt = d["y_test"]
    test_r2 = float(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)) if np.all(np.isfinite(pred)) else float("nan")
    out = {"spec_id": eq, "seed": seed, "test_r2": test_r2, "complexity": int(best["complexity"]),
           "sympy": str(best["sympy_format"]), "scale": float(d["scale"]), "elapsed_sec": round(elapsed, 2)}
    (HERE / "pysr_out").mkdir(exist_ok=True)
    (HERE / "pysr_out" / f"{eq}_{seed}.json").write_text(json.dumps(out))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
