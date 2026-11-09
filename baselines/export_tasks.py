"""Write the exact training and test data of each held-out task, as run_task builds it, so external systems
(run_pysr.py) fit the same data. Usage: python baselines/export_tasks.py SEEDS -> baselines/tasks/<eq>_<seed>.npz"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_selectors as cs
import nguyen

OUT = Path(__file__).resolve().parent / "tasks"


def main(seeds):
    OUT.mkdir(exist_ok=True)
    specs = cs.all_specs()
    heldout = [e for e in [*range(1, 13), *range(101, 116), *cs.srsd_specs()] if e not in (6, 7, 12, 114)]
    for e in heldout:
        spec = specs[e]
        for seed in range(seeds):
            scale = 1.0
            if "srsd" in spec:
                X, y = cs.load_srsd(spec["srsd"], "train")
                keep = np.random.default_rng(seed + e * 1000).choice(len(y), nguyen.SAMPLE_SIZE, replace=False)
                X, y = X[keep], y[keep]
                Xt, yt = cs.load_srsd(spec["srsd"], "test")
                scale = float(np.std(y)) or 1.0
            else:
                X, y = cs.sample(spec, np.random.default_rng(seed + e * 1000), nguyen.SAMPLE_SIZE)
                Xt, yt = cs.sample(spec, np.random.default_rng(10_000 + seed + e * 1000), nguyen.SAMPLE_SIZE)
            np.savez(OUT / f"{e}_{seed}.npz", X=X, y=y / scale, X_test=Xt, y_test=yt, scale=scale)
        meta = {"names": [v["name"] for v in spec["variables"]], "unary": cs.OPERATOR_SETS[cs.benchmark_of(e)]}
        (OUT / f"{e}.json").write_text(json.dumps(meta))
    print("exported", len(heldout), "equations x", seeds, "seeds")


if __name__ == "__main__":
    main(int(sys.argv[1]))
