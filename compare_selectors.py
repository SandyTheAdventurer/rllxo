"""Crossover-selector comparison on hard Nguyen/Keijzer equations (revision 1 experiments).

Arms (all share the paper's GA config from nguyen.py):
  random     uniform crossover points
  heuristic  pairs sampled from gamma*H alone, no learning
  mlp        per-node MLP selector + H (the paper's method)
  mlp_noH    per-node MLP selector, gamma = 0
  pair       cross-attention PairSelector + H
  pair_noH   cross-attention PairSelector, gamma = 0
  pretrained frozen pair scorer pretrained offline on other equations (pretrain_selector.py)
  pretrained_T05  the same at sampling temperature 0.5

Per run it records train and held-out test R², exact symbolic recovery, expression size,
generations used, wall time, and the per-generation best-R² curve.
Rows are appended as they finish, so an interrupted run resumes where it stopped.

Usage: python compare_selectors.py [--equations 6,7,12,114] [--arms ...] [--seeds 30] [--jobs 12]
"""
from __future__ import annotations
import argparse, csv, json, multiprocessing as mp, signal, time
from pathlib import Path
import numpy as np, sympy as sp, torch

ARMS = {
    "random": {"use_learned_crossover": False},
    "heuristic": {"selector_type": "heuristic"},
    "mlp": {"selector_type": "mlp"},
    "mlp_noH": {"selector_type": "mlp", "heuristic_bias_scale": 0.0},
    "pair": {"selector_type": "pair"},
    "pair_noH": {"selector_type": "pair", "heuristic_bias_scale": 0.0},
    # Frozen scorer from pretrain_selector.py, acting from generation 1 with no H
    "pretrained": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector.pt",
                   "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0},
    "pretrained_T05": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector.pt",
                       "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0, "temperature": 0.5},
    # Same, trained to predict that a child beats the better parent (pretrain_selector.py --label improves)
    "pretrained_imp": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector_improves.pt",
                       "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0},
    "pretrained_imp_T2": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector_improves.pt",
                          "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0, "temperature": 2.0},
}
FIELDS = ["arm", "spec_id", "seed", "status", "train_r2", "test_r2", "recovered", "size", "gens_used",
          "elapsed_sec", "best_expr", "error"]
OUT_DIR = Path("results/compare_selectors")


def all_specs():
    import nguyen
    return {**{s["id"]: s for s in nguyen.NGUYEN_SPECS}, **{s["id"] + 100: s for s in nguyen.KEIJZER_SPECS}}


def sample(spec, rng, n):
    cols = [rng.uniform(*v["sampler"], size=n) for v in spec["variables"]]
    y = np.asarray(spec["func"](*cols), dtype=float).ravel()
    ok = np.isfinite(y)
    return np.column_stack(cols)[ok], y[ok]


def _timeout(*_):
    raise TimeoutError


def is_recovered(model: sp.Expr, target: sp.Expr, seconds=30) -> bool:
    """Exact symbolic recovery: model - target simplifies to 0 once evolved constants are
    snapped to nearby simple rationals (within 1e-3)."""
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        model = model.xreplace({f: sp.nsimplify(f, rational=True, tolerance=1e-3) for f in model.atoms(sp.Float)})
        diff = model - target
        # Merge logs (log(a) - log(b) -> log(a/b)), which plain simplify won't do without sign assumptions
        return sp.simplify(diff) == 0 or sp.simplify(sp.logcombine(sp.expand_log(diff, force=True), force=True)) == 0
    except Exception:
        return False
    finally:
        signal.alarm(0)


def run_task(task):
    arm, spec_id, seed = task
    torch.set_num_threads(1)
    import nguyen
    from main import run_ga
    spec = all_specs()[spec_id]
    row = {"arm": arm, "spec_id": spec_id, "seed": seed, "status": "ok", "error": ""}
    curve = []
    t0 = time.time()
    try:
        # Same training sample as nguyen.py; the test sample is fresh points from the same ranges
        X, y = sample(spec, np.random.default_rng(seed + spec_id * 1000), nguyen.SAMPLE_SIZE)
        X_test, y_test = sample(spec, np.random.default_rng(10_000 + seed + spec_id * 1000), nguyen.SAMPLE_SIZE)
        names = [v["name"] for v in spec["variables"]]
        unary = {n: nguyen._UNARY_REG[n] for n in spec.get("allowed_unary") or [] if n in nguyen._UNARY_REG}
        kw = {**nguyen.ga_config, **nguyen.learned_overrides, **ARMS[arm], "verbose": False, "seed": seed,
              "allow_pow": bool(spec.get("allow_pow", False)),
              "metrics_callback": lambda gen, r2: curve.append(round(r2, 6))}
        best, train_r2 = run_ga(X, y, var_names=names, unary_funcs=unary, **kw)
        pred = best.evaluate({n: X_test[:, i] for i, n in enumerate(names)})
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        row.update(train_r2=float(train_r2), size=best.size(), gens_used=len(curve), best_expr=best.to_string(),
                   test_r2=float(1 - np.sum((y_test - pred) ** 2) / ss_tot) if np.all(np.isfinite(pred)) else float("nan"))
        row["elapsed_sec"] = round(time.time() - t0, 2)  # GA time only, excludes the recovery check
        row["recovered"] = int(is_recovered(best.to_sympy(), spec["sympy"]))
    except Exception as exc:
        row.update(status="error", error=repr(exc), elapsed_sec=round(time.time() - t0, 2))
    return row, curve


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--equations", default="6,7,12,114", help="spec ids; Keijzer ids are offset by 100")
    pa.add_argument("--arms", default=",".join(ARMS))
    pa.add_argument("--seeds", type=int, default=30)
    pa.add_argument("--jobs", type=int, default=8)
    pa.add_argument("--tag", default="main", help="output file name prefix")
    args = pa.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv, out_curves = OUT_DIR / f"{args.tag}.csv", OUT_DIR / f"{args.tag}_curves.jsonl"
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["arm"], int(r["spec_id"]), int(r["seed"])) for r in csv.DictReader(f)}
    tasks = [(a, int(e), s) for s in range(args.seeds) for e in args.equations.split(",")
             for a in args.arms.split(",") if (a, int(e), s) not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} already done), {args.jobs} workers", flush=True)

    new_file = not out_csv.exists()
    t0 = time.time()
    with open(out_csv, "a", newline="") as fc, open(out_curves, "a") as fj, \
            mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=4) as pool:
        writer = csv.DictWriter(fc, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, (row, curve) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            writer.writerow(row)
            fc.flush()
            fj.write(json.dumps({"arm": row["arm"], "spec_id": row["spec_id"], "seed": row["seed"], "r2": curve}) + "\n")
            fj.flush()
            eta = (time.time() - t0) / k * (len(tasks) - k)
            print(f"[{k}/{len(tasks)}] {row['arm']:9s} eq={row['spec_id']:3d} seed={row['seed']:2d} {row['status']} "
                  f"r2={row.get('train_r2', float('nan')):.4f} rec={row.get('recovered', '-')} "
                  f"{row['elapsed_sec']:.0f}s  eta {eta / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
