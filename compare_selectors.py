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
  pretrained_imp(_T2)  scorer trained to predict a child beating the better parent (temperature 1 / 2)
  imp_mix50/25    that scorer picks the points of 50% / 25% of crossovers, the rest uniform
  ssc        semantic similarity-based crossover (Uy et al., 2011)
  imp_sem    improvement scorer with the swapped subtrees' semantic distance
  no_crossover, *_m01  leverage checks (mutation only; mutation rate 0.1)
  greedy8/32 upper bound: best of 8 / 32 uniform point pairs by child fitness
  mlp_rank(_novel)  mlp rewarded by the child's population rank (zero for semantically neutral swaps)

Every equation in a benchmark gets that benchmark's standard function set (OPERATOR_SETS).
Per run it records train and held-out test R², exact symbolic recovery, tree edit distance to the
true equation (TED, and SRSD's normalized NED), expression size,
generations used, wall time, and the per-generation best-R² curve.
Rows are appended as they finish, so an interrupted run resumes where it stopped.

Equations: Nguyen 1-12, Keijzer 101-115, SRSD-Feynman medium 2000-2039 (dataset/srsd_medium).

Usage: python compare_selectors.py [--equations 6,7,12,114|nguyen,keijzer,srsd] [--arms ...] [--seeds 30] [--jobs 12]
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
    # The improvement scorer picks the points of only a share of crossovers; the rest stay uniform
    "imp_mix50": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector_improves.pt",
                  "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0, "selector_mix": 0.5},
    "imp_mix25": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector_improves.pt",
                  "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0, "selector_mix": 0.25},
    # Semantic similarity-based crossover (Uy et al., 2011), the semantic-crossover baseline
    "ssc": {"selector_type": "ssc", "selector_warmup_gens": 1},
    # Improvement scorer that also sees the swapped subtrees' semantic distance
    "imp_sem": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector_improves_sem.pt",
                "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0},
    # Leverage checks: no crossover at all (mutation only), and a standard 0.1 mutation rate
    "no_crossover": {"use_learned_crossover": False, "crossover_rate": 0.0},
    # MLP selector rewarded by the best child's rank in the population instead of its gain over
    # the parents, optionally with no reward for semantically neutral swaps
    "mlp_rank": {"selector_type": "mlp", "selector_reward": "rank"},
    "mlp_rank_novel": {"selector_type": "mlp", "selector_reward": "rank_novel"},
    # Upper bound: score the children of k uniform point pairs and keep the best (uses fitness evaluations)
    "greedy8": {"selector_type": "greedy", "selector_warmup_gens": 1, "greedy_trials": 8},
    "greedy32": {"selector_type": "greedy", "selector_warmup_gens": 1, "greedy_trials": 32},
}
for _arm in ["random", "mlp", "ssc", "imp_sem"]:
    ARMS[f"{_arm}_m01"] = {**ARMS[_arm], "mutation_rate": 0.1}
FIELDS = ["arm", "spec_id", "seed", "status", "train_r2", "test_r2", "recovered", "ted", "ned", "size",
          "gens_used", "elapsed_sec", "best_expr", "error"]
OUT_DIR = Path("results/compare_selectors")


SRSD_DIR = Path("dataset/srsd_medium")
# Each benchmark's standard function set, the same for every equation in it: Nguyen's is Uy et al.'s
# (2011) {+, -, *, /, sin, cos, exp, log}; Keijzer and SRSD add sqrt. No pow anywhere
NGUYEN_UNARY = ["sin", "cos", "exp", "log"]
OPERATOR_SETS = {"nguyen": NGUYEN_UNARY, "keijzer": NGUYEN_UNARY + ["sqrt"], "srsd": NGUYEN_UNARY + ["sqrt"]}


def benchmark_of(spec_id):
    return "nguyen" if spec_id < 100 else "keijzer" if spec_id < 1000 else "srsd"


SRSD_OFFSET = 2000  # SRSD-Feynman medium equation k (sorted by name) has spec id 2000 + k


def srsd_specs():
    """SRSD-Feynman medium (Matsubara et al., 2024) from dataset/srsd_medium; the true equation comes
    from supp_info.json (not the pickles)."""
    info = SRSD_DIR / "supp_info.json"
    if not info.exists():
        return {}
    specs = {}
    for k, (name, v) in enumerate(sorted(json.loads(info.read_text()).items())):
        target = sp.sympify(v["sympy_eq_str"])
        n_vars = max(int(str(x)[1:]) for x in target.free_symbols) + 1
        specs[SRSD_OFFSET + k] = {"id": SRSD_OFFSET + k, "name": name, "sympy": target, "srsd": name,
                                  "variables": [{"name": f"x{i}"} for i in range(n_vars)]}
    return specs


def all_specs():
    import nguyen
    return {**{s["id"]: s for s in nguyen.NGUYEN_SPECS}, **{s["id"] + 100: s for s in nguyen.KEIJZER_SPECS},
            **srsd_specs()}


def load_srsd(name, split):
    data = np.loadtxt(SRSD_DIR / split / f"{name}.txt")
    return data[:, :-1], data[:, -1]


def sample(spec, rng, n):
    cols = [rng.uniform(*v["sampler"], size=n) for v in spec["variables"]]
    y = np.asarray(spec["func"](*cols), dtype=float).ravel()
    ok = np.isfinite(y)
    return np.column_stack(cols)[ok], y[ok]


def _timeout(*_):
    raise TimeoutError


# Small rationals times a nonzero power of e or pi (e.g. exp(-1), pi/2), either sign: function sets
# with exp and log produce these, which no nearby rational can stand for
_SPECIAL = [sign * q * base ** k for q in (1, 2, 3, 4, sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4))
            for base in (sp.E, sp.pi) for k in (-2, -1, 1, 2) for sign in (1, -1)]


def snap_constants(model: sp.Expr, target: sp.Expr, physical=False) -> sp.Expr:
    """Model with real-valued variables and evolved constants snapped: within 0.1% of a float
    constant of the target to that, within 0.01% of a small rational times a power of e or pi to
    that, else to a nearby simple rational (within 1e-3). With
    physical (SRSD), constants within 0.1% of one of the target's constants, possibly times a power
    of pi, snap to that constant first."""
    model = model.xreplace({x: sp.Symbol(x.name, real=True) for x in model.free_symbols})
    if physical:
        model = sp.expand_mul(model)
        consts = [sign * c * sp.pi ** k for c in [*target.atoms(sp.Float), sp.Integer(1)]
                  for k in range(-2, 3) for sign in (1, -1)]
        return model.xreplace({f: snap_constant(f, consts, 1e-3) for f in model.atoms(sp.Float)})
    # Float constants in the target itself (Keijzer-6's Euler-Mascheroni constant) come first
    own = [sign * c for c in target.atoms(sp.Float) for sign in (1, -1)]
    return model.xreplace({f: snap_constant(f, own, 1e-3) if any(abs(float(f) - float(c)) <= 1e-3 * abs(float(c))
                                                                for c in own) else snap_constant(f, _SPECIAL, 1e-4)
                           for f in model.atoms(sp.Float)})


def snap_constant(f: sp.Float, candidates, rel_tol) -> sp.Expr:
    """The first candidate within rel_tol (relative) of f, else the nearby simple rational (within 1e-3)."""
    for c in candidates:
        if abs(float(f) - float(c)) <= rel_tol * abs(float(c)):
            return c
    return sp.nsimplify(f, rational=True, tolerance=1e-3)


def is_recovered(model: sp.Expr, target: sp.Expr, seconds=30, physical=False) -> bool:
    """Exact symbolic recovery: model - target simplifies to 0 once evolved constants are
    snapped (snap_constants)."""
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        model = snap_constants(model, target, physical)
        target = target.xreplace({x: sp.Symbol(x.name, real=True) for x in target.free_symbols})
        diff = model - target
        # Merge logs (log(a) - log(b) -> log(a/b)), which plain simplify won't do without sign assumptions
        return sp.simplify(diff) == 0 or sp.simplify(sp.logcombine(sp.expand_log(diff, force=True), force=True)) == 0
    except Exception:
        return False
    finally:
        signal.alarm(0)


def _zss_tree(expr: sp.Expr):
    """zss tree of a sympy expression: operator class names, variable names, numbers as 'const'."""
    import zss
    if expr.is_Number or expr.is_NumberSymbol:
        return zss.Node("const")
    if expr.is_Symbol:
        return zss.Node(str(expr))
    node = zss.Node(type(expr).__name__)
    for arg in expr.args:
        node.addkid(_zss_tree(arg))
    return node


def tree_edit_distance(model: sp.Expr, target: sp.Expr, seconds=30, physical=False):
    """(TED, NED): Zhang-Shasha tree edit distance between the model (constants snapped as for
    recovery) and the true equation, both simplified so they share a canonical form, and
    NED = min(1, TED / target tree size) as in SRSD (Matsubara et al., 2024). An expression is left
    unsimplified if simplify times out."""
    import zss
    exprs = []
    # The target goes through the same snapping (which expands products) so both share a form
    for e in (snap_constants(model, target, physical), snap_constants(target, target, physical)):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(seconds)
        try:
            e = sp.simplify(e)
        except Exception:
            pass
        finally:
            signal.alarm(0)
        exprs.append(e)
    ted = int(zss.simple_distance(_zss_tree(exprs[0]), _zss_tree(exprs[1])))
    return ted, float(min(1.0, ted / sum(1 for _ in sp.preorder_traversal(exprs[1]))))


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
        scale = 1.0
        if "srsd" in spec:
            # SAMPLE_SIZE rows of the SRSD train split, scored on its test split. y is divided by its
            # standard deviation for the GA (outputs range from 1e-80 to 1e90); predictions are rescaled
            X, y = load_srsd(spec["srsd"], "train")
            keep = np.random.default_rng(seed + spec_id * 1000).choice(len(y), nguyen.SAMPLE_SIZE, replace=False)
            X, y = X[keep], y[keep]
            X_test, y_test = load_srsd(spec["srsd"], "test")
            scale = float(np.std(y)) or 1.0
            y = y / scale
        else:
            # Same training sample as nguyen.py; the test sample is fresh points from the same ranges
            X, y = sample(spec, np.random.default_rng(seed + spec_id * 1000), nguyen.SAMPLE_SIZE)
            X_test, y_test = sample(spec, np.random.default_rng(10_000 + seed + spec_id * 1000), nguyen.SAMPLE_SIZE)
        names = [v["name"] for v in spec["variables"]]
        unary = {n: nguyen._UNARY_REG[n] for n in OPERATOR_SETS[benchmark_of(spec_id)]}
        kw = {**nguyen.ga_config, **nguyen.learned_overrides,
              **ARMS[arm], "verbose": False, "seed": seed,
              "allow_pow": False,
              "metrics_callback": lambda gen, r2: curve.append(round(r2, 6))}
        best, train_r2 = run_ga(X, y, var_names=names, unary_funcs=unary, **kw)
        with np.errstate(all="ignore"):
            pred = scale * best.evaluate({n: X_test[:, i] for i, n in enumerate(names)})
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        row.update(train_r2=float("nan") if train_r2 is None else float(train_r2), size=best.size(), gens_used=len(curve), best_expr=best.to_string(),
                   test_r2=float(1 - np.sum((y_test - pred) ** 2) / ss_tot) if np.all(np.isfinite(pred)) else float("nan"))
        row["elapsed_sec"] = round(time.time() - t0, 2)  # GA time only, excludes the recovery check
        model = scale * best.to_sympy()
        row["recovered"] = int(is_recovered(model, spec["sympy"], physical="srsd" in spec))
        # A recovered model is the true equation, whatever form simplify leaves it in
        row["ted"], row["ned"] = (0, 0.0) if row["recovered"] else \
            tree_edit_distance(model, spec["sympy"], physical="srsd" in spec)
    except Exception as exc:
        row.update(status="error", error=repr(exc), elapsed_sec=round(time.time() - t0, 2))
    return row, curve


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--equations", default="6,7,12,114",
                    help="spec ids (Keijzer offset by 100) or the groups nguyen, keijzer, srsd")
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
    groups = {"nguyen": [str(k) for k in range(1, 13)], "keijzer": [str(k) for k in range(101, 116)],
              "srsd": [str(k) for k in srsd_specs()]}
    equations = [e for tok in args.equations.split(",") for e in groups.get(tok, [tok])]
    tasks = [(a, int(e), s) for s in range(args.seeds) for e in equations
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
