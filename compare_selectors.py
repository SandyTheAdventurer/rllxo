"""Crossover-selector comparison on hard Nguyen/Keijzer equations (revision 1 experiments).

Arms (all share the paper's GA config from nguyen.py):
  random     uniform crossover points
  heuristic  pairs sampled from gamma*H alone, no learning
  mlp        per-node MLP selector + H (the paper's method)
  mlp_noH    per-node MLP selector, gamma = 0
  pair       cross-attention PairSelector + H
  pair_noH   cross-attention PairSelector, gamma = 0
  pretrained frozen pair scorer pretrained offline on other equations (legacy/pretrain_selector.py)
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

Usage: python compare_selectors.py [--equations 6,7,12,114|nguyen,keijzer,srsd|dev|heldout] [--arms ...] [--seeds 30] [--jobs 12]
       python compare_selectors.py --selftest    (six short dev runs printed as JSON, to compare machines)

Runs on Linux, macOS and Windows; see README.md.
"""
from __future__ import annotations
import os
# One thread per run, CPU only. Set before numpy and torch load so every platform computes the same way;
# variables already set in the environment win
for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "TQDM_DISABLE": "1", "RLLXO_SELECTOR_DEVICE": "cpu"}.items():
    os.environ.setdefault(_k, _v)
import argparse, atexit, csv, ctypes, json, multiprocessing as mp, pickle, signal, subprocess, sys, threading, time
from pathlib import Path
import numpy as np, sympy as sp  # torch loads in run_task, so the recovery-check helper stays small

ARMS = {
    "random": {"use_learned_crossover": False},
    "heuristic": {"selector_type": "heuristic"},
    "mlp": {"selector_type": "mlp"},
    "mlp_noH": {"selector_type": "mlp", "heuristic_bias_scale": 0.0},
    "pair": {"selector_type": "pair"},
    "pair_noH": {"selector_type": "pair", "heuristic_bias_scale": 0.0},
    # Frozen scorer from legacy/pretrain_selector.py, acting from generation 1 with no H
    "pretrained": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector.pt",
                   "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0},
    "pretrained_T05": {"selector_type": "pretrained", "selector_path": "results/pretrain/selector.pt",
                       "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0, "temperature": 0.5},
    # Same, trained to predict that a child beats the better parent (legacy/pretrain_selector.py --label improves)
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
_S = {"selector_type": "sem", "selector_warmup_gens": 1}
_B = {"selector_type": "mlp", "selector_warmup_gens": 1, "sem_points": 64, "sem_bias": True, "heuristic_bias_scale": 0.0}
ARMS.update({
    # dev1
    "koza": {"use_learned_crossover": False, "internal_node_prob": 0.9},
    "noop": {**_S, "sem_mode": "none", "sem_points": 128},
    "ex_t1": {**_S, "sem_mode": "exact", "sem_temp": 0.1, "sem_points": 128},
    "ex_t03": {**_S, "sem_mode": "exact", "sem_temp": 0.03, "sem_points": 128},
    "fo_t1": {**_S, "sem_mode": "fo", "sem_temp": 0.1, "sem_points": 128},
    # dev2
    "random_x2": {"use_learned_crossover": False, "gens": 400},
    "random_x4": {"use_learned_crossover": False, "gens": 800},
    "random_p4": {"use_learned_crossover": False, "pop_size": 1200},
    "ex64_t1": {**_S, "sem_mode": "exact", "sem_temp": 0.1, "sem_points": 64},
    "ex64_t03": {**_S, "sem_mode": "exact", "sem_temp": 0.03, "sem_points": 64},
    "ex64_t01": {**_S, "sem_mode": "exact", "sem_temp": 0.01, "sem_points": 64},
    "dir_t1": {**_S, "sem_mode": "exact", "sem_temp": 0.1, "sem_directed": True, "sem_points": 64},
    "dir_t03": {**_S, "sem_mode": "exact", "sem_temp": 0.03, "sem_directed": True, "sem_points": 64},
    "tabu_t03": {**_S, "sem_mode": "exact", "sem_temp": 0.03, "sem_tabu": True, "sem_points": 64},
    "dirtabu_t03": {**_S, "sem_mode": "exact", "sem_temp": 0.03, "sem_directed": True, "sem_tabu": True, "sem_points": 64},
    "mix50_t03": {**_S, "sem_mode": "exact", "sem_temp": 0.03, "sem_mix": 0.5, "sem_points": 64},
    "fo64_t03": {**_S, "sem_mode": "fo", "sem_temp": 0.03, "sem_points": 64},
    "a2c_t03": {**_B, "sem_mode": "exact", "sem_temp": 0.03, "selector_reward": "logratio"},
    "a2c_t03_fast": {**_B, "sem_mode": "exact", "sem_temp": 0.03, "selector_reward": "logratio", "selector_lr": 3e-3, "entropy_weight": 0.01},
    "a2c_t1_fast": {**_B, "sem_mode": "exact", "sem_temp": 0.1, "selector_reward": "logratio", "selector_lr": 3e-3, "entropy_weight": 0.01},
    "mlp_fast": {"selector_type": "mlp", "selector_warmup_gens": 1, "selector_reward": "logratio", "selector_lr": 3e-3, "entropy_weight": 0.01},
    # Ablation of a2c_t03 for the revision (results/EXTRA_ARMS_PLAN.md): the same policy, warmup and reward, no H
    "a2c_noH": {"selector_type": "mlp", "selector_warmup_gens": 1, "heuristic_bias_scale": 0.0, "selector_reward": "logratio"},
})
# Reviewer 5's ablations (results/ABLATION_NOISE_PLAN.md): a2c_t03 without its critic; a2c_t03 and random
# without constant optimization, and at mutation rate 0.1, where crossover dominates the search; and
# size-fair crossover (Langdon, 2000) as a second informed baseline
ARMS.update({
    "a2c_nocritic": {**ARMS["a2c_t03"], "use_critic": False},
    "a2c_noconst": {**ARMS["a2c_t03"], "opt_constants": False},
    "random_noconst": {**ARMS["random"], "opt_constants": False},
    "a2c_m01": {**ARMS["a2c_t03"], "mutation_rate": 0.1},
    "sizefair": {"selector_type": "sizefair", "selector_warmup_gens": 1},
})
for _arm in ["random", "mlp", "ssc", "imp_sem"]:  # random_m01 is reviewer 5's low-mutation baseline
    ARMS[f"{_arm}_m01"] = {**ARMS[_arm], "mutation_rate": 0.1}
FIELDS = ["arm", "spec_id", "seed", "status", "train_r2", "test_r2", "recovered", "ted", "ned", "size",
          "gens_used", "elapsed_sec", "best_expr", "error", "test_r2_orig"]
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results/compare_selectors"


SRSD_DIR = ROOT / "dataset/srsd_medium"
# Each benchmark's standard function set, the same for every equation in it: Nguyen's is Uy et al.'s
# (2011) {+, -, *, /, sin, cos, exp, log}; Keijzer and SRSD add sqrt. No pow anywhere
NGUYEN_UNARY = ["sin", "cos", "exp", "log"]
OPERATOR_SETS = {"nguyen": NGUYEN_UNARY, "keijzer": NGUYEN_UNARY + ["sqrt"], "srsd": NGUYEN_UNARY + ["sqrt"]}


def benchmark_of(spec_id):
    return "nguyen" if spec_id < 100 else "keijzer" if spec_id < 1000 else "srsd"


# The original Nguyen protocol (Uy et al., 2011; as tabulated by Petersen et al., 2021): 20 training points
# from these ranges. compare_selectors normally uses 512 points from the wider ranges in nguyen.py.
# --protocol orig switches to these for Nguyen; the test sample is 512 fresh points from the same ranges
NGUYEN_ORIG_RANGES = {**{k: [(-1.0, 1.0)] for k in range(1, 7)}, 7: [(0.0, 2.0)], 8: [(0.0, 4.0)],
                      **{k: [(0.0, 1.0), (0.0, 1.0)] for k in range(9, 13)}}
NGUYEN_ORIG_POINTS = 20


def _grid(*axes):
    """Every combination of the given 1-D point sets, one column per variable."""
    return np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])


def _even(a, b, step):
    return np.linspace(a, b, int(round((b - a) / step)) + 1)


def keijzer_test_set(k):
    """Keijzer's (2003) test set for Keijzer-k: mostly a dense grid over the training range; K4 is shifted
    and K6 extends from x <= 50 to x <= 120, so those two extrapolate."""
    E = _even
    if k in (1, 2, 3):
        return _grid(E(-k, k, 0.001))
    if k == 4:
        return _grid(E(0.05, 10.05, 0.05))
    if k == 5:
        rng = np.random.default_rng(5)
        return np.column_stack([rng.uniform(-1, 1, 10000), rng.uniform(1, 2, 10000), rng.uniform(-1, 1, 10000)])
    if k == 6:
        return _grid(E(1, 120, 1))
    if k == 7:
        return _grid(E(1, 100, 0.1))
    if k in (8, 9):
        return _grid(E(0, 100, 0.1))
    if k == 10:
        return _grid(E(0, 1, 0.01), E(0, 1, 0.01))
    return _grid(E(-3, 3, 0.01), E(-3, 3, 0.01))  # 11-15


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


def dev_specs():
    """Fresh tuning equations outside every test set: Jin, Neat and SRSD-easy formulas on log-uniform inputs."""
    out = {}
    jin = {1: "2.5*x**4 - 1.3*x**3 + 0.5*y**2 - 1.7*y", 2: "8.0*x**2 + 8.0*y**3 - 15.0",
           3: "0.2*x**3 + 0.5*y**3 - 1.2*y - 0.5*x", 4: "1.5*exp(x) + 5.0*cos(y)", 5: "6.0*sin(x)*cos(y)",
           6: "1.35*x*y + 5.5*sin((x - 1.0)*(y - 1.0))"}
    for k, e in jin.items():
        out[200 + k] = _dev(200 + k, e, {"x": (-3, 3), "y": (-3, 3)})
    neat = {1: ("x**4 + 3*x**3 + x**2 + x", {"x": (-1, 1)}), 2: ("x**5 + x**4 + x**3 + x**2 + x", {"x": (-1, 1)}),
            3: ("sin(x**2)*cos(x) - 1", {"x": (-1, 1)}), 4: ("log(x + 1) + log(x**2 + 1)", {"x": (0, 2)}),
            5: ("2*sin(x)*cos(y)", {"x": (0, 2), "y": (0, 2)}), 7: ("2 - 2.1*cos(9.8*x)*sin(1.3*y)", {"x": (-5, 5), "y": (-5, 5)}),
            8: ("exp(-(x - 2)**2) / (1.2 + (y - 2.5)**2)", {"x": (0.3, 4), "y": (0.3, 4)}),
            9: ("1/(1 + x**(-4)) + 1/(1 + y**(-4))", {"x": (-5, 5), "y": (-5, 5)})}
    for k, (e, r) in neat.items():
        out[300 + k] = _dev(300 + k, e, r)
    info = json.loads((ROOT / "dataset/srsd_easy/supp_info.json").read_text())
    names = ["feynman-i.18.12", "feynman-i.18.16", "feynman-i.26.2", "feynman-i.27.6", "feynman-i.30.5", "feynman-i.47.23",
             "feynman-ii.2.42", "feynman-ii.15.4", "feynman-ii.38.14", "feynman-ii.10.9", "feynman-iii.15.27", "feynman-ii.34.11"]
    for k, n in enumerate(names):
        e = info[n]["sympy_eq_str"]
        syms = sorted(sp.sympify(e).free_symbols, key=lambda s: int(str(s)[1:]))
        out[401 + k] = _dev(401 + k, e, {str(s): (0.1, 10.0) for s in syms}, log=True)
    return out


def _dev(i, expr, ranges, log=False):
    target = sp.sympify(expr)
    syms = [sp.Symbol(n) for n in ranges]
    f = sp.lambdify(syms, target, "numpy")
    return {"id": i, "name": expr, "sympy": target, "func": lambda *c: np.asarray(f(*c), float) * np.ones_like(c[0]),
            "variables": [{"name": n, "sampler": r, "log": log} for n, r in ranges.items()], "dev_log": log}


def all_specs():
    import nguyen
    return {**{s["id"]: s for s in nguyen.NGUYEN_SPECS}, **{s["id"] + 100: s for s in nguyen.KEIJZER_SPECS},
            **srsd_specs(), **dev_specs()}


def load_srsd(name, split):
    data = np.loadtxt(SRSD_DIR / split / f"{name}.txt")
    return data[:, :-1], data[:, -1]


def sample(spec, rng, n):
    cols = [np.exp(rng.uniform(np.log(v["sampler"][0]), np.log(v["sampler"][1]), size=n)) if v.get("log")
            else rng.uniform(*v["sampler"], size=n) for v in spec["variables"]]
    y = np.asarray(spec["func"](*cols), dtype=float).ravel()
    ok = np.isfinite(y)
    return np.column_stack(cols)[ok], y[ok]


def _timeout(*_):
    raise TimeoutError


_alarm_state = {"timer": None, "gen": 0, "lock": threading.Lock()}


def _async_raise(exc):
    """Raise exc in the main thread at its next bytecode boundary."""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(threading.main_thread().ident), ctypes.py_object(exc))


def _alarm(seconds):
    """Raise TimeoutError in the main thread after `seconds` (0 cancels), like signal.alarm with
    _timeout as the handler. Windows has no SIGALRM, so a timer thread raises it instead; like a
    signal handler, that only interrupts between bytecodes."""
    if hasattr(signal, "SIGALRM"):
        if seconds:
            signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(seconds)
        return
    st = _alarm_state

    def fire(gen):
        with st["lock"]:
            if st["gen"] == gen:  # neither cancelled nor replaced since it was armed
                st["timer"] = None
                _async_raise(TimeoutError)

    with st["lock"]:
        if st["timer"] is not None:
            st["timer"].cancel()
            st["timer"] = None
        st["gen"] += 1
        if seconds:
            st["timer"] = threading.Timer(seconds, fire, args=(st["gen"],))
            st["timer"].daemon = True
            st["timer"].start()


def _limit_memory(nbytes):
    """Make a run that allocates more than nbytes fail on its own (status error) instead of pushing
    the machine into swap: RLIMIT_DATA where it exists, else a watchdog thread on the resident set
    (needs psutil; without it there is no limit)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_DATA, (nbytes, nbytes))
        return
    except ImportError:
        pass
    try:
        import psutil
    except ImportError:
        return
    if getattr(_limit_memory, "started", False):
        return
    _limit_memory.started = True

    def watch():
        proc = psutil.Process()
        while True:
            time.sleep(0.5)
            if proc.memory_info().rss > nbytes:
                _async_raise(MemoryError)
                time.sleep(5)

    threading.Thread(target=watch, daemon=True).start()


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
    _alarm(seconds)
    try:
        model = snap_constants(model, target, physical)
        target = target.xreplace({x: sp.Symbol(x.name, real=True) for x in target.free_symbols})
        diff = model - target
        # Merge logs (log(a) - log(b) -> log(a/b)), which plain simplify won't do without sign assumptions
        return sp.simplify(diff) == 0 or sp.simplify(sp.logcombine(sp.expand_log(diff, force=True), force=True)) == 0
    except Exception:
        return False
    finally:
        _alarm(0)


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
        _alarm(seconds)
        try:
            e = sp.simplify(e)
        except Exception:
            pass
        finally:
            _alarm(0)
        exprs.append(e)
    ted = int(zss.simple_distance(_zss_tree(exprs[0]), _zss_tree(exprs[1])))
    return ted, float(min(1.0, ted / sum(1 for _ in sp.preorder_traversal(exprs[1]))))



class _Checker:
    """is_recovered and tree_edit_distance in a helper process, killed after a hard limit.

    SymPy can get stuck in one long C call (a huge integer power), which no in-process alarm can interrupt:
    7 of the 11,340 held-out runs hung there (results/HELDOUT_DEVIATION.md). The helper starts once per
    worker and is reused. A killed recovery check counts as not recovered, as when the 30 s alarm fires; a
    killed TED is NaN. The limits count from the start of each call, not the helper's start-up. The helper
    runs with PYTHONHASHSEED=0 because SymPy's results can depend on the order of hashed arguments.
    """
    LIMITS = {"is_recovered": (60, False), "tree_edit_distance": (150, (float("nan"), float("nan")))}
    proc = None

    @classmethod
    def call(cls, name, *args, **kw):
        limit, default = cls.LIMITS[name]
        if cls.proc is None or cls.proc.poll() is not None:
            cls.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--checker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env={**os.environ, "PYTHONHASHSEED": "0"})
            atexit.register(cls.proc.kill)
            if cls._recv(600) != ("ok", "ready"):
                raise RuntimeError("the recovery-check helper did not start")
        cls._send(cls.proc.stdin, (name, args, kw))
        res = cls._recv(limit)
        if res is None:  # over the limit, or the helper died
            cls.proc.kill()
            cls.proc.wait()
            cls.proc = None
            return default
        if res[0] == "exc":
            raise res[1]
        return res[1]

    @staticmethod
    def _send(f, obj):
        b = pickle.dumps(obj)
        f.write(len(b).to_bytes(8, "little") + b)
        f.flush()

    @staticmethod
    def _read(f):
        h = f.read(8)
        if len(h) < 8:
            raise EOFError
        return pickle.loads(f.read(int.from_bytes(h, "little")))

    @classmethod
    def _recv(cls, timeout):
        """The helper's next reply, or None if none arrives within timeout seconds."""
        box = []

        def read(f=cls.proc.stdout):
            try:
                box.append(cls._read(f))
            except Exception:
                pass
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        return box[0] if box else None


def _checker_main():
    """The helper's loop: read (name, args, kwargs), reply ("ok", result) or ("exc", exception)."""
    _limit_memory(3 << 30)
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # keep stray prints off the reply pipe
    _Checker._send(out, ("ok", "ready"))
    funcs = {"is_recovered": is_recovered, "tree_edit_distance": tree_edit_distance}
    while True:
        try:
            name, args, kw = _Checker._read(inp)
        except EOFError:
            return
        try:
            res = ("ok", funcs[name](*args, **kw))
        except Exception as exc:
            res = ("exc", exc)
        try:
            _Checker._send(out, res)
        except Exception:
            _Checker._send(out, ("exc", RuntimeError(repr(res[1]))))


def run_task(task):
    arm, spec_id, seed = task
    import torch
    torch.set_num_threads(1)
    _limit_memory(3 << 30)
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
            n_train = nguyen.SAMPLE_SIZE
            if os.environ.get("RLLXO_PROTOCOL") == "orig" and benchmark_of(spec_id) == "nguyen":
                spec = {**spec, "variables": [{**v, "sampler": r} for v, r in zip(spec["variables"], NGUYEN_ORIG_RANGES[spec_id])]}
                n_train = NGUYEN_ORIG_POINTS
            # Same training sample as nguyen.py; the test sample is fresh points from the same ranges
            X, y = sample(spec, np.random.default_rng(seed + spec_id * 1000), n_train)
            noise = float(os.environ.get("RLLXO_NOISE", "0"))
            if noise:
                # Gaussian noise on the training targets only, sd = noise * RMS(y) as in SRBench; the test set stays clean
                y = y + np.random.default_rng(20_000 + seed + spec_id * 1000).normal(0.0, noise * np.sqrt(np.mean(y ** 2)), len(y))
            X_test, y_test = sample(spec, np.random.default_rng(10_000 + seed + spec_id * 1000), nguyen.SAMPLE_SIZE)
            if spec.get("dev_log"):
                scale = float(np.std(y)) or 1.0
                y = y / scale
        names = [v["name"] for v in spec["variables"]]
        unary = {n: nguyen._UNARY_REG[n] for n in OPERATOR_SETS[benchmark_of(spec_id)]}
        kw = {**nguyen.ga_config, **nguyen.learned_overrides,
              **ARMS[arm], "verbose": False, "seed": seed,
              "allow_pow": False,
              "metrics_callback": lambda gen, r2: curve.append(round(r2, 6))}
        import semx
        semx.configure(**{k: kw.pop(k) for k in list(kw) if k.startswith("sem_")})
        best, train_r2 = run_ga(X, y, var_names=names, unary_funcs=unary, **kw)
        with np.errstate(all="ignore"):
            pred = scale * best.evaluate({n: X_test[:, i] for i, n in enumerate(names)})
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        row.update(train_r2=float("nan") if train_r2 is None else float(train_r2), size=best.size(), gens_used=len(curve), best_expr=best.to_string(),
                   test_r2=float(1 - np.sum((y_test - pred) ** 2) / ss_tot) if np.all(np.isfinite(pred)) else float("nan"))
        if 101 <= spec_id <= 115:  # Keijzer only (benchmark_of also counts the 200-400 dev ids)
            Xk = keijzer_test_set(spec_id - 100)
            with np.errstate(all="ignore"):
                yk = np.asarray(spec["func"](*Xk.T), dtype=float)
                pk = best.evaluate({n: Xk[:, i] for i, n in enumerate(names)})
            row["test_r2_orig"] = float(1 - np.sum((yk - pk) ** 2) / np.sum((yk - yk.mean()) ** 2)) \
                if np.all(np.isfinite(pk)) else float("nan")
        row["elapsed_sec"] = round(time.time() - t0, 2)  # GA time only, excludes the recovery check
        model = scale * best.to_sympy()
        row["recovered"] = int(_Checker.call("is_recovered", model, spec["sympy"], physical="srsd" in spec or bool(spec.get("dev_log"))))
        # A recovered model is the true equation, whatever form simplify leaves it in
        row["ted"], row["ned"] = (0, 0.0) if row["recovered"] else \
            _Checker.call("tree_edit_distance", model, spec["sympy"], physical="srsd" in spec or bool(spec.get("dev_log")))
    except Exception as exc:
        row.update(status="error", error=repr(exc), elapsed_sec=round(time.time() - t0, 2))
    return row, curve


def main():
    if sys.argv[1:] == ["--checker"]:
        return _checker_main()
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--equations", default="6,7,12,114",
                    help="spec ids (Keijzer offset by 100) or the groups nguyen, keijzer, srsd, dev, heldout")
    pa.add_argument("--arms", default=",".join(ARMS))
    pa.add_argument("--seeds", type=int, default=30)
    pa.add_argument("--seed0", type=int, default=0)
    pa.add_argument("--jobs", type=int, default=8)
    pa.add_argument("--tag", default="main", help="output file name prefix")
    pa.add_argument("--selftest", action="store_true", help="run SELFTEST and print one JSON line per run")
    pa.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian noise on the training targets, sd = noise * RMS(y) (Nguyen, Keijzer, dev)")
    pa.add_argument("--protocol", choices=["paper", "orig"], default="paper",
                    help="orig: Nguyen with the original 20 points and ranges (NGUYEN_ORIG_RANGES)")
    args = pa.parse_args()
    os.environ["RLLXO_NOISE"] = str(args.noise)
    os.environ["RLLXO_PROTOCOL"] = args.protocol  # workers are spawned and inherit the environment
    if args.selftest:
        return selftest()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv, out_curves = OUT_DIR / f"{args.tag}.csv", OUT_DIR / f"{args.tag}_curves.jsonl"
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            done = {(r["arm"], int(r["spec_id"]), int(r["seed"])) for r in csv.DictReader(f)}
    dev = [6, 7, 12, 114, *range(201, 207), *[301, 302, 303, 304, 305, 307, 308, 309], *range(401, 413)]
    groups = {"dev": [str(k) for k in dev], "nguyen": [str(k) for k in range(1, 13)], "keijzer": [str(k) for k in range(101, 116)],
              "srsd": [str(k) for k in srsd_specs()]}
    # The held-out test set (results/HELDOUT_PLAN.md): every Nguyen, Keijzer and SRSD-medium equation not used in development
    groups["heldout"] = [e for g in ("nguyen", "keijzer", "srsd") for e in groups[g] if int(e) not in dev]
    equations = [e for tok in args.equations.split(",") for e in groups.get(tok, [tok])]
    tasks = [(a, int(e), s) for s in range(args.seed0, args.seed0 + args.seeds) for e in equations
             for a in args.arms.split(",") if (a, int(e), s) not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} already done), {args.jobs} workers", flush=True)

    new_file = not out_csv.exists()
    t0 = time.time()
    with open(out_csv, "a", newline="", encoding="utf-8") as fc, open(out_curves, "a", encoding="utf-8") as fj, \
            mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=12) as pool:
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


# Fixed development runs (one per arm family). Their output is identical on one platform and library
# set; across platforms, numpy's sin/cos/exp/log can differ in the last bits, so some runs diverge
SELFTEST = [("random", 203, 1000), ("ex64_t01", 203, 1000), ("random_x4", 204, 1000), ("a2c_t03", 304, 1000),
            ("dir_t03", 401, 1000), ("mlp", 12, 1000)]


def selftest():
    for t in SELFTEST:
        r, _ = run_task(t)
        print(json.dumps({k: r.get(k) for k in ("arm", "spec_id", "seed", "status", "train_r2", "test_r2", "recovered",
                                                "ted", "size", "gens_used", "elapsed_sec", "best_expr")}), flush=True)


if __name__ == "__main__":
    main()
