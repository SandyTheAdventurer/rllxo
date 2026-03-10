"""Nguyen + Keijzer benchmark: learned vs random crossover.
Usage: python nguyen.py [--jobs N] [--shard K/N] [--merge] [--benchmark nguyen|keijzer|all]
"""
from __future__ import annotations
import argparse, csv, glob, importlib, multiprocessing as mp, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np, sympy as sp
from main import run_ga, Node, Const, Var, Unary, Binary

SEED, SAMPLE_SIZE, SEEDS = 1, 512, list(range(30))

_UNARY_REG = {
    "sin": (np.sin, sp.sin), "cos": (np.cos, sp.cos), "exp": (np.exp, sp.exp),
    "log": (np.log, sp.log), "sqrt": (np.sqrt, sp.sqrt), "abs": (np.abs, sp.Abs),
    "tanh": (np.tanh, sp.tanh),
}

# ============ Tree Edit Distance (Zhang-Shasha) ============
def _get_label(node: Node) -> str:
    """Get canonical label for a node."""
    if isinstance(node, Const):
        return "CONST"  # Treat all constants as equivalent
    elif isinstance(node, Var):
        return f"VAR:{node.name}"
    elif isinstance(node, Unary):
        return f"UNARY:{node.func_name}"
    elif isinstance(node, Binary):
        return f"BINARY:{node.func_name}"
    return "UNKNOWN"

def _get_children(node: Node) -> List[Node]:
    """Get children of a node."""
    if isinstance(node, Const) or isinstance(node, Var):
        return []
    elif isinstance(node, Unary):
        return [node.child]
    elif isinstance(node, Binary):
        return [node.left, node.right]
    return []

def _tree_to_list(node: Node) -> Tuple[List[str], List[int], List[int]]:
    """Convert tree to post-order list with leftmost leaf descendants."""
    labels, lefts, parents = [], [], []
    stack = [(node, False, -1)]
    node_to_idx = {}
    
    while stack:
        n, visited, parent_idx = stack.pop()
        if visited:
            idx = len(labels)
            node_to_idx[id(n)] = idx
            labels.append(_get_label(n))
            parents.append(parent_idx)
            children = _get_children(n)
            if not children:
                lefts.append(idx)  # Leaf: leftmost is itself
            else:
                lefts.append(lefts[node_to_idx[id(children[0])]])
        else:
            stack.append((n, True, parent_idx))
            children = _get_children(n)
            for child in reversed(children):
                stack.append((child, False, len(labels) + sum(1 for s in stack if s[1])))
    
    return labels, lefts, parents

def _simple_ted(node1: Node, node2: Node) -> int:
    """Simplified Tree Edit Distance using dynamic programming."""
    def get_nodes_postorder(node: Node) -> List[Node]:
        result = []
        stack = [(node, False)]
        while stack:
            n, visited = stack.pop()
            if visited:
                result.append(n)
            else:
                stack.append((n, True))
                for child in reversed(_get_children(n)):
                    stack.append((child, False))
        return result
    
    nodes1 = get_nodes_postorder(node1)
    nodes2 = get_nodes_postorder(node2)
    n1, n2 = len(nodes1), len(nodes2)
    
    # Simple DP: dp[i][j] = TED between first i nodes of tree1 and first j nodes of tree2
    # This is a simplification - full Zhang-Shasha is more complex
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    
    for i in range(n1 + 1):
        dp[i][0] = i  # Delete all nodes
    for j in range(n2 + 1):
        dp[0][j] = j  # Insert all nodes
    
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            label1, label2 = _get_label(nodes1[i-1]), _get_label(nodes2[j-1])
            cost = 0 if label1 == label2 else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Delete from tree1
                dp[i][j-1] + 1,      # Insert into tree1
                dp[i-1][j-1] + cost  # Match/replace
            )
    
    return dp[n1][n2]

def compute_ted(evolved: Node, target: Node) -> int:
    """Compute Tree Edit Distance between evolved expression and target."""
    if evolved is None or target is None:
        return -1
    try:
        return _simple_ted(evolved, target)
    except Exception:
        return -1

def sympy_to_node(expr: sp.Expr, var_names: List[str]) -> Node:
    """Convert sympy expression to Node tree."""
    if isinstance(expr, sp.Number):
        return Const(float(expr))
    elif isinstance(expr, sp.Symbol):
        return Var(str(expr))
    elif isinstance(expr, sp.Add):
        args = list(expr.args)
        result = sympy_to_node(args[0], var_names)
        for a in args[1:]:
            result = Binary("add", result, sympy_to_node(a, var_names))
        return result
    elif isinstance(expr, sp.Mul):
        args = list(expr.args)
        result = sympy_to_node(args[0], var_names)
        for a in args[1:]:
            result = Binary("mul", result, sympy_to_node(a, var_names))
        return result
    elif isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == sp.Rational(1, 2):
            return Unary("sqrt", sympy_to_node(base, var_names))
        elif exp == -1:
            return Binary("div", Const(1.0), sympy_to_node(base, var_names))
        return Binary("pow", sympy_to_node(base, var_names), sympy_to_node(exp, var_names))
    elif isinstance(expr, sp.sin):
        return Unary("sin", sympy_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.cos):
        return Unary("cos", sympy_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.exp):
        return Unary("exp", sympy_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.log):
        return Unary("log", sympy_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.sqrt):
        return Unary("sqrt", sympy_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.Abs):
        return Unary("abs", sympy_to_node(expr.args[0], var_names))
    elif isinstance(expr, sp.tanh):
        return Unary("tanh", sympy_to_node(expr.args[0], var_names))
    else:
        # Fallback: try to expand and retry
        expanded = sp.expand(expr)
        if expanded != expr:
            return sympy_to_node(expanded, var_names)
        return Const(0.0)  # Fallback

# Sympy symbols for ground truth
_x, _y, _z, _r, _t, _w, _v, _m, _s, _i = sp.symbols('x y z r t w v m s i')

NGUYEN_SPECS = [
    {"id": 1, "func": lambda x: x**3 + x**2 + x, "expr": "x³+x²+x",
     "sympy": _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 2, "func": lambda x: x**4 + x**3 + x**2 + x, "expr": "x⁴+x³+x²+x",
     "sympy": _x**4 + _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 3, "func": lambda x: x**5 + x**4 + x**3 + x**2 + x, "expr": "x⁵+x⁴+x³+x²+x",
     "sympy": _x**5 + _x**4 + _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 4, "func": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x, "expr": "x⁶+x⁵+x⁴+x³+x²+x",
     "sympy": _x**6 + _x**5 + _x**4 + _x**3 + _x**2 + _x,
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": []},
    {"id": 5, "func": lambda x: np.sin(x**2) * np.cos(x) - 1, "expr": "sin(x²)cos(x)-1",
     "sympy": sp.sin(_x**2) * sp.cos(_x) - 1,
     "variables": [{"name": "x", "sampler": (-2.0, 2.0)}], "allowed_unary": ["sin", "cos"]},
    {"id": 6, "func": lambda x: np.sin(x) + np.sin(x + x**2), "expr": "sin(x)+sin(x+x²)",
     "sympy": sp.sin(_x) + sp.sin(_x + _x**2),
     "variables": [{"name": "x", "sampler": (-2.0, 2.0)}], "allowed_unary": ["sin"]},
    {"id": 7, "func": lambda x: np.log(x + 1) + np.log(x**2 + 1), "expr": "log(x+1)+log(x²+1)",
     "sympy": sp.log(_x + 1) + sp.log(_x**2 + 1),
     "variables": [{"name": "x", "sampler": (0.0, 50.0)}], "allowed_unary": ["log"]},
    {"id": 8, "func": lambda x: np.sqrt(np.abs(x)), "expr": "sqrt(|x|)",
     "sympy": sp.sqrt(sp.Abs(_x)),
     "variables": [{"name": "x", "sampler": (-5.0, 5.0)}], "allowed_unary": ["abs", "sqrt"]},
    {"id": 9, "func": lambda x, y: np.sin(x) + np.sin(y**2), "expr": "sin(x)+sin(y²)",
     "sympy": sp.sin(_x) + sp.sin(_y**2),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}, {"name": "y", "sampler": (-1.0, 1.0)}],
     "allowed_unary": ["sin"]},
    {"id": 10, "func": lambda x, y: 2 * np.sin(x) * np.cos(y), "expr": "2sin(x)cos(y)",
     "sympy": 2 * sp.sin(_x) * sp.cos(_y),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}, {"name": "y", "sampler": (-1.0, 1.0)}],
     "allowed_unary": ["sin", "cos"]},
    {"id": 11, "func": lambda x, y: x**y, "expr": "x^y",
     "sympy": _x**_y,
     "variables": [{"name": "x", "sampler": (0.1, 3.0)}, {"name": "y", "sampler": (0.1, 3.0)}],
     "allowed_unary": [], "allow_pow": True},
    {"id": 12, "func": lambda x, y: x**4 - x**3 + (y**2) / 2 - y, "expr": "x⁴-x³+y²/2-y",
     "sympy": _x**4 - _x**3 + _y**2/2 - _y,
     "variables": [{"name": "x", "sampler": (-2.5, 2.5)}, {"name": "y", "sampler": (-2.5, 2.5)}],
     "allowed_unary": []},
]

# Keijzer benchmark: standard symbolic regression problems
KEIJZER_SPECS = [
    # K1: Oscillating polynomial (narrow range)
    {"id": 1, "func": lambda x: 0.3 * x * np.sin(2 * np.pi * x), "expr": "0.3x·sin(2πx)",
     "sympy": sp.Rational(3,10) * _x * sp.sin(2 * sp.pi * _x),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}], "allowed_unary": ["sin"]},
    # K2: Oscillating polynomial (medium range)
    {"id": 2, "func": lambda x: 0.3 * x * np.sin(2 * np.pi * x), "expr": "0.3x·sin(2πx)",
     "sympy": sp.Rational(3,10) * _x * sp.sin(2 * sp.pi * _x),
     "variables": [{"name": "x", "sampler": (-2.0, 2.0)}], "allowed_unary": ["sin"]},
    # K3: Oscillating polynomial (wide range)
    {"id": 3, "func": lambda x: 0.3 * x * np.sin(2 * np.pi * x), "expr": "0.3x·sin(2πx)",
     "sympy": sp.Rational(3,10) * _x * sp.sin(2 * sp.pi * _x),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}], "allowed_unary": ["sin"]},
    # K4: Complex trigonometric with exponential decay
    {"id": 4, "func": lambda x: x**3 * np.exp(-x) * np.cos(x) * np.sin(x) * (np.sin(x)**2 * np.cos(x) - 1),
     "expr": "x³e⁻ˣcos(x)sin(x)(sin²(x)cos(x)-1)",
     "sympy": _x**3 * sp.exp(-_x) * sp.cos(_x) * sp.sin(_x) * (sp.sin(_x)**2 * sp.cos(_x) - 1),
     "variables": [{"name": "x", "sampler": (0.0, 10.0)}], "allowed_unary": ["sin", "cos", "exp"]},
    # K5: Rational function (3 variables) - simplified version
    {"id": 5, "func": lambda x, y, z: (30 * x * z) / ((x - 10) * y**2 + 1e-6),
     "expr": "30xz/((x-10)y²)",
     "sympy": 30 * _x * _z / ((_x - 10) * _y**2),
     "variables": [{"name": "x", "sampler": (-1.0, 1.0)}, {"name": "y", "sampler": (1.0, 2.0)},
                   {"name": "z", "sampler": (-1.0, 1.0)}], "allowed_unary": []},
    # K6: Harmonic series approximation (use continuous approx: ln(x) + gamma)
    {"id": 6, "func": lambda x: np.log(x) + 0.5772156649, "expr": "ln(x)+γ",
     "sympy": sp.log(_x) + sp.Float(0.5772156649),
     "variables": [{"name": "x", "sampler": (1.0, 50.0)}], "allowed_unary": ["log"]},
    # K7: Natural logarithm
    {"id": 7, "func": lambda x: np.log(x), "expr": "ln(x)",
     "sympy": sp.log(_x),
     "variables": [{"name": "x", "sampler": (1.0, 100.0)}], "allowed_unary": ["log"]},
    # K8: Square root
    {"id": 8, "func": lambda x: np.sqrt(x), "expr": "√x",
     "sympy": sp.sqrt(_x),
     "variables": [{"name": "x", "sampler": (0.0, 100.0)}], "allowed_unary": ["sqrt"]},
    # K9: Inverse hyperbolic sine (arcsinh)
    {"id": 9, "func": lambda x: np.arcsinh(x), "expr": "arcsinh(x)",
     "sympy": sp.asinh(_x),
     "variables": [{"name": "x", "sampler": (0.0, 100.0)}], "allowed_unary": ["log", "sqrt"]},
    # K10: Power function
    {"id": 10, "func": lambda x, y: x**y, "expr": "x^y",
     "sympy": _x**_y,
     "variables": [{"name": "x", "sampler": (0.1, 1.0)}, {"name": "y", "sampler": (0.1, 1.0)}],
     "allowed_unary": [], "allow_pow": True},
    # K11: Product with trigonometric
    {"id": 11, "func": lambda x, y: x * y + np.sin((x - 1) * (y - 1)), "expr": "xy+sin((x-1)(y-1))",
     "sympy": _x * _y + sp.sin((_x - 1) * (_y - 1)),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": ["sin"]},
    # K12: Polynomial (same as Nguyen-12)
    {"id": 12, "func": lambda x, y: x**4 - x**3 + y**2/2 - y, "expr": "x⁴-x³+y²/2-y",
     "sympy": _x**4 - _x**3 + _y**2/2 - _y,
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": []},
    # K13: Scaled trigonometric product
    {"id": 13, "func": lambda x, y: 6 * np.sin(x) * np.cos(y), "expr": "6sin(x)cos(y)",
     "sympy": 6 * sp.sin(_x) * sp.cos(_y),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": ["sin", "cos"]},
    # K14: Rational function (2D)
    {"id": 14, "func": lambda x, y: 8 / (2 + x**2 + y**2), "expr": "8/(2+x²+y²)",
     "sympy": 8 / (2 + _x**2 + _y**2),
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": []},
    # K15: Mixed polynomial
    {"id": 15, "func": lambda x, y: x**3/5 + y**3/2 - y - x, "expr": "x³/5+y³/2-y-x",
     "sympy": _x**3/5 + _y**3/2 - _y - _x,
     "variables": [{"name": "x", "sampler": (-3.0, 3.0)}, {"name": "y", "sampler": (-3.0, 3.0)}],
     "allowed_unary": []},
]

# Combined specs (use benchmark arg to select)
SPECS = NGUYEN_SPECS  # Default

ga_config = {
    "pop_size": 300, "gens": 200, "max_init_depth": 6, "max_depth": 30,
    "crossover_rate": 0.85, "mutation_rate": 0.5, "tournament_k": 9, "elitism": 24,
    "use_learned_crossover": False, "selector_lr": 0.001,
    "mlp_layers": 16, "mlp_hidden_dim": 64, "mlp_activation": "leaky_relu", "mlp_dropout": 0.1,
    "use_critic": True, "stopping_metric": "r2", "stopping_threshold": 1.0,
    "complexity_coef": 1e-4, "inclusion_coef": 1.0, "prob_const": 0.3, "const_range": (-5, 5),
    "point_mutation_prob": 0.1, "simplify_new_genomes": False, "simplify_timeout": 0.2,
    "selector_update_freq": 1, "temperature": 0.7, "entropy_weight": 0.1,
    "opt_constants": True, "const_opt_freq": 10, "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 100, "verbose": True, "use_pareto": True, "print_freq": 10, "seed": SEED,
}
learned_overrides = {
    "use_learned_crossover": True, "mlp_layers": 4, "mlp_hidden_dim": 64, "mlp_dropout": 0.0,
    "temperature": 1.0, "entropy_weight": 0.08, "heuristic_bias_scale": 0.08,
    "selector_update_freq": 1, "selector_warmup_gens": 30, "selector_recent_gen_window": 50,
    "selector_min_history": 64, "selector_reward_clip": 3.0, "selector_lr": 3e-4,
    "use_critic": True, "selector_device": "cpu",
}
random_overrides = {"use_learned_crossover": False, "selector_device": "cpu"}

@dataclass(frozen=True)
class Task:
    mode: str
    spec_id: int
    seed: int
    benchmark: str = "nguyen"

def run_single_task(t: Task) -> Dict:
    # Select correct specs based on benchmark
    if t.benchmark == "nguyen":
        specs_list = NGUYEN_SPECS
    elif t.benchmark == "keijzer":
        specs_list = KEIJZER_SPECS
    else:  # all
        specs_list = NGUYEN_SPECS + [{**s, "id": s["id"] + 100} for s in KEIJZER_SPECS]
    spec = next((s for s in specs_list if s["id"] == t.spec_id), None)
    if spec is None:
        return {"mode": t.mode, "spec_id": t.spec_id, "seed": t.seed, "expr_target": "?",
                "n_vars": 0, "status": "error", "best_r2": float("nan"),
                "best_expr": "", "elapsed_sec": 0.0, "error": f"spec_id {t.spec_id} not found"}
    started = time.time()
    status, error, best_expr, best_r2 = "ok", "", "", float("nan")
    try:
        rng = np.random.default_rng(t.seed + t.spec_id * 1000)
        vnames = [v["name"] for v in spec["variables"]]
        cols = [rng.uniform(v["sampler"][0], v["sampler"][1], size=SAMPLE_SIZE) for v in spec["variables"]]
        try:
            y = spec["func"](*cols)
        except Exception:
            y = np.array([spec["func"](*[c[i] for c in cols]) for i in range(SAMPLE_SIZE)])
        X = np.column_stack(cols)
        y = np.asarray(y, dtype=float).ravel()
        ok = np.isfinite(y)
        if not ok.all():
            X, y = X[ok], y[ok]
        if len(y) < 10:
            raise ValueError(f"Too few valid samples ({len(y)}) for Nguyen-{spec['id']}")
        allowed = spec.get("allowed_unary") or []
        unary = {n: _UNARY_REG[n] for n in allowed if n in _UNARY_REG}
        kw = {**ga_config, **(learned_overrides if t.mode == "learned" else random_overrides),
              "seed": t.seed, "allow_pow": bool(spec.get("allow_pow", False))}
        best, r2v = run_ga(X, y, var_names=vnames, unary_funcs=unary, **kw)
        best_r2 = float(r2v) if r2v is not None else float("nan")
        best_expr = best.to_string() if best is not None else ""
        # Compute TED if ground truth sympy is available
        if best is not None and "sympy" in spec:
            try:
                target_node = sympy_to_node(spec["sympy"], vnames)
                ted = compute_ted(best, target_node)
            except Exception:
                ted = -1
        else:
            ted = -1
    except Exception as exc:
        status, error, ted = "error", str(exc), -1
    return {"mode": t.mode, "spec_id": t.spec_id, "seed": t.seed, "expr_target": spec["expr"],
            "n_vars": len(spec["variables"]), "status": status, "best_r2": best_r2,
            "best_expr": best_expr, "ted": ted, "elapsed_sec": time.time() - started, "error": error}

def _get_range_str(spec: Dict) -> str:
    """Format input range as compact string."""
    if not spec or "variables" not in spec:
        return "?"
    ranges = []
    for v in spec["variables"]:
        lo, hi = v["sampler"]
        ranges.append(f"[{lo:.0f},{hi:.0f}]" if lo == int(lo) and hi == int(hi) else f"[{lo:.1f},{hi:.1f}]")
    return ",".join(ranges)

def summarize_results(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for mode, sid in sorted({(r["mode"], r["spec_id"]) for r in rows}):
        sub = [r for r in rows if r["mode"] == mode and r["spec_id"] == sid and r["status"] == "ok"]
        r2 = np.array([r["best_r2"] for r in sub if np.isfinite(r["best_r2"])], dtype=float)
        ts = np.array([r["elapsed_sec"] for r in sub], dtype=float)
        ted_vals = np.array([r.get("ted", -1) for r in sub if r.get("ted", -1) >= 0], dtype=float)
        sp_ = next((s for s in SPECS if s["id"] == sid), {})
        f = lambda a, fn: float(fn(a)) if a.size else float("nan")
        out.append({"mode": mode, "spec_id": sid, "expr_target": sp_.get("expr", "?"),
                     "input_range": _get_range_str(sp_),
                     "n_vars": len(sp_.get("variables", [])), "runs_ok": len(sub),
                     "runs_total": sum(1 for r in rows if r["mode"] == mode and r["spec_id"] == sid),
                     "r2_mean": f(r2, np.mean), "r2_median": f(r2, np.median),
                     "r2_std": f(r2, np.std), "r2_min": f(r2, np.min), "r2_max": f(r2, np.max),
                     "ted_mean": f(ted_vals, np.mean), "ted_median": f(ted_vals, np.median),
                     "ted_min": f(ted_vals, np.min),
                     "time_mean_sec": f(ts, np.mean), "time_std_sec": f(ts, np.std)})
    return out

def results_to_latex_table(summary: List[Dict]) -> str:
    """Convert summarized results to a LaTeX table string."""
    winners = []
    for i in summary:
        l_mean = i["r2_mean"]
        r_mean = next((s["r2_mean"] for s in summary if s["spec_id"] == i["spec_id"] and s["mode"] == "random"), float("nan"))
        if abs(l_mean - r_mean) < 1e-4:
            winners.append("TIE")
        elif l_mean > r_mean:
            winners.append("Learned")
        else:
            winners.append("Random")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Comparison of RL-LXO (Learned) vs.\ Random Crossover on the Nguyen and Keijzer benchmarks. Mean $R^2$ and standard deviation are computed over 30 independent runs. Winner indicates which method achieved higher mean $R^2$ (TIE if difference $< 1e-4$).}",
        r"\label{tab:recovery}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lcc cc cc c}"
        r"\toprule",
        r"& & & \multicolumn{2}{c}{\textbf{RL-LXO (Learned)}} & \multicolumn{2}{c}{\textbf{Random Crossover}} & \\"
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\textbf{Equation} & \textbf{Formula} & \textbf{Range} & \textbf{Mean $R^2$} & \textbf{Std $R^2$} & \textbf{Mean $R^2$} & \textbf{Std $R^2$} & \textbf{Winner} \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Nguyen Polynomial Equations}} \\"
    ]
    for i in summary[:4]:
        spec = next((s for s in NGUYEN_SPECS if s["id"] == i["spec_id"]), None)
        formula = sp.latex(spec["sympy"]) if spec else "?"
        lines.append(rf"Eq. N{i['spec_id']} & ${formula}$ & {i['input_range']} & "
                         rf"{i['r2_mean']:.4f} & {i['r2_std']:.3f} & "
                         rf"{next((s['r2_mean'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.4f} & "
                         rf"{next((s['r2_std'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.3f} & "
                         rf"{winners[summary.index(i)]} \\" 
                        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Nguyen Trigonometric Equations}} \\")
    for i in summary[4:6]:
        spec = next((s for s in NGUYEN_SPECS if s["id"] == i["spec_id"]), None)
        formula = sp.latex(spec["sympy"]) if spec else "?"
        lines.append(rf"Eq. N{i['spec_id']} & ${formula}$ & {i['input_range']} & "
                         rf"{i['r2_mean']:.4f} & {i['r2_std']:.3f} & "
                         rf"{next((s['r2_mean'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.4f} & "
                         rf"{next((s['r2_std'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.3f} & "
                         rf"{winners[summary.index(i)]} \\" 
                        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Nguyen Transcendental Equations}} \\")
    for i in summary[6:8]:
        spec = next((s for s in NGUYEN_SPECS if s["id"] == i["spec_id"]), None)
        formula = sp.latex(spec["sympy"]) if spec else "?"
        lines.append(rf"Eq. N{i['spec_id']} & ${formula}$ & {i['input_range']} & "
                         rf"{i['r2_mean']:.4f} & {i['r2_std']:.3f} & "
                         rf"{next((s['r2_mean'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.4f} & "
                         rf"{next((s['r2_std'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.3f} & "
                         rf"{winners[summary.index(i)]} \\" 
                        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Nguyen Multivariate Equations}} \\")
    for i in summary[8:13]:
        spec = next((s for s in NGUYEN_SPECS if s["id"] == i["spec_id"]), None)
        formula = sp.latex(spec["sympy"]) if spec else "?"
        lines.append(rf"Eq. N{i['spec_id']} & ${formula}$ & {i['input_range']} & "
                         rf"{i['r2_mean']:.4f} & {i['r2_std']:.3f} & "
                         rf"{next((s['r2_mean'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.4f} & "
                         rf"{next((s['r2_std'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.3f} & "
                         rf"{winners[summary.index(i)]} \\" 
                        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Keijzer Equations \cite{keijzer2003improving}}} \\")
    for i in summary[12:27]:
        kid = i['spec_id'] - 100
        spec = next((s for s in KEIJZER_SPECS if s['id'] == kid), None)
        formula = sp.latex(spec['sympy']) if spec else '?'
        lines.append(rf"Eq. K{kid} & ${formula}$ & {i['input_range']} & "
                         rf"{i['r2_mean']:.4f} & {i['r2_std']:.3f} & "
                         rf"{next((s['r2_mean'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.4f} & "
                         rf"{next((s['r2_std'] for s in summary if s['spec_id'] == i['spec_id'] and s['mode'] == 'random'), float('nan')):.3f} & "
                         rf"{winners[summary.index(i)]} \\" 
                        )
    lines.append(
        rf"\multicolumn{{8}}{{l}}{{\textbf{{Summary (27 equations)}}: Learned: {winners.count('Learned')}, Random: {winners.count('Random')}, TIE: {27- winners.count('Learned')- winners.count('Random')}}} \\"
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}"
    ])
    return "\n".join(lines)

def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode","spec_id","seed","expr_target","n_vars",
                                          "status","best_r2","ted","best_expr","elapsed_sec","error"])
        w.writeheader(); w.writerows(rows)

def write_summary_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mode","spec_id","expr_target","input_range","n_vars","runs_ok","runs_total",
            "r2_mean","r2_median","r2_std","r2_min","r2_max",
            "ted_mean","ted_median","ted_min",
            "time_mean_sec","time_std_sec"
        ])
        w.writeheader(); w.writerows(summarize_results(rows))

def fmt_dur(s: float) -> str:
    if not np.isfinite(s): return "?"
    t = max(0, int(round(s))); h, r = divmod(t, 3600); m, s2 = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s2:02d}" if h else f"{m:02d}:{s2:02d}"

def print_comparison(rows: List[Dict], bench_name: str = "NGUYEN") -> None:
    from scipy.stats import wilcoxon
    summary = summarize_results(rows)
    sids = sorted({s["spec_id"] for s in summary})
    W = 160
    print("\n" + "=" * W + f"\n{bench_name} BENCHMARK — LEARNED vs RANDOM CROSSOVER\n" + "=" * W)
    lw = rw = ties = 0
    deltas: List[float] = []
    ted_deltas: List[float] = []
    print(f"{'Spec':>6} {'Expr':>20} {'Range':>18} {'#v':>3} {'L mean±std':>14} {'L med':>7}"
          f" {'R mean±std':>14} {'R med':>7} {'Delta':>8} {'L TED':>6} {'R TED':>6} {'L t/s':>7} {'R t/s':>7} {'Winner':>8}")
    print("-" * W)
    for sid in sids:
        lr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "learned"), None)
        rr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "random"), None)
        if not lr or not rr: continue
        d = lr["r2_mean"] - rr["r2_mean"]; deltas.append(d)
        # TED comparison (lower is better)
        l_ted = lr.get("ted_mean", float("nan"))
        r_ted = rr.get("ted_mean", float("nan"))
        if np.isfinite(l_ted) and np.isfinite(r_ted):
            ted_deltas.append(r_ted - l_ted)  # Positive = learned better
        if abs(d) < 1e-4: w, ties = "TIE", ties + 1
        elif d > 0: w = "LEARNED"; lw += 1
        else: w = "RANDOM"; rw += 1
        l_ted_str = f"{l_ted:>6.1f}" if np.isfinite(l_ted) else "   N/A"
        r_ted_str = f"{r_ted:>6.1f}" if np.isfinite(r_ted) else "   N/A"
        range_str = lr.get("input_range", "?")[:18]
        print(f"{sid:>6} {lr['expr_target'][:20]:>20} {range_str:>18} {lr['n_vars']:>3}"
              f" {lr['r2_mean']:>7.4f}±{lr['r2_std']:<5.3f} {lr['r2_median']:>7.4f}"
              f" {rr['r2_mean']:>7.4f}±{rr['r2_std']:<5.3f} {rr['r2_median']:>7.4f}"
              f" {d:>+8.4f} {l_ted_str} {r_ted_str} {lr['time_mean_sec']:>7.1f} {rr['time_mean_sec']:>7.1f} {w:>8}")
    print("-" * W)
    da = np.array(deltas) if deltas else np.array([])
    print(f"Learned wins: {lw}  |  Random wins: {rw}  |  Ties: {ties}")
    if da.size:
        print(f"Mean   R² delta: {float(np.mean(da)):+.4f}")
        print(f"Median R² delta: {float(np.median(da)):+.4f}")
    if da.size >= 6:
        try:
            stat, pv = wilcoxon(da, alternative="two-sided")
            ne = np.sum(da != 0)
            rb = 1.0 - (2.0 * stat) / (ne * (ne + 1) / 2) if ne > 0 else 0.0
            sig = " *" if pv < 0.05 else " (*)" if pv < 0.10 else ""
            print(f"Wilcoxon: W={stat:.1f}, p={pv:.4f}{sig}  |  Rank-biserial r={rb:+.3f}")
        except Exception as e:
            print(f"Wilcoxon failed: {e}")
    if da.size >= 2:
        sd = np.std(da, ddof=1)
        cd = float(np.mean(da) / sd) if sd > 0 else 0.0
        mg = "large" if abs(cd) >= 0.8 else "medium" if abs(cd) >= 0.5 else "small" if abs(cd) >= 0.2 else "negligible"
        print(f"Cohen's d: {cd:+.3f} ({mg})")
    # TED summary
    if ted_deltas:
        tda = np.array(ted_deltas)
        print(f"TED: Learned better in {sum(1 for t in tda if t > 0)}/{len(tda)} eqs  |  "
              f"Mean TED advantage: {float(np.mean(tda)):+.2f}")
    print(f"Equations: {len(sids)}  |  Seeds: {len(SEEDS)}\n" + "=" * W)

# --- Entry point ---

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Nguyen + Keijzer benchmark")
    pa.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 1))
    pa.add_argument("--output", type=Path, default=None, help="Output CSV (default: {benchmark}_results.csv)")
    pa.add_argument("--summary-output", type=Path, default=None)
    pa.add_argument("--progress-every", type=int, default=5)
    pa.add_argument("--tensorboard-logdir", type=Path, default=None)
    pa.add_argument("--shard", type=str, default=None, help="K/N (e.g. 1/4)")
    pa.add_argument("--merge", action="store_true", help="Merge shard CSVs")
    pa.add_argument("--benchmark", type=str, default="all", choices=["nguyen", "keijzer", "all"],
                    help="Which benchmark to run (default: nguyen)")
    args = pa.parse_args()

    # Select benchmark
    if args.benchmark == "nguyen":
        SPECS = NGUYEN_SPECS
        bench_name = "NGUYEN"
    elif args.benchmark == "keijzer":
        SPECS = KEIJZER_SPECS
        bench_name = "KEIJZER"
    else:  # all
        # Offset Keijzer IDs to avoid collision
        SPECS = NGUYEN_SPECS + [
            {**s, "id": s["id"] + 100, "expr": f"K{s['id']}:{s['expr']}"} for s in KEIJZER_SPECS
        ]
        bench_name = "NGUYEN+KEIJZER"

    # Set default output paths based on benchmark
    if args.output is None:
        args.output = Path(f"{args.benchmark}_results.csv")
    if args.summary_output is None:
        args.summary_output = Path(f"{args.benchmark}_results_summary.csv")
    if args.tensorboard_logdir is None:
        args.tensorboard_logdir = Path(f"logs/{args.benchmark}_tensorboard")

    if args.merge:
        shard_pattern = f"{args.benchmark}_results_shard*.csv"
        files = sorted(glob.glob(str(args.output.parent / "shards" / shard_pattern)))
        if not files:
            raise SystemExit(f"No shard files found matching {shard_pattern}")
        all_rows: List[Dict] = []
        for p in files:
            print(f"Reading {p} ...", flush=True)
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    r["spec_id"], r["seed"], r["n_vars"] = int(r["spec_id"]), int(r["seed"]), int(r["n_vars"])
                    r["best_r2"], r["elapsed_sec"] = float(r["best_r2"]), float(r["elapsed_sec"])
                    r["ted"] = int(r.get("ted", -1)) if r.get("ted", "-1") not in ("", "-1", "nan") else -1
                    all_rows.append(r)
        all_rows.sort(key=lambda r: (r["mode"], r["spec_id"], r["seed"]))
        d = args.output.parent
        write_csv(all_rows, d / f"{args.benchmark}_results.csv")
        write_summary_csv(all_rows, d / f"{args.benchmark}_results_summary.csv")
        print(f"\nMerged {len(all_rows)} rows from {len(files)} shards")
        print_comparison(all_rows, bench_name)
        with open(args.output.parent / f"{args.benchmark}_results.tex", "w", encoding="utf-8") as f:
            f.write(results_to_latex_table(summarize_results(all_rows)))
        raise SystemExit(0)

    shard_idx = num_shards = None
    spec_sub: List[Dict] | None = None
    if args.shard:
        try:
            k, n = args.shard.split("/"); shard_idx, num_shards = int(k), int(n)
        except (ValueError, AttributeError):
            raise SystemExit(f"ERROR: --shard must be K/N, got '{args.shard}'")
        if not (1 <= shard_idx <= num_shards):
            raise SystemExit(f"ERROR: shard {shard_idx} out of 1..{num_shards}")
        ss = sorted(SPECS, key=lambda s: s["id"])
        spec_sub = [s for i, s in enumerate(ss) if i % num_shards == shard_idx - 1]
        if not spec_sub:
            raise SystemExit(f"Shard {shard_idx}/{num_shards} is empty.")

    specs = spec_sub or SPECS
    tasks = [Task(m, s["id"], sd, args.benchmark) for m in ("random", "learned") for s in specs for sd in SEEDS]
    nt = len(tasks)

    if shard_idx:
        out_csv = args.output.with_name(f"{args.output.stem}_shard{shard_idx}.csv")
        sum_csv = args.summary_output.with_name(f"{args.summary_output.stem}_shard{shard_idx}.csv")
        tb_dir = args.tensorboard_logdir / f"shard{shard_idx}" if args.tensorboard_logdir else None
        tag = f"  [shard {shard_idx}/{num_shards}]"
    else:
        out_csv, sum_csv, tb_dir, tag = args.output, args.summary_output, args.tensorboard_logdir, ""

    jobs = max(1, min(args.jobs, nt))
    print(f"{bench_name}{tag}: {len(specs)} eq × {len(SEEDS)} seeds × 2 = {nt} tasks | Workers: {jobs}")
    print(f"IDs: {sorted(s['id'] for s in specs)}")

    tb = None
    if tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter; tb = SummaryWriter(log_dir=str(tb_dir))
        except Exception:
            try: tb = getattr(importlib.import_module("tensorboardX"), "SummaryWriter")(log_dir=str(tb_dir))
            except Exception: pass
        if tb: print(f"TensorBoard: {tb_dir}", flush=True)

    t0 = time.time(); rows: List[Dict] = []; mc = {"random": 0, "learned": 0}; errs = 0
    pf = max(1, args.progress_every)
    try:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for done, row in enumerate(pool.imap_unordered(run_single_task, tasks), 1):
                rows.append(row); m = row["mode"]; mc[m] += 1
                if row["status"] != "ok": errs += 1
                dt = time.time() - t0; rate = done / dt if dt > 0 else 0; eta = (nt - done) / rate if rate > 0 else float("inf")
                if done == 1 or done == nt or done % pf == 0 or row["status"] != "ok":
                    r2t = f"{row['best_r2']:.4f}" if np.isfinite(row["best_r2"]) else "nan"
                    print(f"[{done}/{nt}] {m} spec={row['spec_id']} seed={row['seed']} "
                          f"{row['status']} r2={r2t} {row['elapsed_sec']:.1f}s errs={errs} "
                          f"eta={fmt_dur(eta)}", flush=True)
                if tb:
                    tb.add_scalar("progress/done", done, done)
                    tb.add_scalar("timing/task_sec", float(row["elapsed_sec"]), done)
                    if row["status"] == "ok" and np.isfinite(row["best_r2"]):
                        tb.add_scalar(f"metrics/{m}_r2", float(row["best_r2"]), done)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")
    finally:
        if tb: tb.flush(); tb.close()

    rows.sort(key=lambda r: (r["mode"], r["spec_id"], r["seed"]))
    write_csv(rows, out_csv); write_summary_csv(rows, sum_csv)
    print(f"\nDone in {fmt_dur(time.time() - t0)}. {len(rows)} rows -> {out_csv} | Errors: {sum(1 for r in rows if r['status'] != 'ok')}")
    print_comparison(rows, bench_name)