import numpy as np
import pandas as pd
import sympy as sp
from pathlib import Path
from functools import lru_cache
from main import run_ga
import main as ga_main


def uniform_sampler(low: float, high: float):
    def _sample(rng: np.random.Generator, size: int):
        return rng.uniform(low, high, size)

    return _sample


def dataset_from_spec(spec: dict, size: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else SEED)
    variables = spec.get("variables", [])
    data: dict[str, np.ndarray] = {}
    for var in variables:
        sampler = var.get("sampler")
        name = var.get("name")
        if sampler is None:
            raise ValueError(f"Missing sampler for variable {name}")
        if callable(sampler):
            data[name] = sampler(rng, size)
        else:
            low, high = sampler
            data[name] = rng.uniform(low, high, size)
    cols = [data[v["name"]] for v in variables]
    try:
        target = spec["func"](*cols)
    except Exception:
        target = np.array([spec["func"](*[c[i] for c in cols]) for i in range(size)])
    data["target"] = target
    return pd.DataFrame(data)


def select_unary_funcs(spec: dict, base_unary: dict) -> tuple[dict, list[str] | None]:
    allowed = spec.get("allowed_unary")
    if allowed is None:
        return base_unary, None
    if len(allowed) == 0:
        return {}, []
    return {name: base_unary[name] for name in allowed}, allowed


def run_equation(spec: dict) -> tuple[float, dict]:
    dataset = dataset_from_spec(spec, size=NGUYEN_SAMPLE_SIZE, seed=SEED + spec["id"])
    dataset = dataset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    feature_cols = [col for col in dataset.columns if col != "target"]
    X = dataset[feature_cols].to_numpy()
    y = dataset["target"].to_numpy()

    unary_for_run, _ = select_unary_funcs(spec, nguyen_unary_funcs)

    best, best_r2 = run_ga(
        X,
        y,
        var_names=feature_cols,
        unary_funcs=unary_for_run,
        allow_pow=bool(spec.get("allow_pow", False)),
        **ga_config,
    )

    result = {
        "id": spec.get("id"),
        "expr": spec.get("expr"),
        "best_r2": float(best_r2) if best_r2 is not None else None,
        "best_str": best.to_string() if best is not None else None,
        "best_sympy": best.to_sympy() if best is not None else None,
    }
    nguyen_results.append(result)
    return float(best_r2) if best_r2 is not None else 0.0, result


def run_equation_with_seed(spec: dict, seed: int) -> tuple[float, dict]:
    dataset = dataset_from_spec(spec, size=NGUYEN_SAMPLE_SIZE, seed=SEED + spec["id"])
    dataset = dataset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    feature_cols = [col for col in dataset.columns if col != "target"]
    X = dataset[feature_cols].to_numpy()
    y = dataset["target"].to_numpy()

    unary_for_run, _ = select_unary_funcs(spec, nguyen_unary_funcs)
    ga_kwargs = dict(ga_config)
    ga_kwargs["seed"] = seed

    best, best_r2 = run_ga(
        X,
        y,
        var_names=feature_cols,
        unary_funcs=unary_for_run,
        allow_pow=bool(spec.get("allow_pow", False)),
        **ga_kwargs,
    )

    result = {
        "id": spec.get("id"),
        "expr": spec.get("expr"),
        "best_r2": float(best_r2) if best_r2 is not None else None,
        "best_str": best.to_string() if best is not None else None,
        "best_sympy": best.to_sympy() if best is not None else None,
    }
    return float(best_r2) if best_r2 is not None else 0.0, result


def _sympy_locals() -> dict:
    return {
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
    }


def _parse_target_expr(spec: dict) -> sp.Expr | None:
    expr = spec.get("expr")
    if not expr:
        return None
    expr = expr.replace("np.", "")
    return sp.sympify(expr, locals=_sympy_locals())


def _sympy_label(expr: sp.Expr) -> tuple:
    if expr.is_Number:
        return ("num",)
    if expr.is_Symbol:
        return ("sym", str(expr))
    return ("func", str(expr.func))


@lru_cache(maxsize=None)
def _sympy_tree_size(expr: sp.Expr) -> int:
    return 1 + sum(_sympy_tree_size(arg) for arg in expr.args)


def _sympy_tree_edit_distance(a: sp.Expr | None, b: sp.Expr | None) -> int | None:
    if a is None or b is None:
        return None

    cache: dict[tuple[sp.Expr, sp.Expr], int] = {}

    def _distance(x: sp.Expr, y: sp.Expr) -> int:
        key = (x, y)
        if key in cache:
            return cache[key]

        cost_sub = 0 if _sympy_label(x) == _sympy_label(y) else 1
        x_children = list(x.args)
        y_children = list(y.args)
        m, n = len(x_children), len(y_children)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + _sympy_tree_size(x_children[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + _sympy_tree_size(y_children[j - 1])

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                del_cost = dp[i - 1][j] + _sympy_tree_size(x_children[i - 1])
                ins_cost = dp[i][j - 1] + _sympy_tree_size(y_children[j - 1])
                sub_cost = dp[i - 1][j - 1] + _distance(x_children[i - 1], y_children[j - 1])
                dp[i][j] = min(del_cost, ins_cost, sub_cost)

        dist = cost_sub + dp[m][n]
        cache[key] = dist
        return dist

    return _distance(a, b)


def _is_recovered(pred: sp.Expr | None, target: sp.Expr | None) -> bool:
    if pred is None or target is None:
        return False
    try:
        return sp.simplify(pred - target) == 0
    except Exception:
        return False


def run_nguyen_seeds(
    seeds: list[int] | None = None,
    out_path: str | Path = "nguyen_results.csv",
) -> pd.DataFrame:
    if seeds is None:
        seeds = list(range(1, 11))

    records = []
    for spec in NGUYEN_SPECS:
        target_expr = _parse_target_expr(spec)
        r2_vals: list[float] = []
        ted_vals: list[int] = []
        recovered_flags: list[bool] = []

        for seed in seeds:
            r2, result = run_equation_with_seed(spec, seed)
            pred_expr = result.get("best_sympy")
            r2_vals.append(r2)
            ted = _sympy_tree_edit_distance(pred_expr, target_expr)
            if ted is not None:
                ted_vals.append(ted)
            recovered_flags.append(_is_recovered(pred_expr, target_expr))

        record = {
            "id": spec.get("id"),
            "expr": spec.get("expr"),
            "avg_r2": float(np.mean(r2_vals)) if r2_vals else None,
            "std_r2": float(np.std(r2_vals)) if r2_vals else None,
            "avg_ted": float(np.mean(ted_vals)) if ted_vals else None,
            "std_ted": float(np.std(ted_vals)) if ted_vals else None,
            "recovery_rate": float(np.mean(recovered_flags)) if recovered_flags else None,
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    return df

NGUYEN_SPECS = [
    {
        "id": 1,
        "func": lambda x: x**3 + x**2 + x,
        "expr": "x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": (-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 2,
        "func": lambda x: x**4 + x**3 + x**2 + x,
        "expr": "x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": (-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 3,
        "func": lambda x: x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": (-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 4,
        "func": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": (-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 5,
        "func": lambda x: np.sin(x**2) * np.cos(x) - 1,
        "expr": "np.sin(x**2)*np.cos(x) - 1",
        "variables": [{"name": "x", "sampler": (-2.0, 2.0)}],
        "allowed_unary": ["sin", "cos"],
    },
    {
        "id": 6,
        "func": lambda x: np.sin(x) + np.sin(x + x**2),
        "expr": "np.sin(x) + np.sin(x + x**2)",
        "variables": [{"name": "x", "sampler": (-2.0, 2.0)}],
        "allowed_unary": ["sin"],
    },
    {
        "id": 7,
        "func": lambda x: np.log(x + 1) + np.log(x**2 + 1),
        "expr": "np.log(x + 1) + np.log(x**2 + 1)",
        "variables": [{"name": "x", "sampler": (0.0, 50.0)}],
        "allowed_unary": ["log"],
    },
    {
        "id": 8,
        "func": lambda x: np.sqrt(np.abs(x)),
        "expr": "np.sqrt(np.abs(x))",
        "variables": [{"name": "x", "sampler": (-5.0, 5.0)}],
        "allowed_unary": ["abs", "sqrt"],
    },
    {
        "id": 9,
        "func": lambda x, y: np.sin(x) + np.sin(y**2),
        "expr": "np.sin(x) + np.sin(y**2)",
        "variables": [
            {"name": "x", "sampler": (-1.0, 1.0)},
            {"name": "y", "sampler": (-1.0, 1.0)},
        ],
        "allowed_unary": ["sin"],
    },
    {
        "id": 10,
        "func": lambda x, y: 2 * np.sin(x) * np.cos(y),
        "expr": "2*np.sin(x)*np.cos(y)",
        "variables": [
            {"name": "x", "sampler": (-1.0, 1.0)},
            {"name": "y", "sampler": (-1.0, 1.0)},
        ],
        "allowed_unary": ["sin", "cos"],
    },
    {
        "id": 11,
        "func": lambda x, y: x**y,
        "expr": "x**y",
        "variables": [
            {"name": "x", "sampler": (0.1, 3.0)},
            {"name": "y", "sampler": (0.1, 3.0)},
        ],
        "allowed_unary": [],
        "allow_pow": True,
    },
    {
        "id": 12,
        "func": lambda x, y: x**4 - x**3 + (y**2) / 2 - y,
        "expr": "x**4 - x**3 + (y**2)/2 - y",
        "variables": [
            {"name": "x", "sampler": (-2.5, 2.5)},
            {"name": "y", "sampler": (-2.5, 2.5)},
        ],
        "allowed_unary": [],
    },
]

SEED = 1

ga_config = {
    "pop_size": 1500,
    "gens": 150,
    "max_init_depth": 6,
    "max_depth": 30,
    "crossover_rate": 0.85,
    "mutation_rate": 0.5,
    "tournament_k": 9,
    "elitism": 12,
    "use_learned_crossover": True,
    "selector_lr": 0.0003,
    "mlp_layers": 16,
    "mlp_hidden_dim": 128,
    "mlp_activation": "leaky_relu",
    "mlp_dropout": 0.1,
    "use_critic": True,
    "stopping_metric": "r2",
    "stopping_threshold": 1.0,
    "complexity_coef": 1e-4,
    "inclusion_coef": 1.0,
    "prob_const": 0.3,
    "const_range": (-5, 5),
    "point_mutation_prob": 0.1,
    "simplify_new_genomes": False,
    "simplify_timeout": 0.2,
    "selector_update_freq": 3,
    "temperature": 0.7,
    "entropy_weight": 0.1,
    "opt_constants": True,
    "const_opt_freq": 3,
    "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 500,
    "verbose": True,
    "use_pareto": True,
    "print_freq": 5,
    "seed": SEED,
}

# Basic unary functions expected by run_ga: (numpy_func, sympy_func) tuples
default_unary_funcs = {
    "sin": (np.sin, sp.sin),
    "cos": (np.cos, sp.cos),
    "exp": (np.exp, sp.exp),
    "log": (np.log, sp.log),
    "sqrt": (np.sqrt, sp.sqrt),
    "abs": (np.abs, sp.Abs),
}

# Per-benchmark unary maps (provide sympy conversion alongside numpy callables)
nguyen_unary_funcs = dict(default_unary_funcs)

# Result collectors & sample sizes
nguyen_results = []

NGUYEN_SAMPLE_SIZE = 512
# %%
"""# Nguyen equation 1
print("Running Nguyen equation 1...")
nguyen_score_1, nguyen_result_1 = run_equation(
    NGUYEN_SPECS[0],
)
nguyen_result_1

# %%
# Nguyen equation 2
print("Running Nguyen equation 2...")
nguyen_score_2, nguyen_result_2 = run_equation(
    NGUYEN_SPECS[1],
)
nguyen_result_2

# %%
# Nguyen equation 3
print("Running Nguyen equation 3...")
nguyen_score_3, nguyen_result_3 = run_equation(
    NGUYEN_SPECS[2],
)
nguyen_result_3

# %%
# Nguyen equation 4
print("Running Nguyen equation 4...")
nguyen_score_4, nguyen_result_4 = run_equation(
    NGUYEN_SPECS[3],
)
nguyen_result_4

# %%
# Nguyen equation 5
print("Running Nguyen equation 5...")
nguyen_score_5, nguyen_result_5 = run_equation(
    NGUYEN_SPECS[4],
)
nguyen_result_5

# %%
# Nguyen equation 6
print("Running Nguyen equation 6...")
nguyen_score_6, nguyen_result_6 = run_equation(
    NGUYEN_SPECS[5],
)
nguyen_result_6

# %%
# Nguyen equation 7
print("Running Nguyen equation 7...")
nguyen_score_7, nguyen_result_7 = run_equation(
    NGUYEN_SPECS[6],
)
nguyen_result_7

# %%
# Nguyen equation 8
print("Running Nguyen equation 8...")
nguyen_score_8, nguyen_result_8 = run_equation(
    NGUYEN_SPECS[7],
)
nguyen_result_8

# %%
# Nguyen equation 9
print("Running Nguyen equation 9...")
nguyen_score_9, nguyen_result_9 = run_equation(
    NGUYEN_SPECS[8],
)
nguyen_result_9

# %%
# Nguyen equation 10
print("Running Nguyen equation 10...")
nguyen_score_10, nguyen_result_10 = run_equation(
    NGUYEN_SPECS[9],
)
nguyen_result_10

# %%
# Nguyen equation 11
print("Running Nguyen equation 11...")
nguyen_score_11, nguyen_result_11 = run_equation(
    NGUYEN_SPECS[10],
)
nguyen_result_11

# %%
# Nguyen equation 12
print("Running Nguyen equation 12...")
nguyen_score_12, nguyen_result_12 = run_equation(
    NGUYEN_SPECS[11],
)
nguyen_result_12"""


run_nguyen_seeds()