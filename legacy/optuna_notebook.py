"""
Optuna tuning for Nguyen notebook settings.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import sympy as sp
from optuna.samplers import TPESampler

from main import run_ga

SEED = 1
NGUYEN_SAMPLE_SIZE = 512


def uniform_sampler(low: float, high: float, gap_frac: float = 0.3):
    """Sample from two disjoint bands using a trig warp to avoid local Taylor regimes."""

    def _sample(rng: np.random.Generator, size: int):
        mid = 0.5 * (low + high)
        span = high - low
        gap = max(1e-9, abs(span) * gap_frac)
        left = (low, mid - 0.5 * gap)
        right = (mid + 0.5 * gap, high)
        if left[1] <= left[0] or right[1] <= right[0]:
            theta = rng.uniform(0.0, np.pi, size)
            return low + np.sin(theta) * span

        bands = rng.integers(0, 2, size)
        theta = rng.uniform(0.0, np.pi, size)
        weights = np.sin(theta)
        lows = np.where(bands == 0, left[0], right[0])
        highs = np.where(bands == 0, left[1], right[1])
        return lows + weights * (highs - lows)

    return _sample


def dataset_from_spec(spec: Dict, size: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else SEED)
    variables = spec.get("variables", [])
    data: Dict[str, np.ndarray] = {}
    for var in variables:
        sampler = var.get("sampler")
        name = var.get("name")
        if sampler is None:
            raise ValueError(f"Missing sampler for variable {name}")
        data[name] = sampler(rng, size)
    cols = [data[v["name"]] for v in variables]
    try:
        target = spec["func"](*cols)
    except Exception:
        target = np.array([spec["func"](*[c[i] for c in cols]) for i in range(size)])
    data["target"] = target
    return pd.DataFrame(data)


def select_unary_funcs(spec: Dict, base_unary: Dict) -> Tuple[Dict, List[str] | None]:
    allowed = spec.get("allowed_unary")
    if allowed is None:
        return base_unary, None
    if len(allowed) == 0:
        return {}, []
    return {name: base_unary[name] for name in allowed}, allowed


NGUYEN_SPECS: List[Dict] = [
    {
        "id": 1,
        "func": lambda x: x**3 + x**2 + x,
        "expr": "x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 2,
        "func": lambda x: x**4 + x**3 + x**2 + x,
        "expr": "x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 3,
        "func": lambda x: x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 4,
        "func": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
        "allowed_unary": [],
    },
    {
        "id": 5,
        "func": lambda x: np.sin(x**2) * np.cos(x) - 1,
        "expr": "np.sin(x**2)*np.cos(x) - 1",
        "variables": [{"name": "x", "sampler": uniform_sampler(-2.0, 2.0)}],
        "allowed_unary": ["sin", "cos"],
    },
    {
        "id": 6,
        "func": lambda x: np.sin(x) + np.sin(x + x**2),
        "expr": "np.sin(x) + np.sin(x + x**2)",
        "variables": [{"name": "x", "sampler": uniform_sampler(-2.0, 2.0)}],
        "allowed_unary": ["sin"],
    },
    {
        "id": 7,
        "func": lambda x: np.log(x + 1) + np.log(x**2 + 1),
        "expr": "np.log(x + 1) + np.log(x**2 + 1)",
        "variables": [{"name": "x", "sampler": uniform_sampler(0.0, 50.0)}],
        "allowed_unary": ["log"],
    },
    {
        "id": 8,
        "func": lambda x: np.sqrt(np.abs(x)),
        "expr": "np.sqrt(np.abs(x))",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
        "allowed_unary": ["abs", "sqrt"],
    },
    {
        "id": 9,
        "func": lambda x, y: np.sin(x) + np.sin(y**2),
        "expr": "np.sin(x) + np.sin(y**2)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-1.0, 1.0)},
            {"name": "y", "sampler": uniform_sampler(-1.0, 1.0)},
        ],
        "allowed_unary": ["sin"],
    },
    {
        "id": 10,
        "func": lambda x, y: 2 * np.sin(x) * np.cos(y),
        "expr": "2*np.sin(x)*np.cos(y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-1.0, 1.0)},
            {"name": "y", "sampler": uniform_sampler(-1.0, 1.0)},
        ],
        "allowed_unary": ["sin", "cos"],
    },
    {
        "id": 11,
        "func": lambda x, y: x**y,
        "expr": "x**y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(0.1, 3.0)},
            {"name": "y", "sampler": uniform_sampler(0.1, 3.0)},
        ],
        "allowed_unary": [],
    },
    {
        "id": 12,
        "func": lambda x, y: x**4 - x**3 + (y**2) / 2 - y,
        "expr": "x**4 - x**3 + (y**2)/2 - y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-2.5, 2.5)},
            {"name": "y", "sampler": uniform_sampler(-2.5, 2.5)},
        ],
        "allowed_unary": [],
    },
]


default_unary_funcs = {
    "sin": (np.sin, sp.sin),
    "cos": (np.cos, sp.cos),
    "exp": (np.exp, sp.exp),
    "log": (np.log, sp.log),
    "sqrt": (np.sqrt, sp.sqrt),
    "abs": (np.abs, sp.Abs),
}

nguyen_unary_funcs = dict(default_unary_funcs)

BASE_GA_CONFIG: Dict = {
    "pop_size": 1000,
    "gens": 150,
    "max_init_depth": 4,
    "max_depth": 20,
    "crossover_rate": 0.85,
    "mutation_rate": 0.4,
    "tournament_k": 10,
    "elitism": 5,
    "use_learned_crossover": True,
    "selector_lr": 0.0003,
    "mlp_layers": 6,
    "mlp_hidden_dim": 128,
    "mlp_activation": "gelu",
    "mlp_dropout": 0.1,
    "use_critic": True,
    "stopping_metric": "r2",
    "stopping_threshold": 1.0,
    "complexity_coef": 5e-4,
    "inclusion_coef": 1.0,
    "prob_const": 0.3,
    "const_range": (-5, 5),
    "point_mutation_prob": 0.1,
    "simplify_new_genomes": False,
    "simplify_timeout": 0.2,
    "selector_update_freq": 3,
    "temperature": 0.75,
    "entropy_weight": 0.075,
    "opt_constants": True,
    "const_opt_freq": 5,
    "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 120,
    "verbose": False,
    "use_pareto": True,
    "print_freq": 1,
    "seed": SEED,
}


def run_equation(spec: Dict, sample_size: int, unary_funcs: Dict, override_kwargs: Dict, seed_offset: int = 0) -> float:
    dataset = dataset_from_spec(spec, size=sample_size, seed=SEED + seed_offset + spec["id"])
    dataset = dataset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    feature_cols = [col for col in dataset.columns if col != "target"]
    X = dataset[feature_cols].to_numpy()
    y = dataset["target"].to_numpy()

    ga_kwargs = {**BASE_GA_CONFIG}
    ga_kwargs.update(override_kwargs)

    unary_for_run, _ = select_unary_funcs(spec, unary_funcs)

    best, best_r2 = run_ga(
        X,
        y,
        var_names=feature_cols,
        unary_funcs=unary_for_run,
        **ga_kwargs,
    )

    return float(best_r2) if best_r2 is not None else 0.0


def suggest_params(trial: optuna.Trial) -> Dict:
    pop_size = trial.suggest_int("pop_size", 500, 3000, step=100)
    gens = trial.suggest_int("gens", 50, 150, step=10)
    max_init_depth = trial.suggest_int("max_init_depth", 2, 8)
    max_depth = trial.suggest_int("max_depth", 6, 12)
    crossover_rate = trial.suggest_float("crossover_rate", 0.5, 0.95)
    mutation_rate = trial.suggest_float("mutation_rate", 0.2, 0.7)
    tournament_k = trial.suggest_int("tournament_k", 2, 10)
    elitism = trial.suggest_int("elitism", 2, 20)
    selector_lr = trial.suggest_float("selector_lr", 1e-4, 1e-3, log=True)
    mlp_layers = trial.suggest_int("mlp_layers", 4, 32, step=4)
    mlp_hidden_dim = trial.suggest_int("mlp_hidden_dim", 32, 256, step=32)
    complexity_coef = trial.suggest_float("complexity_coef", 1e-8, 1e-4, log=True)
    temperature = trial.suggest_float("temperature", 0.5, 1.0)
    critic_weight = trial.suggest_float("critic_weight", 0.5, 0.9)
    entropy_weight = trial.suggest_float("entropy_weight", 0.01, 0.2)
    use_pareto = trial.suggest_categorical("use_pareto", [True, False])

    if max_depth <= max_init_depth:
        max_depth = max_init_depth + 1

    return {
        "pop_size": pop_size,
        "gens": gens,
        "max_init_depth": max_init_depth,
        "max_depth": max_depth,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "tournament_k": tournament_k,
        "elitism": elitism,
        "selector_lr": selector_lr,
        "mlp_layers": mlp_layers,
        "mlp_hidden_dim": mlp_hidden_dim,
        "complexity_coef": complexity_coef,
        "temperature": temperature,
        "critic_weight": critic_weight,
        "entropy_weight": entropy_weight,
        "use_pareto": use_pareto,
        "verbose": False,
        "print_freq": 0,
    }


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)
    params["seed"] = SEED + trial.number

    scores: List[float] = []
    for spec in NGUYEN_SPECS:
        r2_val = run_equation(spec, NGUYEN_SAMPLE_SIZE, nguyen_unary_funcs, params, seed_offset=trial.number * 1000)
        scores.append(r2_val)

    return float(np.mean(scores))


@dataclass
class StudyConfig:
    n_trials: int = 10
    timeout: int | None = None
    study_name: str = "Nguyen_Notebook_Tuning"
    storage: str | None = "sqlite:///nguyen_optuna.db"


def build_study(cfg: StudyConfig) -> optuna.study.Study:
    sampler = TPESampler(
        n_startup_trials=5,
        n_ei_candidates=32,
        multivariate=True,
        seed=SEED,
    )
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=cfg.study_name,
        storage=cfg.storage,
        load_if_exists=True,
    )


def run_study(cfg: StudyConfig) -> optuna.study.Study:
    study = build_study(cfg)
    print(f"Starting Optuna TPE study: {cfg.study_name}")
    print(f"Trials: {cfg.n_trials}, Timeout: {cfg.timeout}")
    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout, show_progress_bar=True)

    best = study.best_trial
    print("\nBest average R²: {:.6f}".format(best.value))
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for Nguyen notebook settings")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--storage", type=str, default="sqlite:///nguyen_optuna.db", help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="Nguyen_Notebook_Tuning", help="Optuna study name")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = StudyConfig(
        n_trials=args.trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
    )
    run_study(cfg)


if __name__ == "__main__":
    main()
