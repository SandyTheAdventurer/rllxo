from main import run_ga, set_global_seed, simplify_equation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import seaborn as sns
import sympy as sp
import re

SEED = 42
set_global_seed(SEED)

# Load daily data
df = pd.read_csv("monsoon_daily.csv")
df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')
df.set_index('DATE', inplace=True)
df = df.drop(columns=['YEAR', 'DOY'])
df.columns = ['Te', 'Sp', 'Pr', 'So']  # T2M, QV2M, PRECTOTCORR, GWETROOT

# Filter monsoon season (Jun 1 - Sep 30)
df = df[(df.index.month >= 6) & (df.index.month <= 9)].copy()

# Sort by date
df = df.sort_index()

# Feature engineering for daily data
df_feat = df.copy()

# Seasonal features (annual cycle) - multiple harmonics
doy = df_feat.index.dayofyear
for n in [1, 2, 3]:
    df_feat[f"Ms{n}"] = np.sin(2 * np.pi * n * doy / 365.25)
    df_feat[f"Mc{n}"] = np.cos(2 * np.pi * n * doy / 365.25)

targets = ['Pr', 'So', 'Sp', 'Te']

# Extended lags
for c in targets:
    for lag in [1, 2, 3, 7, 14, 30]:
        df_feat[f"{c}{lag}"] = df_feat[c].shift(lag)

# Rolling windows (mean, std, max)
for c in targets:
    for w in [7, 14, 30]:
        df_feat[f"{c}{w}m"] = df_feat[c].rolling(w, min_periods=1).mean().shift(1)
        df_feat[f"{c}{w}std"] = df_feat[c].rolling(w, min_periods=1).std().shift(1)
        df_feat[f"{c}{w}max"] = df_feat[c].rolling(w, min_periods=1).max().shift(1)

# Cross-variable interactions (relevant for Sp, Te)
df_feat["So1_Sp1"] = df_feat["So1"] * df_feat["Sp1"]
df_feat["Te1_Sp1"] = df_feat["Te1"] * df_feat["Sp1"]
df_feat["So1_Te1"] = df_feat["So1"] * df_feat["Te1"]
df_feat["So7m_Sp7m"] = df_feat["So7m"] * df_feat["Sp7m"]
df_feat["Te7m_Sp7m"] = df_feat["Te7m"] * df_feat["Sp7m"]

df_feat = df_feat.dropna()
print(f"Feature rows: {len(df_feat)}, columns: {len(df_feat.columns)}")

# Chronological 80/20 split
split_at = int(np.floor(0.80 * len(df_feat)))
train_df = df_feat.iloc[:split_at].copy()
test_df = df_feat.iloc[split_at:].copy()

print(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
print(f"Test:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

default_ga_config = {
    "pop_size": 1000, "gens": 150, "max_init_depth": 4, "max_depth": 8,
    "crossover_rate": 0.9, "mutation_rate": 0.3, "tournament_k": 5, "elitism": 15,
    "use_learned_crossover": True,
    "mlp_layers": 3, "mlp_hidden_dim": 128, "mlp_activation": "leaky_relu", "mlp_dropout": 0.1,
    "stopping_metric": "r2", "stopping_threshold": 1.0,
    "complexity_coef": 1e-6, "inclusion_coef": 0.0, "prob_const": 0.3, "const_range": (-5, 5),
    "point_mutation_prob": 0.1, "simplify_new_genomes": False, "simplify_timeout": 0.2,
    "temperature": 1.0, "entropy_weight": 0.08, "heuristic_bias_scale": 0.1,
    "opt_constants": True, "const_opt_freq": 5, "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 200, "verbose": True, "use_pareto": True, "print_freq": 5, "seed": SEED,
    "selector_update_freq": 5, "selector_warmup_gens": 5, "selector_recent_gen_window": 50,
    "selector_min_history": 64, "selector_reward_clip": 3.0, "selector_lr": 5e-4,
    "use_critic": True, "selector_device": "cpu",
    "linear_terms": True, "max_terms": 30, "add_term_prob": 0.2, "quadratic_init_frac": 0.3,
    "allow_pow": True,
}

target_ga_configs = {
    'Pr': {"use_pareto": False, "complexity_coef": 1e-3},
    'So': {"stopping_threshold": 0.95},
    'Sp': {"stopping_threshold": 0.95, "complexity_coef": 5e-7, "max_depth": 8},
    'Te': {"stopping_threshold": 0.95, "complexity_coef": 5e-7, "max_depth": 8},
}

def predict_expression(expr, feature_cols, X):
    fn = sp.lambdify(sp.symbols(feature_cols), expr, "numpy")
    return np.broadcast_to(np.asarray(fn(*X.T), dtype=float), X.shape[0])

results = {}

for target in targets:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    
    feature_cols = [c for c in df_feat.columns if c != target]
    
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)
    
    # Scaling from training data only
    x_mu, x_sd = X_train.mean(axis=0), X_train.std(axis=0)
    y_mu, y_sd = y_train.mean(), y_train.std()
    if np.any(x_sd == 0) or y_sd == 0:
        raise ValueError(f"Cannot standardize {target}: zero variance in training data.")
    
    ga_config = default_ga_config.copy()
    ga_config.update(target_ga_configs.get(target, {}))
    ga_config["var_names"] = feature_cols
    
    print(f"Running GA: {len(X_train)} train / {len(X_test)} test")
    eqn, ga_train_r2 = run_ga(
        (X_train - x_mu) / x_sd, (y_train - y_mu) / y_sd, **ga_config
    )
    
    z_expr = sp.powsimp(sp.trigsimp(sp.cancel(eqn.to_sympy())))
    unscale = {
        symbol: (sp.Symbol(symbol.name) - x_mu[feature_cols.index(symbol.name)])
        / x_sd[feature_cols.index(symbol.name)]
        for symbol in z_expr.free_symbols
    }
    expr = y_mu + y_sd * z_expr.xreplace(unscale)
    train_data_dict = {col: X_train[:, i] for i, col in enumerate(feature_cols)}
    simplified_expr, _ = simplify_equation(expr, train_data_dict, y_train)
    
    train_r2 = r2_score(y_train, predict_expression(expr, feature_cols, X_train))
    test_r2 = r2_score(y_test, predict_expression(expr, feature_cols, X_test))
    
    results[target] = {
        'equation': expr,
        'simplified': simplified_expr,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'ga_train_r2': ga_train_r2
    }
    
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"Equation: {expr}\n")

# Summary
print("\n" + "="*60)
print("SUMMARY - Daily Data Chronological 80/20 Split")
print("="*60)
for target, r in results.items():
    print(f"{target}: Train R²={r['train_r2']:.4f}, Test R²={r['test_r2']:.4f}")

# Save results
import json
with open("results_daily.json", "w") as f:
    json.dump({k: {kk: str(vv) if isinstance(vv, sp.Expr) else vv for kk, vv in v.items()} 
               for k, v in results.items()}, f, indent=2)

with open("eqns_daily.tex", "w") as f:
    for target, r in results.items():
        f.write(f"{target} & {sp.latex(r['simplified'])} & {r['test_r2']:.4f} \\\\\n")