import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # main.py lives in the repo root
from main import run_ga, set_global_seed, simplify_equation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import seaborn as sns
import sympy as sp
import re
import time

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # STIX matches Times New Roman for math

SEED = 42
set_global_seed(SEED)

color_mapping = {
    'Temperature': 'darkred',
    'Specific Humidity': 'blue',
    'Precipitation': 'darkviolet',
    'Soil Wetness': 'forestgreen'
}

monsoon = pd.read_csv("monsoon.csv")
df = pd.read_csv("monsoon.csv")

df_long = df.melt(
    id_vars=["PARAMETER", "YEAR"],
    var_name="MONTH",
    value_name="VALUE"
)

df_pivot = df_long.pivot_table(
    index=["YEAR", "MONTH"],
    columns="PARAMETER",
    values="VALUE"
).reset_index()

month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
             "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
             "OCT": 10, "NOV": 11, "DEC": 12}

df_pivot["MONTH_NUM"] = df_pivot["MONTH"].map(month_map)

df_pivot["DATE"] = pd.to_datetime(
    df_pivot["YEAR"].astype(str) + "-" + df_pivot["MONTH_NUM"].astype(str)
)
df_pivot.set_index("DATE", inplace=True)

df_final = df_pivot.drop(columns=["YEAR", "MONTH", "MONTH_NUM"])

def sanitize(name: str) -> str:
    name_map = {
        'Precipitation': 'P',
        'Soil Moisture ': 'W',   # trailing space in CSV
        'Soil Moisture': 'W',
        'Specific Humidity': 'Q',
        'Temperature': 'T',
    }
    stripped = name.strip()
    if name in name_map:
        return name_map[name]
    if stripped in name_map:
        return name_map[stripped]
    s = re.sub(r"\W+", "_", stripped)
    if s and s[0].isdigit():
        s = f"v_{s}"
    s = s[:2]
    return s

col_map = {c: sanitize(c) for c in df_final.columns}
df_final.rename(columns=col_map, inplace=True)

df_final = (
    df_final.apply(pd.to_numeric, errors="coerce")
    .replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    .dropna()
)
targets = list(df_final.columns)
if len(df_final.columns) <= 1:
    print("Not enough columns to compute mutual information (need at least 1 feature and 1 target).")
else:
    mi_results = {}
    for t in df_final.columns:
        # use all other columns as features for this target
        feature_df = df_final.drop(columns=[t])
        X = feature_df.to_numpy(dtype=float)
        feature_names = feature_df.columns.tolist()

        y = df_final[t].to_numpy(dtype=float)
        mi = mutual_info_regression(X, y, discrete_features=False, random_state=SEED)
        mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
        mi_results[t] = mi_series

    mi_df = pd.DataFrame(mi_results).T.fillna(0)
    plt.figure(figsize=(max(8, 0.8 * mi_df.shape[1]), 3 + mi_df.shape[0]))
    sns.heatmap(mi_df, annot=True, fmt=".4f", cmap="viridis")
    plt.title("Mutual Information (targets x features)")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(f"mi.pdf", bbox_inches='tight')
    

print("Correlation matrix:", df_final.corr())
sns.heatmap(data=df_final.corr(), annot=True, cmap="coolwarm")

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

vars = ['T', 'Q', 'P', 'W']
chvars = {'P': 'P', 'W': 'W', 'Q': 'Q', 'T': 'T'}
var_to_full = {'P': 'Precipitation', 'W': 'Soil Wetness', 'Q': 'Specific Humidity', 'T': 'Temperature'}
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        var_i_full = var_to_full[vars[i]]
        color_i = color_mapping[var_i_full]
        if i == j:
            ax.hist(df_final[vars[i]], bins=20, color=color_i, edgecolor='black')
            ax.set_xlabel(chvars[vars[i]])
            ax.set_ylabel('Density')
        else:
            ax.scatter(df_final[vars[j]], df_final[vars[i]], color=color_i, alpha=0.6, s=10)
            ax.set_xlabel(chvars[vars[j]])
            ax.set_ylabel(chvars[vars[i]])
        ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.savefig(f"pairplot.pdf", bbox_inches='tight')

def add_seasonal_features(df):
    """Add seasonal features (month sin/cos) - always available at prediction time."""
    df = df.copy()
    month = df.index.month
    df["Ms"] = np.sin(2 * np.pi * month / 12)
    df["Mc"] = np.cos(2 * np.pi * month / 12)
    return df

def add_lag_features_full(df, lag_vars):
    """Add previous-month lag features within each year (full dataset, no split)."""
    df = df.copy()
    for c in lag_vars:
        df[f"{c}1"] = df.groupby(df.index.year)[c].shift(1)
    return df

# Use all data — no train/test split
df_sorted = df_final.sort_index()
df_all = add_seasonal_features(df_sorted)
df_all = add_lag_features_full(df_all, ["P", "W", "Q", "T"])

feature_vars = ["Ms", "Mc", "P", "W", "Q", "T"]
lag_vars = [f"{c}1" for c in ["P", "W", "Q", "T"]]
all_feature_vars = feature_vars + lag_vars

target_data = {}
for t in targets:
    cols = [c for c in all_feature_vars if c != t]
    data = df_all[cols + [t]].dropna()
    target_data[t] = data
    print(f"{t}: n={len(data)}")

default_ga_config = {
    "pop_size": 1000, "gens": 150, "max_init_depth": 4, "max_depth": 4,
    "crossover_rate": 0.9, "mutation_rate": 0.3, "tournament_k": 5, "elitism": 15,
    "use_learned_crossover": True,
    "mlp_layers": 3, "mlp_hidden_dim": 128, "mlp_activation": "leaky_relu", "mlp_dropout": 0.1,
    "stopping_metric": "r2", "stopping_threshold": 1.0,
    "complexity_coef": 5e-5, "inclusion_coef": 0.0, "prob_const": 0.3, "const_range": (-5, 5),
    "point_mutation_prob": 0.1, "simplify_new_genomes": False, "simplify_timeout": 0.2,
    "temperature": 1.0, "entropy_weight": 0.08, "heuristic_bias_scale": 0.1,
    "opt_constants": True, "const_opt_freq": 10, "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 100, "verbose": True, "use_pareto": True, "print_freq": 5, "seed": SEED,
    "selector_update_freq": 5, "selector_warmup_gens": 5, "selector_recent_gen_window": 50,
    "selector_min_history": 64, "selector_reward_clip": 3.0, "selector_lr": 5e-4,
    "use_critic": True, "selector_device": "cpu",
    # Fit top-level term weights by least squares and bias the search toward quadratic terms
    "linear_terms": True, "max_terms": 15, "add_term_prob": 0.2, "quadratic_init_frac": 0.3,
}

target_ga_configs = {
    'P': {"use_pareto": False, "complexity_coef": 1e-3},
    'W': {"stopping_threshold": 0.91},
    'Q': {"stopping_threshold": 0.91},
    'T': {"stopping_threshold": 0.91},
}

def predict_expression(expr, feature_cols, X):
    """Evaluate a symbolic equation and return one prediction per input row."""
    fn = sp.lambdify(sp.symbols(feature_cols), expr, "numpy")
    return np.broadcast_to(np.asarray(fn(*X.T), dtype=float), X.shape[0])


def run_experiment():
    """Fit equations on 100% of data."""
    records, equations, simplified_equations = [], {}, {}

    for target in reversed(targets):
        target_full = next((orig for orig, alias in col_map.items() if alias == target), target)
        data = target_data[target]

        feature_cols = [c for c in data.columns if c != target]
        X = data[feature_cols].to_numpy(dtype=float)
        y = data[target].to_numpy(dtype=float)

        x_mu, x_sd = X.mean(axis=0), X.std(axis=0)
        y_mu, y_sd = y.mean(), y.std()
        if np.any(x_sd == 0) or y_sd == 0:
            raise ValueError(f"Cannot standardize {target}: zero variance.")

        ga_config = default_ga_config.copy()
        ga_config.update(target_ga_configs.get(target, {}))
        ga_config["var_names"] = feature_cols

        print(f"Running: {target} (n={len(X)})")
        eqn, ga_r2 = run_ga(
            (X - x_mu) / x_sd, (y - y_mu) / y_sd, **ga_config
        )
        z_expr = sp.powsimp(sp.trigsimp(sp.cancel(eqn.to_sympy())))
        unscale = {
            symbol: (sp.Symbol(symbol.name) - x_mu[feature_cols.index(symbol.name)])
            / x_sd[feature_cols.index(symbol.name)]
            for symbol in z_expr.free_symbols
        }
        expr = y_mu + y_sd * z_expr.xreplace(unscale)
        simplified_expr, _ = simplify_equation(expr, data, y)

        r2 = r2_score(y, predict_expression(expr, feature_cols, X))
        equations[target] = expr
        simplified_equations[target] = simplified_expr
        records.append({
            "target": target, "target_name": target_full,
            "n": len(X), "r2": r2,
            "ga_r2_standardized": ga_r2,
            "equation": str(expr), "simplified_equation": str(simplified_expr),
        })
        print(f"{target_full}: R²={r2:.4f}")
        print(f"Equation: {expr}\n")

    results = pd.DataFrame(records)
    results.to_csv("results.csv", index=False)
    with open("eqns.tex", "w", encoding="utf-8") as f:
        for row in records:
            f.write(f"{row['target']} & {sp.latex(equations[row['target']])} & {row['r2']:.4f} \\\\\n")
    return results, equations, simplified_equations


# Run on full data
start_time = time.time()
results, equations, simplified_equations = run_experiment()
print("\nFull-data R²:")
print(results[["target", "target_name", "r2"]])
elapsed = time.time() - start_time
print(f"\nTotal runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

# Save simplified results
results_simplified = {row['target']: (row['simplified_equation'], row['r2']) for _, row in results.iterrows()}
pd.DataFrame(results_simplified, index=["Equation", "R^2 Score"]).T.to_csv("results_simplified.csv")
with open("eqns_simplified.tex", "w") as f:
    for target, (expr, score) in results_simplified.items():
        f.write(f"{target} & {sp.latex(simplified_equations[target])} & {score:.4f} \\\\\n")
