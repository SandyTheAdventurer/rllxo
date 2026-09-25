from main import run_ga, set_global_seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import sympy as sp
import re

SEED = 42
set_global_seed(SEED)

color_mapping = {
    'Temperature': 'darkred',
    'Specific Humidity': 'blue',
    'Precipitation': 'darkviolet',
    'Soil Moisture': 'forestgreen'
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
    s = name.strip()
    s = re.sub(r"\W+", "_", s)
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

vars = ['Pr', 'So', 'Sp', 'Te']
chvars = {'Pr': 'P', 'So': 'W', 'Sp': 'Q', 'Te': 'T'}
var_to_full = {'Pr': 'Precipitation', 'So': 'Soil Moisture', 'Sp': 'Specific Humidity', 'Te': 'Temperature'}
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

default_ga_config = {
    "pop_size": 1500, "gens": 200, "max_init_depth": 4, "max_depth": 10,
    "crossover_rate": 0.9, "mutation_rate": 0.3, "tournament_k": 5, "elitism": 15,
    "use_learned_crossover": True,
    "mlp_layers": 8, "mlp_hidden_dim": 1024, "mlp_activation": "leaky_relu", "mlp_dropout": 0.1,
    "stopping_metric": "r2", "stopping_threshold": 1.0,
    "complexity_coef": 1e-6, "inclusion_coef": 1.0, "prob_const": 0.3, "const_range": (-5, 5),
    "point_mutation_prob": 0.1, "simplify_new_genomes": False, "simplify_timeout": 0.2,
    "temperature": 1.0, "entropy_weight": 0.08, "heuristic_bias_scale": 0.1,
    "opt_constants": True, "const_opt_freq": 10, "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 100, "verbose": True, "use_pareto": True, "print_freq": 10, "seed": SEED,
    "selector_update_freq": 1, "selector_warmup_gens": 5, "selector_recent_gen_window": 50,
    "selector_min_history": 64, "selector_reward_clip": 3.0, "selector_lr": 5e-4,
    "use_critic": True, "selector_device": "cpu"
}

target_ga_configs = {
    'Pr': {"use_pareto": False},
    'So': {"stopping_threshold": 0.9},
    'Sp': {},
    'Te': {},
}

results = {i: [None, None] for i in targets}

targets = targets[::-1]


for target in targets:
    target_full = next((orig for orig, alias in col_map.items() if alias == target), target)
    feature_cols = [c for c in df_final.columns if c != target]
    # Merge default config with any target-specific overrides
    ga_config = default_ga_config.copy()
    ga_config.update(target_ga_configs.get(target, {}))
    ga_config["var_names"] = feature_cols

    X = df_final[feature_cols].to_numpy(dtype=float)
    y = df_final[target].to_numpy(dtype=float)
    print(f"Running {target} against {feature_cols}")
    eqn, score = run_ga(X, y, **ga_config)

    sym_expr = sp.cancel(eqn.to_sympy())
    sym_expr = sp.trigsimp(sym_expr)
    sym_expr = sp.powsimp(sym_expr)
    results[target] = [sym_expr, score]
    print(f"Best equation for {target_full} (R^2={score:.4f}): {sym_expr}\n")

for target, (expr, score) in results.items():
    print(f"Target: {target}, R^2 Score: {score:.4f}")
    print(f"Equation: {expr}\n")
    feature_cols = [c for c in df_final.columns if c != target]
    y = df_final[target].to_numpy(dtype=float)
    X = df_final[feature_cols].to_numpy(dtype=float)
    lambdified = sp.lambdify(sp.symbols(feature_cols), expr, "numpy")
    y_pred = lambdified(*X.T)
    residuals = y - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(df_final.index, y, alpha=0.7, zorder=5, color="#001a4d")
    plt.plot(df_final.index, y_pred, color=color_mapping.get(var_to_full.get(target)), zorder=6)
    sigma = float(np.std(residuals))
    upper1 = y_pred + 0.5 * sigma
    lower1 = y_pred - 0.5 * sigma
    upper2 = y_pred + 1 * sigma
    lower2 = y_pred - 1 * sigma
    plt.fill_between(df_final.index, lower2, upper2, color=color_mapping.get(var_to_full.get(target)), alpha=0.12, label=f"±σ (σ={sigma:.3f})", zorder=2)
    plt.fill_between(df_final.index, lower1, upper1, color=color_mapping.get(var_to_full.get(target)), alpha=0.18, label=f"±0.5σ (σ={sigma:.3f})", zorder=3)
    plt.xlabel("Year")
    plt.ylabel(f"{var_to_full.get(target, target)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"predict_{target}.pdf", bbox_inches='tight')


print(f"Total score: {sum(score for _, score in results.values()):.4f}")

pd.DataFrame(results, index=["Equation", "R^2 Score"]).T.to_csv("results.csv")
with open("eqns.tex", "w") as f:
    for target, (expr, score) in results.items():
        f.write(f"{target} & {sp.latex(expr)} & {score:.4f} \\\\\n")