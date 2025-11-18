import numpy as np
import pandas as pd
import re
from main import run_ga
import sympy as sp

def sanitize(name: str) -> str:
    s = name.strip()
    s = re.sub(r"\W+", "_", s)
    if s and s[0].isdigit():
        s = f"v_{s}"
    return s

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return r2, rmse

# Load and preprocess data
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
col_map = {c: sanitize(c) for c in df_final.columns}
df_final.rename(columns=col_map, inplace=True)

df_final = (
    df_final.apply(pd.to_numeric, errors="coerce")
    .replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    .dropna()
)

# Test configuration
target = "Soil_Moisture"
feature_cols = [c for c in df_final.columns if c != target]

X = df_final[feature_cols].to_numpy(dtype=float)
y = df_final[target].to_numpy(dtype=float)

unary_funcs = {
    'sin': (np.sin, sp.sin),
    'cos': (np.cos, sp.cos),
    'tan': (np.tan, sp.tan),
    'exp': (np.exp, sp.exp),
    'sqrt': (np.sqrt, sp.sqrt)
}

# Run tests with 20 different seeds
num_seeds = 20
r2_scores = []

print("Testing configuration across 20 different seeds...")
print("-" * 60)

for seed in range(num_seeds):
    print(f"\nRunning seed {seed + 1}/{num_seeds} (SEED={seed})")
    
    best, best_score = run_ga(
        X,
        y,
        pop_size=1500,
        gens=300,
        max_init_depth=4,
        max_depth=10,
        crossover_rate=0.85,
        mutation_rate=0.2,
        tournament_k=5,
        elitism=15,
        var_names=feature_cols,
        unary_funcs=unary_funcs,
        use_learned_crossover=True,
        selector_lr=0.0005,
        use_transformer=False,
        transformer_heads=8,
        transformer_layers=4,
        mlp_layers=10,
        mlp_hidden_dim=128,
        mlp_activation="gelu",
        mlp_dropout=0.1,
        stopping_metric="r2",
        stopping_threshold=None,
        complexity_coef=1e-5,
        inclusion_coef=0.5,
        prob_const=0.35,
        const_range=(-3, 3),
        point_mutation_prob=0.1,
        selector_update_freq=3,
        temperature=0.7,
        critic_weight=0.6,
        entropy_weight=0.02,
        verbose=False,
        print_freq=10,
        seed=seed,
    )
    
    if best is not None:
        feature_values = {name: df_final[name].to_numpy(dtype=float) for name in feature_cols}
        y_pred = best.evaluate(feature_values)
        r2, rmse = regression_metrics(y, y_pred)
        r2_scores.append(r2)
        print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, Best Score: {best_score:.6f}")
    else:
        print(f"  WARNING: Seed {seed} returned None")

# Calculate statistics
r2_scores = np.array(r2_scores)
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
min_r2 = np.min(r2_scores)
max_r2 = np.max(r2_scores)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Number of successful runs: {len(r2_scores)}/{num_seeds}")
print(f"Mean R²: {mean_r2:.4f}")
print(f"Std R²:  {std_r2:.4f}")
print(f"Min R²:  {min_r2:.4f}")
print(f"Max R²:  {max_r2:.4f}")
print("=" * 60)

# Display all scores
print("\nAll R² scores:")
for i, score in enumerate(r2_scores):
    print(f"  Seed {i}: {score:.4f}")