"""Generate prediction plots from saved equations — no GA run needed."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sp
import re

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # STIX matches Times New Roman for math

# --- Load and prepare data (same pipeline as climate.py) ---
df = pd.read_csv("monsoon.csv")
df_long = df.melt(id_vars=["PARAMETER", "YEAR"], var_name="MONTH", value_name="VALUE")
df_pivot = df_long.pivot_table(index=["YEAR", "MONTH"], columns="PARAMETER", values="VALUE").reset_index()

month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
             "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
             "OCT": 10, "NOV": 11, "DEC": 12}
df_pivot["MONTH_NUM"] = df_pivot["MONTH"].map(month_map)
df_pivot["DATE"] = pd.to_datetime(df_pivot["YEAR"].astype(str) + "-" + df_pivot["MONTH_NUM"].astype(str))
df_pivot.set_index("DATE", inplace=True)
df_final = df_pivot.drop(columns=["YEAR", "MONTH", "MONTH_NUM"])

# Rename columns
name_map = {'Precipitation': 'P', 'Soil Moisture ': 'W', 'Soil Moisture': 'W',
            'Specific Humidity': 'Q', 'Temperature': 'T'}
df_final.rename(columns={c: name_map.get(c, name_map.get(c.strip(), c)) for c in df_final.columns}, inplace=True)
df_final = df_final.apply(pd.to_numeric, errors="coerce").replace([pd.NA, float("inf"), -float("inf")], pd.NA).dropna()

# Seasonal + lag features
df_sorted = df_final.sort_index()
month = df_sorted.index.month
df_sorted["Ms"] = np.sin(2 * np.pi * month / 12)
df_sorted["Mc"] = np.cos(2 * np.pi * month / 12)
for c in ["P", "W", "Q", "T"]:
    df_sorted[f"{c}1"] = df_sorted.groupby(df_sorted.index.year)[c].shift(1)
df_sorted = df_sorted.dropna()

# --- Load results ---
results = pd.read_csv("results.csv")

color_mapping = {'T': 'darkred', 'Q': 'blue', 'P': 'darkviolet', 'W': 'forestgreen'}
var_to_full = {
    'T': '$T$ (Temperature)',
    'Q': '$Q$ (Specific Humidity)',
    'P': '$P$ (Precipitation)',
    'W': '$W$ (Soil Wetness)',
}

feature_vars = ["Ms", "Mc", "P", "W", "Q", "T"] + [f"{c}1" for c in ["P", "W", "Q", "T"]]

# Store per-target results for combined figure
target_results = {}

for _, row in results.iterrows():
    target = row['target']
    expr_str = row['equation']

    # Parse equation — need local_dict because 'Q' conflicts with sympy's Q
    feature_cols = [c for c in feature_vars if c != target]
    all_var_names = feature_cols + [target]
    local_dict = {name: sp.Symbol(name) for name in all_var_names}
    local_dict.update({"Abs": sp.Abs})
    expr = sp.parsing.sympy_parser.parse_expr(expr_str, local_dict=local_dict)
    data = df_sorted[feature_cols + [target]].dropna()

    y = data[target].to_numpy(dtype=float)
    X = data[feature_cols].to_numpy(dtype=float)

    fn = sp.lambdify(sp.symbols(feature_cols), expr, "numpy")
    y_pred = np.broadcast_to(np.asarray(fn(*X.T), dtype=float), y.shape)

    residuals = y - y_pred
    sigma = float(np.std(residuals))

    target_results[target] = dict(data=data, y=y, y_pred=y_pred, sigma=sigma)

    plt.figure(figsize=(10, 5))
    plt.scatter(data.index, y, alpha=0.7, zorder=5, color="#001a4d", s=15)
    plt.plot(data.index, y_pred, color=color_mapping.get(target, 'black'), zorder=6)

    upper1 = y_pred + 0.5 * sigma
    lower1 = y_pred - 0.5 * sigma
    upper2 = y_pred + 1 * sigma
    lower2 = y_pred - 1 * sigma
    plt.fill_between(data.index, lower2, upper2, color=color_mapping.get(target, 'grey'),
                     alpha=0.12, label=f"$\\pm\\sigma$ ($\\sigma$={sigma:.3f})", zorder=2)
    plt.fill_between(data.index, lower1, upper1, color=color_mapping.get(target, 'grey'),
                     alpha=0.18, label=f"$\\pm 0.5\\sigma$ ($\\sigma$={sigma:.3f})", zorder=3)
    plt.xlabel("$t$ (in Years)")
    plt.ylabel(f"{var_to_full.get(target, target)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"predict_{target}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved predict_{target}.pdf")

# --- Combined 2×2 figure matching image.png (T, Q, P, W order) ---
panel_order = ['T', 'Q', 'P', 'W']   # (a) top-left, (b) top-right, (c) bottom-left, (d) bottom-right
subcaptions = ['(a)', '(b)', '(c)', '(d)']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

for ax, target, subcap in zip(axes_flat, panel_order, subcaptions):
    res = target_results[target]
    data, y, y_pred, sigma = res['data'], res['y'], res['y_pred'], res['sigma']
    color = color_mapping.get(target, 'black')
    upper1 = y_pred + 0.5 * sigma
    lower1 = y_pred - 0.5 * sigma
    upper2 = y_pred + 1 * sigma
    lower2 = y_pred - 1 * sigma

    ax.fill_between(data.index, lower2, upper2, color=color,
                    alpha=0.12, label=f"$\\pm\\sigma$ ($\\sigma$={sigma:.3f})", zorder=2)
    ax.fill_between(data.index, lower1, upper1, color=color,
                    alpha=0.18, label=f"$\\pm 0.5\\sigma$ ($\\sigma$={sigma:.3f})", zorder=3)
    ax.plot(data.index, y_pred, color=color, zorder=6)
    ax.scatter(data.index, y, alpha=0.7, zorder=5, color="#001a4d", s=15)

    ax.set_xlabel("$t$ (in Years)")
    ax.set_ylabel(var_to_full.get(target, target))
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(subcap, y=-0.18, fontsize=12)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("predict_combined.pdf", bbox_inches='tight')
plt.close()
print("Saved predict_combined.pdf")

print("Done — all prediction plots generated.")
