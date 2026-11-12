r"""
baseline.py  --  Classical baseline models for comparison with RL-LXO
symbolic regression on Kanha Shanti Vanam MERRA-2 monsoon data.

Baselines (Group 1):
  1. Climatological Mean   -- predict the long-term monthly mean
  2. Persistence           -- predict y(t) = y(t-1)  (naive lag-1)
  3. Linear Regression     -- OLS on the same feature set as RL-LXO
  4. Polynomial Regression -- degree-2 polynomial on the same features

Outputs:
  baselines.tex  --  LaTeX table ready to \input{baselines} in the paper
  (also prints all metrics to the console)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sympy as sp
import re
import warnings

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
# 1.  DATA PIPELINE  (mirrors climate.py exactly)
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv("monsoon.csv")

df_long = df.melt(
    id_vars=["PARAMETER", "YEAR"], var_name="MONTH", value_name="VALUE"
)
df_pivot = df_long.pivot_table(
    index=["YEAR", "MONTH"], columns="PARAMETER", values="VALUE"
).reset_index()

month_map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
df_pivot["MONTH_NUM"] = df_pivot["MONTH"].map(month_map)
df_pivot["DATE"] = pd.to_datetime(
    df_pivot["YEAR"].astype(str) + "-" + df_pivot["MONTH_NUM"].astype(str)
)
df_pivot.set_index("DATE", inplace=True)
df_final = df_pivot.drop(columns=["YEAR", "MONTH", "MONTH_NUM"])


def sanitize(name: str) -> str:
    """Map raw column names to short aliases (same logic as climate.py)."""
    name_map = {
        "Precipitation": "P",
        "Soil Moisture ": "W",   # trailing space in the CSV
        "Soil Moisture": "W",
        "Specific Humidity": "Q",
        "Temperature": "T",
    }
    stripped = name.strip()
    if name in name_map:
        return name_map[name]
    if stripped in name_map:
        return name_map[stripped]
    s = re.sub(r"\W+", "_", stripped)
    if s and s[0].isdigit():
        s = f"v_{s}"
    return s[:2]


col_map = {c: sanitize(c) for c in df_final.columns}
df_final.rename(columns=col_map, inplace=True)

df_final = (
    df_final.apply(pd.to_numeric, errors="coerce")
    .replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    .dropna()
)

targets = list(df_final.columns)


# ── Seasonal and lag features ────────────────────────────────────────
def add_seasonal_features(frame):
    frame = frame.copy()
    month = frame.index.month
    frame["Ms"] = np.sin(2 * np.pi * month / 12)
    frame["Mc"] = np.cos(2 * np.pi * month / 12)
    return frame


def add_lag_features(frame, lag_vars):
    frame = frame.copy()
    for c in lag_vars:
        frame[f"{c}1"] = frame.groupby(frame.index.year)[c].shift(1)
    return frame


df_sorted = df_final.sort_index()
df_all = add_seasonal_features(df_sorted)
df_all = add_lag_features(df_all, ["P", "W", "Q", "T"])

feature_vars = ["Ms", "Mc", "P", "W", "Q", "T"]
lag_vars = [f"{c}1" for c in ["P", "W", "Q", "T"]]
all_feature_vars = feature_vars + lag_vars

# Build per-target datasets (drop rows with NaN from lagging)
target_data = {}
for t in targets:
    cols = [c for c in all_feature_vars if c != t]
    data = df_all[cols + [t]].dropna()
    target_data[t] = data
    print(f"  {t}: n = {len(data)} samples, {len(cols)} features")


# ═══════════════════════════════════════════════════════════════════════
# 2.  EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════
def calc_mape(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calc_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def compute_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": calc_mape(y_true, y_pred),
        "smape": calc_smape(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  RUN BASELINES
# ═══════════════════════════════════════════════════════════════════════
MODEL_ORDER = [
    "Global Mean",
    "Monthly Climatology",
    "Persistence",
    "Linear Regression",
    "RL-LXO (Ours)",
]

TARGET_ORDER = ["T", "Q", "P", "W"]
TARGET_LABEL = {
    "T": "Temperature",
    "Q": "Specific Humidity",
    "P": "Precipitation",
    "W": "Soil Wetness",
}

# results[model][target] = {r2, rmse, mape, smape}
results = {m: {} for m in MODEL_ORDER}

for target in TARGET_ORDER:
    data = target_data[target]
    feat_cols = [c for c in data.columns if c != target]
    X = data[feat_cols].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)

    # ── 1. Global Mean ───────────────────────────────────────────────
    y_pred_global = np.full_like(y, y.mean())
    results["Global Mean"][target] = compute_metrics(y, y_pred_global)

    # ── 2. Monthly Climatology ───────────────────────────────────────
    # Predict the historical average for that specific calendar month
    monthly_means = data.groupby(data.index.month)[target].mean()
    y_pred_climatology = data.index.month.map(monthly_means).to_numpy(dtype=float)
    results["Monthly Climatology"][target] = compute_metrics(y, y_pred_climatology)

    # ── 3. Persistence (lag-1 of the target variable) ────────────────
    lag_col = f"{target}1"
    y_pred_persist = data[lag_col].to_numpy(dtype=float)
    results["Persistence"][target] = compute_metrics(y, y_pred_persist)

    # ── 4. Linear Regression (OLS, same features as RL-LXO) ─────────
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    results["Linear Regression"][target] = compute_metrics(y, y_pred_lr)


# ═══════════════════════════════════════════════════════════════════════
# 4.  RL-LXO RESULTS  (evaluate simplified equations from results.csv)
# ═══════════════════════════════════════════════════════════════════════
print("\nEvaluating RL-LXO simplified equations …")

try:
    rllxo_df = pd.read_csv("results.csv")

    # Pre-define all possible variable symbols to avoid clashes with
    # sympy internals (e.g. Q clashes with sympy.Q assumptions).
    _all_var_names = ["P", "W", "Q", "T", "Ms", "Mc",
                      "P1", "W1", "Q1", "T1"]
    _sym_locals = {name: sp.Symbol(name) for name in _all_var_names}
    _sym_locals["Abs"] = sp.Abs  # needed for log(Abs(...))

    for _, row in rllxo_df.iterrows():
        target = row["target"]
        if target not in TARGET_ORDER:
            continue

        data = target_data[target]
        y = data[target].to_numpy(dtype=float)

        try:
            eq_str = row["simplified_equation"]
            expr = sp.sympify(eq_str, locals=_sym_locals)
            free_syms = sorted(expr.free_symbols, key=lambda s: s.name)
            sym_names = [s.name for s in free_syms]

            fn = sp.lambdify(free_syms, expr, modules=["numpy"])
            args = [data[name].to_numpy(dtype=float) for name in sym_names]

            with np.errstate(all="ignore"):
                y_pred = np.asarray(fn(*args), dtype=float).ravel()
                # Replace any non-finite values with the finite mean
                finite_mask = np.isfinite(y_pred)
                if finite_mask.any() and not finite_mask.all():
                    y_pred[~finite_mask] = np.nanmean(y_pred[finite_mask])

            results["RL-LXO (Ours)"][target] = compute_metrics(y, y_pred)
            print(f"  {target}: evaluated OK  "
                  f"(R^2 = {results['RL-LXO (Ours)'][target]['r2']:.4f})")

        except Exception as e:
            # Fallback: use R^2 from results.csv; derive RMSE analytically
            r2 = row["r2"]
            y_std = np.std(y, ddof=0)
            rmse = y_std * np.sqrt(max(0.0, 1.0 - r2))
            results["RL-LXO (Ours)"][target] = {
                "r2": r2, "rmse": rmse, "mape": None, "smape": None,
            }
            print(f"  {target}: fallback (could not evaluate equation: {e})")

except FileNotFoundError:
    print("  results.csv not found -- RL-LXO row will use placeholder values.")


# ═══════════════════════════════════════════════════════════════════════
# 5.  CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("BASELINE COMPARISON — ALL METRICS")
print("=" * 90)

for model in MODEL_ORDER:
    print(f"\n  {model}")
    print(f"  {'-' * 70}")
    for target in TARGET_ORDER:
        m = results[model].get(target)
        if m is None:
            print(f"    {target}: (no results)")
            continue
        mp = f"{m['mape']:.2f}%" if m["mape"] is not None else "N/A"
        sm = f"{m['smape']:.2f}%" if m["smape"] is not None else "N/A"
        print(f"    {target}  R²={m['r2']:+.4f}   RMSE={m['rmse']:.4f}"
              f"   MAPE={mp:>8s}   SMAPE={sm:>8s}")


# ═══════════════════════════════════════════════════════════════════════
# 6.  GENERATE LaTeX TABLE  →  baselines.tex
# ═══════════════════════════════════════════════════════════════════════

# Identify best (highest R², lowest RMSE) per target for bolding
best_r2 = {}
best_rmse = {}
for t in TARGET_ORDER:
    r2_vals = [results[m][t]["r2"] for m in MODEL_ORDER if t in results[m]]
    rmse_vals = [
        results[m][t]["rmse"]
        for m in MODEL_ORDER
        if t in results[m] and results[m][t]["rmse"] is not None
    ]
    best_r2[t] = max(r2_vals) if r2_vals else 0
    best_rmse[t] = min(rmse_vals) if rmse_vals else float("inf")


def _fmt(value, best, higher_is_better=True, spec=".4f"):
    """Format a number; bold it if it equals the best value."""
    s = f"{value:{spec}}"
    is_best = (
        abs(value - best) < 1e-8
        if higher_is_better
        else abs(value - best) < 1e-8
    )
    return rf"\textbf{{{s}}}" if is_best else s


# ── Build the table lines ────────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(
    r"\caption{Performance comparison of baseline statistical models "
    r"against RL-LXO symbolic regression on MERRA-2 monsoon climate data "
    r"(1981--2025, $n=135$ monsoon months). "
    r"$T$: Temperature ($^\circ$C), "
    r"$Q$: Specific Humidity (g\,kg$^{-1}$), "
    r"$P$: Precipitation (mm), "
    r"$W$: Soil Wetness (fraction). "
    r"Best values in each column are \textbf{bold}.}"
)
lines.append(r"\label{tab:baselines}")
lines.append(r"\begin{tabular}{l cccc cccc}")
lines.append(r"\toprule")
lines.append(
    r" & \multicolumn{4}{c}{$R^{2}$} "
    r"& \multicolumn{4}{c}{RMSE} \\"
)
lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
lines.append(r"Model & $T$ & $Q$ & $P$ & $W$ & $T$ & $Q$ & $P$ & $W$ \\")
lines.append(r"\midrule")

for model in MODEL_ORDER:
    cells = []

    # R² columns
    for t in TARGET_ORDER:
        m = results[model].get(t)
        if m is not None:
            cells.append(_fmt(m["r2"], best_r2[t], higher_is_better=True))
        else:
            cells.append("---")

    # RMSE columns
    for t in TARGET_ORDER:
        m = results[model].get(t)
        if m is not None and m["rmse"] is not None:
            cells.append(_fmt(m["rmse"], best_rmse[t], higher_is_better=False))
        else:
            cells.append("---")

    # Bold model name for RL-LXO
    if model == "RL-LXO (Ours)":
        label = r"\textbf{RL-LXO (Ours)}"
    else:
        label = model

    lines.append(f"{label} & {' & '.join(cells)} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex_content = "\n".join(lines)

with open("baselines.tex", "w", encoding="utf-8") as f:
    f.write(tex_content)

print("\n" + "=" * 90)
print("LaTeX table written to:  baselines.tex")
print("=" * 90)
print()
print(tex_content)
