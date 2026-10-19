"""Regenerate pairplot.pdf from monsoon.csv — no GA run needed."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # STIX matches Times New Roman for math

# --- Load and prepare data (same pipeline as climate.py) ---
df = pd.read_csv("monsoon.csv")

df_long = df.melt(id_vars=["PARAMETER", "YEAR"], var_name="MONTH", value_name="VALUE")
df_pivot = df_long.pivot_table(
    index=["YEAR", "MONTH"], columns="PARAMETER", values="VALUE"
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
        'Soil Moisture ': 'W',
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
    return s[:2]

col_map = {c: sanitize(c) for c in df_final.columns}
df_final.rename(columns=col_map, inplace=True)
df_final = (
    df_final.apply(pd.to_numeric, errors="coerce")
    .replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    .dropna()
)

# --- Pairplot ---
color_mapping = {
    'Temperature': 'darkred',
    'Specific Humidity': 'blue',
    'Precipitation': 'darkviolet',
    'Soil Wetness': 'forestgreen',
}
vars_order = ['T', 'Q', 'P', 'W']
chvars = {'T': 'T', 'Q': 'Q', 'P': 'P', 'W': 'W'}
var_to_full = {'T': 'Temperature', 'Q': 'Specific Humidity', 'P': 'Precipitation', 'W': 'Soil Wetness'}

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        color_i = color_mapping[var_to_full[vars_order[i]]]
        if i == j:
            ax.hist(df_final[vars_order[i]], bins=20, color=color_i, edgecolor='black')
            ax.set_xlabel(chvars[vars_order[i]])
            ax.set_ylabel('Density')
        else:
            ax.scatter(df_final[vars_order[j]], df_final[vars_order[i]],
                       color=color_i, alpha=0.6, s=10)
            ax.set_xlabel(chvars[vars_order[j]])
            ax.set_ylabel(chvars[vars_order[i]])
        ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.savefig("pairplot.pdf", bbox_inches='tight')
plt.close()
print("Saved pairplot.pdf")
