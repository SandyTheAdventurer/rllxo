"""Statistics and figures for the revised paper, from the frozen held-out run.

Reads results/heldout/heldout.csv and heldout_curves.jsonl (63 equations x 30 seeds x 6 arms).
Writes figures/*.pdf and the numbers quoted in the paper (figures/numbers.txt). Usage: python analysis/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/heldout"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical slots (dataviz reference palette, light mode): fixed per arm, never by rank
COLOR = {"random": "#2a78d6", "a2c_t03": "#eb6834", "ex64_t03": "#1baf7a", "random_x4": "#eda100"}
STYLE = {"random": "-", "a2c_t03": "-", "ex64_t03": "--", "random_x4": ":"}
LABEL = {"random": "Random crossover", "a2c_t03": "GP-SR-RLC (policy + H)", "ex64_t03": "H only, no policy",
         "random_x4": "Random, 4x generations"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "legend.frameon": False})


def family(s):
    return "Nguyen" if s <= 12 else "Keijzer" if s <= 115 else "SRSD"


def name(s):
    return f"N{s}" if s <= 12 else f"K{s - 100}" if s <= 115 else f"SRSD-{s - 2000}"


df = pd.read_csv(DATA / "heldout.csv")
df["score"] = df["test_r2"].where(np.isfinite(df["test_r2"]) & (df["test_r2"] > 0) & (df["status"] == "ok"), 0.0)
df["solved"] = (df["score"] >= 0.999).astype(float)
df["recovered"] = df["recovered"].fillna(0).where(df["status"] == "ok", 0)
med = df.pivot_table(index="spec_id", columns="arm", values="score", aggfunc="median")
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def paired(a, b, metric="score", agg="median"):
    t = df.pivot_table(index="spec_id", columns="arm", values=metric, aggfunc=agg)
    d = t[a] - t[b]
    nz = d[d != 0]
    rk = rankdata(np.abs(nz))
    wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
    r = wilcoxon(nz, zero_method="wilcox")  # as analyze_heldout.py: non-zero differences only (exact test)
    rbc = (wp - wm) / (wp + wm)  # matched-pairs rank-biserial correlation
    say(f"{a} vs {b} [{metric}/{agg}]: W+={wp:.0f} W-={wm:.0f} W(min)={r.statistic:.0f} p={r.pvalue:.3g} "
        f"r_rb={rbc:.2f} wins={int((d > 0).sum())} losses={int((d < 0).sum())} ties={int((d == 0).sum())} "
        f"mean diff={d.mean():.4f} median diff={d.median():.4f}")


say("== Aggregate tests (per-equation medians, 63 equations)")
for b in ["random", "ex64_t03", "random_x4"]:
    paired("a2c_t03", b)
paired("a2c_t03", "random", "solved", "mean")
paired("a2c_t03", "random", "recovered", "mean")
for a in ["ex64_t03", "dir_t03", "mlp", "random_x4"]:
    paired(a, "random")
for fam in ["Nguyen", "Keijzer", "SRSD"]:
    ids = [s for s in med.index if family(s) == fam]
    d = med.loc[ids, "a2c_t03"] - med.loc[ids, "random"]
    r = wilcoxon(d[d != 0], zero_method="wilcox")
    say(f"family {fam}: n={len(ids)} W={r.statistic:.0f} p={r.pvalue:.3g} wins={(d > 0).sum()} losses={(d < 0).sum()}")

say("== Per arm")
for arm, g in df.groupby("arm"):
    ok = g[g.status == "ok"]
    say(f"{arm}: mean median R2={med[arm].mean():.4f} solved={g.solved.mean():.3f} recovered={g.recovered.mean():.3f} "
        f"sec={ok.elapsed_sec.mean():.1f} size={ok['size'].mean():.1f} median size={ok['size'].median():.0f} "
        f"gens={ok.gens_used.mean():.1f} errors={(g.status != 'ok').sum()}")

say("== Per-equation table rows (a2c_t03 vs random), Vargha-Delaney A12 and Holm-corrected Mann-Whitney U")
rows = []
for s in sorted(df.spec_id.unique()):
    a = df[(df.spec_id == s) & (df.arm == "a2c_t03")]
    b = df[(df.spec_id == s) & (df.arm == "random")]
    if np.array_equal(np.sort(a.score.values), np.sort(b.score.values)):
        p = 1.0
    else:
        p = mannwhitneyu(a.score, b.score, alternative="two-sided").pvalue
    u = mannwhitneyu(a.score, b.score, alternative="two-sided").statistic if p < 1 else len(a) * len(b) / 2
    rows.append(dict(eq=name(s), spec=s, fam=family(s), ma=a.score.median(), mb=b.score.median(), sa=a.solved.mean(),
                     sb=b.solved.mean(), ra=a.recovered.mean(), rb=b.recovered.mean(), za=a["size"].median(),
                     zb=b["size"].median(), A12=u / (len(a) * len(b)), p=p))
tab = pd.DataFrame(rows)
order = np.argsort(tab.p.values)
m = len(tab)
holm = np.empty(m)
running = 0.0
for k, i in enumerate(order):
    running = max(running, (m - k) * tab.p.values[i])
    holm[i] = min(1.0, running)
tab["holm"] = holm
tab.to_csv(OUT / "per_equation.csv", index=False)
say(f"Holm-significant: a2c better {((tab.holm < .05) & (tab.A12 > .5)).sum()}, random better "
    f"{((tab.holm < .05) & (tab.A12 < .5)).sum()}")
for fam in ["Nguyen", "Keijzer", "SRSD"]:
    t = tab[tab.fam == fam]
    say(f"  {fam}: better {((t.holm < .05) & (t.A12 > .5)).sum()} of {len(t)}")
say(f"A12 large (>=0.71): {(tab.A12 >= .71).sum()}, A12 <= 0.29: {(tab.A12 <= .29).sum()}")


def fmt_p(p):
    if p >= 0.995:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


with open(OUT / "per_equation_rows.tex", "w") as f:
    for fam in ["Nguyen", "Keijzer", "SRSD"]:
        f.write(f"\\midrule\n\\multicolumn{{9}}{{l}}{{\\textit{{{fam}}}}} \\\\\n")
        for r in tab[tab.fam == fam].itertuples():
            star = "$^{*}$" if r.holm < 0.05 else ""
            f.write(f"{r.eq} & {r.ma:.4f} & {r.mb:.4f} & {r.sa:.2f} & {r.sb:.2f} & {r.ra:.2f} & {r.rb:.2f} & "
                    f"{r.A12:.2f} & {fmt_p(r.holm)}{star} \\\\\n")

# ---- Convergence: best-so-far training R² vs generation and vs wall time, mean over all 1890 runs per arm
curves = {}
with open(DATA / "heldout_curves.jsonl") as f:
    for line in f:
        j = json.loads(line)
        curves[(j["arm"], j["spec_id"], j["seed"])] = j["r2"]
el = {(r.arm, r.spec_id, r.seed): (r.elapsed_sec, r.gens_used) for r in df.itertuples()}
arms = ["random", "a2c_t03", "ex64_t03", "random_x4"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
G = 800
tgrid = np.linspace(0, 120, 241)
for arm in arms:
    keys = [k for k in curves if k[0] == arm]
    gmax = 800 if arm == "random_x4" else 200
    M = np.zeros((len(keys), gmax))
    T = np.zeros((len(keys), len(tgrid)))
    for n, k in enumerate(keys):
        c = np.clip(np.nan_to_num(np.maximum.accumulate(np.clip(np.nan_to_num(curves[k], nan=0.0), 0, 1))), 0, 1)
        if len(c) == 0:
            c = np.zeros(1)
        row = np.full(gmax, c[-1])  # early-stopped runs keep their final value
        row[:len(c)] = c[:gmax]
        M[n] = row
        sec, gens = el.get(k, (np.nan, len(c)))
        per = sec / max(len(c), 1) if np.isfinite(sec) else np.nan
        idx = np.minimum((tgrid / per).astype(int), len(c) - 1) if per and np.isfinite(per) and per > 0 else np.full(len(tgrid), len(c) - 1)
        T[n] = c[np.clip(idx, 0, len(c) - 1)]
    axes[0].plot(np.arange(1, gmax + 1), M.mean(0), STYLE[arm], color=COLOR[arm], label=LABEL[arm])
    axes[1].plot(tgrid, T.mean(0), STYLE[arm], color=COLOR[arm], label=LABEL[arm])
axes[0].set_xscale("log")
axes[0].set_xlabel("Generation (log scale)")
axes[0].set_ylabel("Mean best training $R^2$")
axes[1].set_xlabel("Wall-clock time per run (s)")
axes[0].set_title("(a) Against generations", loc="left", fontsize=9)
axes[1].set_title("(b) Against time", loc="left", fontsize=9)
for ax in axes:
    ax.set_ylim(0.2, 0.86)
axes[1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "convergence.pdf")
plt.close(fig)

# ---- Distributions: test R² over 30 seeds per equation, random vs GP-SR-RLC, Nguyen + Keijzer
nk = tab[tab.fam != "SRSD"]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
pos = np.arange(len(nk))
for off, arm in [(-0.18, "random"), (0.18, "a2c_t03")]:
    data = [df[(df.spec_id == s) & (df.arm == arm)].score.values for s in nk.spec]
    bp = ax.boxplot(data, positions=pos + off, widths=0.32, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=COLOR[arm], markeredgewidth=0),
                    medianprops=dict(color="black", linewidth=1.2), whiskerprops=dict(color=COLOR[arm]),
                    capprops=dict(color=COLOR[arm]), boxprops=dict(edgecolor=COLOR[arm]))
    for b in bp["boxes"]:
        b.set_facecolor(COLOR[arm])
        b.set_alpha(0.55)
    ax.plot([], [], "s", color=COLOR[arm], alpha=0.7, label=LABEL[arm])
for i, r in enumerate(nk.itertuples()):
    if r.holm < 0.05:
        ax.text(i, 1.02, "*", ha="center", fontsize=10)
ax.set_xticks(pos)
ax.set_xticklabels(list(nk["eq"]), rotation=90)
ax.set_ylabel("Test $R^2$ (30 seeds)")
ax.set_ylim(0, 1.08)
ax.legend(loc="lower left", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "boxplots_nk.pdf")
plt.close(fig)

# ---- Per-equation medians, GP-SR-RLC vs random, all 63 equations
fig, ax = plt.subplots(figsize=(3.5, 3.3))
marks = {"Nguyen": "o", "Keijzer": "s", "SRSD": "^"}
for fam, mk in marks.items():
    t = tab[tab.fam == fam]
    ax.scatter(t.mb, t.ma, marker=mk, s=22, facecolor="none" if fam == "SRSD" else COLOR["a2c_t03"],
               edgecolor=COLOR["a2c_t03"], linewidth=1.0, label=fam)
ax.plot([0, 1], [0, 1], color="#888888", linewidth=1)
ax.set_xlabel("Random crossover, median test $R^2$")
ax.set_ylabel("GP-SR-RLC, median test $R^2$")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "scatter_medians.pdf")
plt.close(fig)

(OUT / "numbers.txt").write_text("\n".join(lines) + "\n")
