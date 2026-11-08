"""Analysis of the held-out test, exactly as fixed in results/HELDOUT_PLAN.md before the run.

Usage: python analyze_heldout.py [results/compare_selectors/heldout.csv]

Per equation, each arm is scored by its median test R² over the seeds (non-finite or negative R² and
error rows count as 0). The primary test is a2c_t03 vs random: two-sided Wilcoxon signed-rank over the
63 per-equation medians, zero differences dropped. Four secondary tests are Holm-corrected as one
family. Everything else is descriptive.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEV = {6, 7, 12, 114}
EQUATIONS = [e for e in [*range(1, 13), *range(101, 116), *range(2000, 2040)] if e not in DEV]
ARMS = ["random", "a2c_t03", "ex64_t03", "dir_t03", "random_x4", "mlp"]
SEEDS = range(30)
SOLVED = 0.999


def family(e):
    return "Nguyen" if e < 100 else "Keijzer" if e < 1000 else "SRSD"


def load(path):
    d = pd.read_csv(path)
    d = d[d.arm.isin(ARMS) & d.spec_id.isin(EQUATIONS) & d.seed.isin(SEEDS)]
    d = d.drop_duplicates(["arm", "spec_id", "seed"], keep="last")
    ok = d.status == "ok"
    r2 = pd.to_numeric(d.test_r2, errors="coerce")
    d["r"] = r2.where(ok & np.isfinite(r2), 0.0).clip(lower=0.0)
    d["rec"] = pd.to_numeric(d.recovered, errors="coerce").where(ok, 0).fillna(0).astype(float)
    d["solved"] = (d.r >= SOLVED).astype(float)
    return d


def per_equation(d, col, how):
    return d.groupby(["arm", "spec_id"])[col].agg(how).unstack("arm")


def test(x, y):
    """(wins, losses, ties, p): two-sided Wilcoxon signed-rank on x - y, zero differences dropped."""
    z = (x - y).dropna()
    w, l = int((z > 0).sum()), int((z < 0).sum())
    nz = z[z != 0]
    p = float(wilcoxon(nz, zero_method="wilcox").pvalue) if len(nz) else 1.0
    return w, l, len(z) - w - l, p


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main(path):
    d = load(path)
    expected = len(ARMS) * len(EQUATIONS) * len(SEEDS)
    print(f"# Held-out analysis ({path})\n")
    print(f"Rows: {len(d)} of {expected} expected; error rows: {int((d.status != 'ok').sum())}")
    missing = expected - len(d)
    if missing:
        print(f"**INCOMPLETE: {missing} runs missing.** The plan's test is on the full set.")
    med = per_equation(d, "r", "median")
    sol = per_equation(d, "solved", "mean")
    rec = per_equation(d, "rec", "mean")

    w, l, t, p = test(med["a2c_t03"], med["random"])
    print("\n## Primary: a2c_t03 vs random, median test R² per equation")
    print(f"{len(med)} equations: {w} wins, {l} losses, {t} ties; two-sided Wilcoxon p = {p:.3g}"
          f" ({'significant' if p < 0.05 else 'not significant'} at 0.05)")
    print(f"Mean of per-equation medians: a2c_t03 {med['a2c_t03'].mean():.4f}, random {med['random'].mean():.4f}")

    sec = [("a2c_t03 vs ex64_t03 (does the policy add anything), median R²", med["a2c_t03"], med["ex64_t03"]),
           ("a2c_t03 vs random_x4 (equal compute), median R²", med["a2c_t03"], med["random_x4"]),
           ("a2c_t03 vs random, fraction of seeds solved (R² >= 0.999)", sol["a2c_t03"], sol["random"]),
           ("a2c_t03 vs random, exact recovery rate", rec["a2c_t03"], rec["random"])]
    res = [test(x, y) for _, x, y in sec]
    adj = holm(np.array([r[3] for r in res]))
    print("\n## Secondary (Holm-corrected as one family)")
    print("| Comparison | Wins | Losses | Ties | p | Holm p |\n|---|---|---|---|---|---|")
    for (name, _, _), (w, l, t, p), pa in zip(sec, res, adj):
        print(f"| {name} | {w} | {l} | {t} | {p:.3g} | {pa:.3g} |")

    print("\n## Descriptive (no correction, not part of the claim)")
    print("| Comparison | Wins | Losses | Ties | p |\n|---|---|---|---|---|")
    for a, b in [("dir_t03", "random"), ("mlp", "random"), ("ex64_t03", "random"), ("random_x4", "random")]:
        w, l, t, p = test(med[a], med[b])
        print(f"| {a} vs {b}, median R² | {w} | {l} | {t} | {p:.3g} |")
    print("\n| Arm | Mean median R² | Solved | Recovered | Mean seconds per run | Mean size |\n|---|---|---|---|---|---|")
    for a in ARMS:
        s = d[d.arm == a]
        print(f"| {a} | {med[a].mean():.4f} | {sol[a].mean():.3f} | {rec[a].mean():.3f} | {s.elapsed_sec.mean():.1f} | "
              f"{pd.to_numeric(s['size'], errors='coerce').mean():.1f} |")
    print("\n### a2c_t03 vs random by benchmark family (median R²)")
    for fam in ["Nguyen", "Keijzer", "SRSD"]:
        idx = [e for e in med.index if family(e) == fam]
        w, l, t, p = test(med.loc[idx, "a2c_t03"], med.loc[idx, "random"])
        print(f"- {fam} ({len(idx)} equations): {w} wins, {l} losses, {t} ties, p = {p:.3g}")
    errs = d[d.status != "ok"]
    if len(errs):
        print("\n### Error rows")
        print(errs.groupby(["arm", "error"]).size().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/compare_selectors/heldout.csv")
