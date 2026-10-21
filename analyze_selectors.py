"""Summarise compare_selectors.py output: per equation and arm, recovery rate, train/test R²,
size, generations and wall time, plus Mann-Whitney U tests (Holm-corrected within each
reference arm) of test R² against the random baseline and the paper's MLP selector.

Usage: python analyze_selectors.py [results/compare_selectors/main.csv]
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main(path):
    df = pd.read_csv(path)
    df = df[df.status == "ok"]
    arms = [a for a in ["random", "heuristic", "mlp", "mlp_noH", "pair", "pair_noH", "pretrained", "pretrained_T05", "pretrained_imp", "pretrained_imp_T2"] if a in set(df.arm)]
    summary = df.groupby(["spec_id", "arm"]).agg(
        n=("seed", "size"), recovery=("recovered", "mean"), train_r2=("train_r2", "mean"),
        test_r2=("test_r2", "mean"), test_r2_median=("test_r2", "median"), test_r2_sd=("test_r2", "std"),
        size=("size", "mean"), gens=("gens_used", "mean"), sec=("elapsed_sec", "mean")).reset_index()
    summary["arm"] = pd.Categorical(summary.arm, arms)
    pd.set_option("display.width", 200)
    print(summary.sort_values(["spec_id", "arm"]).round(4).to_string(index=False))

    print("\nOverall recovery rate and mean wall time per arm:")
    print(df.groupby("arm").agg(recovery=("recovered", "mean"), sec=("elapsed_sec", "mean")).reindex(arms).round(3))

    for ref in ("random", "mlp"):
        if ref not in arms:
            continue
        tests = []
        for (sid, arm), g in df.groupby(["spec_id", "arm"]):
            if arm == ref:
                continue
            base = df[(df.spec_id == sid) & (df.arm == ref)].test_r2.dropna()
            other = g.test_r2.dropna()
            if len(base) and len(other):
                p = mannwhitneyu(other, base, alternative="two-sided").pvalue
                tests.append((sid, arm, other.median() - base.median(), p))
        if not tests:
            continue
        t = pd.DataFrame(tests, columns=["spec_id", "arm", "median_diff", "p"])
        t["p_holm"] = holm(t.p.to_numpy())
        print(f"\nMann-Whitney U on test R² vs {ref} (Holm over {len(t)} comparisons):")
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/compare_selectors/main.csv")
