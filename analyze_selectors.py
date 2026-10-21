"""Summarise compare_selectors.py output: per equation and arm, recovery rate, train/test R²,
size, generations and wall time; pooled recovery with Fisher exact tests; and Mann-Whitney U
tests (Holm-corrected within each reference arm) of test R² against the random baseline and
the paper's MLP selector.

Usage: python analyze_selectors.py [results/compare_selectors/main.csv] [--specs 6,7,12] [--arms a,b,...]
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

ARMS = ["random", "heuristic", "mlp", "mlp_noH", "pair", "pair_noH", "ssc"]  # display order; others follow
REFS = ("random", "mlp")


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main(path, specs=None, arms=None):
    df = pd.read_csv(path)
    df = df[df.status == "ok"]
    if specs:
        df = df[df.spec_id.isin(specs)]
    if arms:
        df = df[df.arm.isin(set(arms) | set(REFS))]
    arms = [a for a in ARMS if a in set(df.arm)] + sorted(set(df.arm) - set(ARMS))
    summary = df.groupby(["spec_id", "arm"]).agg(
        n=("seed", "size"), recovery=("recovered", "mean"), train_r2=("train_r2", "mean"),
        test_r2=("test_r2", "mean"), test_r2_median=("test_r2", "median"), test_r2_sd=("test_r2", "std"),
        size=("size", "mean"), gens=("gens_used", "mean"), sec=("elapsed_sec", "mean")).reset_index()
    summary["arm"] = pd.Categorical(summary.arm, arms)
    pd.set_option("display.width", 200)
    print(summary.sort_values(["spec_id", "arm"]).round(4).to_string(index=False))

    pooled = df.groupby("arm").agg(recovered=("recovered", "sum"), runs=("recovered", "size"),
                                   sec=("elapsed_sec", "mean")).reindex(arms)
    pooled["recovery"] = pooled.recovered / pooled.runs
    for ref in REFS:
        if ref in pooled.index:
            r = pooled.loc[ref]
            pooled[f"fisher_p_vs_{ref}"] = [
                np.nan if a == ref else fisher_exact([[x.recovered, x.runs - x.recovered],
                                                      [r.recovered, r.runs - r.recovered]])[1]
                for a, x in pooled.iterrows()]
    print("\nPooled recovery and mean wall time per arm:")
    print(pooled.round(3).to_string())

    for ref in REFS:
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
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("path", nargs="?", default="results/compare_selectors/main.csv")
    pa.add_argument("--specs", help="comma-separated spec ids to include")
    pa.add_argument("--arms", help="comma-separated arms to include (random and mlp are always kept)")
    args = pa.parse_args()
    main(args.path, [int(s) for s in args.specs.split(",")] if args.specs else None,
         args.arms.split(",") if args.arms else None)
