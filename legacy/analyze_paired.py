"""Per-equation paired comparison of crossover arms (the SRSD protocol in results/PROTOCOL.md).

Per equation, each arm's median test R² over seeds (non-finite or negative counts as 0), recovery
rate, share of runs with test R² >= 0.999 and median NED (normalized tree edit distance, lower is
better; wins count lower NED); then two-sided Wilcoxon signed-rank tests across
equations (ties dropped) for each arm against each reference.

Usage: python analyze_paired.py [results/compare_selectors/v2.csv] [--refs random,mlp] [--specs 2000-2039]
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon


def per_equation(df):
    df = df.assign(r2=df.test_r2.where(np.isfinite(df.test_r2) & (df.test_r2 > 0), 0.0))
    return df.groupby(["spec_id", "arm"]).agg(
        n=("seed", "size"), median_r2=("r2", "median"), recovery=("recovered", "mean"),
        solved=("r2", lambda r: np.mean(r >= 0.999)), ned=("ned", "median"),
        sec=("elapsed_sec", "mean")).reset_index()


def main(path, refs, spec_range=None):
    df = pd.read_csv(path)
    df = df[df.status == "ok"]
    if spec_range:
        df = df[df.spec_id.between(*spec_range)]
    eq = per_equation(df)
    pd.set_option("display.width", 200)
    wide = eq.pivot(index="spec_id", columns="arm")
    print(wide[["median_r2", "recovery", "ned"]].round(4).to_string())
    print("\nMean over equations:")
    print(eq.groupby("arm")[["median_r2", "recovery", "solved", "ned", "sec"]].mean().round(4).to_string())
    print(f"\nRuns per arm: {df.groupby('arm').size().to_dict()}")
    rows = []
    for ref in refs:
        for arm in sorted(set(eq.arm) - {ref}):
            for metric in ["median_r2", "recovery", "solved", "ned"]:
                both = wide[metric][[arm, ref]].dropna()
                diff = (both[arm] - both[ref]) * (-1 if metric == "ned" else 1)
                nz = diff[diff != 0]
                p = wilcoxon(nz, zero_method="wilcox").pvalue if len(nz) else np.nan
                rows.append((arm, ref, metric, len(both), (nz > 0).sum(), (nz < 0).sum(), diff.median(), p))
    print("\nWilcoxon signed-rank across equations (wins/losses exclude ties):")
    print(pd.DataFrame(rows, columns=["arm", "ref", "metric", "eqs", "wins", "losses", "median_diff", "p"])
          .round(4).to_string(index=False))


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("path", nargs="?", default="results/compare_selectors/v2.csv")
    pa.add_argument("--refs", default="random,mlp")
    pa.add_argument("--specs", help="spec id range lo-hi, e.g. 2000-2039 for SRSD or 1-115 for Nguyen+Keijzer")
    args = pa.parse_args()
    main(args.path, args.refs.split(","), tuple(map(int, args.specs.split("-"))) if args.specs else None)
