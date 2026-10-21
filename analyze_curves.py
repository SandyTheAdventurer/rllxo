"""Convergence analysis of compare_selectors.py curves (best training R² per generation).

Per equation and arm: median best R² at checkpoint generations, the share of runs reaching
R² >= 0.99 / 0.999 and the median generation they got there, and the area under the
log10(1 - R²) curve (lower = converged faster and further), tested against random and mlp
with Mann-Whitney U, Holm-corrected within each reference arm.

Usage: python analyze_curves.py [results/compare_selectors/final_curves.jsonl]
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from analyze_selectors import holm

GENS = 200
CHECKPOINTS = (10, 25, 50, 100, 200)
ARMS = ["random", "heuristic", "mlp", "mlp_noH", "pair", "pair_noH", "ssc"]  # display order; others follow


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            c = np.array(r["r2"], dtype=float)
            # Runs that hit the stopping R² end early; hold their last value to the full length
            c = np.concatenate([c, np.full(GENS - len(c), c[-1])]) if len(c) < GENS else c[:GENS]
            rows.append({"arm": r["arm"], "spec_id": r["spec_id"], "seed": r["seed"], "curve": c})
    return pd.DataFrame(rows)


def first_gen(c, thr):
    hit = np.nonzero(c >= thr)[0]
    return hit[0] + 1 if len(hit) else np.nan


def main(path):
    df = load(path)
    df["auc"] = df.curve.apply(lambda c: np.mean(np.log10(np.clip(1 - c, 1e-12, None))))
    for thr in (0.99, 0.999):
        df[f"gen{thr}"] = df.curve.apply(lambda c: first_gen(c, thr))
    out = []
    for (sid, arm), g in df.groupby(["spec_id", "arm"]):
        stack = np.stack(g.curve)
        row = {"spec_id": sid, "arm": arm, "n": len(g)}
        row.update({f"r2@{k}": np.median(stack[:, k - 1]) for k in CHECKPOINTS})
        for thr in (0.99, 0.999):
            row[f"hit{thr}"] = g[f"gen{thr}"].notna().mean()
            row[f"gen{thr}"] = g[f"gen{thr}"].median()
        row["auc"] = g.auc.median()
        out.append(row)
    s = pd.DataFrame(out)
    s["arm"] = pd.Categorical(s.arm, [a for a in ARMS if a in set(s.arm)] + sorted(set(s.arm) - set(ARMS)))
    pd.set_option("display.width", 220)
    print("Median best training R² at generation k, share of runs reaching a threshold and median generation\n"
          "to reach it, median AUC of log10(1 - R²) (lower is better):")
    print(s.sort_values(["spec_id", "arm"]).round(4).to_string(index=False))

    for ref in ("random", "mlp"):
        tests = []
        for (sid, arm), g in df.groupby(["spec_id", "arm"]):
            if arm == ref:
                continue
            base = df[(df.spec_id == sid) & (df.arm == ref)].auc
            tests.append((sid, arm, g.auc.median() - base.median(), mannwhitneyu(g.auc, base).pvalue))
        t = pd.DataFrame(tests, columns=["spec_id", "arm", "median_auc_diff", "p"])
        t["p_holm"] = holm(t.p.to_numpy())
        print(f"\nMann-Whitney U on AUC vs {ref} (negative diff = faster than {ref}; Holm over {len(t)}):")
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/compare_selectors/final_curves.jsonl")
