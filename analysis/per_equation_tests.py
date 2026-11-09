"""Per-equation tests (reviewer 4, R4.4/4.5): for each equation, a two-sided Mann-Whitney U test of an arm's
30 test R² values against random's, Holm-corrected across the equations. Scoring as in the held-out plan
(negative, non-finite or error = 0). Not part of the pre-registered analysis.

Usage: python analysis/per_equation_tests.py results/compare_selectors/heldout.csv [arm] > table.md
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, ".")
from analyze_heldout import load, EQUATIONS, family, holm  # noqa: E402

NAMES = {}


def main(path, arm="a2c_t03", base="random"):
    d = load(path)
    rows = []
    for e in EQUATIONS:
        a = d[(d.arm == arm) & (d.spec_id == e)].r.to_numpy()
        b = d[(d.arm == base) & (d.spec_id == e)].r.to_numpy()
        p = 1.0 if np.all(a == a[0]) and np.all(b == a[0]) else float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append(dict(eq=e, family=family(e), med_a=np.median(a), med_b=np.median(b), mean_a=a.mean(), sd_a=a.std(ddof=1),
                         mean_b=b.mean(), sd_b=b.std(ddof=1), solved_a=(a >= 0.999).mean(), solved_b=(b >= 0.999).mean(), p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.to_numpy())
    t["verdict"] = np.where(t.p_holm >= 0.05, "n.s.", np.where(t.med_a > t.med_b, arm, base))
    print(f"# Per-equation Mann-Whitney U: {arm} vs {base} (30 seeds each, Holm across {len(t)} equations)\n")
    print(f"Significant after Holm: {arm} better on {(t.verdict == arm).sum()}, {base} better on {(t.verdict == base).sum()}, "
          f"not significant on {(t.verdict == 'n.s.').sum()}. Uncorrected p < 0.05: {(t.p < 0.05).sum()}.\n")
    for fam in ["Nguyen", "Keijzer", "SRSD"]:
        s = t[t.family == fam]
        print(f"- {fam}: {arm} better on {(s.verdict == arm).sum()}, {base} better on {(s.verdict == base).sum()}, n.s. on {(s.verdict == 'n.s.').sum()}")
    print(f"\n| Eq | Median {arm} | Median {base} | Mean ± SD {arm} | Mean ± SD {base} | Solved {arm} | Solved {base} | p | Holm p | Verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in t.itertuples():
        name = f"N{r.eq}" if r.eq < 100 else f"K{r.eq - 100}" if r.eq < 1000 else f"SRSD-{r.eq - 2000}"
        print(f"| {name} | {r.med_a:.4f} | {r.med_b:.4f} | {r.mean_a:.4f} ± {r.sd_a:.4f} | {r.mean_b:.4f} ± {r.sd_b:.4f} | "
              f"{r.solved_a:.2f} | {r.solved_b:.2f} | {r.p:.2g} | {r.p_holm:.2g} | {r.verdict} |")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []))
