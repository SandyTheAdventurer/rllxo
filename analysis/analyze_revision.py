"""Analysis of the revision-1 runs, as fixed in the plans before the runs:
results/EXTRA_ARMS_PLAN.md, PROTOCOL_CHECKS_PLAN.md, ABLATION_NOISE_PLAN.md and CIRCUIT_PLAN.md.

Usage: python analysis/analyze_revision.py HELDOUT_CSV RESULTS_DIR > results/pc_rev1/analysis.md
  HELDOUT_CSV  the frozen held-out run (for the a2c_t03 and ex64_t03 rows)
  RESULTS_DIR  folder with compare_selectors/{extra,keijzer_orig,nguyen_orig,ablations,noise01,noise10}.csv
               and circuits/circuits.csv

Scoring is the held-out plan's: per equation, each arm's median test R² over seeds 0-29, with negative,
non-finite or error rows counted as 0. Tests are two-sided Wilcoxon signed-rank over equations, zero
differences dropped (the exact test, as in analyze_heldout.py). Wins count equations where the first arm
is higher.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_heldout import EQUATIONS, holm, family  # noqa: E402

NK = [1, 2, 3, 4, 5, 8, 9, 10, 11, *range(101, 114), 115]
KEIJZER = [*range(101, 114), 115]
NGUYEN = [1, 2, 3, 4, 5, 8, 9, 10, 11]
SEEDS = range(30)
SOLVED = 0.999


def load(path, eqs, col="test_r2"):
    d = pd.read_csv(path)
    d = d[d.spec_id.isin(eqs) & d.seed.isin(SEEDS)].drop_duplicates(["arm", "spec_id", "seed"], keep="last")
    ok = d.status == "ok"
    r2 = pd.to_numeric(d[col], errors="coerce")
    d["r"] = r2.where(ok & np.isfinite(r2), 0.0).clip(lower=0.0)
    d["rec"] = pd.to_numeric(d.recovered, errors="coerce").where(ok, 0).fillna(0).astype(float)
    d["solved"] = (d.r >= SOLVED).astype(float)
    d["size"] = pd.to_numeric(d["size"], errors="coerce")
    return d


def per_eq(d, col="r", how="median"):
    return d.groupby(["arm", "spec_id"])[col].agg(how).unstack("arm")


def test(x, y, lower_is_better=False):
    """(wins, losses, ties, p, r_rb): wins = first arm better. r_rb is the matched-pairs rank-biserial."""
    z = (x - y).dropna()
    if lower_is_better:
        z = -z
    w, l = int((z > 0).sum()), int((z < 0).sum())
    nz = z[z != 0]
    if len(nz) == 0:
        return w, l, len(z), 1.0, 0.0
    p = float(wilcoxon(nz, zero_method="wilcox").pvalue)
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    r_rb = float((ranks[nz.to_numpy() > 0].sum() - ranks[nz.to_numpy() < 0].sum()) / ranks.sum())
    return w, l, len(z) - w - l, p, r_rb


def family_table(rows, title):
    """rows: [(name, x, y)] or [(name, x, y, lower_is_better)]; one Holm family."""
    res = [test(r[1], r[2], *(r[3:] or [False])) for r in rows]
    adj = holm(np.array([r[3] for r in res]))
    print(f"\n## {title}\n")
    print("| Comparison | n | Wins | Losses | Ties | p | Holm p | r |\n|---|---|---|---|---|---|---|---|")
    for r, (w, l, t, p, rb), pa in zip(rows, res, adj):
        print(f"| {r[0]} | {w + l + t} | {w} | {l} | {t} | {p:.3g} | {pa:.3g} | {rb:.2f} |")


def desc_rows(rows, title):
    print(f"\n### {title}\n")
    print("| Comparison | n | Wins | Losses | Ties | p (uncorrected) | r |\n|---|---|---|---|---|---|---|")
    for name, x, y in rows:
        w, l, t, p, rb = test(x, y)
        print(f"| {name} | {w + l + t} | {w} | {l} | {t} | {p:.3g} | {rb:.2f} |")


def arm_table(d, arms, title):
    med, sol, rec = per_eq(d), per_eq(d, "solved", "mean"), per_eq(d, "rec", "mean")
    print(f"\n### {title}\n")
    print("| Arm | Runs | Errors | Mean median R² | Solved | Recovered | Mean s/run | Mean size |\n|---|---|---|---|---|---|---|---|")
    for a in arms:
        s = d[d.arm == a]
        print(f"| {a} | {len(s)} | {int((s.status != 'ok').sum())} | {med[a].mean():.4f} | {sol[a].mean():.3f} | "
              f"{rec[a].mean():.3f} | {s.elapsed_sec.mean():.1f} | {s['size'].mean():.1f} |")


def errors(d, title):
    e = d[d.status != "ok"]
    if len(e):
        print(f"\nError rows ({title}):\n")
        for r in e.itertuples():
            print(f"- {r.arm}, eq {r.spec_id}, seed {r.seed}: `{r.error}`")


def same_runs(a, b, arm, title):
    m = a[a.arm == arm].merge(b[b.arm == arm], on=["spec_id", "seed"])
    same = (m.best_expr_x.astype(str) == m.best_expr_y.astype(str))
    print(f"- {title}: {int(same.sum())} of {len(m)} {arm} runs give the identical final expression.")


def main(heldout_csv, rdir):
    rdir = Path(rdir)
    cs = rdir / "compare_selectors"
    H = load(heldout_csv, EQUATIONS)
    print("# Revision-1 analysis (runs on the Windows PC, 20 workers)\n")
    print(f"Held-out reference rows: {heldout_csv}. Run folder: {rdir}.")

    # ---------------- Extra arms ----------------
    X = load(cs / "extra.csv", EQUATIONS)
    print("\n# 1. Extra arms (EXTRA_ARMS_PLAN.md)\n")
    print(f"Rows: {len(X)} of {3 * 63 * 30}.")
    both = pd.concat([H[H.arm.isin(["a2c_t03", "ex64_t03"])], X])
    med = per_eq(both)
    family_table([("a2c_t03 vs ssc", med.a2c_t03, med.ssc),
                  ("ssc vs random", med.ssc, med.random),
                  ("a2c_noH vs random", med.a2c_noH, med.random),
                  ("a2c_t03 vs a2c_noH", med.a2c_t03, med.a2c_noH)],
                 "Extra arms: one Holm family, median test R², 63 equations (random = this run's rows)")
    sol, rec = per_eq(both, "solved", "mean"), per_eq(both, "rec", "mean")
    desc_rows([("a2c_noH vs ex64_t03", med.a2c_noH, med.ex64_t03),
               ("ex64_t03 vs ssc", med.ex64_t03, med.ssc),
               ("a2c_t03 vs ssc, solved fraction", sol.a2c_t03, sol.ssc),
               ("a2c_t03 vs ssc, recovery", rec.a2c_t03, rec.ssc),
               ("a2c_noH vs random, solved fraction", sol.a2c_noH, sol.random)], "Extra arms: descriptive")
    for fam in ["Nguyen", "Keijzer", "SRSD"]:
        idx = [e for e in med.index if family(e) == fam]
        w, l, t, p, _ = test(med.loc[idx, "a2c_t03"], med.loc[idx, "ssc"])
        print(f"- a2c_t03 vs ssc, {fam} ({len(idx)}): {w} wins, {l} losses, {t} ties, p = {p:.3g}")
    arm_table(both, ["random", "ssc", "a2c_noH", "ex64_t03", "a2c_t03"], "Extra arms: per arm (63 equations)")
    print("\nMachine check (the plan compared laptop and PC random; both random sets are now from the PC):")
    same_runs(H, X, "random", "held-out random vs this run's random")
    hr, xr = H[H.arm == "random"], X[X.arm == "random"]
    print(f"- Mean seconds per random run: held-out {hr.elapsed_sec.mean():.2f} (18 workers), this run "
          f"{xr.elapsed_sec.mean():.2f} (20 workers).")
    errors(X, "extra")

    # ---------------- Protocol checks ----------------
    print("\n# 2. Protocol checks (PROTOCOL_CHECKS_PLAN.md)\n")
    KA = load(cs / "keijzer_orig.csv", KEIJZER, col="test_r2_orig")
    KA_ours = load(cs / "keijzer_orig.csv", KEIJZER)
    NB = load(cs / "nguyen_orig.csv", NGUYEN)
    print(f"Rows: keijzer_orig {len(KA)} of {3 * 14 * 30}, nguyen_orig {len(NB)} of {3 * 9 * 30}.")
    mk, mn, rn = per_eq(KA), per_eq(NB), per_eq(NB, "rec", "mean")
    desc_rows([("Run A: a2c_t03 vs random, Keijzer's test sets", mk.a2c_t03, mk.random),
               ("Run A: a2c_t03 vs ssc, Keijzer's test sets", mk.a2c_t03, mk.ssc),
               ("Run A (reference): a2c_t03 vs random, our test sets", per_eq(KA_ours).a2c_t03, per_eq(KA_ours).random),
               ("Run B: a2c_t03 vs random, median R²", mn.a2c_t03, mn.random),
               ("Run B: a2c_t03 vs random, recovery rate", rn.a2c_t03, rn.random),
               ("Run B: a2c_t03 vs ssc, median R²", mn.a2c_t03, mn.ssc)],
              "Protocol checks (each test on its own, as the plan lists them; 14 and 9 equations)")
    print("\nK4 and K6 on Keijzer's test sets (extrapolation), per-equation Mann-Whitney U, a2c_t03 vs random:\n")
    print("| Eq | Median a2c_t03 | Median random | Median ssc | p |\n|---|---|---|---|---|")
    for e in (104, 106):
        a = KA[(KA.arm == "a2c_t03") & (KA.spec_id == e)].r.to_numpy()
        b = KA[(KA.arm == "random") & (KA.spec_id == e)].r.to_numpy()
        c = KA[(KA.arm == "ssc") & (KA.spec_id == e)].r.to_numpy()
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        print(f"| K{e - 100} | {np.median(a):.4f} | {np.median(b):.4f} | {np.median(c):.4f} | {p:.3g} |")
    arm_table(KA, ["random", "ssc", "a2c_t03"], "Run A per arm (score on Keijzer's test sets)")
    arm_table(NB, ["random", "ssc", "a2c_t03"], "Run B per arm (original Nguyen protocol)")
    same_runs(H, KA, "a2c_t03", "Run A vs held-out (same training data)")
    errors(KA, "keijzer_orig"); errors(NB, "nguyen_orig")

    # ---------------- Ablations ----------------
    print("\n# 3. Ablations (ABLATION_NOISE_PLAN.md, run C)\n")
    A = load(cs / "ablations.csv", NK)
    print(f"Rows: {len(A)} of {7 * 23 * 30}.")
    XA = X[X.spec_id.isin(NK)]
    allA = pd.concat([A, XA])
    ma = per_eq(allA)
    family_table([("a2c_t03 vs a2c_nocritic", ma.a2c_t03, ma.a2c_nocritic),
                  ("a2c_noconst vs random_noconst", ma.a2c_noconst, ma.random_noconst),
                  ("a2c_m01 vs random_m01", ma.a2c_m01, ma.random_m01),
                  ("a2c_t03 vs sizefair", ma.a2c_t03, ma.sizefair),
                  ("sizefair vs random", ma.sizefair, ma.random)],
                 "Ablations: one Holm family, median test R², 23 Nguyen and Keijzer equations")
    desc_rows([("a2c_t03 vs a2c_noH (joint vs independent sampling)", ma.a2c_t03, ma.a2c_noH),
               ("a2c_t03 vs random", ma.a2c_t03, ma.random),
               ("a2c_noconst vs a2c_t03", ma.a2c_noconst, ma.a2c_t03),
               ("random_noconst vs random", ma.random_noconst, ma.random)], "Ablations: descriptive")
    arm_table(allA, ["random", "sizefair", "random_noconst", "random_m01", "a2c_noH", "a2c_nocritic", "a2c_noconst",
                     "a2c_m01", "a2c_t03"], "Ablations: per arm (23 equations)")
    same_runs(H, A, "a2c_t03", "ablation a2c_t03 vs held-out a2c_t03")
    errors(A, "ablations")

    # ---------------- Noise ----------------
    print("\n# 4. Noise (ABLATION_NOISE_PLAN.md, run D)\n")
    N1, N10 = load(cs / "noise01.csv", NK), load(cs / "noise10.csv", NK)
    print(f"Rows: noise01 {len(N1)}, noise10 {len(N10)} (of {3 * 23 * 30} each). Test sets are noise-free.")
    m1, m10 = per_eq(N1), per_eq(N10)
    family_table([("1% noise: a2c_t03 vs random", m1.a2c_t03, m1.random),
                  ("1% noise: a2c_t03 vs ssc", m1.a2c_t03, m1.ssc),
                  ("10% noise: a2c_t03 vs random", m10.a2c_t03, m10.random),
                  ("10% noise: a2c_t03 vs ssc", m10.a2c_t03, m10.ssc)],
                 "Noise: one Holm family, median clean-test R², 23 equations")
    m0 = per_eq(pd.concat([H[H.arm == "a2c_t03"], XA]))
    print("\n| Noise | random | ssc | a2c_t03 (mean of per-equation median R²) |\n|---|---|---|---|")
    print(f"| 0 (held-out / extra rows) | {m0.random.loc[NK].mean():.4f} | {m0.ssc.loc[NK].mean():.4f} | {m0.a2c_t03.loc[NK].mean():.4f} |")
    print(f"| 1% | {m1.random.mean():.4f} | {m1.ssc.mean():.4f} | {m1.a2c_t03.mean():.4f} |")
    print(f"| 10% | {m10.random.mean():.4f} | {m10.ssc.mean():.4f} | {m10.a2c_t03.mean():.4f} |")
    arm_table(N1, ["random", "ssc", "a2c_t03"], "1% noise per arm")
    arm_table(N10, ["random", "ssc", "a2c_t03"], "10% noise per arm")
    errors(N1, "noise01"); errors(N10, "noise10")

    # ---------------- Circuits ----------------
    print("\n# 5. Resistor networks (CIRCUIT_PLAN.md)\n")
    C = pd.read_csv(rdir / "circuits" / "circuits.csv")
    print(f"Rows: {len(C)} of 1500 (5 arms x 30 targets x 10 seeds).")
    C["lerr"] = np.log10(C.best_error.clip(lower=1e-12))  # exact matches (error 0) are set to 1e-12
    C["solved"] = C.solved.astype(str).str.lower().isin(["1", "true", "1.0"]).astype(float)
    ml = C.groupby(["arm", "target"]).lerr.median().unstack("arm")
    ms = C.groupby(["arm", "target"]).solved.mean().unstack("arm")
    w, l, t, p, rb = test(ml["a2c_sem"], ml["swap"], lower_is_better=True)
    print("## Primary: a2c_sem vs swap, median log10 relative error per target (wins = lower error)\n")
    print(f"{len(ml)} targets: {w} wins, {l} losses, {t} ties; two-sided Wilcoxon p = {p:.3g}, r = {rb:.2f} "
          f"({'significant' if p < 0.05 else 'not significant'} at 0.05)")
    family_table([("sem vs swap", ml["sem"], ml["swap"], True),
                  ("a2c_sem vs sem", ml["a2c_sem"], ml["sem"], True),
                  ("learned vs swap", ml["learned"], ml["swap"], True),
                  ("a2c_sem vs uniform", ml["a2c_sem"], ml["uniform"], True),
                  ("a2c_sem vs swap, solved fraction", ms["a2c_sem"], ms["swap"])],
                 "Circuits secondary: one Holm family (log-error comparisons: wins = lower error)")
    print("\n| Arm | Median of target medians, log10 error | Solved | Mean resistors | Mean evaluations | Mean s/run |\n|---|---|---|---|---|---|")
    for a in ["swap", "uniform", "learned", "sem", "a2c_sem"]:
        s = C[C.arm == a]
        print(f"| {a} | {ml[a].median():.2f} | {ms[a].mean():.3f} | {s.best_resistors.mean():.2f} | "
              f"{s.evaluations.mean():.0f} | {s.elapsed_sec.mean():.1f} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
