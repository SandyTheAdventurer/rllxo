"""PySR context table (results/PYSR_CONTEXT_PLAN.md): per-equation median test R², solved fraction, recovery and
size for PySR, random and a2c_t03 over seeds 0-9. Scoring as in the held-out plan (negative, non-finite or
error = 0). The Wilcoxon over the 63 equations is for information only; no claim depends on it.

Usage: python analysis/pysr_context.py HELDOUT_CSV PYSR_SOURCE
  PYSR_SOURCE  baselines/pysr.csv from score_pysr.py (with recovery), or a folder of run_pysr.py JSON files
               (test R² and size only; recovery is then reported as missing)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_revision import load, per_eq, test, EQUATIONS, family  # noqa: E402

SEEDS = range(10)


def pysr_rows(src):
    src = Path(src)
    if src.is_dir():
        d = pd.DataFrame([json.loads(f.read_text()) for f in sorted(src.glob("*.json"))])
        d["arm"], d["status"], d["recovered"], d["size"] = "pysr", "ok", np.nan, d["complexity"]
        return d
    return pd.read_csv(src)


def main(heldout_csv, src):
    H = load(heldout_csv, EQUATIONS)
    H = H[H.arm.isin(["random", "a2c_t03"]) & H.seed.isin(SEEDS)]
    P = pysr_rows(src)
    has_rec = P.recovered.notna().any()
    P = P[P.spec_id.isin(EQUATIONS) & P.seed.isin(SEEDS)].copy()
    ok = P.status == "ok"
    r2 = pd.to_numeric(P.test_r2, errors="coerce")
    P["r"] = r2.where(ok & np.isfinite(r2), 0.0).clip(lower=0.0)
    P["rec"] = pd.to_numeric(P.recovered, errors="coerce").where(ok, 0).astype(float)
    P["solved"] = (P.r >= 0.999).astype(float)
    P["size"] = pd.to_numeric(P["size"], errors="coerce")
    D = pd.concat([H, P])
    med, sol = per_eq(D), per_eq(D, "solved", "mean")
    rec = per_eq(D, "rec", "mean")
    print("# PySR as context (seeds 0-9, 63 held-out equations)\n")
    print(f"PySR rows: {len(P)} of 630; errors {int((P.status != 'ok').sum())}."
          + ("" if has_rec else " Recovery not yet scored (score_pysr.py still running)."))
    print("\n| Arm | Mean median R² | Solved | Recovered | Mean size | Mean s/run |\n|---|---|---|---|---|---|")
    for a in ["random", "a2c_t03", "pysr"]:
        s = D[D.arm == a]
        r = f"{rec[a].mean():.3f}" if (a != "pysr" or has_rec) else "pending"
        print(f"| {a} | {med[a].mean():.4f} | {sol[a].mean():.3f} | {r} | {s['size'].mean():.1f} | {s.elapsed_sec.mean():.1f} |")
    print("\nBy family (mean of per-equation median R²):\n")
    print("| Family | n | random | a2c_t03 | pysr |\n|---|---|---|---|---|")
    for fam in ["Nguyen", "Keijzer", "SRSD"]:
        idx = [e for e in med.index if family(e) == fam]
        print(f"| {fam} | {len(idx)} | {med.loc[idx, 'random'].mean():.4f} | {med.loc[idx, 'a2c_t03'].mean():.4f} | "
              f"{med.loc[idx, 'pysr'].mean():.4f} |")
    print("\nFor information only (two-sided Wilcoxon over the 63 equations, ties dropped):\n")
    for x, y in [("pysr", "a2c_t03"), ("pysr", "random")]:
        w, l, t, p, rb = test(med[x], med[y])
        print(f"- {x} vs {y}, median R²: {w} wins, {l} losses, {t} ties, p = {p:.3g}, r = {rb:.2f}")
        w, l, t, p, rb = test(sol[x], sol[y])
        print(f"- {x} vs {y}, solved fraction: {w} wins, {l} losses, {t} ties, p = {p:.3g}, r = {rb:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
