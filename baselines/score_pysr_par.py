"""Parallel version of score_pysr.py: the same row for each PySR output (same checks, same limits, same
PYTHONHASHSEED=0 helper), scored in a pool of worker processes. Writes baselines/pysr.csv in sorted order.
Usage: python baselines/score_pysr_par.py [jobs]"""
import sys, csv, json
import multiprocessing as mp
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def score(path):
    import sympy as sp
    import compare_selectors as cs
    specs = cs.all_specs()
    o = json.loads(Path(path).read_text())
    e, spec = o["spec_id"], specs[o["spec_id"]]
    names = json.loads((HERE / "tasks" / f"{e}.json").read_text())["names"]
    row = {"arm": "pysr", "spec_id": e, "seed": o["seed"], "status": "ok", "error": "", "test_r2": o["test_r2"],
           "size": o["complexity"], "elapsed_sec": o["elapsed_sec"], "best_expr": o["sympy"]}
    try:
        model = o["scale"] * sp.sympify(o["sympy"], locals={n: sp.Symbol(n) for n in names})
        physical = "srsd" in spec
        row["recovered"] = int(cs._Checker.call("is_recovered", model, spec["sympy"], physical=physical))
        row["ted"], row["ned"] = (0, 0.0) if row["recovered"] else \
            cs._Checker.call("tree_edit_distance", model, spec["sympy"], physical=physical)
    except Exception as exc:
        row.update(status="error", error=repr(exc))
    return row


if __name__ == "__main__":
    import compare_selectors as cs
    files = sorted((HERE / "pysr_out").glob("*.json"))
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows = []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for i, r in enumerate(pool.imap(score, [str(f) for f in files])):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(i + 1, "of", len(files), flush=True)
    with open(HERE / "pysr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cs.FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", HERE / "pysr.csv", flush=True)
