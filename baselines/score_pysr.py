"""Turn PySR outputs (run_pysr.py) into rows in the compare_selectors CSV format, arm "pysr": test R² as
PySR computed it, and recovery and TED from the same checks as the GP runs. Size is PySR's complexity
(node count). Usage: python baselines/score_pysr.py -> baselines/pysr.csv"""
import sys, csv, json
from pathlib import Path
import sympy as sp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_selectors as cs

HERE = Path(__file__).resolve().parent


def main():
    specs = cs.all_specs()
    rows = []
    for f in sorted((HERE / "pysr_out").glob("*.json")):
        o = json.loads(f.read_text())
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
        rows.append(row)
    with open(HERE / "pysr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cs.FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", HERE / "pysr.csv")


if __name__ == "__main__":
    main()
