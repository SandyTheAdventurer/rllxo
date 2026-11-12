"""Count the batch child evaluations of GP-SR-RLC (a2c_t03) per run. Usage (from rllxo-abl):
python count_evals.py HELDOUT_CSV OUT_CSV SEED JOBS"""
import csv, sys, os, multiprocessing as mp
sys.path.insert(0, os.getcwd())

def task(t):
    import semx, compare_selectors as cs
    cnt = {"computed": 0, "cached": 0, "pairs_scored": 0, "pairs_capped": 0}
    orig_directed, orig_bias = semx.directed, semx.bias_grid
    def directed(a, b, ka, kb):
        if (ka, kb) in semx._grids: cnt["cached"] += a.size() * b.size()
        else: cnt["computed"] += a.size() * b.size()
        return orig_directed(a, b, ka, kb)
    def bias_grid(a, b):
        if semx.CFG["mode"] == "exact" and a.size() * b.size() > semx.CFG["cap"]: cnt["pairs_capped"] += 1
        else: cnt["pairs_scored"] += 1
        return orig_bias(a, b)
    semx.directed, semx.bias_grid = directed, bias_grid
    row, _ = cs.run_task(t)
    semx.directed, semx.bias_grid = orig_directed, orig_bias
    return {"spec_id": row["spec_id"], "seed": row["seed"], "status": row["status"], "best_expr": row.get("best_expr", ""),
            "elapsed_sec": row.get("elapsed_sec"), "gens_used": row.get("gens_used"), **cnt}

if __name__ == "__main__":
    held, out, seed, jobs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ref = {int(r["spec_id"]): r["best_expr"] for r in csv.DictReader(open(held)) if r["arm"] == "a2c_t03" and int(r["seed"]) == seed}
    tasks = [("a2c_t03", e, seed) for e in sorted(ref)]
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool, open(out, "w", newline="") as f:
        w = None
        for k, r in enumerate(pool.imap_unordered(task, tasks), 1):
            r["same_as_heldout"] = int(r["best_expr"] == ref[r["spec_id"]])
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
            w.writerow(r); f.flush()
            print(k, r["spec_id"], r["computed"], r["same_as_heldout"], flush=True)
