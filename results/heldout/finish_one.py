"""Run one held-out task with the frozen compare_selectors.run_task, except that is_recovered and
tree_edit_distance run in a child process that is killed after a hard limit, counted from the start of the call (not the child's start-up). A killed recovery check
counts as not recovered (what the frozen code returns when its 30 s alarm fires); a killed TED is NaN.
Writes the row and curve as JSON to rllxo_sel/finish/<arm>_<eq>_<seed>.json."""
import sys, json, time, multiprocessing as mp
sys.path.insert(0, 'rllxo_win')
import compare_selectors as cs

LIMITS = {"is_recovered": (60, False), "tree_edit_distance": (150, (float("nan"), float("nan")))}

def _child(q, name, args, kw):
    q.put("started")
    q.put(getattr(cs, name)(*args, **kw))

def hard(name):
    limit, default = LIMITS[name]
    def call(*args, **kw):
        ctx = mp.get_context("spawn"); q = ctx.Queue()
        p = ctx.Process(target=_child, args=(q, name, args, kw), daemon=True); p.start()
        q.get(timeout=600)  # child start-up (imports) is not counted
        try:
            res = q.get(timeout=limit)
        except Exception:
            res = default; KILLED.append(name)
        p.kill(); p.join()
        return res
    return call

if __name__ == "__main__":
    import os
    arm, eq, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    KILLED = []
    cs.is_recovered, cs.tree_edit_distance = hard("is_recovered"), hard("tree_edit_distance")
    row, curve = cs.run_task((arm, eq, seed))
    os.makedirs("rllxo_sel/finish", exist_ok=True)
    with open(f"rllxo_sel/finish/{arm}_{eq}_{seed}.json", "w", encoding="utf-8") as f:
        json.dump({"row": row, "curve": curve, "hard_killed": KILLED}, f)
    print(arm, eq, seed, row["status"], "killed:", KILLED, flush=True)
