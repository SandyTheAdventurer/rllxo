"""Re-run one circuit run of the revision experiment, check it matches circuits.csv, and draw its pruned circuit.
Usage (from rllxo-abl): python circuit_fig.py ARM TARGET SEED OUT_PDF"""
import sys, os, csv
sys.path.insert(0, os.getcwd())
import circuit_experiment as cx
src = open(cx.__file__).read()
old = '"evaluations": len(cache), "elapsed_sec": round(time.time() - t0, 2)}'
assert src.count(old) == 1
ns = {"__file__": cx.__file__, "__name__": "cxcopy"}
exec(compile(src.replace(old, old[:-1] + ', "top": pop[best]}').replace('if __name__ == "__main__":', 'if False:'), cx.__file__, "exec"), ns)
arm, target, seed, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
row = ns["run"]((arm, target, seed))
ref = next(r for r in csv.DictReader(open("results/pc_rev1/circuits/circuits.csv")) if r["arm"] == arm and float(r["target"]) == target and int(r["seed"]) == seed)
print("rerun:", row["best_error"], row["best_resistors"], row["evaluations"])
print("csv:  ", ref["best_error"], ref["best_resistors"], ref["evaluations"])
import circuit_evolve as ce
top = ce.prune_unused(row["top"])
print("pruned:", top, "R_eq =", ce.solve_dc(top), "currents", ce.branch_currents(top))

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif"})
used = sorted({n for r in top.resistors for n in (r.n1, r.n2)})
order, frontier = [0], [0]
while frontier:  # breadth-first from V+, so nodes appear left to right along the path to GND
    nxt = []
    for u in frontier:
        for r in top.resistors:
            for v, w in ((r.n1, r.n2), (r.n2, r.n1)):
                if v == u and w not in order and w != 1:
                    order.append(w); nxt.append(w)
    frontier = nxt
order.append(1)
X = {n: 2.7 * k for k, n in enumerate(order)}
groups = {}
for r in top.resistors:
    k = tuple(sorted((r.n1, r.n2), key=lambda n: X[n])); groups.setdefault(k, []).append(r)
keys = sorted(groups, key=lambda k: (X[k[0]], X[k[1]]))
fig, ax = plt.subplots(figsize=(7.0, 2.1))
idx = 0
for k in keys:
    rs = groups[k]; x1, x2 = X[k[0]], X[k[1]]
    skip = x2 - x1 > 2.7 + 1e-9
    ys = [0.9] if skip else ([0.0] if len(rs) == 1 else list(np.linspace(0.55, -0.55, len(rs))))
    for r, y in zip(rs, ys):
        idx += 1
        ax.plot([x1, x1, x2, x2], [0, y, y, 0], color="#1A6EBD", lw=1.8, zorder=1, solid_capstyle="round")
        val = f"{r.value/1000:g} k$\\Omega$" if r.value >= 1000 else f"{r.value:g} $\\Omega$"
        ax.text((x1 + x2) / 2, y, f"R{idx} = {val}", ha="center", va="center", fontsize=11, zorder=3,
                bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="black", lw=1.0))
for n in order:
    ax.scatter([X[n]], [0], s=60, color="black", zorder=4)
    lab = {0: "V+", 1: "GND"}.get(n, f"N{n}")
    if n == 0:
        ax.text(X[n] - 0.15, 0, lab, ha="right", va="center", fontsize=11)
    elif n == 1:
        ax.text(X[n] + 0.15, 0, lab, ha="left", va="center", fontsize=11)
    else:
        ax.text(X[n] - 0.1, -0.12, lab, ha="right", va="top", fontsize=11)
ax.set_xlim(-0.8, X[1] + 0.9); ax.set_ylim(-0.9, 1.0); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(out, bbox_inches="tight"); print("saved", out)
