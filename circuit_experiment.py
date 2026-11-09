"""Resistor-network application, rerun for the revision (results/CIRCUIT_PLAN.md).

The GA (from circuit_evolve.py) evolves a resistor network whose equivalent resistance between V+ and GND
matches a target, with NSGA-II ranking on (relative error, resistor count). The arms differ only in how
crossover picks the resistor pair to swap between two parents:
  uniform  circuit_evolve's uniform crossover (each position swapped with probability 0.5), for reference
  swap     one uniformly random pair (the random-crossover baseline)
  learned  the paper's original selector (MLP node scores, REINFORCE)
  sem      the semantic ranking alone: softmax of H = -rank/tau over the exact child scores of every pair
  a2c_sem  the method: the selector's pair logits plus H, as a2c_t03 does for trees

Usage: python circuit_experiment.py --arms swap,sem,a2c_sem,learned,uniform --targets 30 --seeds 10 --jobs 4
"""
from __future__ import annotations
import argparse, csv, math, multiprocessing as mp, os, random, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

E6 = [m * 10 ** k for k in range(3) for m in (10, 15, 22, 33, 47, 68)]  # 10 ohm to 6.8 kohm
NODES, RESISTORS, POP, GENS, TAU = 6, 6, 100, 50, 0.03
FIELDS = ["arm", "target", "seed", "best_error", "best_resistors", "solved", "front_size", "evaluations", "elapsed_sec"]
SOLVED = 1e-4  # relative error
OUT = Path(__file__).resolve().parent / "results" / "circuits"


def targets(n):
    """n targets, log-uniform between 5 ohm and 5 kohm, fixed by seed 123."""
    return [round(float(t), 2) for t in np.exp(np.random.default_rng(123).uniform(math.log(5), math.log(5000), n))]


def run(task):
    arm, target, seed = task
    import torch
    torch.set_num_threads(1)
    import circuit_evolve as ce
    ce.R_VALUES[:] = E6
    ce.DEVICE = torch.device("cpu")  # main.DEVICE may be a GPU; these networks are tiny
    t0 = time.time()
    ce.set_global_seed(seed)
    cache = {}

    def objs(top):
        k = ce._sig(top)
        if k not in cache:
            cache[k] = ce.evaluate_pareto(top, target)
        return cache[k]

    def score(top):
        e, c = objs(top)
        return e + 0.01 * c

    def child(top, i, r):
        c = top.clone()
        c.resistors[i] = ce.Resistor(r.n1, r.n2, r.value)
        ce._repair(c)
        return c

    def sem_grid(a, b):
        """H[i, j] = -rank/TAU of the better child of swapping a[i] with b[j]; no-op swaps get -inf."""
        L = np.full((len(a.resistors), len(b.resistors)), np.inf)
        for i, ra in enumerate(a.resistors):
            for j, rb in enumerate(b.resistors):
                if (ra.n1, ra.n2, ra.value) != (rb.n1, rb.n2, rb.value):
                    L[i, j] = min(score(child(a, i, rb)), score(child(b, j, ra)))
        H = np.full(L.shape, -np.inf)
        ok = np.isfinite(L)
        if ok.any():
            ranks = np.argsort(np.argsort(L[ok], kind="stable"), kind="stable")
            H[ok] = -(ranks / max(ok.sum() - 1, 1)) / TAU
        return H

    selector = opt = None
    history = []
    if arm in ("learned", "a2c_sem"):
        selector = ce.SubtreeSelector(feature_dim=ce.FEATURE_DIM, hidden_dim=32, mlp_layers=2, use_critic=True).to(ce.DEVICE)
        opt = torch.optim.Adam(selector.parameters(), lr=3e-4)

    def feats(top):
        return torch.stack([ce._resistor_features(top, r, i) for i, r in enumerate(top.resistors)])

    def pick(a, b):
        na, nb = len(a.resistors), len(b.resistors)
        if arm == "swap":
            return random.randrange(na), random.randrange(nb), None
        H = sem_grid(a, b) if arm in ("sem", "a2c_sem") else np.zeros((na, nb))
        if arm == "sem":
            logits = H.ravel()
        else:
            fa, fb = feats(a), feats(b)
            with torch.no_grad():
                la, _ = selector(fa, return_value=True)
                lb, _ = selector(fb, return_value=True)
            logits = (np.clip((la[:, None] + lb[None, :]).double().numpy() / 0.6, -10, 10) + H).ravel()
        if not np.isfinite(logits).any():
            return random.randrange(na), random.randrange(nb), None
        p = np.exp(logits - logits[np.isfinite(logits)].max())
        p[~np.isfinite(p)] = 0
        k = int(np.searchsorted(np.cumsum(p), random.random() * p.sum(), side="right"))
        i, j = divmod(min(k, na * nb - 1), nb)
        rec = None if arm == "sem" else {"fa": fa, "fb": fb, "H": torch.tensor(H, dtype=torch.float64), "i": i, "j": j}
        return i, j, rec

    def update():
        """REINFORCE with standardized rewards (circuit_evolve's update), the log-probabilities taken under the
        same logits the pair was sampled from (H included for a2c_sem)."""
        if not history:
            return
        r = torch.tensor([h["reward"] for h in history], dtype=torch.float64)
        if r.std() < 1e-12:
            history.clear()
            return
        r = (r - r.mean()) / (r.std() + 1e-8)
        lps = []
        for h in history:
            la, _ = selector(h["fa"], return_value=True)
            lb, _ = selector(h["fb"], return_value=True)
            lg = torch.clamp((la[:, None] + lb[None, :]).double() / 0.6, -10, 10) + h["H"]
            lps.append(torch.log_softmax(lg.flatten(), 0)[h["i"] * lg.shape[1] + h["j"]])
        loss = -(torch.stack(lps) * r).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        opt.step()
        history.clear()

    pop = [ce.random_topology(NODES, RESISTORS) for _ in range(POP)]
    for gen in range(GENS):
        ob = [objs(t) for t in pop]
        ranks = ce.pareto_rank(ob)
        if selector is not None and gen % 3 == 0:
            update()
        crowd = {}
        for rk in range(max(ranks) + 1):
            crowd.update(ce.crowding_distance(ob, [i for i in range(POP) if ranks[i] == rk]))
        key = lambda i: (ranks[i], -crowd.get(i, 0))
        nxt = [pop[i].clone() for i in sorted(range(POP), key=key)[:2]]
        while len(nxt) < POP:
            p1, p2 = (pop[min(random.sample(range(POP), 3), key=key)].clone() for _ in range(2))
            if random.random() < 0.85:
                if arm == "uniform":
                    c1, c2 = ce.crossover(p1, p2)
                else:
                    i, j, rec = pick(p1, p2)
                    c1, c2 = child(p1, i, p2.resistors[j]), child(p2, j, p1.resistors[i])
                    if rec is not None:
                        before = min(score(p1), score(p2))
                        rec["reward"] = before - min(score(c1), score(c2))
                        history.append(rec)
            else:
                c1, c2 = p1, p2
            nxt.append(ce.mutate(c1, 0.3))
            if len(nxt) < POP:
                nxt.append(ce.mutate(c2, 0.3))
        pop = nxt
    ob = [objs(t) for t in pop]
    front = [i for i, rk in enumerate(ce.pareto_rank(ob)) if rk == 0]
    best = min(front, key=lambda i: (ob[i][0], ob[i][1]))
    err, nres = ob[best]
    return {"arm": arm, "target": target, "seed": seed, "best_error": err, "best_resistors": nres,
            "solved": int(err < SOLVED), "front_size": len(front), "evaluations": len(cache), "elapsed_sec": round(time.time() - t0, 2)}


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--arms", default="swap,sem,a2c_sem,learned,uniform")
    pa.add_argument("--targets", type=int, default=30)
    pa.add_argument("--seeds", type=int, default=10)
    pa.add_argument("--jobs", type=int, default=4)
    pa.add_argument("--tag", default="circuits")
    a = pa.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.tag}.csv"
    done = set()
    if path.exists():
        with open(path, newline="") as f:
            done = {(r["arm"], float(r["target"]), int(r["seed"])) for r in csv.DictReader(f)}
    tasks = [(arm, t, s) for s in range(a.seeds) for t in targets(a.targets) for arm in a.arms.split(",") if (arm, t, s) not in done]
    print(f"{len(tasks)} tasks", flush=True)
    new = not path.exists()
    with open(path, "a", newline="") as f, mp.get_context("spawn").Pool(a.jobs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, row in enumerate(pool.imap_unordered(run, tasks), 1):
            w.writerow(row)
            f.flush()
            if k % 50 == 0:
                print(f"[{k}/{len(tasks)}]", flush=True)


if __name__ == "__main__":
    main()
