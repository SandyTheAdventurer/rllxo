"""Offline pretraining of the crossover pair scorer (selector_type='pretrained' in main.run_ga).

collect: run the GA with uniformly random crossover points and log, for a subsample of
         crossovers, the canonical features of the two swapped nodes, their semantic distance
         (mean absolute output difference) and the reward the A2C update would have seen. Runs
         on the training equations, plus one seed of the dev equations kept only for reporting.
train:   fit main.pair_scorer_net to predict a crossover label, select by AUC on held-out
         training runs, and report AUC on the dev equations. Labels: 'median' = reward above the
         median of its generation (the signal the A2C update sees, since it standardizes rewards
         per batch); 'improves' = a child beats the better parent (reward > 0). --semantic adds
         the semantic-distance channels.

Usage: python pretrain_selector.py collect [--jobs 6]
       python pretrain_selector.py train [--label median|improves] [--semantic]
"""
import argparse, glob, os, re, multiprocessing as mp
import numpy as np, torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# Dev equations (N6, N7, N12), confirmation equations (N3, N4, N9, N10, see results/PROTOCOL.md)
# and K14 are never pretrained on
TEST_SPECS = [6, 7, 12]
CONFIRM_SPECS = [3, 4, 9, 10]
TRAIN_SPECS = [1, 2, 5, 8, 11] + [100 + k for k in range(1, 14)] + [115]
SEEDS = [100, 101, 102]
TEST_SEEDS = [100]
KEEP = 0.3  # fraction of crossovers logged per run
DATA_DIR = "results/pretrain_sem"
MODEL_DIR = "results/pretrain"


def ops_for(spec):
    """The run's ALL_UNARY + ALL_BINARY order, as run_ga sets it for this spec."""
    import nguyen
    unary = [n for n in spec.get("allowed_unary") or [] if n in nguyen._UNARY_REG]
    return unary + ["add", "sub", "mul", "div"] + (["pow"] if spec.get("allow_pow") else [])


def collect_run(task):
    spec_id, seed = task
    torch.set_num_threads(1)
    import main, nguyen, compare_selectors as cs
    spec = cs.all_specs()[spec_id]
    X, y = cs.sample(spec, np.random.default_rng(seed + spec_id * 1000), nguyen.SAMPLE_SIZE)
    names = [v["name"] for v in spec["variables"]]
    unary = {n: nguyen._UNARY_REG[n] for n in spec.get("allowed_unary") or [] if n in nguyen._UNARY_REG}
    proj = main.canonical_feature_map(ops_for(spec))
    rng, rows = np.random.default_rng(seed), []
    orig_update = main.update_selector

    def logging_update(score_fn, **kw):
        assert main.ALL_UNARY + main.ALL_BINARY == ops_for(spec)
        for r in main.crossover_history:
            if not r.get("_seen"):
                r["_seen"] = True
                if rng.random() < KEEP:
                    child_best = min(score_fn(c)[0] for c in r["offspring"])
                    sa = main.subtree_semantics(r["parents"][0], main.CROSSOVER_INPUTS)[r["a_idx"]]
                    sb = main.subtree_semantics(r["parents"][1], main.CROSSOVER_INPUTS)[r["b_idx"]]
                    rows.append((r["gen"], min(r["parent_scores"]) - child_best,
                                 (r["a"][0][r["a_idx"]] @ proj).numpy(), (r["b"][0][r["b_idx"]] @ proj).numpy(),
                                 float((sa - sb).abs().mean())))
        return orig_update(score_fn, **kw)

    main.update_selector = logging_update
    # An 'mlp' selector at huge temperature with no H samples crossover points uniformly while
    # still recording every crossover; it acts from generation 1
    kw = {**nguyen.ga_config, **nguyen.learned_overrides, "selector_type": "mlp", "temperature": 1e6,
          "heuristic_bias_scale": 0.0, "selector_warmup_gens": 1, "verbose": False, "seed": seed,
          "allow_pow": bool(spec.get("allow_pow", False))}
    try:
        main.run_ga(X, y, var_names=names, unary_funcs=unary, **kw)
    finally:
        main.update_selector = orig_update
    np.savez(os.path.join(DATA_DIR, f"{spec_id}_{seed}.npz"), gen=np.array([r[0] for r in rows]),
             reward=np.array([r[1] for r in rows]), fa=np.array([r[2] for r in rows]).reshape(-1, proj.shape[1]),
             fb=np.array([r[3] for r in rows]).reshape(-1, proj.shape[1]), ssd=np.array([r[4] for r in rows]))
    return task, len(rows)


def collect(jobs):
    os.makedirs(DATA_DIR, exist_ok=True)
    tasks = [(e, s) for e in TRAIN_SPECS for s in SEEDS] + [(e, s) for e in TEST_SPECS for s in TEST_SEEDS]
    tasks = [t for t in tasks if not os.path.exists(f"{DATA_DIR}/{t[0]}_{t[1]}.npz")]
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        for k, res in enumerate(pool.imap_unordered(collect_run, tasks), 1):
            print(f"[{k}/{len(tasks)}]", *res, flush=True)


def model_path(label, semantic=False):
    name = "selector" + ("" if label == "median" else f"_{label}") + ("_sem" if semantic else "")
    return os.path.join(MODEL_DIR, name + ".pt")


def labels(gen, reward, label="median"):
    """1 where a crossover's reward is above the median of its generation, or (label='improves')
    where a child beats the better parent."""
    if label == "improves":
        return (reward > 0).astype(np.float32)
    pct = np.empty(len(reward))
    for g in np.unique(gen):
        m = gen == g
        pct[m] = (rankdata(reward[m]) - 0.5) / m.sum()
    return (pct > 0.5).astype(np.float32)


def load_runs(specs, label):
    """[(path, fa, fb, ssd, y)] for every logged run of the given equations."""
    runs = []
    for f in sorted(glob.glob(f"{DATA_DIR}/*_*.npz")):
        if int(re.findall(r"(\d+)_\d+\.npz$", f)[0]) in specs:
            z = np.load(f)
            if len(z["gen"]):
                runs.append((f, z["fa"], z["fb"], z["ssd"], labels(z["gen"], z["reward"], label)))
    return runs


def net_inputs(fa, fb, ssd, semantic):
    """Inputs for logged crossovers, each one pair: a (N, 1, 1, ...) grid."""
    import main
    x = main.pair_inputs(fa[:, None], fb[:, None])
    return torch.cat([x, main.semantic_channels(ssd)[:, None, None]], -1) if semantic else x


def predict(net, fa, fb, ssd, semantic):
    with torch.no_grad():
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        return net(net_inputs(t(fa), t(fb), t(ssd), semantic)).reshape(-1).numpy()


def train(label="median", semantic=False, epochs=30, batch=1024, lr=1e-3):
    import main
    torch.manual_seed(0)
    runs = load_runs(TRAIN_SPECS, label)
    rng = np.random.default_rng(0)
    val_idx = set(rng.choice(len(runs), max(2, len(runs) // 6), replace=False))
    cat = lambda part: tuple(np.concatenate(p) for p in zip(*[r[1:] for i, r in enumerate(runs) if part(i)]))
    tr, va = cat(lambda i: i not in val_idx), cat(lambda i: i in val_idx)
    print(f"{len(runs)} runs: {len(tr[0])} train / {len(va[0])} validation crossovers "
          f"({len(val_idx)} held-out runs), positive rate {tr[-1].mean():.3f}")
    net = main.pair_scorer_net(semantic=semantic)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    fa, fb, ssd, y = (torch.as_tensor(a, dtype=torch.float32) for a in tr)
    pos_weight = (1 - y.mean()) / y.mean()
    best, best_auc = None, 0.0
    for epoch in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            j = perm[i:i + batch]
            logit = net(net_inputs(fa[j], fb[j], ssd[j], semantic)).reshape(-1)
            # Rare positives ('improves') are reweighted to balance the classes
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[j], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(va[-1], predict(net, *va[:-1], semantic))
        if auc > best_auc:
            best, best_auc = {k: v.clone() for k, v in net.state_dict().items()}, auc
        print(f"epoch {epoch + 1:2d} loss {loss.item():.4f} validation AUC {auc:.4f}", flush=True)
    net.load_state_dict(best)
    torch.save(best, model_path(label, semantic))
    print(f"saved {model_path(label, semantic)} (validation AUC {best_auc:.4f})")
    for spec_id in TEST_SPECS:
        *x_t, y_t = (np.concatenate(p) for p in zip(*[r[1:] for r in load_runs([spec_id], label)]))
        logits = predict(net, *x_t, semantic)
        print(f"dev equation {spec_id}: AUC {roc_auc_score(y_t, logits):.4f} on {len(y_t)} crossovers, "
              f"logit spread (5-95%) {np.percentile(logits, 5):.2f} to {np.percentile(logits, 95):.2f}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("command", choices=["collect", "train"])
    pa.add_argument("--jobs", type=int, default=6)
    pa.add_argument("--label", choices=["median", "improves"], default="median")
    pa.add_argument("--semantic", action="store_true")
    args = pa.parse_args()
    collect(args.jobs) if args.command == "collect" else train(args.label, args.semantic)
