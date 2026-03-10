"""Feynman benchmark: learned vs random crossover on ≤3-var Feynman equations.
Usage: python feynman.py [--jobs N] [--shard K/N] [--merge]
"""
from __future__ import annotations
import argparse, csv, glob, importlib, multiprocessing as mp, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np, sympy as sp
from main import run_ga

SEED, SAMPLE_SIZE, MAX_VARS, SEEDS = 1, 512, 3, list(range(30))

_FNS: Dict[str, Any] = {
    "sqrt": np.sqrt, "exp": np.exp, "sin": np.sin, "cos": np.cos,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "tanh": np.tanh, "ln": np.log, "log": np.log, "abs": np.abs, "pi": np.pi,
}
_UNARY = {
    "sin(": "sin", "cos(": "cos", "exp(": "exp", "sqrt(": "sqrt",
    "arcsin(": "arcsin", "arccos(": "arccos", "tanh(": "tanh",
    "ln(": "log", "log(": "log", "abs(": "abs",
}
_UNARY_REG = {
    "sin": (np.sin, sp.sin), "cos": (np.cos, sp.cos), "exp": (np.exp, sp.exp),
    "log": (np.log, sp.log), "sqrt": (np.sqrt, sp.sqrt), "abs": (np.abs, sp.Abs),
    "arcsin": (np.arcsin, sp.asin), "arccos": (np.arccos, sp.acos), "tanh": (np.tanh, sp.tanh),
}

# Load specs at import time
SPECS: List[Dict] = []
_csv = Path(__file__).parent / "dataset" / "FeynmanEquations.csv"
if _csv.exists():
    with open(_csv, encoding="utf-8-sig") as _f:
        for _r in csv.DictReader(_f):
            try:
                _nv = int(float(_r.get("# variables", "").strip()))
            except (ValueError, TypeError):
                continue
            if not 1 <= _nv <= MAX_VARS:
                continue
            _fk = [k for k in _r if "Filename" in k][0]
            _fm = _r["Formula"].strip()
            _vars = []
            for _i in range(1, 11):
                _vn, _vl, _vh = (_r.get(f"v{_i}_{s}", "").strip() for s in ("name", "low", "high"))
                if not _vn or not _vl or not _vh:
                    break
                _vars.append({"name": _vn, "low": float(_vl), "high": float(_vh)})
            if not 1 <= len(_vars) <= MAX_VARS:
                continue
            SPECS.append({
                "id": int(float(_r["Number"])), "filename": _r[_fk].strip(),
                "formula": _fm, "n_vars": len(_vars), "variables": _vars,
                "allowed_unary": sorted({_UNARY[k] for k in _UNARY if k in _fm}),
            })

ga_config = {
    "pop_size": 200, "gens": 200, "max_init_depth": 6, "max_depth": 30,
    "crossover_rate": 0.85, "mutation_rate": 0.5, "tournament_k": 9, "elitism": 24,
    "use_learned_crossover": False, "selector_lr": 0.001,
    "mlp_layers": 16, "mlp_hidden_dim": 64, "mlp_activation": "leaky_relu", "mlp_dropout": 0.1,
    "use_critic": True, "stopping_metric": "r2", "stopping_threshold": 1.0,
    "complexity_coef": 1e-4, "inclusion_coef": 1.0, "prob_const": 0.3, "const_range": (-5, 5),
    "point_mutation_prob": 0.1, "simplify_new_genomes": False, "simplify_timeout": 0.2,
    "selector_update_freq": 1, "temperature": 0.7, "entropy_weight": 0.1,
    "opt_constants": True, "const_opt_freq": 10, "const_opt_method": "L-BFGS-B",
    "const_opt_maxiter": 100, "verbose": True, "use_pareto": True, "print_freq": 10, "seed": SEED,
}
learned_overrides = {
    "use_learned_crossover": True, "mlp_layers": 4, "mlp_hidden_dim": 64, "mlp_dropout": 0.0,
    "temperature": 1.0, "entropy_weight": 0.08, "heuristic_bias_scale": 0.0,
    "selector_update_freq": 1, "selector_warmup_gens": 30, "selector_recent_gen_window": 50,
    "selector_min_history": 64, "selector_reward_clip": 3.0, "selector_lr": 3e-4,
    "use_critic": True, "selector_device": "cuda",
}
random_overrides = {"use_learned_crossover": False, "selector_device": "cuda"}

@dataclass(frozen=True)
class Task:
    mode: str
    spec_id: int
    seed: int

def run_single_task(t: Task) -> Dict:
    spec = next((s for s in SPECS if s["id"] == t.spec_id), None)
    if spec is None:
        return {"mode": t.mode, "spec_id": t.spec_id, "seed": t.seed, "filename": "?",
                "formula": "?", "n_vars": 0, "status": "error", "best_r2": float("nan"),
                "best_expr": "", "elapsed_sec": 0.0, "error": f"spec_id {t.spec_id} not found"}
    started = time.time()
    status, error, best_expr, best_r2 = "ok", "", "", float("nan")
    try:
        rng = np.random.default_rng(t.seed + t.spec_id * 1000)
        vnames = [v["name"] for v in spec["variables"]]
        cols = {v["name"]: rng.uniform(v["low"], v["high"], size=SAMPLE_SIZE) for v in spec["variables"]}
        ns = dict(_FNS)
        exec(compile(f"def _f({','.join(vnames)}):\n return {spec['formula']}\n",
                      f"<feynman:{spec['formula']}>", "exec"), ns)
        fn = ns["_f"]
        try:
            y = fn(*[cols[n] for n in vnames])
        except Exception:
            y = np.array([fn(*[cols[n][i] for n in vnames]) for i in range(SAMPLE_SIZE)])
        X = np.column_stack([cols[n] for n in vnames])
        y = np.asarray(y, dtype=float).ravel()
        ok = np.isfinite(y)
        if not ok.all():
            X, y = X[ok], y[ok]
        if len(y) < 10:
            raise ValueError(f"Too few valid samples ({len(y)}) for {spec['filename']}")
        unary = {n: _UNARY_REG[n] for n in (spec.get("allowed_unary") or []) if n in _UNARY_REG}
        kw = {**ga_config, **(learned_overrides if t.mode == "learned" else random_overrides), "seed": t.seed}
        best, r2v = run_ga(X, y, var_names=vnames, unary_funcs=unary, **kw)
        best_r2 = float(r2v) if r2v is not None else float("nan")
        best_expr = best.to_string() if best is not None else ""
    except Exception as exc:
        status, error = "error", str(exc)
    return {"mode": t.mode, "spec_id": t.spec_id, "seed": t.seed, "filename": spec["filename"],
            "formula": spec["formula"], "n_vars": spec["n_vars"], "status": status,
            "best_r2": best_r2, "best_expr": best_expr, "elapsed_sec": time.time() - started, "error": error}

# --- Shared helpers (called 2+ times) ---

def summarize_results(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for mode, sid in sorted({(r["mode"], r["spec_id"]) for r in rows}):
        sub = [r for r in rows if r["mode"] == mode and r["spec_id"] == sid and r["status"] == "ok"]
        r2 = np.array([r["best_r2"] for r in sub if np.isfinite(r["best_r2"])], dtype=float)
        ts = np.array([r["elapsed_sec"] for r in sub], dtype=float)
        sp_ = next((s for s in SPECS if s["id"] == sid), {})
        f = lambda a, fn: float(fn(a)) if a.size else float("nan")
        out.append({"mode": mode, "spec_id": sid, "filename": sp_.get("filename", "?"),
                     "n_vars": sp_.get("n_vars", 0), "runs_ok": len(sub),
                     "runs_total": sum(1 for r in rows if r["mode"] == mode and r["spec_id"] == sid),
                     "r2_mean": f(r2, np.mean), "r2_median": f(r2, np.median),
                     "r2_std": f(r2, np.std), "r2_min": f(r2, np.min), "r2_max": f(r2, np.max),
                     "time_mean_sec": f(ts, np.mean), "time_std_sec": f(ts, np.std)})
    return out

def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode","spec_id","filename","formula","n_vars",
                                          "seed","status","best_r2","best_expr","elapsed_sec","error"])
        w.writeheader(); w.writerows(rows)

def write_summary_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode","spec_id","filename","n_vars","runs_ok","runs_total",
                                          "r2_mean","r2_median","r2_std","r2_min","r2_max",
                                          "time_mean_sec","time_std_sec"])
        w.writeheader(); w.writerows(summarize_results(rows))

def fmt_dur(s: float) -> str:
    if not np.isfinite(s): return "?"
    t = max(0, int(round(s))); h, r = divmod(t, 3600); m, s2 = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s2:02d}" if h else f"{m:02d}:{s2:02d}"

def print_comparison(rows: List[Dict]) -> None:
    from scipy.stats import wilcoxon
    summary = summarize_results(rows)
    sids = sorted({s["spec_id"] for s in summary})
    W = 130
    print("\n" + "=" * W + "\nFEYNMAN BENCHMARK — LEARNED vs RANDOM CROSSOVER\n" + "=" * W)
    lw = rw = ties = 0
    deltas: List[float] = []
    print(f"{'Spec':>6} {'File':>12} {'#v':>3} {'L mean±std':>14} {'L med':>7}"
          f" {'R mean±std':>14} {'R med':>7} {'Delta':>8} {'L t/s':>7} {'R t/s':>7} {'Winner':>8}")
    print("-" * W)
    for sid in sids:
        lr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "learned"), None)
        rr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "random"), None)
        if not lr or not rr: continue
        d = lr["r2_mean"] - rr["r2_mean"]; deltas.append(d)
        if abs(d) < 1e-4: w, ties = "TIE", ties + 1
        elif d > 0: w = "LEARNED"; lw += 1
        else: w = "RANDOM"; rw += 1
        print(f"{sid:>6} {lr['filename'][:12]:>12} {lr['n_vars']:>3}"
              f" {lr['r2_mean']:>7.4f}±{lr['r2_std']:<5.3f} {lr['r2_median']:>7.4f}"
              f" {rr['r2_mean']:>7.4f}±{rr['r2_std']:<5.3f} {rr['r2_median']:>7.4f}"
              f" {d:>+8.4f} {lr['time_mean_sec']:>7.1f} {rr['time_mean_sec']:>7.1f} {w:>8}")
    print("-" * W)
    da = np.array(deltas) if deltas else np.array([])
    print(f"Learned wins: {lw}  |  Random wins: {rw}  |  Ties: {ties}")
    if da.size:
        print(f"Mean   R² delta: {float(np.mean(da)):+.4f}")
        print(f"Median R² delta: {float(np.median(da)):+.4f}")
    if da.size >= 6:
        try:
            stat, pv = wilcoxon(da, alternative="two-sided")
            ne = np.sum(da != 0)
            rb = 1.0 - (2.0 * stat) / (ne * (ne + 1) / 2) if ne > 0 else 0.0
            sig = " *" if pv < 0.05 else " (*)" if pv < 0.10 else ""
            print(f"Wilcoxon: W={stat:.1f}, p={pv:.4f}{sig}  |  Rank-biserial r={rb:+.3f}")
        except Exception as e:
            print(f"Wilcoxon failed: {e}")
    if da.size >= 2:
        sd = np.std(da, ddof=1)
        cd = float(np.mean(da) / sd) if sd > 0 else 0.0
        mg = "large" if abs(cd) >= 0.8 else "medium" if abs(cd) >= 0.5 else "small" if abs(cd) >= 0.2 else "negligible"
        print(f"Cohen's d: {cd:+.3f} ({mg})")
    print(f"Equations: {len(sids)}  |  Seeds: {len(SEEDS)}\n" + "=" * W)

def write_latex_table(rows: List[Dict], path: Path) -> None:
    from scipy.stats import wilcoxon
    summary = summarize_results(rows)
    sids = sorted({s["spec_id"] for s in summary})
    L: List[str] = [r"\begin{table*}[t]", r"\centering",
        rf"\caption{{Feynman benchmark: learned vs.\ random crossover ({len(SEEDS)} seeds per config).}}",
        r"\label{tab:feynman_results}", r"\begin{tabular}{rl c rr rr r r l}", r"\toprule",
        r"\# & Equation & $n$ & \multicolumn{2}{c}{Learned $R^2$} & "
        r"\multicolumn{2}{c}{Random $R^2$} & $\Delta$ & Time (s) & Winner \\",
        r" &  &  & mean & med & mean & med &  & L / R &  \\", r"\midrule"]
    deltas: List[float] = []; lw = rw = tc = 0
    for sid in sids:
        lr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "learned"), None)
        rr = next((s for s in summary if s["spec_id"] == sid and s["mode"] == "random"), None)
        if not lr or not rr: continue
        lm, rm = lr["r2_mean"], rr["r2_mean"]; d = lm - rm; deltas.append(d)
        fn = lr["filename"].replace("_", r"\_")
        if abs(d) < 1e-4: w, tc = "Tie", tc + 1
        elif d > 0: w = r"\textbf{Learned}"; lw += 1
        else: w = "Random"; rw += 1
        ls = f"\\textbf{{{lm:.4f}}}" if d > 1e-4 else f"{lm:.4f}"
        rs = f"\\textbf{{{rm:.4f}}}" if d < -1e-4 else f"{rm:.4f}"
        L.append(f"  {sid} & {fn} & {lr['n_vars']} & {ls} & {lr['r2_median']:.4f} & "
                 f"{rs} & {rr['r2_median']:.4f} & {d:+.4f} & "
                 f"{lr['time_mean_sec']:.0f}/{rr['time_mean_sec']:.0f} & {w} \\\\")
    L.append(r"\midrule")
    da = np.array(deltas) if deltas else np.array([])
    md = float(np.mean(da)) if da.size else 0.0
    mdd = float(np.median(da)) if da.size else 0.0
    L.append(f"  \\multicolumn{{9}}{{l}}{{Learned wins: {lw}, Random wins: {rw}, Ties: {tc}. "
             f"Mean $\\Delta R^2$: {md:+.4f}, Median $\\Delta R^2$: {mdd:+.4f}.}} \\\\")
    if da.size >= 6:
        try:
            st, pv = wilcoxon(da, alternative="two-sided"); ne = np.sum(da != 0)
            rb = 1.0 - (2.0 * st) / (ne * (ne + 1) / 2) if ne > 0 else 0.0
            sd = np.std(da, ddof=1); cd = float(np.mean(da) / sd) if sd > 0 else 0.0
            L.append(f"  \\multicolumn{{9}}{{l}}{{Wilcoxon $W={st:.1f}$, $p={pv:.4f}$. "
                     f"Rank-biserial $r={rb:+.3f}$. Cohen's $d={cd:+.3f}$.}} \\\\")
        except Exception: pass
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"LaTeX table: {path}")

# --- Entry point ---

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Feynman benchmark")
    pa.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 1))
    pa.add_argument("--output", type=Path, default=Path("feynman_results.csv"))
    pa.add_argument("--summary-output", type=Path, default=Path("feynman_results_summary.csv"))
    pa.add_argument("--progress-every", type=int, default=10)
    pa.add_argument("--tensorboard-logdir", type=Path, default=Path("logs/feynman_tensorboard"))
    pa.add_argument("--shard", type=str, default=None, help="K/N (e.g. 1/8)")
    pa.add_argument("--merge", action="store_true", help="Merge shard CSVs")
    pa.add_argument("--latex-output", type=Path, default=Path("feynman_results.tex"))
    args = pa.parse_args()

    if args.merge:
        files = sorted(glob.glob(str(args.output.parent / "shards" / "feynman_results_shard*.csv")))
        if not files:
            raise SystemExit(f"No shard files found")
        all_rows: List[Dict] = []
        for p in files:
            print(f"Reading {p} ...", flush=True)
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    r["spec_id"], r["seed"], r["n_vars"] = int(r["spec_id"]), int(r["seed"]), int(r["n_vars"])
                    r["best_r2"], r["elapsed_sec"] = float(r["best_r2"]), float(r["elapsed_sec"])
                    all_rows.append(r)
        all_rows.sort(key=lambda r: (r["mode"], r["spec_id"], r["seed"]))
        d = args.output.parent
        write_csv(all_rows, d / "feynman_results.csv")
        write_summary_csv(all_rows, d / "feynman_results_summary.csv")
        print(f"\nMerged {len(all_rows)} rows from {len(files)} shards")
        print_comparison(all_rows)
        write_latex_table(all_rows, d / "feynman_results.tex")
        raise SystemExit(0)

    shard_idx = num_shards = None
    spec_sub: List[Dict] | None = None
    if args.shard:
        try:
            k, n = args.shard.split("/"); shard_idx, num_shards = int(k), int(n)
        except (ValueError, AttributeError):
            raise SystemExit(f"ERROR: --shard must be K/N, got '{args.shard}'")
        if not (1 <= shard_idx <= num_shards):
            raise SystemExit(f"ERROR: shard {shard_idx} out of 1..{num_shards}")
        ss = sorted(SPECS, key=lambda s: s["id"])
        spec_sub = [s for i, s in enumerate(ss) if i % num_shards == shard_idx - 1]
        if not spec_sub:
            raise SystemExit(f"Shard {shard_idx}/{num_shards} is empty.")

    specs = spec_sub or SPECS
    tasks = [Task(m, s["id"], sd) for m in ("random", "learned") for s in specs for sd in SEEDS]
    nt = len(tasks)

    if shard_idx:
        out_csv = args.output.with_name(f"{args.output.stem}_shard{shard_idx}.csv")
        sum_csv = args.summary_output.with_name(f"{args.summary_output.stem}_shard{shard_idx}.csv")
        tb_dir = args.tensorboard_logdir / f"shard{shard_idx}" if args.tensorboard_logdir else None
        tag = f"  [shard {shard_idx}/{num_shards}]"
    else:
        out_csv, sum_csv, tb_dir, tag = args.output, args.summary_output, args.tensorboard_logdir, ""

    jobs = max(1, min(args.jobs, nt))
    print(f"Feynman{tag}: {len(specs)} eq × {len(SEEDS)} seeds × 2 = {nt} tasks | Workers: {jobs}")
    print(f"IDs: {sorted(s['id'] for s in specs)}")

    tb = None
    if tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter; tb = SummaryWriter(log_dir=str(tb_dir))
        except Exception:
            try: tb = getattr(importlib.import_module("tensorboardX"), "SummaryWriter")(log_dir=str(tb_dir))
            except Exception: pass
        if tb: print(f"TensorBoard: {tb_dir}", flush=True)

    t0 = time.time(); rows: List[Dict] = []; mc = {"random": 0, "learned": 0}; errs = 0
    pf = max(1, args.progress_every)
    try:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for done, row in enumerate(pool.imap_unordered(run_single_task, tasks), 1):
                rows.append(row); m = row["mode"]; mc[m] += 1
                if row["status"] != "ok": errs += 1
                dt = time.time() - t0; rate = done / dt if dt > 0 else 0; eta = (nt - done) / rate if rate > 0 else float("inf")
                if done == 1 or done == nt or done % pf == 0 or row["status"] != "ok":
                    r2t = f"{row['best_r2']:.4f}" if np.isfinite(row["best_r2"]) else "nan"
                    print(f"[{done}/{nt}] {m} spec={row['spec_id']} {row.get('filename','?')} "
                          f"seed={row['seed']} {row['status']} r2={r2t} "
                          f"{row['elapsed_sec']:.1f}s errs={errs} eta={fmt_dur(eta)}", flush=True)
                if tb:
                    tb.add_scalar("progress/done", done, done)
                    tb.add_scalar("timing/task_sec", float(row["elapsed_sec"]), done)
                    if row["status"] == "ok" and np.isfinite(row["best_r2"]):
                        tb.add_scalar(f"metrics/{m}_r2", float(row["best_r2"]), done)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving partial results...")
    finally:
        if tb: tb.flush(); tb.close()

    rows.sort(key=lambda r: (r["mode"], r["spec_id"], r["seed"]))
    write_csv(rows, out_csv); write_summary_csv(rows, sum_csv)
    print(f"\nDone in {fmt_dur(time.time() - t0)}. {len(rows)} rows -> {out_csv} | Errors: {sum(1 for r in rows if r['status'] != 'ok')}")
    print_comparison(rows); write_latex_table(rows, args.latex_output)
