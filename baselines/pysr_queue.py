"""Waits for run_queue.py to finish the GP runs, then fits PySR to all 630 exported tasks with K long-lived
worker processes (Julia starts once per worker, not once per task), then scores them.
Each task is still fit by run_pysr.main(eq, seed): same settings, serial, deterministic, random_state=seed.
Usage: python baselines/pysr_queue.py [K]          (worker mode: --worker i K, run inside .venv-pysr)"""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def tasks():
    eqs = sorted(int(p.stem) for p in (HERE / "tasks").glob("*.json"))
    return [(e, s) for s in range(10) for e in eqs]

def worker(i, k):
    sys.path.insert(0, str(HERE))
    import run_pysr
    for j, (e, s) in enumerate(tasks()):
        if j % k == i and not (HERE / "pysr_out" / f"{e}_{s}.json").exists():
            try:
                run_pysr.main(e, s)
            except Exception as ex:
                print("FAILED", e, s, repr(ex), flush=True)

if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(int(sys.argv[2]), int(sys.argv[3])); sys.exit()
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    status = ROOT / "results" / "queue_status.txt"
    while not (status.exists() and "end circuits" in status.read_text()):
        time.sleep(60)
    log = ROOT / "results" / "compare_selectors" / "pysr.log"
    py = ROOT / ".venv-pysr" / "Scripts" / "python.exe"
    with open(status, "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start pysr k={k}\n")
    t0 = time.time()
    with open(log, "a") as lf:
        ps = [subprocess.Popen([str(py), __file__, "--worker", str(i), str(k)], cwd=ROOT, stdout=lf,
                               stderr=subprocess.STDOUT) for i in range(k)]
        rcs = [p.wait() for p in ps]
        subprocess.run([sys.executable, str(HERE / "score_pysr.py")], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    done = len(list((HERE / "pysr_out").glob("*.json")))
    with open(status, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end pysr rcs={rcs} done={done}/630 minutes={(time.time()-t0)/60:.1f}\n")
