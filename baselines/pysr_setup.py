"""PC setup for the PySR context runs (all inside rllxo_rev1): export the 630 tasks, create .venv-pysr with
PySR 2.7.2, let juliacall install Julia, and warm it up with one tiny fit."""
import subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent.parent
def run(*cmd):
    t = time.time(); print(">", *cmd, flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=root)
    print("rc", r.returncode, f"{time.time()-t:.0f}s", flush=True)
    if r.returncode: sys.exit(r.returncode)
run(sys.executable, "baselines/export_tasks.py", "10")
venv = root / ".venv-pysr"
if not venv.exists():
    run(sys.executable, "-m", "venv", venv)
py = venv / "Scripts" / "python.exe"
run(py, "-m", "pip", "install", "--quiet", "pysr==2.7.2")
run(py, "-c", "import numpy as np; from pysr import PySRRegressor as R; X=np.random.rand(30,1); "
              "R(niterations=1, parallelism='serial', deterministic=True, random_state=0, progress=False, "
              "verbosity=0, temp_equation_file=True).fit(X, X[:,0]**2); print('pysr ok')")
