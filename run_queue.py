"""Revision-1 run queue on the PC (20 workers): the GP runs one after the other, each resumable.
Order: extra, keijzer_orig, nguyen_orig, ablations, noise01, noise10, circuits. Writes results/queue_status.txt."""
import subprocess, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
J = "20"
NK = "1,2,3,4,5,8,9,10,11,101,102,103,104,105,106,107,108,109,110,111,112,113,115"
cs = ["compare_selectors.py"]
std = ["--seeds", "30", "--seed0", "0", "--jobs", J]
Q = [
    ("extra", cs + ["--equations", "heldout", "--arms", "ssc,a2c_noH,random", *std, "--tag", "extra"]),
    ("keijzer_orig", cs + ["--equations", "101,102,103,104,105,106,107,108,109,110,111,112,113,115",
                           "--arms", "random,a2c_t03,ssc", *std, "--tag", "keijzer_orig"]),
    ("nguyen_orig", cs + ["--equations", "1,2,3,4,5,8,9,10,11", "--arms", "random,a2c_t03,ssc", *std,
                          "--tag", "nguyen_orig", "--protocol", "orig"]),
    ("ablations", cs + ["--equations", NK, "--arms",
                        "a2c_t03,a2c_nocritic,a2c_noconst,random_noconst,a2c_m01,random_m01,sizefair", *std, "--tag", "ablations"]),
    ("noise01", cs + ["--equations", NK, "--arms", "random,a2c_t03,ssc", *std, "--tag", "noise01", "--noise", "0.01"]),
    ("noise10", cs + ["--equations", NK, "--arms", "random,a2c_t03,ssc", *std, "--tag", "noise10", "--noise", "0.1"]),
    ("circuits", ["circuit_experiment.py", "--arms", "swap,sem,a2c_sem,learned,uniform", "--targets", "30",
                  "--seeds", "10", "--jobs", J]),
]
out = here / "results" / "compare_selectors"; out.mkdir(parents=True, exist_ok=True)
status = here / "results" / "queue_status.txt"
for name, cmd in Q:
    t0 = time.time()
    with open(status, "a") as s: s.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {name}\n")
    print("start", name, flush=True)
    with open(out / f"{name}.log", "a") as log:
        rc = subprocess.run([sys.executable, *cmd], cwd=here, stdout=log, stderr=subprocess.STDOUT).returncode
    msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} end {name} rc={rc} minutes={(time.time()-t0)/60:.1f}"
    with open(status, "a") as s: s.write(msg + "\n")
    print(msg, flush=True)
