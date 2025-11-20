import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def f1(x): return x**3 + x**2 + x
def f2(x): return x**4 + x**3 + x**2 + x
def f3(x): return x**5 + x**4 + x**3 + x**2 + x
def f4(x): return x**6 + x**5 + x**4 + x**3 + x**2 + x
def f5(x): return np.sin(x**2)*np.cos(x) - 1
def f6(x): return np.sin(x) + np.sin(x + x**2)
def f7(x): return np.log(x + 1) + np.log(x**2 + 1)
def f8(x): return np.sqrt(np.abs(x))
def f9(x, y): return np.sin(x) + np.sin(y**2)
def f10(x, y): return 2*np.sin(x)*np.cos(y)
def f11(x, y): return x**y
def f12(x, y): return x**4 - x**3 + (y**2)/2 - y

def uniform_sampler(low, high):
    def _sample(rng, size):
        return rng.uniform(low, high, size)
    return _sample


FUNCTION_SPECS = [
    {
        "id": 1,
        "func": f1,
        "expr": "x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 2,
        "func": f2,
        "expr": "x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 3,
        "func": f3,
        "expr": "x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 4,
        "func": f4,
        "expr": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 5,
        "func": f5,
        "expr": "np.sin(x**2)*np.cos(x) - 1",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 6,
        "func": f6,
        "expr": "np.sin(x) + np.sin(x + x**2)",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 7,
        "func": f7,
        "expr": "np.log(x + 1) + np.log(x**2 + 1)",
        "variables": [{"name": "x", "sampler": uniform_sampler(-0.9, 5.0)}],
    },
    {
        "id": 8,
        "func": f8,
        "expr": "np.sqrt(np.abs(x))",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 9,
        "func": f9,
        "expr": "np.sin(x) + np.sin(y**2)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-5.0, 5.0)},
            {"name": "y", "sampler": uniform_sampler(-5.0, 5.0)},
        ],
    },
    {
        "id": 10,
        "func": f10,
        "expr": "2*np.sin(x)*np.cos(y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-5.0, 5.0)},
            {"name": "y", "sampler": uniform_sampler(-5.0, 5.0)},
        ],
    },
    {
        "id": 11,
        "func": f11,
        "expr": "x**y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(0.1, 5.0)},
            {"name": "y", "sampler": uniform_sampler(0.1, 5.0)},
        ],
    },
    {
        "id": 12,
        "func": f12,
        "expr": "x**4 - x**3 + (y**2)/2 - y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-5.0, 5.0)},
            {"name": "y", "sampler": uniform_sampler(-5.0, 5.0)},
        ],
    },
]


def generate_dataset(size: int, seed: int = 42, out_dir: Path | None = None):
    rng = np.random.default_rng(seed)
    base_dir = Path("dataset") / "nguyen" if out_dir is None else Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    eq_records = []
    for spec in FUNCTION_SPECS:
        data = {}
        inputs = []
        for var_spec in spec["variables"]:
            values = var_spec["sampler"](rng, size)
            data[var_spec["name"]] = values
            inputs.append(values)

        targets = spec["func"](*inputs)
        df = pd.DataFrame(data)
        df["y"] = targets

        csv_path = base_dir / f"{spec['id']}.csv"
        df.to_csv(csv_path, index=False)
        eq_records.append({"path": str(csv_path), "equation": spec["expr"]})
        print(f"Wrote {csv_path} ({len(df)} rows)")

    eqs_df = pd.DataFrame(eq_records)
    eqs_df.to_csv(base_dir / "eqs.csv", index=False)
    print(f"Wrote {base_dir / 'eqs.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Nguyen datasets.")
    parser.add_argument(
        "--size",
        "-n",
        type=int,
        required=True,
        help="Number of samples per equation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom output directory (defaults to dataset/nguyen).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(size=args.size, seed=args.seed, out_dir=args.output)