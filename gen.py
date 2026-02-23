import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def uniform_sampler(low, high):
    def _sample(rng, size):
        return rng.uniform(low, high, size)
    return _sample

def int_sampler(low, high):
    def _sample(rng, size):
        return rng.integers(low, high + 1, size)
    return _sample


FUNCTION_SPECS = [
    {
        "id": 1,
        "func": lambda x: x**3 + x**2 + x,
        "expr": "x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 2,
        "func": lambda x: x**4 + x**3 + x**2 + x,
        "expr": "x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 3,
        "func": lambda x: x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-5.0, 5.0)}],
    },
    {
        "id": 4,
        "func": lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-3.0, 3.0)}],
    },
    {
        "id": 5,
        "func": lambda x: np.sin(x**2) * np.cos(x) - 1,
        "expr": "np.sin(x**2)*np.cos(x) - 1",
        "variables": [{"name": "x", "sampler": uniform_sampler(-2.0, 2.0)}],
    },
    {
        "id": 6,
        "func": lambda x: np.sin(x) + np.sin(x + x**2),
        "expr": "np.sin(x) + np.sin(x + x**2)",
        "variables": [{"name": "x", "sampler": uniform_sampler(-10.0, 10.0)}],
    },
    {
        "id": 7,
        "func": lambda x: np.log(x + 1) + np.log(x**2 + 1),
        "expr": "np.log(x + 1) + np.log(x**2 + 1)",
        "variables": [{"name": "x", "sampler": uniform_sampler(100.0, 2000.0)}],
    },
    {
        "id": 8,
        "func": lambda x: np.sqrt(np.abs(x)),
        "expr": "np.sqrt(np.abs(x))",
        "variables": [{"name": "x", "sampler": uniform_sampler(-10.0, 10.0)}],
    },
    {
        "id": 9,
        "func": lambda x, y: np.sin(x) + np.sin(y**2),
        "expr": "np.sin(x) + np.sin(y**2)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-10.0, 10.0)},
            {"name": "y", "sampler": uniform_sampler(-10.0, 10.0)},
        ],
    },
    {
        "id": 10,
        "func": lambda x, y: 2 * np.sin(x) * np.cos(y),
        "expr": "2*np.sin(x)*np.cos(y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-10.0, 10.0)},
            {"name": "y", "sampler": uniform_sampler(-10.0, 10.0)},
        ],
    },
    {
        "id": 11,
        "func": lambda x, y: x**y,
        "expr": "x**y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(0.1, 2.0)},
            {"name": "y", "sampler": uniform_sampler(0.1, 2.0)},
        ],
    },
    {
        "id": 12,
        "func": lambda x, y: x**4 - x**3 + (y**2) / 2 - y,
        "expr": "x**4 - x**3 + (y**2)/2 - y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-4.0, 4.0)},
            {"name": "y", "sampler": uniform_sampler(-4.0, 4.0)},
        ],
    },
]


# Jin benchmark (all variables sampled U(-3, 3))
JIN_SPECS = [
    {
        "id": 1,
        "func": lambda x, y: 2.5 * x**4 - 1.3 * x**3 + 0.5 * y**2 - 1.7 * y,
        "expr": "2.5*x**4 - 1.3*x**3 + 0.5*y**2 - 1.7*y",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-3, 3)},
            {"name": "y", "sampler": uniform_sampler(-3, 3)},
        ],
    },
    {
        "id": 2,
        "func": lambda x, y: 8.0 * x**2 + 8.0 * y**3 - 15.0,
        "expr": "8.0*x**2 + 8.0*y**3 - 15.0",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-3, 3)},
            {"name": "y", "sampler": uniform_sampler(-3, 3)},
        ],
    },
    {
        "id": 3,
        "func": lambda x, y: 0.2 * x**3 + 0.5 * y**3 - 1.2 * y - 0.5 * x,
        "expr": "0.2*x**3 + 0.5*y**3 - 1.2*y - 0.5*x",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-3, 3)},
            {"name": "y", "sampler": uniform_sampler(-3, 3)},
        ],
    },
    {
        "id": 4,
        "func": lambda x, y: 1.5 * np.exp(x) + 5.0 * np.cos(y),
        "expr": "1.5*exp(x) + 5.0*cos(y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-3, 3)},
            {"name": "y", "sampler": uniform_sampler(-3, 3)},
        ],
    },
    {
        "id": 5,
        "func": lambda x, y: 6.0 * np.sin(x) * np.cos(y),
        "expr": "6.0*sin(x)*cos(y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-3, 3)},
            {"name": "y", "sampler": uniform_sampler(-3, 3)},
        ],
    },
    {
        "id": 6,
        "func": lambda x, y: 1.35 * x * y + 5.5 * np.sin((x - 1.0) * (y - 1.0)),
        "expr": "1.35*x*y + 5.5*sin((x - 1.0)*(y - 1.0))",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-3, 3)},
            {"name": "y", "sampler": uniform_sampler(-3, 3)},
        ],
    },
]


# Neat benchmark
NEAT_SPECS = [
    {
        "id": 1,
        "func": lambda x: x**4 + 3 * x**3 + x**2 + x,
        "expr": "x**4 + 3*x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-1, 1)}],
    },
    {
        "id": 2,
        "func": lambda x: x**5 + x**4 + x**3 + x**2 + x,
        "expr": "x**5 + x**4 + x**3 + x**2 + x",
        "variables": [{"name": "x", "sampler": uniform_sampler(-1, 1)}],
    },
    {
        "id": 3,
        "func": lambda x: np.sin(x**2) * np.cos(x) - 1,
        "expr": "sin(x**2)*cos(x) - 1",
        "variables": [{"name": "x", "sampler": uniform_sampler(-1, 1)}],
    },
    {
        "id": 4,
        "func": lambda x: np.log(x + 1) + np.log(x**2 + 1),
        "expr": "log(x + 1) + log(x**2 + 1)",
        "variables": [{"name": "x", "sampler": uniform_sampler(0, 2)}],
    },
    {
        "id": 5,
        "func": lambda x, y: 2 * np.sin(x) * np.cos(y),
        "expr": "2*sin(x)*cos(y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(0, 2)},
            {"name": "y", "sampler": uniform_sampler(0, 2)},
        ],
    },
    {
        "id": 6,
        "func": lambda x: np.array([np.sum(1.0 / np.arange(1, int(n) + 1)) for n in x.astype(int)]),
        "expr": "sum_{k=1}^x 1/k",
        "variables": [{"name": "x", "sampler": int_sampler(1, 50)}],
    },
    {
        "id": 7,
        "func": lambda x, y: 2 - 2.1 * np.cos(9.8 * x) * np.sin(1.3 * y),
        "expr": "2 - 2.1*cos(9.8*x)*sin(1.3*y)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-50, 50)},
            {"name": "y", "sampler": uniform_sampler(-50, 50)},
        ],
    },
    {
        "id": 8,
        "func": lambda x, y: np.exp(-(x - 2) ** 2) / (1.2 + (y - 2.5) ** 2),
        "expr": "exp(-(x - 2)**2) / (1.2 + (y - 2.5)**2)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(0.3, 4)},
            {"name": "y", "sampler": uniform_sampler(0.3, 4)},
        ],
    },
    {
        "id": 9,
        "func": lambda x, y: 1 / (1 + x ** -4) + 1 / (1 + y ** -4),
        "expr": "1/(1 + x**-4) + 1/(1 + y**-4)",
        "variables": [
            {"name": "x", "sampler": uniform_sampler(-5, 5)},
            {"name": "y", "sampler": uniform_sampler(-5, 5)},
        ],
    },
]


def _generate_benchmark(specs, size: int, seed: int, out_dir: Path, *,
                        extrapolation_size: int | None = None,
                        extrapolation_ratio: float = 0.2,
                        extrapolation_bounds: tuple[float, float] | None = None,
                        name: str = "benchmark"):
    """Generic dataset generator shared by Nguyen, Jin, and Neat benchmarks."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    if extrapolation_size is None:
        extrapolation_size = int(size * extrapolation_ratio)

    eq_records = []
    for spec in specs:
        data = {}
        inputs = []
        for var_spec in spec["variables"]:
            values = var_spec["sampler"](rng, size)
            data[var_spec["name"]] = values
            inputs.append(values)

        targets = spec["func"](*inputs)
        df = pd.DataFrame(data)
        df["target"] = targets

        csv_path = out_dir / f"{spec['id']}.csv"
        df.to_csv(csv_path, index=False)
        eq_records.append({"path": str(csv_path), "equation": spec.get("expr", "")})
        print(f"Wrote {csv_path} ({len(df)} rows)")

        # Optional extrapolation generation if bounds are supplied
        if extrapolation_bounds is not None and extrapolation_size > 0:
            low, high = extrapolation_bounds
            extrap_data = {}
            extrap_inputs = []
            for var_spec in spec["variables"]:
                n_low = extrapolation_size // 2
                n_high = extrapolation_size - n_low
                values_low = rng.uniform(low, low + 1.0, n_low)
                values_high = rng.uniform(high - 1.0, high, n_high)
                values = np.concatenate([values_low, values_high])
                extrap_data[var_spec["name"]] = values
                extrap_inputs.append(values)

            extrap_targets = spec["func"](*extrap_inputs)
            extrap_df = pd.DataFrame(extrap_data)
            extrap_df["target"] = extrap_targets

            extrap_csv_path = out_dir / f"{spec['id']}_extrapolation.csv"
            extrap_df.to_csv(extrap_csv_path, index=False)
            print(f"Wrote {extrap_csv_path} ({len(extrap_df)} rows)")

    eqs_df = pd.DataFrame(eq_records)
    eqs_df.to_csv(out_dir / "eqs.csv", index=False)
    print(f"Wrote {out_dir / 'eqs.csv'} for {name}")


def generate_dataset(size: int, seed: int = 42, out_dir: Path | None = None,
                     extrapolation_size: int | None = None,
                     extrapolation_ratio: float = 0.2):
    """
    Generate Nguyen benchmark datasets (backward-compatible wrapper).
    """
    base_dir = Path("dataset") / "nguyen" if out_dir is None else Path(out_dir)
    _generate_benchmark(
        FUNCTION_SPECS,
        size=size,
        seed=seed,
        out_dir=base_dir,
        extrapolation_size=extrapolation_size,
        extrapolation_ratio=extrapolation_ratio,
        name="Nguyen",
    )


def generate_jin_dataset(size: int = 1000, seed: int = 42, out_dir: Path | None = None):
    base_dir = Path("dataset") / "jin" if out_dir is None else Path(out_dir)
    _generate_benchmark(
        JIN_SPECS,
        size=size,
        seed=seed,
        out_dir=base_dir,
        extrapolation_size=0,
        name="Jin",
    )


def generate_neat_dataset(size: int = 1000, seed: int = 42, out_dir: Path | None = None):
    base_dir = Path("dataset") / "neat" if out_dir is None else Path(out_dir)
    # Use a small extrapolation region outside the main support for smoother evaluation
    _generate_benchmark(
        NEAT_SPECS,
        size=size,
        seed=seed,
        out_dir=base_dir,
        extrapolation_size=int(size * 0.2),
        extrapolation_ratio=0.2,
        extrapolation_bounds=(-6.0, 6.0),
        name="Neat",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate symbolic regression benchmark datasets.")
    parser.add_argument(
        "--benchmark",
        choices=["nguyen", "jin", "neat"],
        default="nguyen",
        help="Which benchmark to generate.",
    )
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
    parser.add_argument(
        "--extrapolation-size",
        type=int,
        default=None,
        help="Number of extrapolation samples. If not specified, uses --extrapolation-ratio.",
    )
    parser.add_argument(
        "--extrapolation-ratio",
        type=float,
        default=0.2,
        help="Fraction of training size to use for extrapolation samples (default: 0.2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.benchmark == "nguyen":
        generate_dataset(
            size=args.size,
            seed=args.seed,
            out_dir=args.output,
            extrapolation_size=args.extrapolation_size,
            extrapolation_ratio=args.extrapolation_ratio,
        )
    elif args.benchmark == "jin":
        generate_jin_dataset(
            size=args.size,
            seed=args.seed,
            out_dir=args.output,
        )
    elif args.benchmark == "neat":
        generate_neat_dataset(
            size=args.size,
            seed=args.seed,
            out_dir=args.output,
        )