"""Generate small fake CSVs strictly for software checks, never research results."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ids.schema import FEATURE_SPECS


def make_fixture(destination, rows=1200):
    destination = Path(destination)
    if rows < 100:
        raise ValueError("Use at least 100 rows.")
    for year, seed in [(2017, 11), (2018, 22)]:
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, 0.3, rows)
        x = rng.lognormal(2 + y[:, None] * 0.9, 0.9, (rows, len(FEATURE_SPECS)))
        # Feature-specific magnitudes and overlapping classes; not a benchmark.
        x *= np.linspace(1, 10000, len(FEATURE_SPECS))
        x[rng.random(x.shape) < 0.005] = np.nan
        headers = [spec[1 if year == 2017 else 2] for spec in FEATURE_SPECS]
        frame = pd.DataFrame(x, columns=headers)
        frame["Label"] = np.where(y, "DDoS" if year == 2017 else "DDOS attack-HOIC", "BENIGN" if year == 2017 else "Benign")
        folder = destination / str(year)
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / f"SYNTHETIC_{year}_NOT_BENCHMARK.csv", index=False)
    (destination / "SYNTHETIC_ONLY.txt").write_text(
        "These CSV files are artificial software fixtures, not CIC-IDS2017/2018 data.\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--rows", type=int, default=1200)
    args = parser.parse_args()
    make_fixture(args.output, args.rows)
    print(f"Created SYNTHETIC ONLY fixtures in {args.output}")
