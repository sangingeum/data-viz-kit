"""Demo: interactive CSV coordinate viewer.

Generates a sample CSV with multiple station+sensor identifiers and opens
the interactive viewer with time-range sliders, identifier checkboxes, and
hover tooltips.

Pass --headless to render a PNG proof instead of opening a window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless by default; viewer switches if interactive
import numpy as np
import pandas as pd

from data_viz_kit.csv_viewer import load_csv, view_csv


def make_sample_csv(path: Path, n: int = 300, seed: int = 42) -> Path:
    """Write a sample CSV with 3 identifiers and return the path."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    t0 = 1725270000.0  # 2024-09-02 ~09:00 UTC

    for station, sensor in [("StationA", "S1"), ("StationA", "S2"), ("StationB", "S1")]:
        timestamps = np.sort(t0 + rng.uniform(0.0, 3600.0, size=n))
        for t in timestamps:
            rows.append(
                {
                    "station": station,
                    "sensor": sensor,
                    "Timestamp": round(t, 3),
                    "N": round(100 + np.sin((t - t0) / 600) * 5 + rng.normal(0, 0.3), 4),
                    "U": round(50 + np.cos((t - t0) / 400) * 2 + rng.normal(0, 0.1), 4),
                    "E": round(200 + (t - t0) * 0.005 + rng.normal(0, 0.5), 4),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="path to an existing CSV file")
    parser.add_argument("--headless", action="store_true", help="render PNG proof")
    parser.add_argument("--out", default="examples/csv_viewer_demo.png")
    parser.add_argument(
        "--id-cols",
        nargs="+",
        default=["station", "sensor"],
        help="column names to use as identifiers",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        df = load_csv(csv_path, id_cols=args.id_cols)
    else:
        csv_path = Path(__file__).resolve().parent / "sample_data.csv"
        make_sample_csv(csv_path)
        df = load_csv(csv_path, id_cols=args.id_cols)

    print(
        f"dataset: {len(df)} rows, "
        f"{len(df['_identifier'].unique())} identifiers, "
        f"timestamps {df['Timestamp'].min():.1f} .. {df['Timestamp'].max():.1f} (epoch s)"
    )

    if args.headless:
        fig, viewer = view_csv(
            df,
            id_cols=args.id_cols,
            title="CSV Coordinate Viewer (headless)",
            block=False,
        )
        # Narrow the time window to prove sliders work.
        lo = df["Timestamp"].min() + 600.0
        hi = df["Timestamp"].max() - 600.0
        viewer["sliders"]["t1"].set_val(lo)  # type: ignore[index]
        viewer["sliders"]["t2"].set_val(hi)  # type: ignore[index]
        fig.canvas.draw()
        a, b = viewer["get_visible_range"]()  # type: ignore[misc]
        print(f"headless OK: visible range [{a:.1f}, {b:.1f}]; PNG: {args.out}")
        fig.savefig(args.out, dpi=150)
        return 0

    # Interactive mode
    view_csv(
        df,
        id_cols=args.id_cols,
        title="CSV Coordinate Viewer — drag t1/t2 sliders, toggle checkboxes, hover points",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
