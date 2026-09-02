"""Demo: interactive time-range viewer with hover tooltips.

Generates a sample dataset of epoch-second (float) timestamps and opens
the interactive window: two sliders pick [t1, t2], the plot shows only
data in that range, and hovering a point shows a tooltip with the series
name and exact x/y values.

Requires a GUI backend (QtAgg/TkAgg); the viewer picks one automatically.
For a headless proof, pass --headless (renders a PNG with a mid-range window).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # headless by default; viewer switches backend if interactive
import numpy as np
import pandas as pd

from data_viz_kit.timeline_viewer import view_time_range


def make_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Sample dataset: float epoch-second timestamps + two noisy signals."""
    rng = np.random.default_rng(seed)
    t0 = 1725270000.0  # 2024-09-02 ~09:00 UTC
    df = pd.DataFrame(
        {
            "timestamp": np.sort(t0 + rng.uniform(0.0, 3600.0, size=n)),
            "temperature": 20 + np.sin(np.linspace(0, 8 * np.pi, n)) * 3 + rng.normal(0, 0.3, n),
            "pressure": 1013 + np.cumsum(rng.normal(0, 0.05, n)),
        }
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="render PNG proof instead of opening a window")
    parser.add_argument("--out", default="examples/timestamp_viewer_demo.png")
    args = parser.parse_args()

    df = make_sample_data()
    print(f"dataset: {len(df)} rows, timestamps {df['timestamp'].min():.1f} .. {df['timestamp'].max():.1f} (epoch s)")

    if args.headless:
        # Headless proof: build viewer (block=False), move sliders programmatically,
        # simulate the same redraw the on_changed callbacks drive, save PNG.
        fig, viewer = view_time_range(
            df,
            time_col="timestamp",
            value_cols=["temperature", "pressure"],
            t1=None,
            t2=None,
            title="Timestamp viewer demo (headless render)",
            block=False,
        )
        lo = df["timestamp"].min() + 600.0
        hi = df["timestamp"].max() - 600.0
        # Drive the slider callbacks exactly as a user drag would.
        viewer["sliders"]["t1"].set_val(lo)  # type: ignore[index]
        viewer["sliders"]["t2"].set_val(hi)  # type: ignore[index]
        fig.canvas.draw()
        a, b = viewer["get_visible_range"]()  # type: ignore[misc]
        expected = df[(df.timestamp >= a) & (df.timestamp <= b)]
        n_pts = len(expected)
        for line in fig.axes[0].get_lines():
            if line.get_label() in ("temperature", "pressure"):
                assert np.asarray(line.get_xdata()).size == len(expected), line.get_label()
        fig.savefig(args.out, dpi=150)
        print(f"headless OK: visible range [{a:.1f}, {b:.1f}] -> {n_pts} points per series; PNG: {args.out}")
        return 0

    view_time_range(
        df,
        time_col="timestamp",
        value_cols=["temperature", "pressure"],
        title="Timestamp viewer demo — drag t1/t2 sliders, hover points for tooltips",
        timestamps_are_utc=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
