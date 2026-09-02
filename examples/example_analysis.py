"""Headless (Agg) demo: generates data, analyzes it, renders a PNG plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_viz_kit import plot_scatter, summarize


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = 2.0 * x + rng.normal(scale=0.5, size=200)

    stats = summarize(__import__("pandas").Series(y), name="y")
    print(f"stats: {stats}")

    out = Path(__file__).resolve().parent / "demo_plot.png"
    plot_scatter(x, y, title="data-viz-kit demo (Agg)", out_path=str(out))
    print(f"PNG written: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
