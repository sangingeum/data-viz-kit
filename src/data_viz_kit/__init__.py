"""data_viz_kit — data analysis & visualization toolkit."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless-safe by default
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from data_viz_kit.timeline_viewer import (
    ensure_interactive_backend,
    filter_time_range,
    format_tooltip,
    view_time_range,
)

from data_viz_kit.csv_viewer import (
    load_csv,
    view_csv,
)


@dataclass
class SummaryStats:
    """Summary statistics for a numeric series."""

    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float


def summarize(series: pd.Series, name: str = "series") -> SummaryStats:
    """Compute summary statistics for a pandas Series."""
    s = pd.Series(series).dropna()
    if s.empty:
        raise ValueError("cannot summarize an empty series")
    return SummaryStats(
        name=name,
        count=int(s.count()),
        mean=float(s.mean()),
        std=float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        min=float(s.min()),
        max=float(s.max()),
    )


def plot_scatter(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    title: str = "Scatter plot",
    out_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Draw a scatter plot; optionally save it to `out_path` and return the figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.asarray(x), np.asarray(y), alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
