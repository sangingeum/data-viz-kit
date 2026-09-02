from __future__ import annotations

import pandas as pd
import pytest

from data_viz_kit import plot_scatter, summarize


def test_summarize_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    stats = summarize(s, name="test")
    assert stats.count == 4
    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(1.2909944487358056)
    assert stats.min == 1.0
    assert stats.max == 4.0


def test_summarize_drops_nan() -> None:
    stats = summarize(pd.Series([1.0, float("nan"), 3.0]))
    assert stats.count == 2


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize(pd.Series([float("nan"), None], dtype="float64"))


def test_plot_scatter_writes_png(tmp_path) -> None:
    out = tmp_path / "plot.png"
    fig = plot_scatter([1, 2, 3], [2, 4, 6], out_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
