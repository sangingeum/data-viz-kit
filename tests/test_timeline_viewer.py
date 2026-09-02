"""Tests for non-interactive timeline_viewer logic (filtering, tooltip text)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_viz_kit.timeline_viewer import filter_time_range, format_tooltip


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [1725270000.0, 1725270000.5, 1725270300.0, 1725273600.5, 1725277200.0],
            "temperature": [20.1, 20.3, 21.0, 22.5, np.nan],
            "pressure": [1013.0, 1013.1, 1012.8, 1014.2, 1015.0],
        }
    )


class TestFilterTimeRange:
    def test_inclusive_bounds(self, sample_df: pd.DataFrame) -> None:
        out, t1, t2 = filter_time_range(sample_df, "timestamp", 1725270000.0, 1725270300.0)
        assert list(out["timestamp"]) == [1725270000.0, 1725270000.5, 1725270300.0]
        assert (t1, t2) == (1725270000.0, 1725270300.0)

    def test_none_defaults_to_full_range(self, sample_df: pd.DataFrame) -> None:
        out, t1, t2 = filter_time_range(sample_df, "timestamp", None, None)
        assert len(out) == len(sample_df)
        assert t1 == sample_df["timestamp"].min()
        assert t2 == sample_df["timestamp"].max()

    def test_t1_greater_than_t2_is_normalized(self, sample_df: pd.DataFrame) -> None:
        out, t1, t2 = filter_time_range(sample_df, "timestamp", 1725273600.5, 1725270000.0)
        assert (t1, t2) == (1725270000.0, 1725273600.5)
        assert len(out) == 4

    def test_empty_result_for_out_of_range_window(self, sample_df: pd.DataFrame) -> None:
        out, _, _ = filter_time_range(sample_df, "timestamp", 1.0e12, 1.0e12 + 1)
        assert out.empty

    def test_nan_timestamp_rows_dropped(self) -> None:
        df = pd.DataFrame({"timestamp": [100.0, np.nan, 200.0], "v": [1.0, 2.0, 3.0]})
        out, t1, t2 = filter_time_range(df, "timestamp", None, None)
        assert list(out["timestamp"]) == [100.0, 200.0]
        assert (t1, t2) == (100.0, 200.0)

    def test_missing_column_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            filter_time_range(sample_df, "nope", None, None)


class TestFormatTooltip:
    def test_raw_seconds(self) -> None:
        text = format_tooltip("temperature", 1725270000.5, 20.3)
        assert "temperature" in text
        assert "1725270000.5" in text
        assert "20.3" in text

    def test_utc_datetime(self) -> None:
        text = format_tooltip("pressure", 1725270000.0, 1013.0, timestamp_format="datetime")
        assert "pressure" in text
        assert "2024-09-02" in text
        assert "UTC" in text


class TestViewerHeadless:
    """Build the full viewer on Agg without calling plt.show()."""

    def test_viewer_build_and_slider_redraw(self, sample_df: pd.DataFrame) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from data_viz_kit.timeline_viewer import view_time_range

        fig, viewer = view_time_range(
            sample_df,
            time_col="timestamp",
            value_cols=["temperature", "pressure"],
            block=False,
        )
        assert viewer["get_visible_range"]() == (  # type: ignore[misc]
            float(sample_df["timestamp"].min()),
            float(sample_df["timestamp"].max()),
        )

        from matplotlib.widgets import Slider

        all_sliders = [viewer["sliders"]["t1"], viewer["sliders"]["t2"]]  # type: ignore[misc]
        assert all(isinstance(w, Slider) for w in all_sliders)
        for widget in all_sliders:
            if widget.label.get_text() == "t1":
                widget.set_val(1725270000.0)
            elif widget.label.get_text() == "t2":
                widget.set_val(1725270300.0)
        a, b = viewer["get_visible_range"]()  # type: ignore[misc]
        assert (a, b) == (1725270000.0, 1725270300.0)
        expected = sample_df[
            (sample_df.timestamp >= a) & (sample_df.timestamp <= b)
        ]
        for ax in fig.axes[:1]:
            for line in ax.get_lines():
                if line.get_label() in ("temperature", "pressure"):
                    assert np.asarray(line.get_xdata()).size == len(expected)

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_backend_selection(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from data_viz_kit.timeline_viewer import ensure_interactive_backend

        # On this headless host no GUI toolkit exists; function must not raise
        # and must report some backend string.
        backend = ensure_interactive_backend()
        assert isinstance(backend, str) and backend
