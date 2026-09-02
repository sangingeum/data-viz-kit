"""Tests for csv_viewer: loading, identifier generation, headless viewer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_viz_kit.csv_viewer import load_csv, view_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
station,sensor,Timestamp,N,U,E
A,1,100.0,10.0,50.0,30.0
A,1,200.5,10.5,50.2,30.1
A,2,100.0,11.0,51.0,31.0
A,2,200.5,11.5,51.3,31.2
B,1,100.0,20.0,60.0,40.0
B,1,200.5,20.3,60.1,40.2
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    return p


@pytest.fixture
def sample_df(csv_path):
    return load_csv(csv_path, id_cols=["station", "sensor"])


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_identifier_column_created(self, sample_df: pd.DataFrame) -> None:
        assert "_identifier" in sample_df.columns
        assert sorted(sample_df["_identifier"].unique()) == ["A+1", "A+2", "B+1"]

    def test_all_columns_present(self, sample_df: pd.DataFrame) -> None:
        for col in ("station", "sensor", "Timestamp", "N", "U", "E", "_identifier"):
            assert col in sample_df.columns

    def test_row_count(self, sample_df: pd.DataFrame) -> None:
        assert len(sample_df) == 6

    def test_missing_id_col_raises(self, csv_path) -> None:
        with pytest.raises(KeyError, match="nope"):
            load_csv(csv_path, id_cols=["station", "nope"])

    def test_missing_coord_col_raises(self, csv_path) -> None:
        with pytest.raises(KeyError, match="Z"):
            load_csv(csv_path, id_cols=["station", "sensor"], coord_cols=["N", "U", "Z"])

    def test_missing_time_col_raises(self, csv_path) -> None:
        with pytest.raises(KeyError, match="Time"):
            load_csv(csv_path, id_cols=["station", "sensor"], time_col="Time")

    def test_single_id_col(self, csv_path) -> None:
        df = load_csv(csv_path, id_cols=["station"])
        assert sorted(df["_identifier"].unique()) == ["A", "B"]


# ---------------------------------------------------------------------------
# view_csv  (headless / Agg)
# ---------------------------------------------------------------------------


class TestViewCsvHeadless:
    """Build the full viewer on Agg without calling plt.show()."""

    @pytest.fixture(autouse=True)
    def _force_agg(self, monkeypatch) -> None:
        """Keep Agg backend; prevent ensure_interactive_backend from
        switching to a GUI backend that may not be available."""
        import matplotlib

        matplotlib.use("Agg")
        monkeypatch.setattr(
            "data_viz_kit.timeline_viewer.ensure_interactive_backend",
            lambda: matplotlib.get_backend(),
        )

    def test_viewer_builds_and_returns(self, sample_df: pd.DataFrame) -> None:
        fig, viewer = view_csv(
            sample_df,
            id_cols=["station", "sensor"],
            block=False,
        )
        assert fig is not None
        assert callable(viewer["get_visible_range"])
        assert callable(viewer["get_visible_identifiers"])

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visible_range_defaults(self, sample_df: pd.DataFrame) -> None:
        fig, viewer = view_csv(
            sample_df,
            id_cols=["station", "sensor"],
            block=False,
        )
        lo, hi = viewer["get_visible_range"]()  # type: ignore[misc]
        assert lo == 100.0
        assert hi == 200.5

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_slider_updates_range(self, sample_df: pd.DataFrame) -> None:
        fig, viewer = view_csv(
            sample_df,
            id_cols=["station", "sensor"],
            block=False,
        )
        viewer["sliders"]["t1"].set_val(150.0)  # type: ignore[index]
        lo, hi = viewer["get_visible_range"]()  # type: ignore[misc]
        assert lo == 150.0
        assert hi == 200.5

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_checkbox_toggles_identifiers(self, sample_df: pd.DataFrame) -> None:
        fig, viewer = view_csv(
            sample_df,
            id_cols=["station", "sensor"],
            block=False,
        )
        assert "A+1" in viewer["get_visible_identifiers"]()  # type: ignore[misc]

        # Simulate clicking the "A+1" checkbox (toggles it off).
        viewer["check"].set_active(0)  # type: ignore[union-attr]
        assert "A+1" not in viewer["get_visible_identifiers"]()  # type: ignore[misc]

        # Toggle it back on.
        viewer["check"].set_active(0)  # type: ignore[union-attr]
        assert "A+1" in viewer["get_visible_identifiers"]()  # type: ignore[misc]

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_four_plots_exist(self, sample_df: pd.DataFrame) -> None:
        fig, viewer = view_csv(
            sample_df,
            id_cols=["station", "sensor"],
            block=False,
        )
        # 4 plot axes (1 3D + 3 2D) + 1 checkbox axes + 2 slider axes = 7
        assert len(fig.axes) == 7

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_lines_per_plot(self, sample_df: pd.DataFrame) -> None:
        fig, viewer = view_csv(
            sample_df,
            id_cols=["station", "sensor"],
            block=False,
        )
        # Each of the 4 plot axes (3D + 3 2D) should have 3 lines (3 identifiers)
        for ax in fig.axes[:4]:
            data_lines = [l for l in ax.get_lines() if l.get_label() in ("A+1", "A+2", "B+1")]
            assert len(data_lines) == 3

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_auto_identifier_creation(self) -> None:
        """view_csv should create _identifier if not already present."""
        df = pd.DataFrame(
            {
                "station": ["X", "X"],
                "sensor": ["a", "b"],
                "Timestamp": [0.0, 1.0],
                "N": [1.0, 2.0],
                "U": [3.0, 4.0],
                "E": [5.0, 6.0],
            }
        )
        fig, viewer = view_csv(df, id_cols=["station", "sensor"], block=False)
        assert sorted(viewer["get_visible_identifiers"]()) == ["X+a", "X+b"]  # type: ignore[misc]

        import matplotlib.pyplot as plt

        plt.close(fig)
