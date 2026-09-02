"""Demo: Streamlit CSV viewer.

Streamlit version of ``csv_viewer_demo.py``.  Upload a CSV (or use the bundled
``examples/sample_data.csv``), pick the timestamp and value columns, and drag
an interactive ``[t1, t2]`` range slider to window the data.  The filtered
series is rendered with ``st.line_chart`` and the filtered rows with
``st.dataframe``.

Run with::

    uv run streamlit run examples/csv_viewer_demo_streamlit.py --server.headless true
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SAMPLE_CSV: Path = Path(__file__).resolve().parent / "sample_data.csv"


def filter_time_window(
    df: pd.DataFrame, time_col: str, t1: float, t2: float
) -> pd.DataFrame:
    """Return rows with *time_col* inside the inclusive window [t1, t2]."""
    times = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    lo, hi = min(t1, t2), max(t1, t2)
    mask = (times >= lo) & (times <= hi) & np.isfinite(times)
    return df.loc[mask]


def main() -> None:
    st.set_page_config(page_title="CSV Viewer (Streamlit)", layout="wide")
    st.title("CSV Coordinate Viewer — Streamlit")
    st.caption(
        "Streamlit port of `csv_viewer_demo.py`: upload a CSV, select the "
        "timestamp and value columns, and window the data with a time-range slider."
    )

    # ---- data source ------------------------------------------------------
    source = st.sidebar.radio(
        "Data source",
        ["Sample data (examples/sample_data.csv)", "Upload CSV"],
    )
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV or switch to the sample data option.")
            st.stop()
        df = pd.read_csv(uploaded)
        data_name = uploaded.name
    else:
        df = pd.read_csv(SAMPLE_CSV)
        data_name = SAMPLE_CSV.name

    st.sidebar.caption(f"Loaded `{data_name}`: {len(df)} rows, {len(df.columns)} columns")
    if df.empty:
        st.warning("The CSV contains no data rows.")
        st.stop()

    # ---- column selection (timestamp + value) ------------------------------
    columns = list(df.columns)
    default_time = "Timestamp" if "Timestamp" in columns else columns[0]
    time_col = st.sidebar.selectbox(
        "Timestamp column", columns, index=columns.index(default_time)
    )

    times = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    finite_times = times[np.isfinite(times)]
    if finite_times.size == 0:
        st.warning(f"Column `{time_col}` has no numeric (finite) values.")
        st.stop()
    data_min, data_max = float(finite_times.min()), float(finite_times.max())

    numeric_cols = [
        c for c in columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    default_values = [c for c in ("N", "U", "E") if c in numeric_cols] or numeric_cols[:1]
    value_cols = st.sidebar.multiselect(
        "Value column(s)", numeric_cols, default=default_values
    )
    if not value_cols:
        st.info("Select at least one value column to plot.")
        st.stop()

    # ---- t1/t2 time-range windowing (range slider) -------------------------
    t1, t2 = st.sidebar.slider(
        "Time range [t1, t2]",
        min_value=data_min,
        max_value=data_max,
        value=(data_min, data_max),
        format="%.1f",
    )

    # ---- filter ------------------------------------------------------------
    filtered = filter_time_window(df, time_col, t1, t2)
    st.subheader(
        f"{len(filtered)} / {len(df)} rows in [{min(t1, t2):.1f}, {max(t1, t2):.1f}]"
    )

    # ---- line chart of filtered series -------------------------------------
    if filtered.empty:
        st.warning("No rows fall inside the selected time window.")
    else:
        chart_df = filtered.set_index(time_col)[value_cols].sort_index()
        st.line_chart(chart_df)

    # ---- table of filtered rows --------------------------------------------
    st.subheader("Filtered rows")
    st.dataframe(filtered, use_container_width=True, height=320)


main()
