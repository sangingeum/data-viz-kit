"""Demo: Streamlit CSV coordinate viewer — full parity with csv_viewer_demo.py.

Streamlit version of ``csv_viewer_demo.py``.  Load the bundled
``examples/sample_data.csv`` (or upload a CSV) and get the same 2x2 plot grid
as the matplotlib viewer:

* 3-D scatter of the N/U/E coordinates
* 2-D scatters N–U, N–E, E–U

Charts are rendered with **plotly** (``st.plotly_chart``) so native hover
tooltips work — hovering a point shows the identifier, the exact timestamp,
and the exact N/U/E values.  Identifier selection (streamlit multiselect,
default: all selected) replaces the matplotlib checkboxes, and a ``[t1, t2]``
range slider windows the data.  Column names (timestamp, identifiers, N/U/E
coordinates) are configurable in the sidebar, mirroring the
``--time-col/--id-cols/--n-col/--u-col/--e-col`` CLI options of the
matplotlib demo.

Run with::

    uv run streamlit run examples/csv_viewer_demo_streamlit.py --server.headless true
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_viz_kit.csv_viewer import load_csv  # noqa: F401  (reuse-style reference)

SAMPLE_CSV: Path = Path(__file__).resolve().parent / "sample_data.csv"


def build_identifier(df: pd.DataFrame, id_cols: list[str]) -> pd.Series:
    """Join *id_cols* with ``'+'`` into an ``_identifier`` series.

    Mirrors ``data_viz_kit.csv_viewer.load_csv`` identifier building.
    """
    ident = cast(pd.Series, df[id_cols[0]].astype(str))
    for col in id_cols[1:]:
        ident = ident + "+" + df[col].astype(str)
    return ident


def filter_time_window(
    df: pd.DataFrame, time_col: str, t1: float, t2: float
) -> pd.DataFrame:
    """Return rows with *time_col* inside the inclusive window [t1, t2]."""
    times = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    lo, hi = min(t1, t2), max(t1, t2)
    mask = (times >= lo) & (times <= hi) & np.isfinite(times)
    return df.loc[mask]


def make_scatter(
    df: pd.DataFrame,
    identifiers: list[str],
    x_col: str,
    y_col: str,
    z_col: str | None,
    time_col: str,
    title: str,
):
    """Return a plotly scatter figure (2-D, or 3-D when *z_col* is given)."""
    coords = [x_col, y_col] + ([z_col] if z_col else [])
    if z_col:
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col, color="_identifier", symbol="_identifier",
            hover_data=coords + [time_col], title=title,
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col, color="_identifier", symbol="_identifier",
            hover_data=coords + [time_col], title=title,
        )
    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=4, opacity=0.75),
        mode="markers",
    )
    # Exact hover: identifier + time + all three coordinates.
    custom = coords + [time_col, "_identifier"]
    hover = "<b>%{customdata[-1]}</b><br>" + "<br>".join(
        f"{c} = %{{customdata[{i}]}}"
        for i, c in enumerate(coords + [time_col])
    )
    fig.update_traces(customdata=custom, hovertemplate=hover + "<extra></extra>")
    fig.update_layout(legend_title_text="Identifier", margin=dict(l=10, r=10, t=40, b=10))
    if identifiers and len(identifiers) > 10:
        pass  # plotly colours handle many series fine
    return fig


def main() -> None:
    st.set_page_config(page_title="CSV Viewer (Streamlit)", layout="wide")
    st.title("CSV Coordinate Viewer — Streamlit")
    st.caption(
        "Full-parity Streamlit port of `csv_viewer_demo.py`: 2x2 scatter grid "
        "(3D + N–U, N–E, E–U) with plotly hover tooltips (identifier + exact "
        "time + exact N/U/E), identifier multiselect, and a `[t1, t2]` range "
        "slider."
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
    columns = list(df.columns)

    # ---- column configuration (mirrors csv_viewer_demo.py CLI args) -------
    default_time = "Timestamp" if "Timestamp" in columns else columns[0]
    time_col = st.sidebar.selectbox(
        "Timestamp column (--time-col)", columns, index=columns.index(default_time)
    )

    id_col_default = [c for c in ("station", "sensor") if c in columns]
    if not id_col_default:
        id_col_default = [c for c in columns if c != time_col][:1]
    id_cols = st.sidebar.multiselect(
        "Identifier columns (--id-cols, joined with '+')",
        [c for c in columns if c != time_col],
        default=id_col_default,
    )
    if not id_cols:
        st.info("Select at least one identifier column.")
        st.stop()

    remaining = [c for c in columns if c not in id_cols and c != time_col]
    coord_defaults = [c for c in ("N", "U", "E") if c in remaining] or remaining[:3]
    coord_defaults = coord_defaults[:3] if len(coord_defaults) >= 3 else remaining[:3]
    if len(remaining) < 3:
        st.warning("Need at least three coordinate columns (N/U/E equivalents).")
        st.stop()
    n_col = st.sidebar.selectbox(
        "N coordinate column (--n-col)", remaining, index=remaining.index(coord_defaults[0])
    )
    u_col = st.sidebar.selectbox(
        "U coordinate column (--u-col)",
        remaining,
        index=remaining.index(coord_defaults[1]),
    )
    e_col = st.sidebar.selectbox(
        "E coordinate column (--e-col)",
        remaining,
        index=remaining.index(coord_defaults[2]),
    )
    coord_cols = [n_col, u_col, e_col]

    # ---- identifier column -------------------------------------------------
    df = df.copy()
    df["_identifier"] = build_identifier(df, id_cols)
    identifiers = sorted(df["_identifier"].unique())

    selected_ids = st.sidebar.multiselect(
        "Identifiers", identifiers, default=identifiers
    )

    # ---- time window -------------------------------------------------------
    times = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    finite_times = times[np.isfinite(times)]
    if finite_times.size == 0:
        st.warning(f"Column `{time_col}` has no numeric (finite) values.")
        st.stop()
    data_min, data_max = float(finite_times.min()), float(finite_times.max())
    t1, t2 = st.sidebar.slider(
        "Time range [t1, t2]",
        min_value=data_min,
        max_value=data_max,
        value=(data_min, data_max),
        format="%.1f",
    )

    # ---- filter ------------------------------------------------------------
    filtered = filter_time_window(df, time_col, t1, t2)
    filtered = filtered[filtered["_identifier"].isin(selected_ids)]
    st.subheader(
        f"{len(filtered)} / {len(df)} rows in "
        f"[{min(t1, t2):.1f}, {max(t1, t2):.1f}] — {len(selected_ids)}/{len(identifiers)} identifiers"
    )

    if filtered.empty:
        st.warning("No rows match the selected time window and identifiers.")
    else:
        # ---- 2x2 plot grid: 3D + N–U + N–E + E–U --------------------------
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                make_scatter(
                    filtered, identifiers, n_col, u_col, e_col, time_col,
                    f"3D {n_col}–{u_col}–{e_col}",
                ),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                make_scatter(
                    filtered, identifiers, n_col, u_col, None, time_col,
                    f"{n_col}–{u_col}",
                ),
                use_container_width=True,
            )
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(
                make_scatter(
                    filtered, identifiers, n_col, e_col, None, time_col,
                    f"{n_col}–{e_col}",
                ),
                use_container_width=True,
            )
        with col_d:
            st.plotly_chart(
                make_scatter(
                    filtered, identifiers, e_col, u_col, None, time_col,
                    f"{e_col}–{u_col}",
                ),
                use_container_width=True,
            )

    # ---- table of filtered rows --------------------------------------------
    st.subheader("Filtered rows")
    st.dataframe(filtered, use_container_width=True, height=320)


main()
