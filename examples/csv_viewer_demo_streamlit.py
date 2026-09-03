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

from llh_utils import (
    build_identifier_color_map,
    build_llh,
    detect_llh_layout,
    make_3d_space_figure,
    make_altitude_profile_figure,
    make_flat_map_figure,
    make_globe_figure,
)

PROJECTIONS: list[str] = ["orthographic", "equirectangular", "natural earth"]

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
    # Columns surfaced in the hover tooltip: all coordinates + timestamp + identifier.
    custom_cols = coords + [time_col, "_identifier"]
    if z_col:
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col, color="_identifier", symbol="_identifier",
            custom_data=custom_cols, hover_data=[], title=title,
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col, color="_identifier", symbol="_identifier",
            custom_data=custom_cols, hover_data=[], title=title,
        )
    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=4, opacity=0.75),
        mode="markers",
    )
    # Exact hover: identifier + time + all coordinates, via customdata columns.
    idx = {c: i for i, c in enumerate(custom_cols)}
    hover = f"<b>%{{customdata[{idx['_identifier']}]}}</b><br>" + "<br>".join(
        f"{c} = %{{customdata[{idx[c]}]}}" for c in coords + [time_col]
    )
    fig.update_traces(hovertemplate=hover + "<extra></extra>")
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

    # ---- coordinate mode ----------------------------------------------------
    coord_mode = st.sidebar.selectbox("Coordinate mode", ["NUE", "LLH"], index=0)

    llh_layout: str | None = None
    seconds_scale = 1.0
    llh_cols: dict[str, str] = {}
    if coord_mode == "LLH":
        detected_layout, defaults = detect_llh_layout(columns)
        layout_options = ["3-col (lat/lon/alt decimal deg)", "7-col (DMS: deg/min/sec)"]
        layout_choice = st.sidebar.radio(
            "LLH input layout", layout_options,
            index=layout_options.index(
                "7-col (DMS: deg/min/sec)"
                if detected_layout == "7col"
                else "3-col (lat/lon/alt decimal deg)"
            ),
        )
        llh_layout = "7col" if layout_choice.startswith("7-col") else "3col"
        st.sidebar.caption(
            f"Auto-detected layout: **{detected_layout}** — showing which layout is in use."
        )
        lat_or_none = defaults.get("lat") or defaults.get("lat_d")
        lon_or_none = defaults.get("lon") or defaults.get("lon_d")
        alt_or_none = defaults.get("alt")
        if llh_layout == "3col":
            if len(remaining_llh := [c for c in columns]) < 3:
                st.warning("Need at least three LLH columns.")
                st.stop()
            llh_cols = {
                "lat": st.sidebar.selectbox(
                    "Latitude column (decimal deg)",
                    columns,
                    index=columns.index(lat_or_none) if lat_or_none else 0,
                ),
                "lon": st.sidebar.selectbox(
                    "Longitude column (decimal deg)",
                    columns,
                    index=columns.index(lon_or_none) if lon_or_none else 0,
                ),
                "alt": st.sidebar.selectbox(
                    "Altitude column (m)",
                    columns,
                    index=columns.index(alt_or_none) if alt_or_none else 0,
                ),
            }
            seconds_scale = 1.0
            st.sidebar.selectbox(
                "seconds scale (7-col only)", [1, 0.01], index=0,
                disabled=True,
                help="Applies only in the 7-col (DMS) layout.",
            )
        else:
            def _pick(label: str, field: str) -> str:
                d = defaults.get(field)
                return st.sidebar.selectbox(
                    label, columns, index=columns.index(d) if d else 0,
                )

            llh_cols = {
                "lat_d": _pick("LatDeg column", "lat_d"),
                "lat_m": _pick("LatMin column (1/60 deg)", "lat_m"),
                "lat_s": _pick("LatSec column (1/60 min)", "lat_s"),
                "lon_d": _pick("LonDeg column", "lon_d"),
                "lon_m": _pick("LonMin column (1/60 deg)", "lon_m"),
                "lon_s": _pick("LonSec column (1/60 min)", "lon_s"),
                "alt": _pick("Altitude column (m)", "alt"),
            }
            seconds_scale = float(
                st.sidebar.selectbox(
                    "seconds scale", [1, 0.01], index=0,
                    help="Multiplies LatSec/LonSec before conversion — "
                         "use 0.01 when seconds are pre-multiplied by 100.",
                )
            )
        # ---- numeric validation ---------------------------------------------
        consumed = list(llh_cols.values())
        bad_counts: dict[str, int] = {}
        for c in consumed:
            coerced = cast(pd.Series, pd.to_numeric(df[c], errors="coerce"))
            bad_counts[c] = int((coerced.isna() & df[c].notna()).sum())
        total_bad = sum(bad_counts.values())
        if total_bad:
            st.sidebar.warning(
                "Non-numeric values coerced to NaN: "
                + ", ".join(f"{c}: {n}" for c, n in bad_counts.items() if n)
            )
        else:
            st.sidebar.success("All selected LLH columns are numeric.")

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
    if coord_mode == "LLH" and llh_layout is not None:
        filtered, non_numeric = build_llh(filtered, llh_layout, llh_cols, seconds_scale)
        st.caption(
            f"LLH mode: layout **{llh_layout}**, seconds scale **{seconds_scale}**"
            + (f", {non_numeric} non-numeric coordinate cells coerced to NaN" if non_numeric else "")
        )
    st.subheader(
        f"{len(filtered)} / {len(df)} rows in "
        f"[{min(t1, t2):.1f}, {max(t1, t2):.1f}] — {len(selected_ids)}/{len(identifiers)} identifiers"
    )

    if filtered.empty:
        st.warning("No rows match the selected time window and identifiers.")
    elif coord_mode == "LLH":
        # ---- LLH trajectory-viewer sidebar options --------------------------
        projection = st.sidebar.selectbox(
            "Globe projection", PROJECTIONS, index=0,
        )
        fit_bounds = st.sidebar.toggle(
            "Fit bounds to trajectories", value=True,
            help="Zoom the map to the filtered data's lat/lon bounding box "
                 "instead of showing the whole Earth.",
        )
        globe_df = filtered.dropna(subset=["lat_deg", "lon_deg"])
        ids = sorted(globe_df["_identifier"].unique())
        # one shared identifier->colour mapping across ALL views so each
        # identifier keeps the same colour everywhere
        color_map = build_identifier_color_map(ids)
        # ---- row 1: globe (left) + flat map (right) --------------------------
        col_globe, col_map = st.columns(2)
        with col_globe:
            st.plotly_chart(
                make_globe_figure(
                    globe_df, ids, time_col,
                    f"Trajectory globe ({projection})",
                    color_map=color_map,
                    projection=projection,
                    fit_bounds=fit_bounds,
                ),
                use_container_width=True,
            )
            st.caption(
                "Drag to rotate the globe; scroll (or use the mode bar) to "
                "zoom. Hover a point for identifier, timestamp, UTC time and "
                "exact lat/lon/alt. Markers only — no connecting lines."
            )
        with col_map:
            st.plotly_chart(
                make_flat_map_figure(
                    globe_df, ids, time_col,
                    "Top-down trajectory map (equirectangular)",
                    color_map=color_map,
                    fit_bounds=fit_bounds,
                ),
                use_container_width=True,
            )
            st.caption(
                "Classic top-down view of the same trajectories — pure "
                "point display, identifier-coloured."
            )
        # ---- row 2: 3D space (left) + altitude profile (right) ---------------
        col_3d, col_prof = st.columns(2)
        with col_3d:
            st.plotly_chart(
                make_3d_space_figure(
                    globe_df, ids, time_col,
                    "3D space view (lat/lon/alt)",
                    color_map=color_map,
                ),
                use_container_width=True,
            )
            st.caption(
                "True 3-D scatter: x = lon, y = lat, z = alt_m. Drag to "
                "rotate and inspect the vertical structure in space."
            )
        with col_prof:
            st.plotly_chart(
                make_altitude_profile_figure(
                    globe_df, ids, time_col,
                    "Altitude vs along-track distance",
                    color_map=color_map,
                ),
                use_container_width=True,
            )
            st.caption(
                "Vertical flight profile: altitude (m) against cumulative "
                "great-circle distance along each track (km) — climb, "
                "cruise, and descent per aircraft. x is distance, not time."
            )
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
