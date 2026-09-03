"""LLH (LLA) helpers for the streamlit CSV viewer.

Kept free of streamlit imports so pytest can exercise the conversion,
auto-detection, and figure builders headlessly.

Conversions
-----------
* DMS -> decimal degrees: ``deg = D + M/60 + S * seconds_scale / 3600``
  (``seconds_scale`` defaults to 1; owners' data sometimes stores seconds
  pre-multiplied by 100, in which case pass 0.01).
* Identifier building and time handling live in the streamlit script and are
  reused verbatim (multi-col ``'+'`` join, time-range slider, hover details).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.express as px

COORD_MODES: list[str] = ["NUE", "LLH"]

#: candidate column-name fragments, lower-cased, matched with ``str.contains``
_LAT_PAT = ("latitude", "lat")
_LON_PAT = ("longitude", "lon", "lng")
_ALT_PAT = ("altitude", "alt")
_LAT_DEG_PAT = ("lat_deg", "latd")
_LAT_MIN_PAT = ("lat_min", "latm")
_LAT_SEC_PAT = ("lat_sec", "lats")
_LON_DEG_PAT = ("lon_deg", "lond")
_LON_MIN_PAT = ("lon_min", "lonm")
_LON_SEC_PAT = ("lon_sec", "lons")


def dms_to_decimal_degrees(
    deg: pd.Series,
    minutes: pd.Series,
    seconds: pd.Series,
    seconds_scale: float = 1.0,
) -> pd.Series:
    """Convert deg/min/sec columns to decimal degrees.

    ``deg = D + M / 60 + (S * seconds_scale) / 3600`` where *seconds_scale*
    is applied to the seconds component *before* conversion (use 0.01 when
    the source seconds are pre-multiplied by 100).

    Args:
        deg: degrees component (any numeric).
        minutes: minutes component, 1/60 of a degree.
        seconds: seconds component, 1/60 of a minute.
        seconds_scale: multiplier applied to *seconds* before conversion.

    Returns:
        Decimal degrees as a float series.

    Raises:
        ValueError: if *seconds_scale* is not positive.
    """
    if seconds_scale <= 0:
        raise ValueError(f"seconds_scale must be positive, got {seconds_scale}")
    d = cast(pd.Series, pd.to_numeric(deg, errors="coerce")).astype(float)
    m = cast(pd.Series, pd.to_numeric(minutes, errors="coerce")).astype(float)
    s = cast(pd.Series, pd.to_numeric(seconds, errors="coerce")).astype(float) * seconds_scale
    return d + m / 60.0 + s / 3600.0


def coerce_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    """Coerce *series* to float; return the coerced series and non-numeric count.

    Non-numeric (and missing) entries become NaN; the count excludes rows that
    were already NaN in the source.
    """
    raw = series
    coerced = cast(pd.Series, pd.to_numeric(raw, errors="coerce")).astype(float)
    newly_bad = int((coerced.isna() & raw.notna()).sum())
    return coerced, newly_bad


def _find_col(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    lowered = {c: c.lower() for c in columns}
    # try all fragments in order so 'latitude' wins over the broader 'lat'
    for pat in patterns:
        for col, low in lowered.items():
            if pat in low:
                return col
    return None


def detect_llh_layout(
    columns: list[str],
) -> tuple[str, dict[str, str | None]]:
    """Auto-detect an LLH layout from *columns*.

    Returns:
        ``(layout, defaults)`` where *layout* is ``"3col"`` or ``"7col"``
        and *defaults* maps field names to column names (or ``None``):

        * 3-col: ``lat``, ``lon``, ``alt``
        * 7-col: ``lat_d``, ``lat_m``, ``lat_s``, ``lon_d``, ``lon_m``,
          ``lon_s``, ``alt``
    """
    three = {
        "lat": _find_col(columns, _LAT_PAT),
        "lon": _find_col(columns, _LON_PAT),
        "alt": _find_col(columns, _ALT_PAT),
    }
    seven = {
        "lat_d": _find_col(columns, _LAT_DEG_PAT),
        "lat_m": _find_col(columns, _LAT_MIN_PAT),
        "lat_s": _find_col(columns, _LAT_SEC_PAT),
        "lon_d": _find_col(columns, _LON_DEG_PAT),
        "lon_m": _find_col(columns, _LON_MIN_PAT),
        "lon_s": _find_col(columns, _LON_SEC_PAT),
        "alt": _find_col(columns, _ALT_PAT),
    }
    if all(v is not None for v in seven.values()):
        return "7col", seven
    return "3col", three


def build_llh(
    df: pd.DataFrame,
    layout: str,
    cols: dict[str, str],
    seconds_scale: float = 1.0,
) -> tuple[pd.DataFrame, int]:
    """Return a copy of *df* with ``lat_deg``/``lon_deg``/``alt_m`` added.

    ``lat_deg``/``lon_deg`` are decimal degrees, ``alt_m`` is metres.
    In 7-col mode *seconds_scale* multiplies the seconds fields before
    conversion; it is ignored in 3-col mode.

    Returns:
        ``(out_df, non_numeric_count)`` — the count of source cells that
        failed numeric coercion (across every consumed coordinate column).
    """
    out = df.copy()
    bad = 0
    if layout == "3col":
        lat, n1 = coerce_numeric(out[cols["lat"]])
        lon, n2 = coerce_numeric(out[cols["lon"]])
        alt, n3 = coerce_numeric(out[cols["alt"]])
        bad = n1 + n2 + n3
        out["lat_deg"], out["lon_deg"], out["alt_m"] = lat, lon, alt
    else:
        lat = dms_to_decimal_degrees(
            out[cols["lat_d"]], out[cols["lat_m"]], out[cols["lat_s"]], seconds_scale
        )
        lon = dms_to_decimal_degrees(
            out[cols["lon_d"]], out[cols["lon_m"]], out[cols["lon_s"]], seconds_scale
        )
        alt, n3 = coerce_numeric(out[cols["alt"]])
        bad = n3
        out["lat_deg"], out["lon_deg"], out["alt_m"] = lat, lon, alt
    return out, bad


def make_globe_figure(
    df: pd.DataFrame,
    identifiers: list[str],
    time_col: str,
    title: str,
    color_by_altitude: bool = True,
):
    """Interactive orthographic-globe scatter of the converted LLH data.

    Hover keeps identifier + timestamp + exact lat/lon/alt (plus a UTC
    datetime string when ``_utc_time`` is present), via **per-point**
    custom_data columns — never a list of names (the 7df7e70 bug).
    """
    df = df.copy()
    if "_utc_time" not in df.columns:
        ts = cast(pd.Series, pd.to_numeric(df[time_col], errors="coerce"))
        df["_utc_time"] = cast(
            pd.Series, pd.to_datetime(ts, unit="s", utc=True)
        ).astype(str)

    color = "alt_m" if color_by_altitude else "_identifier"
    custom_cols = ["lat_deg", "lon_deg", "alt_m", time_col, "_utc_time", "_identifier"]
    fig = px.scatter_geo(
        df,
        lat="lat_deg",
        lon="lon_deg",
        color=color,
        custom_data=custom_cols,
        hover_data=[],
        title=title,
        projection="orthographic",
    )
    idx = {c: i for i, c in enumerate(custom_cols)}
    hover = (
        f"<b>%{{customdata[{idx['_identifier']}]}}</b><br>"
        f"lat_deg = %{{customdata[{idx['lat_deg']}]}}<br>"
        f"lon_deg = %{{customdata[{idx['lon_deg']}]}}<br>"
        f"alt_m = %{{customdata[{idx['alt_m']}]}}<br>"
        f"{time_col} = %{{customdata[{idx[time_col]}]}}<br>"
        f"UTC = %{{customdata[{idx['_utc_time']}]}}"
    )
    fig.update_traces(
        selector=dict(type="scattergeo"),
        marker=dict(size=5, opacity=0.85),
        hovertemplate=hover + "<extra></extra>",
    )
    if not color_by_altitude:
        fig.update_traces(selector=dict(type="scattergeo"), symbol="_identifier")
    fig.update_layout(
        geo=dict(
            projection_type="orthographic",
            showcountries=True,
            showcoastlines=True,
            showland=True,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_time_scatter(
    df: pd.DataFrame,
    identifiers: list[str],
    value_col: str,
    time_col: str,
    title: str,
):
    """2-D parity scatter of *value_col* (lat/lon/alt) vs time."""
    custom_cols = [time_col, value_col, "_identifier"]
    fig = px.scatter(
        df, x=time_col, y=value_col, color="_identifier", symbol="_identifier",
        custom_data=custom_cols, hover_data=[], title=title,
    )
    idx = {c: i for i, c in enumerate(custom_cols)}
    hover = (
        f"<b>%{{customdata[{idx['_identifier']}]}}</b><br>"
        f"{time_col} = %{{customdata[{idx[time_col]}]}}<br>"
        f"{value_col} = %{{customdata[{idx[value_col]}]}}"
    )
    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=4, opacity=0.75),
        mode="markers",
        hovertemplate=hover + "<extra></extra>",
    )
    fig.update_layout(legend_title_text="Identifier", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def sample_llh_3col() -> pd.DataFrame:
    """Sample 3-col LLH data (lat/lon/alt decimal degrees) around Seoul."""
    base = pd.read_csv(SAMPLE_CSV_PATH)
    rng = np.random.default_rng(42)
    stations = {"StationA": (0.0, 0.0), "StationB": (0.01, 0.01)}
    lats, lons, alts = [], [], []
    for _, row in base.iterrows():
        dlat, dlon = stations[str(row["station"])]
        lats.append(37.5 + dlat + rng.normal(0, 2e-4))
        lons.append(127.0 + dlon + rng.normal(0, 2e-4))
        alts.append(80.0 + rng.normal(0, 1.5))
    base["Latitude"] = np.round(lats, 6)
    base["Longitude"] = np.round(lons, 6)
    base["Altitude_m"] = np.round(alts, 3)
    return base


def sample_llh_7col() -> pd.DataFrame:
    """Sample 7-col LLH data (DMS columns) derived from the 3-col sample.

    DMS components are computed by *decomposing* the decimal degrees, so the
    round-trip through :func:`dms_to_decimal_degrees` (seconds_scale=1)
    reproduces the same position.  ``LatSec``/``LonSec`` are stored
    pre-multiplied by 100 (matching the owner's data quirk) so the demo also
    exercises the 0.01 seconds scale.
    """
    base = sample_llh_3col()
    for prefix, lat in (("Lat", base["Latitude"]), ("Lon", base["Longitude"])):
        sign = np.sign(lat)
        remain = np.abs(lat) * 3600.0
        d = np.floor(remain / 3600.0)
        m = np.floor((remain - d * 3600.0) / 60.0)
        s = remain - d * 3600.0 - m * 60.0
        # store seconds pre-multiplied by 100 (owner's quirk), keep 4 decimals
        base[f"{prefix}Deg"] = (sign * d).astype(int)
        base[f"{prefix}Min"] = m.astype(int)
        base[f"{prefix}Sec"] = np.round(s * 100.0, 4)
    return base


SAMPLE_CSV_PATH = Path(__file__).resolve().parent / "sample_data.csv"
