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
import plotly.graph_objects as go

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


def _utc_time_series(df: pd.DataFrame, time_col: str) -> pd.Series:
    """UTC datetime strings derived from *time_col* (numeric epoch seconds)."""
    ts = cast(pd.Series, pd.to_numeric(df[time_col], errors="coerce"))
    return cast(pd.Series, pd.to_datetime(ts, unit="s", utc=True)).astype(str)


#: per-point customdata columns shared by every trajectory figure
_TRAJECTORY_CUSTOM_COLS = ["lat_deg", "lon_deg", "alt_m"]


def _trajectory_hover_template(time_col: str, custom_cols: list[str]) -> str:
    """Hover template showing identifier, timestamp, UTC time, exact lat/lon/alt."""
    idx = {c: i for i, c in enumerate(custom_cols)}
    return (
        f"<b>%{{customdata[{idx['_identifier']}]}}</b><br>"
        f"lat_deg = %{{customdata[{idx['lat_deg']}]}}<br>"
        f"lon_deg = %{{customdata[{idx['lon_deg']}]}}<br>"
        f"alt_m = %{{customdata[{idx['alt_m']}]}}<br>"
        f"{time_col} = %{{customdata[{idx[time_col]}]}}<br>"
        f"UTC = %{{customdata[{idx['_utc_time']}]}}<extra></extra>"
    )


def _fit_geo_layout(
    fig: "go.Figure",
    df: pd.DataFrame,
    projection: str,
    fit_bounds: bool,
) -> None:
    """Zoom the geo view to the filtered data's lat/lon bounding box.

    Orthographic projection supports ``center`` + ``projection.scale``; the
    flat projections (equirectangular / natural earth) support lat/lon axis
    ranges.  Padding keeps the outermost points off the plot edge.
    """
    geo: dict[str, object] = {
        "projection_type": projection,
        "showcountries": True,
        "showcoastlines": True,
        "showland": True,
    }
    if fit_bounds and not df.empty:
        lats = df["lat_deg"].dropna().astype(float)
        lons = df["lon_deg"].dropna().astype(float)
        if not lats.empty and not lons.empty:
            lat_pad = max((lats.max() - lats.min()) * 0.15, 0.5)
            lon_pad = max((lons.max() - lons.min()) * 0.15, 0.5)
            lat_lo, lat_hi = lats.min() - lat_pad, lats.max() + lat_pad
            lon_lo, lon_hi = lons.min() - lon_pad, lons.max() + lon_pad
            center = {"lon": float((lon_lo + lon_hi) / 2.0),
                      "lat": float((lat_lo + lat_hi) / 2.0)}
            if projection == "orthographic":
                span = max(lat_hi - lat_lo, lon_hi - lon_lo, 1e-6)
                scale = float(min(12.0, max(1.0, 170.0 / span)))
                geo["center"] = center
                geo["projection"] = {"type": projection, "scale": scale}
            else:
                geo["center"] = center
                geo["lataxis"] = {"range": [float(lat_lo), float(lat_hi)]}
                geo["lonaxis"] = {"range": [float(lon_lo), float(lon_hi)]}
    fig.update_layout(geo=geo)


def make_trajectory_figure(
    df: pd.DataFrame,
    identifiers: list[str],
    time_col: str,
    title: str,
    color_by_altitude: bool = True,
    projection: str = "orthographic",
    fit_bounds: bool = True,
    show_colorbar: bool = True,
):
    """Trajectory viewer figure: one markers-only scattergeo per identifier.

    Pure point display — no connecting lines.  Points are **sorted by
    timestamp** per identifier (matters for hover/tabular consistency).
    When *color_by_altitude* is on, markers are coloured by ``alt_m``
    (Viridis, shared cmin/cmax across identifiers + colorbar); when off,
    markers take their identifier's palette colour.

    Hover shows identifier, timestamp, UTC time, and exact lat/lon/alt via
    **per-point** customdata columns — never a list of names (7df7e70 bug).

    With ``projection="equirectangular"`` and ``show_colorbar=False`` this
    doubles as the flat top-down map panel.
    """
    df = df.copy()
    df["_utc_time"] = _utc_time_series(df, time_col)
    custom_cols = _TRAJECTORY_CUSTOM_COLS + [time_col, "_utc_time", "_identifier"]
    hover = _trajectory_hover_template(time_col, custom_cols)
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    alt_min = float(df["alt_m"].min()) if not df.empty else 0.0
    alt_max = float(df["alt_m"].max()) if not df.empty else 1.0
    if alt_max <= alt_min:
        alt_max = alt_min + 1.0

    for k, ident in enumerate(sorted(identifiers)):
        sub = df[df["_identifier"] == ident].copy()
        sub = sub.dropna(subset=["lat_deg", "lon_deg"])
        if sub.empty:
            continue
        sub = sub.sort_values(time_col, kind="stable")
        color = palette[k % len(palette)]
        cd = sub[custom_cols].to_numpy()
        marker: dict[str, object] = dict(size=4, opacity=0.9)
        if color_by_altitude:
            marker["color"] = sub["alt_m"]
            marker["colorscale"] = "Viridis"
            marker["cmin"] = alt_min
            marker["cmax"] = alt_max
            marker["showscale"] = bool(show_colorbar and k == 0)
            marker["colorbar"] = (
                dict(title="alt_m") if (show_colorbar and k == 0) else None
            )
        else:
            marker["color"] = color
        fig.add_trace(go.Scattergeo(
            lat=sub["lat_deg"], lon=sub["lon_deg"], mode="markers",
            name=ident, legendgroup=ident, showlegend=True,
            marker=marker,
            customdata=cd, hovertemplate=hover,
        ))

    fig.update_layout(
        title=title,
        legend_title_text="Identifier",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    _fit_geo_layout(fig, df, projection, fit_bounds)
    return fig


def make_globe_figure(
    df: pd.DataFrame,
    identifiers: list[str],
    time_col: str,
    title: str,
    color_by_altitude: bool = True,
    projection: str = "orthographic",
    fit_bounds: bool = True,
):
    """Backwards-compatible alias for :func:`make_trajectory_figure` (globe panel)."""
    return make_trajectory_figure(
        df, identifiers, time_col, title,
        color_by_altitude=color_by_altitude,
        projection=projection, fit_bounds=fit_bounds,
    )


def make_flat_map_figure(
    df: pd.DataFrame,
    identifiers: list[str],
    time_col: str,
    title: str,
    color_by_altitude: bool = True,
    fit_bounds: bool = True,
):
    """Flat top-down trajectory map (equirectangular) — second panel."""
    return make_trajectory_figure(
        df, identifiers, time_col, title,
        color_by_altitude=color_by_altitude,
        projection="equirectangular", fit_bounds=fit_bounds,
        show_colorbar=False,
    )


def make_3d_space_figure(
    df: pd.DataFrame,
    identifiers: list[str],
    time_col: str,
    title: str,
    color_by_altitude: bool = True,
):
    """True 3-D lat/lon/alt view (``go.Scatter3d``, markers only).

    Shows the vertical structure of the trajectories in space: x = lon,
    y = lat, z = alt_m.  Colouring follows the same toggle as the maps:
    altitude colour-scale (shared cmin/cmax + colorbar) or per-identifier
    palette colour.  Full hover details on every point.
    """
    df = df.copy()
    df["_utc_time"] = _utc_time_series(df, time_col)
    custom_cols = _TRAJECTORY_CUSTOM_COLS + [time_col, "_utc_time", "_identifier"]
    hover = _trajectory_hover_template(time_col, custom_cols)
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    alt_min = float(df["alt_m"].min()) if not df.empty else 0.0
    alt_max = float(df["alt_m"].max()) if not df.empty else 1.0
    if alt_max <= alt_min:
        alt_max = alt_min + 1.0
    for k, ident in enumerate(sorted(identifiers)):
        sub = df[df["_identifier"] == ident].copy()
        sub = sub.dropna(subset=["lat_deg", "lon_deg", "alt_m"])
        if sub.empty:
            continue
        sub = sub.sort_values(time_col, kind="stable")
        cd = sub[custom_cols].to_numpy()
        marker: dict[str, object] = dict(size=3, opacity=0.85)
        if color_by_altitude:
            marker["color"] = sub["alt_m"]
            marker["colorscale"] = "Viridis"
            marker["cmin"] = alt_min
            marker["cmax"] = alt_max
            marker["showscale"] = bool(k == 0)
            marker["colorbar"] = dict(title="alt_m") if k == 0 else None
        else:
            marker["color"] = palette[k % len(palette)]
        fig.add_trace(go.Scatter3d(
            x=sub["lon_deg"], y=sub["lat_deg"], z=sub["alt_m"],
            mode="markers", name=ident, legendgroup=ident, showlegend=True,
            marker=marker, customdata=cd, hovertemplate=hover,
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="lon_deg", yaxis_title="lat_deg", zaxis_title="alt_m",
        ),
        legend_title_text="Identifier",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Great-circle distance (km) between consecutive point pairs."""
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return pd.Series(6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))),
                     index=lat1.index)


def along_track_distance_km(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Cumulative great-circle ground distance (km) along each identifier's track.

    Rows are sorted per ``_identifier`` by *time_col*; the cumulative
    haversine distance to the identifier's first point is returned (0 for
    the first point of each identifier).  Requires ``lat_deg``/``lon_deg``.
    """
    df = df.copy()
    df["_utc_time"] = _utc_time_series(df, time_col)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, sub in df.sort_values(["_identifier", time_col], kind="stable").groupby(
        "_identifier", sort=False
    ):
        sub = sub.dropna(subset=["lat_deg", "lon_deg"])
        if len(sub) < 2:
            if not sub.empty:
                out.loc[sub.index] = 0.0
            continue
        step = _haversine_km(
            sub["lat_deg"].shift(), sub["lon_deg"].shift(),
            sub["lat_deg"], sub["lon_deg"],
        ).fillna(0.0)
        out.loc[sub.index] = step.cumsum()
    return out


def make_altitude_profile_figure(
    df: pd.DataFrame,
    identifiers: list[str],
    time_col: str,
    title: str,
):
    """Altitude vs along-track distance (km) — vertical flight profile.

    x is cumulative great-circle distance along each identifier's track
    (not time, sidestepping the owner's no-time-x-axis rule).  One
    markers-only scatter per identifier; full hover details.
    """
    df = df.copy()
    df["_utc_time"] = _utc_time_series(df, time_col)
    df["_along_km"] = along_track_distance_km(df, time_col)
    custom_cols = ["_along_km", "alt_m"] + _TRAJECTORY_CUSTOM_COLS + [
        time_col, "_utc_time", "_identifier"
    ]
    idx = {c: i for i, c in enumerate(custom_cols)}
    hover = (
        f"<b>%{{customdata[{idx['_identifier']}]}}</b><br>"
        f"along_track_km = %{{customdata[{idx['_along_km']}]}}<br>"
        f"alt_m = %{{customdata[{idx['alt_m']}]}}<br>"
        f"lat_deg = %{{customdata[{idx['lat_deg']}]}}<br>"
        f"lon_deg = %{{customdata[{idx['lon_deg']}]}}<br>"
        f"{time_col} = %{{customdata[{idx[time_col]}]}}<br>"
        f"UTC = %{{customdata[{idx['_utc_time']}]}}<extra></extra>"
    )
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for k, ident in enumerate(sorted(identifiers)):
        sub = df[df["_identifier"] == ident].copy()
        sub = sub.dropna(subset=["lat_deg", "lon_deg", "alt_m", "_along_km"])
        if sub.empty:
            continue
        sub = sub.sort_values(time_col, kind="stable")
        fig.add_trace(go.Scatter(
            x=sub["_along_km"], y=sub["alt_m"], mode="markers",
            name=ident, legendgroup=ident, showlegend=True,
            marker=dict(size=4, opacity=0.85, color=palette[k % len(palette)]),
            customdata=sub[custom_cols].to_numpy(), hovertemplate=hover,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Along-track distance (km)",
        yaxis_title="Altitude (m)",
        legend_title_text="Identifier",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


#: flight legs: (callsign, aircraft id, start (lat, lon), waypoints, cruise alt m)
#: Routes (realistic Korean city pairs):
#:   KAL7701+HL8201: Incheon -> over Seoul -> Busan
#:   JJA102 +HL8052: Seoul -> Jeju
#:   KAL9903+HL8275: Daegu -> Gwangju
_FLIGHT_ROUTES: list[dict[str, object]] = [
    {
        "callsign": "KAL7701", "aircraft": "HL8201",
        "start": (37.46, 126.44),          # Incheon
        "via": (37.52, 127.02),            # over Seoul (Gimpo)
        "end": (35.18, 128.99),            # Busan (Gimhae)
        "cruise": 10500.0,
    },
    {
        "callsign": "JJA102", "aircraft": "HL8052",
        "start": (37.52, 127.02),          # Seoul
        "via": None,
        "end": (33.51, 126.53),            # Jeju
        "cruise": 9800.0,
    },
    {
        "callsign": "KAL9903", "aircraft": "HL8275",
        "start": (35.87, 128.60),          # Daegu
        "via": None,
        "end": (35.17, 126.89),            # Gwangju
        "cruise": 9000.0,
    },
]

#: sample epoch: 2025-07-01 00:00:00 UTC; one hour of flight
_SAMPLE_EPOCH = 1751328000.0
_SAMPLE_DURATION_S = 3600.0


def _slerp_interp(
    lat1: float, lon1: float, lat2: float, lon2: float, frac: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle interpolation (slerp on the unit sphere) between endpoints."""
    p1 = np.array([
        np.cos(np.radians(lat1)) * np.cos(np.radians(lon1)),
        np.cos(np.radians(lat1)) * np.sin(np.radians(lon1)),
        np.sin(np.radians(lat1)),
    ])
    p2 = np.array([
        np.cos(np.radians(lat2)) * np.cos(np.radians(lon2)),
        np.cos(np.radians(lat2)) * np.sin(np.radians(lon2)),
        np.sin(np.radians(lat2)),
    ])
    omega = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    if omega < 1e-9:
        return (
            lat1 + (lat2 - lat1) * frac,
            lon1 + (lon2 - lon1) * frac,
        )
    s = np.sin(omega)
    a = np.sin((1.0 - frac) * omega) / s
    b = np.sin(frac * omega) / s
    xyz = a[:, None] * p1 + b[:, None] * p2
    lat = np.degrees(np.arcsin(xyz[:, 2]))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lat, lon


def _altitude_profile(
    frac: np.ndarray, cruise: float, rng: np.random.Generator
) -> np.ndarray:
    """Climb/cruise/descent altitude profile with gentle cruise variation."""
    alt = np.full_like(frac, cruise)
    climb = frac < 0.15
    descent = frac > 0.85
    alt[climb] = 500.0 + (cruise - 500.0) * (frac[climb] / 0.15)
    alt[descent] = cruise * (1.0 - (frac[descent] - 0.85) / 0.15) + 300.0
    return alt + rng.normal(0.0, 12.0, size=frac.shape)


def _flight_rows(
    route: dict[str, object], n_points: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Build one aircraft's trajectory rows (lat/lon/alt + derived columns)."""
    start, via, end, cruise = cast(
        tuple[tuple[float, float], tuple[float, float] | None,
              tuple[float, float], float],
        (route["start"], route["via"], route["end"], route["cruise"]),
    )
    t0, t1 = start
    if via is not None:
        v0, v1 = via
        e0, e1 = end
        f1 = np.linspace(0.0, 1.0, n_points // 2 + 1)
        f2 = np.linspace(0.0, 1.0, n_points - len(f1))
        lat_a, lon_a = _slerp_interp(t0, t1, v0, v1, f1)
        lat_b, lon_b = _slerp_interp(v0, v1, e0, e1, f2)
        lat = np.concatenate([lat_a, lat_b[1:]])
        lon = np.concatenate([lon_a, lon_b[1:]])
        frac = np.linspace(0.0, 1.0, len(lat))
    else:
        e0, e1 = end
        frac = np.linspace(0.0, 1.0, n_points)
        lat, lon = _slerp_interp(t0, t1, e0, e1, frac)
    # slight lateral curve so the track is not a perfect line
    lat = lat + 0.03 * np.sin(np.pi * frac) * rng.normal(0.35, 0.1)
    lon = lon + 0.03 * np.sin(np.pi * frac) * rng.normal(-0.5, 0.1)
    alt = _altitude_profile(frac, cruise, rng)
    ts = _SAMPLE_EPOCH + frac * _SAMPLE_DURATION_S + rng.normal(
        0.0, 1.0, size=frac.shape
    )
    ts = np.sort(ts)
    # N/U/E derived from lat/lon deltas (approximate metres)
    dlat = np.gradient(np.radians(lat)) * 6_371_000.0
    dlon = np.gradient(np.radians(lon)) * 6_371_000.0 * np.cos(np.radians(lat))
    return pd.DataFrame(
        {
            "station": route["callsign"],
            "sensor": route["aircraft"],
            "Timestamp": np.round(ts, 3),
            "N": np.round(dlat, 2),
            "U": np.round(alt - alt.mean(), 2),
            "E": np.round(dlon, 2),
            "Latitude": np.round(lat, 6),
            "Longitude": np.round(lon, 6),
            "Altitude_m": np.round(alt, 3),
        }
    )


def sample_llh_3col() -> pd.DataFrame:
    """Sample 3-col LLH data: three aircraft on realistic flight paths.

    * KAL7701+HL8201 — Incheon -> over Seoul -> Busan (~370 km + leg)
    * JJA102+HL8052  — Seoul -> Jeju (~450 km)
    * KAL9903+HL8275 — Daegu -> Gwangju (~200 km)

    Great-circle (slerp) interpolation with a slight lateral curve,
    climb/cruise/descent altitude profiles (cruise 9000-10500 m), and
    timestamps spread over one hour.  ~300 points per aircraft.
    """
    rng = np.random.default_rng(42)
    sizes = [310, 290, 300]
    return pd.concat(
        [_flight_rows(r, n, rng) for r, n in zip(_FLIGHT_ROUTES, sizes)],
        ignore_index=True,
    )


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
