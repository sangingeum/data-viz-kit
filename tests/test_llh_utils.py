"""Headless tests for the LLH (LLA) helpers in examples/llh_utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from llh_utils import (  # noqa: E402
    along_track_distance_km,
    build_llh,
    coerce_numeric,
    detect_llh_layout,
    dms_to_decimal_degrees,
    make_3d_space_figure,
    make_altitude_profile_figure,
    make_flat_map_figure,
    make_globe_figure,
    make_trajectory_figure,
    sample_llh_3col,
    sample_llh_7col,
)


# ---- DMS conversion + seconds scale ---------------------------------------


def test_dms_basic_hand_computed() -> None:
    """37 deg 30 min 21.96 s = 37.5061 deg (hand check: 21.96/3600=0.0061)."""
    df = pd.DataFrame({"d": [37.0], "m": [30.0], "s": [21.96]})
    out = dms_to_decimal_degrees(df["d"], df["m"], df["s"], 1.0)
    assert out.iloc[0] == pytest.approx(37.0 + 30.0 / 60.0 + 21.96 / 3600.0)


def test_dms_seconds_scale_001() -> None:
    """Seconds pre-multiplied by 100: scale 0.01 restores true seconds."""
    df = pd.DataFrame({"d": [37.0], "m": [30.0], "s": [2196.0]})  # = 21.96 s * 100
    out = dms_to_decimal_degrees(df["d"], df["m"], df["s"], 0.01)
    assert out.iloc[0] == pytest.approx(37.0 + 30.0 / 60.0 + 21.96 / 3600.0)


def test_dms_seconds_scale_default_is_one() -> None:
    df = pd.DataFrame({"d": [10.0], "m": [0.0], "s": [3600.0]})  # 3600s scaled=1h
    out = dms_to_decimal_degrees(df["d"], df["m"], df["s"])
    assert out.iloc[0] == pytest.approx(11.0)


def test_dms_rejects_nonpositive_scale() -> None:
    df = pd.DataFrame({"d": [1.0], "m": [0.0], "s": [0.0]})
    with pytest.raises(ValueError):
        dms_to_decimal_degrees(df["d"], df["m"], df["s"], 0.0)


def test_dms_negative_longitude() -> None:
    df = pd.DataFrame({"d": [-127.0], "m": [0.0], "s": [0.0]})
    out = dms_to_decimal_degrees(df["d"], df["m"], df["s"], 1.0)
    assert out.iloc[0] == pytest.approx(-127.0)


# ---- numeric coercion ------------------------------------------------------


def test_coerce_numeric_reports_non_numeric() -> None:
    s = pd.Series(["1.5", "abc", None, "2.0"])
    out, bad = coerce_numeric(s)
    assert bad == 1  # 'abc' only; None was already missing
    assert out.iloc[0] == pytest.approx(1.5)
    assert np.isnan(out.iloc[1])


# ---- auto-detect -----------------------------------------------------------


def test_detect_3col_layout() -> None:
    layout, d = detect_llh_layout(
        ["Timestamp", "sensor", "latitude", "longitude", "altitude"]
    )
    assert layout == "3col"
    assert d == {"lat": "latitude", "lon": "longitude", "alt": "altitude"}


def test_detect_3col_abbreviations() -> None:
    layout, d = detect_llh_layout(["t", "lat", "lng", "alt_m"])
    assert layout == "3col"
    assert d["lat"] == "lat" and d["lon"] == "lng" and d["alt"] == "alt_m"


def test_detect_7col_layout() -> None:
    layout, d = detect_llh_layout(
        ["t", "lat_deg", "lat_min", "lat_sec", "lon_deg", "lon_min",
         "lon_sec", "alt"]
    )
    assert layout == "7col"
    assert d["lat_d"] == "lat_deg"
    assert d["lon_s"] == "lon_sec"
    assert d["alt"] == "alt"


def test_detect_prefers_7col_when_both_present() -> None:
    layout, d = detect_llh_layout(
        ["latd", "latm", "lats", "lond", "lonm", "lons", "alt",
         "latitude", "longitude"]
    )
    assert layout == "7col"
    assert d["lat_d"] == "latd"


# ---- build_llh -------------------------------------------------------------


def _seven_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [1725270026.504],
            "LatDeg": [37.0],
            "LatMin": [30.0],
            "LatSec": [2196.0],  # pre-multiplied by 100 -> 21.96 s
            "LonDeg": [126.0],
            "LonMin": [59.0],
            "LonSec": [5925.12],  # -> 59.2512 s
            "alt_m": [81.1],
        }
    )


def test_build_llh_7col_with_scale() -> None:
    out, bad = build_llh(
        _seven_df(), "7col",
        {"lat_d": "LatDeg", "lat_m": "LatMin", "lat_s": "LatSec",
         "lon_d": "LonDeg", "lon_m": "LonMin", "lon_s": "LonSec",
         "alt": "alt_m"},
        seconds_scale=0.01,
    )
    assert bad == 0
    assert out["lat_deg"].iloc[0] == pytest.approx(37.5061)
    assert out["lon_deg"].iloc[0] == pytest.approx(126.0 + 59 / 60 + 59.2512 / 3600)
    assert out["alt_m"].iloc[0] == pytest.approx(81.1)


def test_build_llh_7col_scale_one() -> None:
    out, _ = build_llh(
        _seven_df(), "7col",
        {"lat_d": "LatDeg", "lat_m": "LatMin", "lat_s": "LatSec",
         "lon_d": "LonDeg", "lon_m": "LonMin", "lon_s": "LonSec",
         "alt": "alt_m"},
        seconds_scale=1.0,
    )
    # seconds used as-is: 2196 s = 36.6 arcmin extra
    assert out["lat_deg"].iloc[0] == pytest.approx(37.0 + 30 / 60 + 2196 / 3600)


def test_build_llh_3col() -> None:
    df = pd.DataFrame({"lat": [37.5], "lon": [127.0], "alt": [80.0], "x": [1]})
    out, bad = build_llh(df, "3col", {"lat": "lat", "lon": "lon", "alt": "alt"})
    assert bad == 0
    assert out["lat_deg"].iloc[0] == 37.5
    assert out["lon_deg"].iloc[0] == 127.0
    assert out["alt_m"].iloc[0] == 80.0
    assert "x" in out.columns  # original columns preserved


def test_build_llh_3col_counts_non_numeric() -> None:
    df = pd.DataFrame({"lat": [37.5, "oops"], "lon": [127.0, 127.1],
                       "alt": [80.0, 81.0]})
    out, bad = build_llh(df, "3col", {"lat": "lat", "lon": "lon", "alt": "alt"})
    assert bad == 1
    assert np.isnan(out["lat_deg"].iloc[1])


# ---- sample generators -----------------------------------------------------


def test_sample_llh_3col_flight_paths() -> None:
    """Three aircraft on realistic trajectories that actually travel."""
    df = sample_llh_3col()
    assert {"station", "sensor", "Timestamp", "N", "U", "E",
            "Latitude", "Longitude", "Altitude_m"}.issubset(df.columns)
    ids = sorted((df["station"] + "+" + df["sensor"]).unique())
    assert len(ids) == 3
    for ident in ids:
        station, sensor = ident.split("+")
        sub = df[(df["station"] == station) & (df["sensor"] == sensor)]
        assert 80 <= len(sub) <= 300, f"{ident}: {len(sub)} points"
        lat_span = sub["Latitude"].max() - sub["Latitude"].min()
        lon_span = sub["Longitude"].max() - sub["Longitude"].min()
        assert max(lat_span, lon_span) > 0.5, f"{ident} must travel"
        assert sub["Altitude_m"].max() > 3000.0, f"{ident} must reach cruise"
        ts = np.sort(sub["Timestamp"].to_numpy())
        assert ts[-1] - ts[0] <= 3600.0 + 30.0  # ~one hour of flight
        assert (sub["Altitude_m"] > 0).all()


def test_sample_llh_7col_roundtrip() -> None:
    df = sample_llh_7col()
    cols = {"lat_d": "LatDeg", "lat_m": "LatMin", "lat_s": "LatSec",
            "lon_d": "LonDeg", "lon_m": "LonMin", "lon_s": "LonSec",
            "alt": "Altitude_m"}
    out, bad = build_llh(df, "7col", cols, seconds_scale=0.01)
    assert bad == 0
    assert np.allclose(out["lat_deg"], df["Latitude"], atol=1e-9)
    assert np.allclose(out["lon_deg"], df["Longitude"], atol=1e-9)


def test_sample_csv_has_llh_columns() -> None:
    csv = pd.read_csv(EXAMPLES_DIR / "sample_data.csv")
    for c in ("station", "sensor", "Timestamp", "N", "U", "E",
              "Latitude", "Longitude", "Altitude_m",
              "LatDeg", "LatMin", "LatSec", "LonDeg", "LonMin", "LonSec"):
        assert c in csv.columns, f"missing column {c}"
    assert 450 <= len(csv) <= 700


def _smoothness_stats(csv: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-station polyfit residual std (deg) and max altitude jump (m)."""
    stats: dict[str, dict[str, float]] = {}
    for ident, sub in csv.groupby("station"):
        sub = sub.sort_values("Timestamp")
        t = sub["Timestamp"].to_numpy(dtype=float)
        s: dict[str, float] = {}
        for col in ("Latitude", "Longitude"):
            coeffs = np.polyfit(t - t[0], sub[col].to_numpy(dtype=float), 3)
            resid = sub[col].to_numpy(dtype=float) - np.polyval(coeffs, t - t[0])
            s[f"{col.lower()}_resid_std_deg"] = float(resid.std())
        s["max_alt_jump_m"] = float(
            np.abs(np.diff(sub["Altitude_m"].to_numpy(dtype=float))).max()
        )
        stats[str(ident)] = s
    return stats


def test_sample_trajectories_are_smooth() -> None:
    """Smooth generator: polyfit residual std < 0.005 deg, alt jumps < 100 m."""
    csv = pd.read_csv(EXAMPLES_DIR / "sample_data.csv")
    stats = _smoothness_stats(csv)
    assert stats, "expected three identifiers"
    for ident, s in stats.items():
        assert s["latitude_resid_std_deg"] < 0.005, (
            f"{ident} lat residual std {s['latitude_resid_std_deg']:.4f} deg"
        )
        assert s["longitude_resid_std_deg"] < 0.005, (
            f"{ident} lon residual std {s['longitude_resid_std_deg']:.4f} deg"
        )
        assert s["max_alt_jump_m"] < 100.0, (
            f"{ident} max altitude jump {s['max_alt_jump_m']:.1f} m"
        )


def test_sample_point_spacing_consistent() -> None:
    """Median step spacing within 0.5-4 km for every aircraft."""
    csv = pd.read_csv(EXAMPLES_DIR / "sample_data.csv")
    R = 6371.0088
    for ident, sub in csv.groupby("station"):
        sub = sub.sort_values("Timestamp")
        lat = np.radians(sub["Latitude"].to_numpy(dtype=float))
        lon = np.radians(sub["Longitude"].to_numpy(dtype=float))
        dlat, dlon = np.diff(lat), np.diff(lon)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2)
        steps = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        median = float(np.median(steps))
        assert 0.5 <= median <= 4.0, f"{ident} median step {median:.3f} km"

# ---- figure builders (headless) -------------------------------------------


def _figure_df() -> pd.DataFrame:
    df = sample_llh_3col().head(50)
    out, _ = build_llh(
        df, "3col", {"lat": "Latitude", "lon": "Longitude", "alt": "Altitude_m"}
    )
    out["_identifier"] = out["station"] + "+" + out["sensor"]
    return out


def test_globe_figure_scattergeo_customdata_is_per_point() -> None:
    """7df7e70 regression: customdata must be per-point values, not names."""
    df = _figure_df()
    fig = make_globe_figure(df, sorted(df["_identifier"].unique()), "Timestamp",
                            "globe")
    traces = [t for t in fig.data if t.type == "scattergeo"]
    assert traces, "expected a scattergeo trace"
    for trace in traces:
        cd = np.asarray(trace.customdata)
        assert cd.ndim == 2 and cd.shape[0] == len(trace.lat), (
            "customdata must be per-point rows"
        )
        # lat column of customdata must equal the actual point values
        assert np.allclose(cd[:, 0].astype(float),
                           np.asarray(trace.lat, dtype=float))
        # no customdata cell may be a bare column name (the old bug)
        names = {"lat_deg", "lon_deg", "alt_m", "Timestamp",
                 "_utc_time", "_identifier"}
        flat = {str(v) for row in cd for v in row}
        assert not (flat & names), "column names leaked into customdata"


def test_trajectory_traces_are_markers_only_sorted_by_time() -> None:
    """Every identifier draws a markers-only path, points sorted by timestamp."""
    df = _figure_df().sort_values("Timestamp", ascending=False)  # shuffled input
    fig = make_globe_figure(df, sorted(df["_identifier"].unique()), "Timestamp",
                            "globe")
    traces = [t for t in fig.data if t.type == "scattergeo"]
    assert traces
    for trace in traces:
        assert trace.mode == "markers", "point display only — no lines"
        ts = pd.to_datetime(np.asarray(trace.customdata)[:, 3], unit="s")
        assert (np.diff(ts.astype(np.int64)) >= np.timedelta64(0, "s")).all(), (
            "trace points must be monotonic non-decreasing in time"
        )


def test_identifier_colors_consistent_across_all_views() -> None:
    """Each identifier keeps ONE colour in globe, flat map, 3D, and profile."""
    from llh_utils import build_identifier_color_map

    df = _figure_df()
    ids = sorted(df["_identifier"].unique())
    color_map = build_identifier_color_map(ids)
    assert len(color_map) == len(ids)
    # same identifier always maps to the same colour, order-independent
    assert build_identifier_color_map(list(reversed(ids))) == color_map

    fig_globe = make_globe_figure(df, ids, "Timestamp", "g", color_map=color_map)
    fig_flat = make_flat_map_figure(df, ids, "Timestamp", "f", color_map=color_map)
    fig_3d = make_3d_space_figure(df, ids, "Timestamp", "3d", color_map=color_map)
    fig_prof = make_altitude_profile_figure(
        df, ids, "Timestamp", "p", color_map=color_map
    )
    for fig in (fig_globe, fig_flat, fig_3d, fig_prof):
        seen = {t.name: t.marker.color for t in fig.data if t.name in color_map}
        assert seen == color_map, f"colour mismatch in {type(fig.data[0]).type}"
    # no altitude-driven colour scale remains anywhere
    for fig in (fig_globe, fig_flat, fig_3d, fig_prof):
        for t in fig.data:
            assert not t.marker.showscale
            assert t.marker.colorscale is None


def test_trajectory_identifier_colors_default_consistent() -> None:
    """Without an explicit color_map, default palette colours are stable."""
    df = _figure_df()
    ids = sorted(df["_identifier"].unique())
    f1 = make_globe_figure(df, ids, "Timestamp", "g")
    f2 = make_3d_space_figure(df, ids, "Timestamp", "3d")
    colors_g = {t.name: t.marker.color for t in f1.data}
    colors_3 = {t.name: t.marker.color for t in f2.data}
    assert colors_g == colors_3


def test_trajectory_hover_template() -> None:
    """Hover shows identifier, timestamp, UTC, and exact lat/lon/alt."""
    df = _figure_df()
    fig = make_trajectory_figure(df, sorted(df["_identifier"].unique()),
                                 "Timestamp", "g")
    tmpl = [t.hovertemplate for t in fig.data if t.hovertemplate]
    assert tmpl
    for t in tmpl:
        assert "_identifier" not in t  # template references indices, not names
        assert "UTC = %{customdata[4]}" in t
        assert "alt_m = %{customdata[2]}" in t
    line_traces = [t for t in fig.data if t.hoverinfo == "skip"]
    assert not line_traces, "no hover-suppressed traces (markers-only now)"


def test_trajectory_fit_bounds_zooms_to_data() -> None:
    df = _figure_df()
    fig = make_globe_figure(df, [], "Timestamp", "g", fit_bounds=True)
    geo = fig.layout.geo
    assert geo.center is not None
    assert geo.projection.scale and geo.projection.scale > 1, "auto-zoomed"
    assert abs(geo.center.lat - df["lat_deg"].mean()) < 1.0

    fig_plain = make_globe_figure(df, [], "Timestamp", "g", fit_bounds=False)
    assert fig_plain.layout.geo.projection.scale in (None, 1)


def test_trajectory_projection_selectable() -> None:
    df = _figure_df()
    for proj in ("orthographic", "equirectangular", "natural earth"):
        fig = make_trajectory_figure(df, [], "Timestamp", "g", projection=proj)
        assert fig.layout.geo.projection.type == proj


def test_flat_map_figure_is_equirectangular() -> None:
    df = _figure_df()
    fig = make_flat_map_figure(df, sorted(df["_identifier"].unique()),
                               "Timestamp", "flat")
    assert fig.layout.geo.projection.type == "equirectangular"
    assert all(t.mode == "markers" for t in fig.data)
    assert not any(t.marker.showscale for t in fig.data)


# ---- new trajectory views ---------------------------------------------------


def test_3d_space_figure_markers_only_customdata() -> None:
    df = _figure_df()
    ids = sorted(df["_identifier"].unique())
    fig = make_3d_space_figure(df, ids, "Timestamp", "3d")
    assert fig.data and all(t.type == "scatter3d" for t in fig.data)
    for t in fig.data:
        assert t.mode == "markers"
        cd = np.asarray(t.customdata)
        assert cd.ndim == 2 and cd.shape[0] == len(t.x)
        # z is altitude, matching actual points
        assert np.allclose(np.asarray(t.z, dtype=float),
                           cd[:, 2].astype(float))
    assert not any(t.marker.showscale for t in fig.data)
    assert any(t.hovertemplate for t in fig.data)


def test_along_track_distance_is_cumulative_monotonic() -> None:
    df = _figure_df()
    dist = along_track_distance_km(df, "Timestamp")
    assert len(dist) == len(df)
    for ident in sorted(df["_identifier"].unique()):
        sub = df[df["_identifier"] == ident]
        d = dist.loc[sub.index].sort_index()
        assert d.iloc[0] == pytest.approx(0.0)
        assert (np.diff(d.to_numpy()) >= -1e-9).all()
    # full-sample trajectories must travel real distances
    full = sample_llh_3col()
    out, _ = build_llh(
        full, "3col",
        {"lat": "Latitude", "lon": "Longitude", "alt": "Altitude_m"},
    )
    out["_identifier"] = out["station"] + "+" + out["sensor"]
    full_dist = along_track_distance_km(out, "Timestamp")
    max_per_id = full_dist.groupby(out["_identifier"]).max()
    assert (max_per_id > 100.0).all()


def test_altitude_profile_figure() -> None:
    df = _figure_df()
    fig = make_altitude_profile_figure(
        df, sorted(df["_identifier"].unique()), "Timestamp", "prof"
    )
    assert fig.data and all(t.type == "scatter" for t in fig.data)
    for t in fig.data:
        assert t.mode == "markers"  # no lines
        x = np.asarray(t.x, dtype=float)
        y = np.asarray(t.y, dtype=float)
        assert (np.diff(x) >= -1e-9).all()  # cumulative distance
        assert y.min() > 0.0
        assert t.hovertemplate and "along_track_km" in t.hovertemplate


def test_globe_figure_layout_and_hover() -> None:
    df = _figure_df()
    fig = make_globe_figure(df, sorted(df["_identifier"].unique()), "Timestamp",
                            "globe")
    geo = fig.layout.geo
    assert geo.projection.type == "orthographic"
    assert geo.showcountries and geo.showcoastlines and geo.showland
    tmpl = [t.hovertemplate for t in fig.data if t.hovertemplate][0]
    assert "_identifier" not in tmpl  # template references indices, not names
    assert "customdata[5]" in tmpl  # identifier slot
    assert "UTC = %{customdata[4]}" in tmpl  # utc datetime slot
    utc = df.assign(_utc_time=pd.to_datetime(df["Timestamp"], unit="s", utc=True)
                    .astype(str))
    assert utc["_utc_time"].str.contains(r"\d{4}-\d{2}-\d{2}").all()
