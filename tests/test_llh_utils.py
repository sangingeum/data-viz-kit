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
    build_llh,
    coerce_numeric,
    detect_llh_layout,
    dms_to_decimal_degrees,
    make_globe_figure,
    make_time_scatter,
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


def test_sample_llh_3col() -> None:
    df = sample_llh_3col()
    assert {"station", "sensor", "Timestamp", "N", "U", "E",
            "Latitude", "Longitude", "Altitude_m"}.issubset(df.columns)
    assert df["Latitude"].between(37.0, 38.0).all()
    assert df["Longitude"].between(126.5, 127.5).all()


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
    assert len(csv) == 900


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
        assert cd.ndim == 2 and cd.shape[0] == len(df), (
            "customdata must be per-point rows"
        )
        # lat column of customdata must equal the actual point values
        assert np.allclose(cd[:, 0].astype(float), df["lat_deg"].to_numpy())
        # no customdata cell may be a bare column name (the old bug)
        names = {"lat_deg", "lon_deg", "alt_m", "Timestamp",
                 "_utc_time", "_identifier"}
        flat = {str(v) for row in cd for v in row}
        assert not (flat & names), "column names leaked into customdata"


def test_globe_figure_layout_and_hover() -> None:
    df = _figure_df()
    fig = make_globe_figure(df, [], "Timestamp", "globe")
    geo = fig.layout.geo
    assert geo.projection.type == "orthographic"
    assert geo.showcountries and geo.showcoastlines and geo.showland
    tmpl = fig.data[0].hovertemplate
    assert "_identifier" not in tmpl  # template references indices, not names
    assert "customdata[5]" in tmpl  # identifier slot
    assert "UTC = %{customdata[4]}" in tmpl  # utc datetime slot
    utc = df.assign(_utc_time=pd.to_datetime(df["Timestamp"], unit="s", utc=True)
                    .astype(str))
    assert utc["_utc_time"].str.contains(r"\d{4}-\d{2}-\d{2}").all()


def test_time_scatter_customdata_per_point() -> None:
    df = _figure_df()
    fig = make_time_scatter(df, [], "lat_deg", "Timestamp", "lat vs time")
    for trace in fig.data:
        cd = np.asarray(trace.customdata)
        assert cd.ndim == 2
        vals = cd[:, 1].astype(float)
        assert set(np.round(vals, 6)).issubset(set(np.round(df["lat_deg"], 6)))
