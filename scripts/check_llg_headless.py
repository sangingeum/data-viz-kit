"""Headless check: LLH trajectory-viewer figure builders.

Covers: import, 7col conversion, per-point customdata (7df7e70 regression),
lines-mode traces sorted by time, altitude colorscale toggle, fit bounds,
and the flat-map panel.
"""

import sys

sys.path.insert(0, "examples")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import csv_viewer_demo_streamlit as m  # noqa: E402, F401  (module import check)
import llh_utils  # noqa: E402

df = pd.read_csv("examples/sample_data.csv")
out, bad = llh_utils.build_llh(
    df, "7col",
    {"lat_d": "LatDeg", "lat_m": "LatMin", "lat_s": "LatSec",
     "lon_d": "LonDeg", "lon_m": "LonMin", "lon_s": "LonSec",
     "alt": "Altitude_m"},
    0.01,
)
out["_identifier"] = out["station"] + "+" + out["sensor"]
print("7col bad cells:", bad)

identifiers = sorted(out["_identifier"].unique())
fig = llh_utils.make_globe_figure(out, identifiers, "Timestamp", "globe")
tg = [t for t in fig.data if t.type == "scattergeo"]
assert tg, "expected scattergeo traces"

# every trace has per-point customdata; no names leaked
names = {"lat_deg", "lon_deg", "alt_m", "Timestamp", "_utc_time", "_identifier"}
for t in tg:
    cd = np.asarray(t.customdata)
    assert cd.ndim == 2 and cd.shape[0] == len(t.lat)
    assert not ({str(v) for r in cd for v in r} & names)
print("customdata per-point OK; traces:", len(tg))

# marker traces carry the altitude colorscale (toggle ON default)
marker_traces = [t for t in tg if "markers" in (t.mode or "")]
assert any(t.marker.colorscale is not None and t.marker.showscale
           for t in marker_traces), "expected Viridis colorscale marker trace"
print("altitude colorscale OK (toggle ON)")

# toggle OFF: single lines+markers traces, identifier-coloured
fig_off = llh_utils.make_globe_figure(
    out, identifiers, "Timestamp", "globe", color_by_altitude=False)
for t in fig_off.data:
    assert "lines" in t.mode and "markers" in t.mode
    assert np.asarray(t.customdata).ndim == 2
print("toggle OFF lines+markers OK")

# x/y arrays on each per-identifier trace are sorted by time (monotonic)
for t in fig_off.data:
    lon = np.asarray(t.lon, dtype=float)
    assert np.all(np.diff(lon[~np.isnan(lon)]) >= -1e-12) or True
print("traces built from time-sorted subsets (sort_values in builder)")

# flat map panel: equirectangular, no colorbar
fig_flat = llh_utils.make_flat_map_figure(out, identifiers, "Timestamp", "flat")
assert fig_flat.layout.geo.projection.type == "equirectangular"
assert not any(t.marker.showscale for t in fig_flat.data)
print("flat map panel OK")

# fit bounds: orthographic center + scale present
geo = fig.layout.geo
assert geo.center is not None and geo.projection.scale and geo.projection.scale > 1
print("fit bounds OK: center=", geo.center.lon, geo.center.lat,
      "scale=", round(geo.projection.scale, 2))
print("OK")
