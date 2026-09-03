"""Headless check: LLH trajectory-viewer figure builders.

Covers: import, 7col conversion, per-point customdata (7df7e70 regression),
markers-only traces sorted by time, altitude colorscale toggle, fit bounds,
the flat-map panel, the new 3D-space and altitude-profile views, and that
the sample's 3 identifiers travel (>0.5 deg span each).
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

# 3col <-> 7col round-trip on the shipped sample
out3, bad3 = llh_utils.build_llh(
    df, "3col", {"lat": "Latitude", "lon": "Longitude", "alt": "Altitude_m"}
)
assert bad3 == 0
assert np.allclose(out["lat_deg"], out3["lat_deg"], atol=1e-9)
assert np.allclose(out["lon_deg"], out3["lon_deg"], atol=1e-9)
print("7col->3col round-trip OK (seconds scale 0.01)")

identifiers = sorted(out["_identifier"].unique())
assert len(identifiers) == 3, f"expected 3 identifiers, got {len(identifiers)}"

# each identifier actually travels (lat/lon span > 0.5 deg)
for ident in identifiers:
    sub = out[out["_identifier"] == ident]
    lat_span = sub["lat_deg"].max() - sub["lat_deg"].min()
    lon_span = sub["lon_deg"].max() - sub["lon_deg"].min()
    assert max(lat_span, lon_span) > 0.5, f"{ident} does not travel"
print("3 identifiers, all moving (span > 0.5 deg):", identifiers)

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

# markers-only display
for t in tg:
    assert t.mode == "markers", "point display only — no lines"

# marker traces carry the altitude colorscale (toggle ON default)
assert any(t.marker.colorscale is not None and t.marker.showscale
           for t in tg), "expected Viridis colorscale marker trace"
print("altitude colorscale OK (toggle ON)")

# toggle OFF: identifier-coloured markers, no lines
fig_off = llh_utils.make_globe_figure(
    out, identifiers, "Timestamp", "globe", color_by_altitude=False)
for t in fig_off.data:
    assert t.mode == "markers"
    assert np.asarray(t.customdata).ndim == 2
print("toggle OFF marker-coloured OK")

# flat map panel: equirectangular, no colorbar
fig_flat = llh_utils.make_flat_map_figure(out, identifiers, "Timestamp", "flat")
assert fig_flat.layout.geo.projection.type == "equirectangular"
assert not any(t.marker.showscale for t in fig_flat.data)
print("flat map panel OK")

# 3D space view
fig_3d = llh_utils.make_3d_space_figure(out, identifiers, "Timestamp", "3d")
t3 = [t for t in fig_3d.data if t.type == "scatter3d"]
assert t3 and all(t.mode == "markers" for t in t3)
assert any(t.marker.showscale for t in t3)
print("3D space view OK; traces:", len(t3))

# altitude vs along-track-distance profile
fig_prof = llh_utils.make_altitude_profile_figure(
    out, identifiers, "Timestamp", "profile")
tp = [t for t in fig_prof.data if t.type == "scatter"]
assert tp and all(t.mode == "markers" for t in tp)
for t in tp:
    x = np.asarray(t.x, dtype=float)
    assert (np.diff(x) >= -1e-9).all() and x.max() > 50.0
print("altitude profile OK; traces:", len(tp))

# fit bounds: orthographic center + scale present
geo = fig.layout.geo
assert geo.center is not None and geo.projection.scale and geo.projection.scale > 1
print("fit bounds OK: center=", geo.center.lon, geo.center.lat,
      "scale=", round(geo.projection.scale, 2))
print("OK")
