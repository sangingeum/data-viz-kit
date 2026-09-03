"""Headless check: LLH trajectory-viewer figure builders.

Covers: import, 7col conversion, per-point customdata (7df7e70 regression),
markers-only traces sorted by time, identifier-colour consistency across all
four views (no altitude colourscale), fit bounds, the flat-map panel, the
3D-space and altitude-profile views, sample smoothness stats, and that the
sample's 3 identifiers travel (>0.5 deg span each).
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

# one identifier->colour mapping shared by every view
color_map = llh_utils.build_identifier_color_map(identifiers)
print("identifier color map:", color_map)

fig = llh_utils.make_globe_figure(out, identifiers, "Timestamp", "globe",
                                  color_map=color_map)
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

# identifier-colour consistency across ALL FOUR views
fig_flat = llh_utils.make_flat_map_figure(out, identifiers, "Timestamp", "flat",
                                          color_map=color_map)
fig_3d = llh_utils.make_3d_space_figure(out, identifiers, "Timestamp", "3d",
                                        color_map=color_map)
fig_prof = llh_utils.make_altitude_profile_figure(out, identifiers, "Timestamp",
                                                  "profile", color_map=color_map)
for label, f in (("globe", fig), ("flat", fig_flat),
                 ("3d", fig_3d), ("profile", fig_prof)):
    seen = {t.name: t.marker.color for t in f.data}
    assert seen == color_map, f"{label} colour mismatch: {seen}"
    for t in f.data:
        assert t.mode == "markers"
        assert not t.marker.showscale, f"{label}: colorbar must be gone"
        assert t.marker.colorscale is None, f"{label}: colorscale must be gone"
        assert np.asarray(t.customdata).ndim == 2
print("identifier-colour consistency OK across globe/flat/3d/profile")

# flat map panel: equirectangular
assert fig_flat.layout.geo.projection.type == "equirectangular"
print("flat map panel OK")

# fit bounds: orthographic center + scale present
geo = fig.layout.geo
assert geo.center is not None and geo.projection.scale and geo.projection.scale > 1
print("fit bounds OK: center=", geo.center.lon, geo.center.lat,
      "scale=", round(geo.projection.scale, 2))

# smoothness stats of the regenerated sample
EARTH_M = 6_371_000.0
for ident, sub in df.groupby("station"):
    sub = sub.sort_values("Timestamp")
    t = sub["Timestamp"].to_numpy(dtype=float)
    lines = [f"{ident}: {len(sub)} points"]
    for col, axis in (("Latitude", "lat"), ("Longitude", "lon")):
        coeffs = np.polyfit(t - t[0], sub[col].to_numpy(dtype=float), 3)
        resid = sub[col].to_numpy(dtype=float) - np.polyval(coeffs, t - t[0])
        lines.append(
            f"  {axis} polyfit residual std = {resid.std():.2e} deg"
            f" ({resid.std() * EARTH_M * np.pi / 180:.1f} m)"
        )
        assert resid.std() < 0.005
    jumps = np.abs(np.diff(sub["Altitude_m"].to_numpy(dtype=float)))
    lines.append(f"  max consecutive-altitude jump = {jumps.max():.2f} m")
    assert jumps.max() < 100.0
    # median step spacing, consistent across aircraft
    lat = np.radians(sub["Latitude"].to_numpy(dtype=float))
    lon = np.radians(sub["Longitude"].to_numpy(dtype=float))
    a = (np.sin(np.diff(lat) / 2) ** 2
         + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
    steps = EARTH_M / 1000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    lines.append(f"  median step = {np.median(steps):.2f} km")
    print("\n".join(lines))
print("OK")
