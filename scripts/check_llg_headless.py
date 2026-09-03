"""Headless check: import streamlit module, build 7-col globe from sample data,
verify per-point customdata (7df7e70 regression)."""

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

fig = llh_utils.make_globe_figure(out.head(200), [], "Timestamp", "globe")
tg = [t for t in fig.data if t.type == "scattergeo"]
assert tg
cd = np.asarray(tg[0].customdata)
print("customdata shape:", cd.shape)
assert cd.shape[0] == 200 and cd.ndim == 2
names = {"lat_deg", "lon_deg", "alt_m", "Timestamp", "_utc_time", "_identifier"}
assert not ({str(v) for r in cd for v in r} & names)
assert np.allclose(cd[:, 0].astype(float), out["lat_deg"].head(200).to_numpy())
print("UTC sample:", cd[0, 4])
print("OK")
