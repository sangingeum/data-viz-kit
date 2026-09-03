#!/usr/bin/env python3
"""Regenerate examples/sample_data.csv from the 3-aircraft flight generators."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from llh_utils import sample_llh_7col  # noqa: E402

out = Path(__file__).resolve().parent.parent / "examples" / "sample_data.csv"
df = sample_llh_7col()
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, {len(df.columns)} columns")
g = df.groupby("station").agg(
    n=("Latitude", "size"),
    lat_span=("Latitude", lambda s: s.max() - s.min()),
    lon_span=("Longitude", lambda s: s.max() - s.min()),
    max_alt=("Altitude_m", "max"),
)
print(g)

# smoothness report: polyfit residual std + consecutive-altitude jumps
EARTH_M = 6_371_000.0
for ident, sub in df.groupby("station"):
    sub = sub.sort_values("Timestamp")
    t = sub["Timestamp"].to_numpy(dtype=float)
    print(f"\n{ident}: {len(sub)} points")
    for col in ("Latitude", "Longitude"):
        coeffs = np.polyfit(t - t[0], sub[col].to_numpy(dtype=float), 3)
        resid = sub[col].to_numpy(dtype=float) - np.polyval(coeffs, t - t[0])
        axis = "lat" if col == "Latitude" else "lon"
        print(
            f"  {axis} polyfit residual std = {resid.std():.2e} deg"
            f" ({resid.std() * EARTH_M * np.pi / 180:.1f} m)"
        )
    jumps = np.abs(np.diff(sub["Altitude_m"].to_numpy(dtype=float)))
    print(f"  max consecutive-altitude jump = {jumps.max():.2f} m")
