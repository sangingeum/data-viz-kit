#!/usr/bin/env python3
"""Regenerate examples/sample_data.csv from the 3-aircraft flight generators."""

import sys
from pathlib import Path

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
