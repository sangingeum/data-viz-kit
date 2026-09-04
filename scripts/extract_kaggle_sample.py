"""Final extraction: pick flights by segment quality measured on CLEANED data.

Cleaned = baro altitude in [0, 13 km], dedup'd consecutive identical
position+altitude reports.  Rank segments by cleaned distinct-position count;
take 3 different aircraft.  Then subsample evenly, rebase timestamps, LLH ->
NUE (WGS84 ENU, base = mean lat/lon), write viewer schema (DMS seconds x100,
sign carried by the 3-col column).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from llh_utils import llh_to_nue  # noqa: E402

SRC = "/home/keum/temp/dataset/9columns_and_nullcallsign_dropped_flights_data.csv"
OUT = REPO / "examples" / "sample_data.csv"
TARGET_POINTS = 200
COMMON_EPOCH = 1751328000.0  # 2025-07-01 00:00 UTC

# chosen from the cleaned-candidate scan (different aircraft, big spans):
#   seg 89509  N369PD (a42f04): 6195 distinct, climb 472 -> 12009 m
#   seg 26479  N2QU   (a18ec2): 7354 distinct, climb  427 ->  8214 m
#   seg 116746 N46CL  (a5977f): 4341 distinct, climb  549 ->  8595 m
SEGMENTS = [
    (89509, "N369PD", "a42f04"),
    (26479, "N2QU", "a18ec2"),
    (116746, "N46CL", "a5977f"),
]


def load_clean_segment(segs: pl.DataFrame, seg_id: int) -> pl.DataFrame:
    df = segs.filter(pl.col("seg") == seg_id).sort("time")
    df = df.filter((pl.col("baroaltitude") >= 0) & (pl.col("baroaltitude") < 13_000))
    df = df.filter(
        (pl.col("lat") != pl.col("lat").shift())
        | (pl.col("lon") != pl.col("lon").shift())
        | (pl.col("baroaltitude") != pl.col("baroaltitude").shift())
    )
    return df


lf = pl.scan_csv(
    SRC,
    schema_overrides={
        "time": pl.Float64,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "baroaltitude": pl.Float64,
        "geoaltitude": pl.Float64,
    },
    null_values=[""],
    ignore_errors=True,
)

segs = (
    lf.sort("icao24", "callsign", "time")
    .with_columns(
        (
            (pl.col("icao24") != pl.col("icao24").shift())
            | (pl.col("callsign") != pl.col("callsign").shift())
            | ((pl.col("time") - pl.col("time").shift()) > 60)
        ).fill_null(True).alias("brk")
    )
    .with_columns(pl.col("brk").cum_sum().alias("seg"))
    .collect(engine="streaming")
)

frames = []
for seg_id, cs, icao in SEGMENTS:
    df = load_clean_segment(segs, seg_id)
    n = df.height
    idx = np.unique(np.linspace(0, n - 1, min(TARGET_POINTS, n)).astype(int))
    sub = df[idx]
    print(f"{cs}: {n} cleaned positions -> {sub.height} sampled, "
          f"baro {sub['baroaltitude'].min():.0f}..{sub['baroaltitude'].max():.0f} m")
    frames.append(sub)

for i, f in enumerate(frames):
    dur = f["time"][-1] - f["time"][0]
    scale = min(1.0, 2400.0 / max(dur, 1.0))
    f = f.with_columns(
        (COMMON_EPOCH + (pl.col("time") - pl.col("time").min()) * scale).alias("time")
    )
    frames[i] = f

df = pl.concat(frames)
df = df.with_columns(
    pl.col("baroaltitude").alias("Altitude_m"),
    pl.col("callsign").alias("station"),
    pl.col("icao24").alias("sensor"),
    pl.col("time").alias("Timestamp"),
)

lat = df["lat"].to_numpy()
lon = df["lon"].to_numpy()
alt = df["Altitude_m"].to_numpy()
base_lat, base_lon = float(np.mean(lat)), float(np.mean(lon))
N, U, E = llh_to_nue(lat, lon, alt, base_lat, base_lon, 0.0)

out = pl.DataFrame(
    {
        "station": df["station"],
        "sensor": df["sensor"],
        "Timestamp": df["Timestamp"],
        "N": N,
        "U": U,
        "E": E,
        "Latitude": lat,
        "Longitude": lon,
        "Altitude_m": alt,
    }
)


def dms(deg):
    """Signed degrees -> (D, M, S*100) positive magnitudes (sign carried by
    the 3-col column: western lon negative)."""
    a = np.abs(deg)
    d = np.floor(a)
    m = np.floor((a - d) * 60)
    s = ((a - d) * 60 - m) * 60
    return d, m, s * 100.0


latd, latm, lats = dms(lat)
lond, lonm, lons = dms(lon)
out = out.with_columns(
    pl.Series("LatDeg", latd),
    pl.Series("LatMin", latm),
    pl.Series("LatSec", lats),
    pl.Series("LonDeg", lond),
    pl.Series("LonMin", lonm),
    pl.Series("LonSec", lons),
)

out = out.select(
    "station", "sensor", "Timestamp", "N", "U", "E",
    "Latitude", "Longitude", "Altitude_m",
    "LatDeg", "LatMin", "LatSec", "LonDeg", "LonMin", "LonSec",
)
out.write_csv(OUT)
print(f"wrote {OUT}: {out.height} rows")
print(
    out.group_by("station")
    .agg(
        pl.len().alias("n"),
        (pl.col("Latitude").max() - pl.col("Latitude").min()).alias("lat_span"),
        (pl.col("Longitude").max() - pl.col("Longitude").min()).alias("lon_span"),
        pl.col("Altitude_m").min().alias("alt_min"),
        pl.col("Altitude_m").max().alias("alt_max"),
    )
    .sort("station")
)
print(f"base lat/lon: {base_lat:.5f}, {base_lon:.5f}")