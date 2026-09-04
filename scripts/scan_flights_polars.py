#!/usr/bin/env python3
"""Scan the Kaggle OpenSky CSV with polars (vectorized, multithreaded).

Stage 1: per-(icao24, callsign) stats in ONE lazy pass; print the best
candidates for a curated 3-flight sample.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SRC = Path("/home/keum/temp/dataset/9columns_and_nullcallsign_dropped_flights_data.csv")

lf = pl.scan_csv(
    SRC,
    schema_overrides={
        "time": pl.Float64,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "velocity": pl.Float64,
        "heading": pl.Float64,
        "baroaltitude": pl.Float64,
        "geoaltitude": pl.Float64,
    },
    null_values=[""],
    ignore_errors=True,
)

stats = (
    lf.group_by("icao24", "callsign")
    .agg(
        pl.len().alias("n"),
        pl.col("time").min().alias("tmin"),
        pl.col("time").max().alias("tmax"),
        pl.col("lat").max().alias("lat_max"),
        pl.col("lat").min().alias("lat_min"),
        pl.col("lon").max().alias("lon_max"),
        pl.col("lon").min().alias("lon_min"),
        pl.col("geoaltitude").drop_nulls().len().alias("geo_n"),
        pl.col("geoaltitude").max().alias("alt_max"),
    )
    .with_columns(
        (pl.col("lat_max") - pl.col("lat_min")).abs().alias("lat_span"),
        (pl.col("lon_max") - pl.col("lon_min")).abs().alias("lon_span"),
        (pl.col("tmax") - pl.col("tmin")).alias("dur_s"),
        (pl.col("geo_n") / pl.col("n")).alias("geo_frac"),
    )
    .filter(
        (pl.col("dur_s") > 1800)
        & (pl.col("dur_s") < 6 * 3600)
        & ((pl.col("lat_span") > 1.0) | (pl.col("lon_span") > 1.0))
        & (pl.col("n") < 20_000)
        & (pl.col("geo_frac") > 0.9)
        & (pl.col("alt_max") > 5000)
    )
    .sort("n", descending=True)
    .head(40)
    .collect(engine="streaming")
)

with pl.Config(tbl_rows=40, tbl_width_chars=220):
    print(stats)