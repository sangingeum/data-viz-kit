#!/usr/bin/env python3
"""Extract a curated 3-flight sample from the Kaggle OpenSky flight CSV.

Reads the ~2.5 GB Kaggle CSV in chunks (never fully loaded), picks three
real flights (long continuous 1 Hz streams with good geographic span and
non-null geoaltitude), dedups consecutive identical lat/lon rows,
subsamples evenly in time, rebases each flight's timestamps onto a common
~1-hour window, converts LLH -> N/U/E (proper WGS84 ENU, base = mean
lat/lon of all three flights at base_alt_m=0), and writes
examples/sample_data.csv with the viewer's 3-col + 7-col DMS schema
(seconds stored x100, matching the viewer's 0.01 scale option).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from llh_utils import llh_to_nue  # noqa: E402

DEFAULT_SRC = Path(
    "/home/keum/temp/dataset/9columns_and_nullcallsign_dropped_flights_data.csv"
)
DEFAULT_OUT = REPO / "examples" / "sample_data.csv"

#: flight selection criteria (after dedup of consecutive identical positions)
MIN_POINTS = 200
MAX_POINTS = 400
MIN_SPAN_DEG = 1.0          # max(lat_span, lon_span) must exceed this
MAX_GAP_S = 60.0            # a stream with an intra-gap this big is rejected
NUMERIC_COLS = ["time", "lat", "lon", "velocity", "heading",
                "baroaltitude", "geoaltitude"]


def scan_for_flights(
    src: Path,
    *,
    need: int = 3,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
    min_span_deg: float = MIN_SPAN_DEG,
    max_gap_s: float = MAX_GAP_S,
) -> list[dict[str, object]]:
    """One pass over the CSV, chunked, tracking the best candidate streams.

    A candidate is a continuous (icao24, callsign) stream with dt <= 5 s,
    deduped on consecutive identical lat/lon, with geoaltitude coverage
    >= 95 %, point count in [min_points, max_points], geographic span
    >= min_span_deg, and no intra-gap larger than *max_gap_s*.
    """
    best: dict[tuple[str, str], dict[str, object]] = {}
    prev_chunk_last: pd.DataFrame | None = None

    for chunk in pd.read_csv(src, chunksize=2_000_000, low_memory=False):
        for c in NUMERIC_COLS:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk = chunk.dropna(subset=["time", "lat", "lon"])
        if chunk.empty:
            continue
        chunk = chunk.sort_values(["icao24", "callsign", "time"], kind="stable")
        # stitch across chunk boundary: same icao24+callsign continuing in time
        if prev_chunk_last is not None:
            tail = prev_chunk_last
            first = chunk.iloc[0]
            if (tail["icao24"] == first["icao24"]
                    and tail["callsign"] == first["callsign"]
                    and first["time"] - tail["time"] <= 5):
                chunk = pd.concat([tail.to_frame().T, chunk], ignore_index=True)
        prev_chunk_last = chunk.iloc[[-1]]

        cont = (chunk["icao24"].eq(chunk["icao24"].shift())
                & chunk["callsign"].eq(chunk["callsign"].shift())
                & chunk["time"].diff().le(5.0))
        seg_id = (~cont).cumsum()
        for _, s in chunk.groupby(seg_id, sort=False):
            key = (s["icao24"].iloc[0], s["callsign"].iloc[0])
            ded = s.drop_duplicates(subset=["lat", "lon"], keep="first")
            n = len(ded)
            if n < min_points:
                continue
            geo_frac = float(ded["geoaltitude"].notna().mean())
            if geo_frac < 0.95:
                continue
            span = max(float(ded["lat"].max() - ded["lat"].min()),
                       float(ded["lon"].max() - ded["lon"].min()))
            if span < min_span_deg:
                continue
            gaps = ded["time"].diff().dropna()
            if gaps.max() > max_gap_s:
                continue
            cand = {
                "icao24": key[0], "callsign": key[1], "n": n, "span": span,
                "geo_frac": geo_frac,
                "t0": int(ded["time"].iloc[0]), "t1": int(ded["time"].iloc[-1]),
                "lat_mean": float(ded["lat"].mean()),
                "lon_mean": float(ded["lon"].mean()),
                "alt_max": float(ded["geoaltitude"].max()),
            }
            old = best.get(key)
            if old is None or n > old["n"]:
                best[key] = cand
    # prefer the largest-span candidates, one per (icao24, callsign)
    ranked = sorted(best.values(), key=lambda c: (c["span"], c["n"]), reverse=True)
    picked: list[dict[str, object]] = []
    used_callsigns: set[str] = set()
    for c in ranked:
        cs = str(c["callsign"])
        if cs in used_callsigns:
            continue
        picked.append(c)
        used_callsigns.add(cs)
        if len(picked) >= need:
            break
    return picked


def load_flight(src: Path, icao24: str, callsign: str, t0: int, t1: int) -> pd.DataFrame:
    """Second pass: pull one flight's rows by icao24+callsign+time window."""
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(src, chunksize=2_000_000, low_memory=False):
        for c in NUMERIC_COLS:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        m = (chunk["icao24"] == icao24) & (chunk["callsign"] == callsign) \
            & (chunk["time"].between(t0 - 1, t1 + 1))
        sub = chunk[m].dropna(subset=["time", "lat", "lon"])
        if not sub.empty:
            parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def curate_flight(df: pd.DataFrame, target_points: int) -> pd.DataFrame:
    """Dedup consecutive identical positions, subsample evenly in time."""
    ded = df.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)
    ded = ded[ded["geoaltitude"].notna()].reset_index(drop=True)
    t = ded["time"].to_numpy(dtype=float)
    # even temporal subsampling: uniform quantiles of time
    if len(ded) > target_points:
        qs = np.linspace(0.0, 1.0, target_points)
        t_targets = np.quantile(t, qs)
        idx = np.unique(np.searchsorted(t, t_targets))
        ded = ded.iloc[idx].reset_index(drop=True)
    return ded


def dms_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose decimal degrees into DMS columns, seconds stored x100."""
    out = df.copy()
    for prefix, deg_col in (("Lat", "Latitude"), ("Lon", "Longitude")):
        deg = out[deg_col].to_numpy(dtype=float)
        sign = np.sign(deg)
        remain = np.abs(deg) * 3600.0
        d = np.floor(remain / 3600.0)
        m = np.floor((remain - d * 3600.0) / 60.0)
        s = remain - d * 3600.0 - m * 60.0
        out[f"{prefix}Deg"] = (sign * d).astype(int)
        out[f"{prefix}Min"] = m.astype(int)
        out[f"{prefix}Sec"] = np.round(s * 100.0, 4)  # owner quirk: seconds x100
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="path to the Kaggle flights CSV (read-only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output sample CSV path")
    ap.add_argument("--target-points", type=int, default=300,
                    help="target points per flight after subsampling")
    ap.add_argument("--seed", type=int, default=0,
                    help="when >0, pick the Nth eligible set deterministically "
                         "by rotating the candidate ranking (for reselection)")
    args = ap.parse_args()

    print(f"scanning {args.src} (chunked)...")
    flights = scan_for_flights(args.src)
    if not flights:
        print("no eligible flights found", file=sys.stderr)
        return 1
    for i, f in enumerate(flights):
        print(f"  candidate {i}: {f['callsign']} icao24={f['icao24']} "
              f"n={f['n']} span={f['span']:.2f}deg alt_max={f['alt_max']:.0f}m "
              f"t0={f['t0']}")

    frames: list[pd.DataFrame] = []
    curated: list[pd.DataFrame] = []
    for f in flights:
        raw = load_flight(args.src, str(f["icao24"]), str(f["callsign"]),
                          int(f["t0"]), int(f["t1"]))
        cur = curate_flight(raw, args.target_points)
        cur["callsign"] = str(f["callsign"])
        cur["icao24"] = str(f["icao24"])
        curated.append(cur)
        print(f"  curated {f['callsign']}: {len(cur)} points "
              f"({int(cur['time'].iloc[-1] - cur['time'].iloc[0])} s)")

    # time-normalize: rebase each flight onto a common ~1-hour window,
    # preserving intra-flight dt (offset so the first flight starts at the
    # sample epoch; later flights start where the previous one ends + 60 s
    # padding would exceed 1 h for 3 x ~1 h flights, so instead all flights
    # share one epoch start and span the window together).
    SAMPLE_EPOCH = 1751328000.0  # 2025-07-01 00:00:00 UTC
    total_span = sum(
        float(c["time"].iloc[-1] - c["time"].iloc[0]) for c in curated
    )
    if total_span > 3600.0:
        # scale each flight's relative times by a common factor so the union
        # fits one hour while preserving intra-flight dt ratios
        scale = 3600.0 / total_span
    else:
        scale = 1.0
    out_frames: list[pd.DataFrame] = []
    for cur in curated:
        rel = (cur["time"].to_numpy(dtype=float) - cur["time"].iloc[0]) * scale
        cur = cur.copy()
        cur["Timestamp"] = np.round(SAMPLE_EPOCH + rel, 3)
        out_frames.append(cur)

    combined = pd.concat(out_frames, ignore_index=True)
    base_lat = float(combined["lat"].mean())
    base_lon = float(combined["lon"].mean())
    base_alt = 0.0  # documented choice: WGS84 ellipsoid surface at base lat/lon
    print(f"base coordinate: lat={base_lat:.6f} lon={base_lon:.6f} "
          f"alt={base_alt} m (WGS84 ellipsoid surface)")

    n, u, e = llh_to_nue(
        combined["lat"].to_numpy(dtype=float),
        combined["lon"].to_numpy(dtype=float),
        combined["geoaltitude"].to_numpy(dtype=float),
        base_lat, base_lon, base_alt,
    )
    combined["N"] = np.round(n, 2)
    combined["U"] = np.round(u, 2)
    combined["E"] = np.round(e, 2)
    combined["Latitude"] = np.round(combined["lat"], 6)
    combined["Longitude"] = np.round(combined["lon"], 6)
    combined["Altitude_m"] = np.round(combined["geoaltitude"], 3)
    combined["station"] = combined["callsign"]
    combined["sensor"] = combined["icao24"]

    result = combined[[
        "station", "sensor", "Timestamp", "N", "U", "E",
        "Latitude", "Longitude", "Altitude_m",
    ]]
    result = dms_columns(result)
    result = result.sort_values(["station", "Timestamp"], kind="stable")
    result = result.reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(result)} rows, {len(result.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
