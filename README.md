# data-viz-kit

Data analysis and visualization toolkit (v2). Provides a small analysis API
over numpy/pandas/polars with matplotlib/seaborn/plotly output, plus
spreadsheet (Excel) and web-scraping utilities.

Requires **Python 3.12** (`>=3.12,<3.13`).


## Interactive timeline viewer

`data_viz_kit.view_time_range(df, time_col, value_cols, ...)` opens an
interactive window for timestamped data (epoch **seconds as floats**, e.g.
`1725270000.5`):

- **Time-range sliders** — two matplotlib `widgets.Slider` handles (`t1`, `t2`)
  select the visible window `[t1, t2]`. Every slider change re-filters the
  data and redraws only the points inside the window.
- **Hover tooltips** (toggle with `show_tooltips=False`) — hovering near a
  point shows a legend-style annotation with the series name and the exact
  `x = <timestamp>` / `y = <value>`. With `timestamps_are_utc=True` the
  timestamp is also rendered as a UTC date-time string.
- If series scales diverge strongly (e.g. ~20 vs ~1013), the last series is
  drawn on a twin y-axis automatically.

Interactive sliders need a GUI backend (QtAgg/TkAgg, not Agg);
`view_time_range` calls `ensure_interactive_backend()` to pick an available
one when the session is not headless.

```python
import pandas as pd
from data_viz_kit import view_time_range

df = pd.DataFrame({
    "timestamp": [1725270000.0, 1725270010.5, 1725270020.0],
    "temperature": [20.1, 20.3, 21.0],
})
view_time_range(df, "timestamp", ["temperature"], timestamps_are_utc=True)
```

Headless (Agg) sessions can still build the viewer programmatically:
`view_time_range(..., block=False)` returns `(figure, viewer)`; the viewer
dict exposes `get_visible_range()` and the `sliders` handles. See
`examples/timestamp_viewer_demo.py --headless` for a full render proof.

## Install / develop

```bash
uv sync            # creates .venv from uv.lock (Python 3.12)
uv run pytest      # test suite
uv run python examples/example_analysis.py            # headless demo, writes a PNG
uv run python examples/timestamp_viewer_demo.py       # interactive sliders + tooltips (GUI)
uv run python examples/timestamp_viewer_demo.py --headless   # headless render proof, writes a PNG
uv run python scripts/smoke_check.py         # imports & exercises every dependency
```

## Layout

- `src/data_viz_kit/` — library code (`timeline_viewer.py` = interactive viewer,
  `csv_viewer.py` = CSV coordinate viewer)
- `examples/example_analysis.py` — headless (Agg) matplotlib demo
- `examples/timestamp_viewer_demo.py` — timestamped-data demo (interactive + `--headless` mode)
- `examples/csv_viewer_demo.py` — matplotlib CSV viewer demo (uses `examples/sample_data.csv`;
  column names configurable via `--time-col/--id-cols/--n-col/--u-col/--e-col`)
- `examples/csv_viewer_demo_streamlit.py` — Streamlit version of the CSV viewer demo
- `scripts/smoke_check.py` — dependency smoke test
- `tests/` — pytest suite

## Streamlit CSV viewer

`examples/csv_viewer_demo_streamlit.py` is a full-parity Streamlit port of
`examples/csv_viewer_demo.py`: upload a CSV (or load the bundled
`examples/sample_data.csv`), pick the timestamp, identifier, and N/U/E
coordinate columns in the sidebar (auto-detected defaults mirroring the
matplotlib demo's CLI args), toggle identifiers with a multiselect, and drag
an interactive `[t1, t2]` range slider to window the data. The same 2x2 plot
grid is rendered (3D N/U/E scatter + N–U, N–E, E–U scatters) using **plotly**
(`st.plotly_chart`), and the filtered rows are tabulated with
`st.dataframe`.

```bash
uv run streamlit run examples/csv_viewer_demo_streamlit.py --server.headless true
# then open http://localhost:8501
```

Note: Streamlit cannot use matplotlib hover annotations, so the plotly charts
provide equivalent native hover tooltips instead — hovering a point shows the
identifier, the exact timestamp, and the exact N/U/E values.

### LLH (LLA) coordinate mode

The sidebar's **Coordinate mode** selector switches the viewer between the
existing `NUE` behaviour and `LLH` (latitude / longitude / altitude).
Identifier columns, the timestamp column, and the `[t1, t2]` range slider work
identically in both modes.

Two LLH input layouts are supported (picked with a sidebar radio, auto-detected
from column names):

- **3-col** — `Latitude`, `Longitude` (decimal degrees), `Altitude` (m).
  Auto-detected defaults: names containing `latitude`/`lat`,
  `longitude`/`lon`/`lng`, `altitude`/`alt`.
- **7-col (DMS)** — `LatDeg`, `LatMin` (1/60 deg), `LatSec` (1/60 min) and the
  `Lon*` equivalents, plus `Altitude` (m). Auto-detected defaults: names
  containing `lat_deg`/`latd`, `lat_min`/`latm`, `lat_sec`/`lats` (and the
  `lon*` analogues). Conversion:
  `deg = D + M/60 + S·seconds_scale/3600` for both lat and lon.

Every consumed column has a sidebar picker; selected columns are validated as
numeric (non-numeric cells are coerced to NaN and counted in a sidebar
warning).

**Seconds scale** — a sidebar selectbox (`1` default, or `0.01`) multiplies
`LatSec`/`LonSec` *before* conversion, for data where the seconds component is
pre-multiplied by 100. It applies only in the 7-col layout and is disabled
(greyed out) in 3-col mode.

**LLH visualization — trajectory viewer** — the primary view is an aircraft
**trajectory viewer** laid out as two compact rows of side-by-side panels
followed by the filtered data table:

- **Row 1 — Trajectory globe** (left, orthographic by default) next to the
  **Top-down trajectory map** (right, equirectangular projection). One
  markers-only trace per identifier; countries, coastlines, and land are
  shown; captions explain rotation/zoom and hovering.
- **Row 2 — 3D space view** (left, `go.Scatter3d`: x = lon, y = lat,
  z = alt_m) next to the **Altitude profile** (right, altitude against
  cumulative along-track distance in km).

Every view colours markers by **identifier only** — one shared identifier →
colour mapping (Plotly qualitative palette) is computed once and passed to
all four figures, so each aircraft keeps the same colour everywhere.

Readability options in the sidebar:

- **Globe projection** — orthographic (default), equirectangular, or natural
  earth.
- **Fit bounds to trajectories** (default ON) — sets the geo center plus
  projection scale (orthographic) or lat/lon axis ranges (flat projections)
  from the filtered data's bounding box with padding, so the view auto-zooms
  to the trajectories instead of the whole Earth.

Hover on any point keeps the identifier, the timestamp, its UTC date-time, and
the exact converted lat/lon/alt (per-point customdata).

The bundled `examples/sample_data.csv` carries extra LLH columns (3-col
`Latitude`/`Longitude`/`Altitude_m` and 7-col `LatDeg…LonSec`) around Seoul
(37.5N, 127.0E) — the demo works in LLH mode without uploading anything;
existing N/U/E columns are unchanged. The sample trajectories are **smooth**:
per-flight lateral curve offsets, GPS-level (~1 m) position noise, a smooth
climb/cruise/descent altitude envelope (~5 m noise, no >100 m jumps), and a
consistent ~1.75 km median point spacing for all three aircraft
(regenerate with `scripts/regenerate_sample.py`).

## License

MIT — see `LICENSE`.
