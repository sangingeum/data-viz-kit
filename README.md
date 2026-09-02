# data-viz-kit

Data analysis and visualization toolkit (v2). Provides a small analysis API
over numpy/pandas/polars with matplotlib/seaborn/plotly output, plus
spreadsheet (Excel) and web-scraping utilities.

Requires **Python 3.12** (`>=3.12,<3.13`).

## Purpose

This toolkit is intended for migration to an **air-gapped environment**.
Migration itself is handled by the owner; this repository deliberately
contains no offline-install or wheelhouse instructions or artifacts.

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
- `examples/csv_viewer_demo.py` — matplotlib CSV viewer demo (uses `examples/sample_data.csv`)
- `examples/csv_viewer_demo_streamlit.py` — Streamlit version of the CSV viewer demo
- `scripts/smoke_check.py` — dependency smoke test
- `tests/` — pytest suite

## Streamlit CSV viewer

`examples/csv_viewer_demo_streamlit.py` is a Streamlit port of
`examples/csv_viewer_demo.py`: upload a CSV (or load the bundled
`examples/sample_data.csv`), pick the timestamp and value columns, and drag an
interactive `[t1, t2]` range slider to window the data — only rows inside the
window are plotted (`st.line_chart`) and tabulated (`st.dataframe`).

```bash
uv run streamlit run examples/csv_viewer_demo_streamlit.py --server.headless true
# then open http://localhost:8501
```

Note: Streamlit charts do not support matplotlib-style hover tooltips; exact
values are available in the filtered-data table.

## License

MIT — see `LICENSE`.
