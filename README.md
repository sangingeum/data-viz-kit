# data-viz-kit

Data analysis and visualization toolkit (v2). Provides a small analysis API
over numpy/pandas/polars with matplotlib/seaborn/plotly output, plus
spreadsheet (Excel) and web-scraping utilities.

## Purpose

This toolkit is intended for migration to an **air-gapped environment**.
Migration itself is handled by the owner; this repository deliberately
contains no offline-install or wheelhouse instructions or artifacts.

## Install / develop

```bash
uv sync            # creates .venv from uv.lock
uv run pytest      # test suite
uv run python examples/example_analysis.py   # headless demo, writes a PNG
uv run python scripts/smoke_check.py         # imports & exercises every dependency
```

## Layout

- `src/data_viz_kit/` — library code
- `examples/example_analysis.py` — headless (Agg) matplotlib demo
- `scripts/smoke_check.py` — dependency smoke test
- `tests/` — pytest suite

## License

MIT — see `LICENSE`.
