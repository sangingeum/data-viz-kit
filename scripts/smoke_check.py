#!/usr/bin/env python3
"""Smoke check: import every declared dependency and do a trivial operation with each.

Exit code 0 = all pass; prints a per-package PASS/FAIL line.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((name, True, detail))
        print(f"PASS {name}: {detail}")
    except Exception as exc:  # noqa: BLE001 - smoke test must report everything
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")


def main() -> int:
    import numpy as np

    check("numpy", lambda: f"array sum={float(np.arange(5).sum())}")

    import pandas as pd

    check("pandas", lambda: f"DataFrame shape={pd.DataFrame({'a': [1, 2]}).shape}")

    import polars as pl

    check("polars", lambda: f"sum={pl.DataFrame({'a': [1, 2, 3]})['a'].sum()}")

    import pyarrow as pa
    import pyarrow.compute as pc

    check(
        "pyarrow",
        lambda: f"max={pc.max(pa.array([1, 5, 3])).as_py()}",
    )

    import scipy.stats as st

    check("scipy", lambda: f"norm pdf(0)={st.norm.pdf(0):.4f}")

    import statsmodels.api as sm

    _model = sm.OLS([1.0, 2.0, 3.0], sm.add_constant([1.0, 2.0, 3.0])).fit()
    check("statsmodels", lambda: f"OLS slope={_model.params[1]:.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _mpl() -> str:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return f"png bytes={len(buf.getvalue())}"

    check("matplotlib", _mpl)

    import seaborn as sns

    def _sns() -> str:
        tips_like = pd.DataFrame({"x": range(10), "y": [i * 2 for i in range(10)]})
        ax = sns.lineplot(data=tips_like, x="x", y="y")
        return f"lineplot drawn, lines={len(ax.lines)}"

    check("seaborn", _sns)

    import plotly.graph_objects as go

    def _plotly() -> str:
        fig = go.Figure(go.Bar(x=["a"], y=[1]))
        return f"fig has {len(fig.data)} trace"

    check("plotly", _plotly)

    def _openpyxl() -> str:
        from openpyxl import Workbook

        with TemporaryDirectory() as d:
            p = Path(d) / "t.xlsx"
            wb = Workbook()
            wb.active["A1"] = "hello"
            wb.save(p)
            return f"xlsx bytes={p.stat().st_size}"

    check("openpyxl", _openpyxl)

    def _xlsxwriter() -> str:
        import xlsxwriter

        buf = io.BytesIO()
        wb = xlsxwriter.Workbook(buf, {"in_memory": True})
        ws = wb.add_worksheet()
        ws.write(0, 0, "hi")
        wb.close()
        return f"xlsx bytes={len(buf.getvalue())}"

    check("xlsxwriter", _xlsxwriter)

    import tabulate

    check("tabulate", lambda: tabulate.tabulate([["a", 1]], headers=["k", "v"])[:12])

    from bs4 import BeautifulSoup

    check(
        "beautifulsoup4",
        lambda: f"parsed title={BeautifulSoup('<title>t</title>', 'html.parser').title.string}",
    )

    from lxml import etree

    check("lxml", lambda: f"root={etree.fromstring(b'<a/>').tag}")

    import requests

    check("requests", lambda: f"version={requests.__version__} (no network call)")

    import sklearn.linear_model

    def _sklearn() -> str:
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 2.0, 4.0]
        model = sklearn.linear_model.LinearRegression().fit(X, y)
        return f"coef={model.coef_[0]:.4f}"

    check("scikit-learn", _sklearn)

    import jupyterlab  # noqa: F401

    check("jupyterlab", lambda: "imported ok")

    import ipykernel

    check("ipykernel", lambda: f"version={ipykernel.__version__}")

    import pytest

    check("pytest", lambda: f"version={pytest.__version__}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} packages passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
