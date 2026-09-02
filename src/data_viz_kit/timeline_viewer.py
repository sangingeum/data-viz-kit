"""Interactive time-range viewer built on matplotlib widgets.

Timestamps are epoch seconds as floats (e.g. ``1725270000.5``). Two
matplotlib ``widgets.Slider`` objects select the visible window
``[t1, t2]``; the axes re-filter the data on every change event. Hover
tooltips (motion-notify_event) show series name plus the exact
``x=timestamp, y=value`` of the nearest data point.

Note: interactive sliders require a GUI backend (QtAgg/TkAgg), not Agg.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

GUI_BACKENDS = ("QtAgg", "TkAgg", "GTK3Agg", "MacOSX")


def ensure_interactive_backend() -> str:
    """Pick an available interactive backend; no-op if one is already active."""
    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        import matplotlib.backends  # noqa: F401

        for name in GUI_BACKENDS:
            try:
                matplotlib.use(name)
                return matplotlib.get_backend()
            except Exception:  # noqa: BLE001 - try next backend
                continue
    return matplotlib.get_backend()


def filter_time_range(
    df,  # pandas.DataFrame
    time_col: str,
    t1: float | None,
    t2: float | None,
):
    """Return rows with time_col in [t1, t2] (inclusive bounds)."""
    import pandas as pd

    if time_col not in df.columns:
        raise KeyError(f"time column {time_col!r} not in DataFrame")
    import numpy as np

    times = np.asarray(pd.to_numeric(pd.Series(df[time_col]), errors="coerce"), dtype=float)
    finite = np.isfinite(times)
    if t1 is None:
        t1 = float(times[finite].min())
    if t2 is None:
        t2 = float(times[finite].max())
    t1, t2 = min(t1, t2), max(t1, t2)
    mask = finite & (times >= t1) & (times <= t2)
    return df.loc[mask], t1, t2


def format_tooltip(series_name: str, t: float, y: float, timestamp_format: str = "raw") -> str:
    """Format hover tooltip text: series name plus exact x/y values."""
    if timestamp_format == "datetime":
        import datetime as _dt

        t_str = _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    else:
        t_str = f"{t}"
    return f"{series_name}\nx = {t_str}\ny = {y:.6g}"


def view_time_range(
    df,  # pandas.DataFrame
    time_col: str,
    value_cols: Sequence[str],
    t1: float | None = None,
    t2: float | None = None,
    title: str = "Time range viewer",
    timestamps_are_utc: bool = False,
    show_tooltips: bool = True,
    block: bool = True,
):
    """Open an interactive window showing only data within [t1, t2].

    Two sliders (t1/t2) update the plotted range live; hover tooltips
    show series name and exact x/y values (toggle with `show_tooltips`).

    Returns ``(figure, viewer)`` where the viewer exposes
    ``get_visible_range()`` for programmatic inspection (useful headless).
    """
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    ensure_interactive_backend()

    for col in [time_col, *value_cols]:
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in DataFrame")

    all_times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    all_times = all_times[np.isfinite(all_times)]
    if all_times.size == 0:
        raise ValueError("no finite timestamps in time_col")
    data_min, data_max = float(all_times.min()), float(all_times.max())
    t1 = data_min if t1 is None else float(t1)
    t2 = data_max if t2 is None else float(t2)
    t1, t2 = min(t1, t2), max(t1, t2)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.09, right=0.91, top=0.92, bottom=0.24)

    lines: dict[str, matplotlib.lines.Line2D] = {}
    # Decide axes placement BEFORE creating lines so each line lives on
    # its proper axes (twin if scales diverge).
    axes_by_col: dict[str, matplotlib.axes.Axes] = {col: ax for col in value_cols}
    if len(value_cols) > 1:
        scales = []
        for col in value_cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            scales.append(abs(float(vals.mean())) if vals.size else 1.0)
        if max(scales) / max(min(scales), 1e-12) > 20:
            twin = ax.twinx()
            axes_by_col[value_cols[-1]] = twin
            twin.set_ylabel(value_cols[-1])
            twin.grid(False)

    for col in value_cols:
        (line,) = axes_by_col[col].plot([], [], lw=1.2, label=col)
        lines[col] = line

    ax.set_xlim(data_min, data_max)
    # Per-axis y-limits: main axes from its own series, twin from its own.
    def _set_initial_ylim(target: matplotlib.axes.Axes, cols: list[str]) -> None:
        vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        lo = float(vals.min()) if vals.size else 0.0
        hi = float(vals.max()) if vals.size else 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        target.set_ylim(lo, hi)

    main_cols = [col for col in value_cols if axes_by_col[col] is ax]
    twin_cols = [col for col in value_cols if axes_by_col[col] is not ax]
    _set_initial_ylim(ax, main_cols)
    for twin_col in twin_cols:
        _set_initial_ylim(axes_by_col[twin_col], [twin_col])
        axes_by_col[twin_col].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:.1f}")
        )
    ax.set_xlabel(time_col)
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax_slider_1 = fig.add_axes((0.12, 0.12, 0.76, 0.03))
    ax_slider_2 = fig.add_axes((0.12, 0.06, 0.76, 0.03))
    slider_t1 = Slider(ax_slider_1, "t1", data_min, data_max, valinit=t1, valfmt="%.1f")
    slider_t2 = Slider(ax_slider_2, "t2", data_min, data_max, valinit=t2, valfmt="%.1f")

    viewer: dict[str, object] = {"t1": t1, "t2": t2, "tooltips_enabled": show_tooltips}

    def _redraw() -> None:
        a, b = sorted((float(viewer["t1"]), float(viewer["t2"])))  # type: ignore[arg-type]
        for col, line in lines.items():
            times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
            values = df[col].to_numpy(dtype=float)
            mask = (times >= a) & (times <= b) & np.isfinite(times) & np.isfinite(values)
            line.set_data(times[mask], values[mask])
            target = axes_by_col[col]
            target.relim()
            target.autoscale_view(scalex=False)
        ax.set_xlim(a, b)
        fig.canvas.draw_idle()

    def _on_t1(_val: float) -> None:
        viewer["t1"] = slider_t1.val
        _redraw()

    def _on_t2(_val: float) -> None:
        viewer["t2"] = slider_t2.val
        _redraw()

    slider_t1.on_changed(_on_t1)
    slider_t2.on_changed(_on_t2)

    # ---- hover tooltips -------------------------------------------------
    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.4", "fc": "lemonchiffon", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=8,
        visible=False,
        annotation_clip=False,
    )

    def _on_motion(event: matplotlib.backend_bases.MouseEvent) -> None:
        # Only show tooltips when the cursor is over a plotting axes.
        plot_axes = {ax_ for ax_ in axes_by_col.values()}
        if not viewer["tooltips_enabled"] or event.inaxes not in plot_axes:  # type: ignore[operator]
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return
        best: tuple[float, float, str, float, float] | None = None
        for col, line in lines.items():
            if axes_by_col[col] is not event.inaxes:
                continue
            xs, ys = line.get_xdata(), line.get_ydata()
            if len(xs) == 0:
                continue
            xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            dist = (xs_a - event.xdata) ** 2 + (ys_a - event.ydata) ** 2  # type: ignore[operator]
            i = int(np.argmin(dist))
            if best is None or dist[i] < best[0]:
                best = (float(dist[i]), xs_a[i], col, ys_a[i], 0.0)
        if best is None or best[0] > (ax.get_xbound()[1] - ax.get_xbound()[0]) ** 2 * 0.01:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return
        _, t_val, col, y_val, _ = best
        if timestamps_are_utc:
            t_str = format_tooltip(col, t_val, y_val, timestamp_format="datetime")
        else:
            t_str = format_tooltip(col, t_val, y_val)
        annotation.set_text(t_str)
        annotation.xy = (t_val, y_val)
        annotation.axes = axes_by_col[col]
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)

    def get_visible_range() -> tuple[float, float]:
        return (float(viewer["t1"]), float(viewer["t2"]))  # type: ignore[arg-type]

    viewer["get_visible_range"] = get_visible_range
    viewer["sliders"] = {"t1": slider_t1, "t2": slider_t2}
    _redraw()

    if block:
        plt.show()
    return fig, viewer
