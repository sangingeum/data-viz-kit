"""Interactive CSV coordinate viewer with time-range sliders, identifier
check-box filtering, and hover tooltips.

Expected CSV layout
-------------------
The first row is a header.  Subsequent rows contain:

* **Two (or more) identifier columns** – concatenated with ``"+"`` to form a
  per-row identifier string (e.g. ``"StationA+Sensor1"``).
* **A timestamp column** – epoch-seconds as ``float`` (fractional OK).
* **Coordinate columns** – typically ``N``, ``U``, ``E`` in metres.

Public API
----------
``load_csv``  – read a CSV file and prepare the ``_identifier`` column.
``view_csv``  – open an interactive matplotlib window.
"""

from __future__ import annotations

import os
from typing import Sequence


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(
    path: str | os.PathLike,
    id_cols: Sequence[str],
    time_col: str = "Timestamp",
    coord_cols: Sequence[str] = ("N", "U", "E"),
):
    """Read *path* and return a :class:`~pandas.DataFrame` with an
    ``_identifier`` column built by joining *id_cols* with ``"+"``.

    Raises :exc:`KeyError` if any required column is missing.
    """
    import pandas as pd

    df = pd.read_csv(path)

    required = list(id_cols) + [time_col] + list(coord_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in CSV: {missing}")

    df["_identifier"] = df[id_cols[0]].astype(str)
    for col in id_cols[1:]:
        df["_identifier"] = df["_identifier"] + "+" + df[col].astype(str)

    return df


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def view_csv(
    df,  # pandas.DataFrame
    id_cols: Sequence[str],
    time_col: str = "Timestamp",
    coord_cols: Sequence[str] = ("N", "U", "E"),
    t1: float | None = None,
    t2: float | None = None,
    title: str = "CSV Coordinate Viewer",
    show_tooltips: bool = True,
    block: bool = True,
):
    """Open an interactive window showing coordinate data from *df*.

    Four plots are shown in a 2×2 grid:

    * **Plot 1** (top-left) — 3-D scatter of all three coordinates.
    * **Plot 2** (top-right) — 2-D scatter of the first two coordinates.
    * **Plot 3** (bottom-left) — 2-D scatter of the first and third.
    * **Plot 4** (bottom-right) — 2-D scatter of the third and second.

    Two sliders filter the visible time window ``[t1, t2]``.
    ``CheckButtons`` on the right toggle visibility per identifier.  Hover
    tooltips display the identifier label and the full row values.

    *coord_cols* must contain exactly three column names (default
    ``("N", "U", "E")``).

    Returns ``(figure, viewer)`` where *viewer* is a ``dict`` exposing
    ``get_visible_range()`` and ``get_visible_identifiers()`` for
    programmatic / headless inspection.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.widgets import Slider, CheckButtons

    from data_viz_kit.timeline_viewer import ensure_interactive_backend

    ensure_interactive_backend()

    # ---- ensure _identifier column exists --------------------------------
    if "_identifier" not in df.columns:
        df = df.copy()
        df["_identifier"] = df[id_cols[0]].astype(str)
        for col in id_cols[1:]:
            df["_identifier"] = df["_identifier"] + "+" + df[col].astype(str)

    # ---- validate columns ------------------------------------------------
    coord_cols = list(coord_cols)
    if len(coord_cols) != 3:
        raise ValueError(
            f"coord_cols must have exactly 3 elements, got {len(coord_cols)}"
        )
    c_n, c_u, c_e = coord_cols

    required = [time_col] + coord_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in DataFrame: {missing}")

    # ---- timestamp range -------------------------------------------------
    all_times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    finite_times = all_times[np.isfinite(all_times)]
    if finite_times.size == 0:
        raise ValueError("no finite timestamps in time column")

    data_min, data_max = float(finite_times.min()), float(finite_times.max())
    t1 = data_min if t1 is None else float(t1)
    t2 = data_max if t2 is None else float(t2)
    t1, t2 = min(t1, t2), max(t1, t2)

    # ---- unique identifiers & colours ------------------------------------
    identifiers = sorted(df["_identifier"].unique())
    cmap = plt.colormaps["tab10"] if len(identifiers) <= 10 else plt.colormaps["tab20"]
    colors = {ident: cmap(i % cmap.N) for i, ident in enumerate(identifiers)}

    # ---- figure layout (2×2 + right checkboxes + bottom sliders) ---------
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.07, right=0.72, top=0.93, bottom=0.16,
        hspace=0.32, wspace=0.28,
    )
    fig.suptitle(title, fontsize=13)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_nu = fig.add_subplot(gs[0, 1])
    ax_ne = fig.add_subplot(gs[1, 0])
    ax_eu = fig.add_subplot(gs[1, 1])

    plot_axes_2d = [ax_nu, ax_ne, ax_eu]

    # axis labels & titles
    ax_3d.set_xlabel(f"{c_n} (m)", fontsize=8, labelpad=4)
    ax_3d.set_ylabel(f"{c_u} (m)", fontsize=8, labelpad=4)
    ax_3d.set_zlabel(f"{c_e} (m)", fontsize=8, labelpad=4)
    ax_3d.set_title(f"3D {c_n}{c_u}{c_e}", fontsize=10)
    ax_3d.tick_params(labelsize=7)

    ax_nu.set_xlabel(f"{c_n} (m)"); ax_nu.set_ylabel(f"{c_u} (m)")
    ax_nu.set_title(f"{c_n}–{c_u}", fontsize=10)

    ax_ne.set_xlabel(f"{c_n} (m)"); ax_ne.set_ylabel(f"{c_e} (m)")
    ax_ne.set_title(f"{c_n}–{c_e}", fontsize=10)

    ax_eu.set_xlabel(f"{c_e} (m)"); ax_eu.set_ylabel(f"{c_u} (m)")
    ax_eu.set_title(f"{c_e}–{c_u}", fontsize=10)

    for ax in plot_axes_2d:
        ax.grid(True, alpha=0.3)

    # ---- draw markers per identifier on all 4 plots ----------------------
    lines_3d: dict[str, matplotlib.lines.Line2D] = {}
    lines_nu: dict[str, matplotlib.lines.Line2D] = {}
    lines_ne: dict[str, matplotlib.lines.Line2D] = {}
    lines_eu: dict[str, matplotlib.lines.Line2D] = {}

    for ident in identifiers:
        c = colors[ident]
        kw = dict(marker=".", markersize=3, linestyle="None", color=c,
                  label=ident, alpha=0.7)
        (l,) = ax_3d.plot([], [], [], **kw)
        lines_3d[ident] = l
        (l,) = ax_nu.plot([], [], **kw)
        lines_nu[ident] = l
        (l,) = ax_ne.plot([], [], **kw)
        lines_ne[ident] = l
        (l,) = ax_eu.plot([], [], **kw)
        lines_eu[ident] = l

    # De-duplicate legend (one entry per identifier, on the 3-D axes)
    handles = [lines_3d[ident] for ident in identifiers]
    ax_3d.legend(
        handles=handles, labels=identifiers,
        loc="upper left", fontsize=7,
        ncol=max(1, len(identifiers) // 8),
    )

    # coordinate-pair mapping for each 2-D axes
    ax_2d_info: dict[int, tuple[str, str, dict]] = {
        id(ax_nu): (c_n, c_u, lines_nu),
        id(ax_ne): (c_n, c_e, lines_ne),
        id(ax_eu): (c_e, c_u, lines_eu),
    }

    # ---- CheckButtons (right panel) --------------------------------------
    check_height = min(0.77, 0.035 * len(identifiers) + 0.05)
    check_bottom = 0.16 + (0.77 - check_height) / 2
    ax_check = fig.add_axes([0.75, check_bottom, 0.23, check_height])
    ax_check.set_title("Identifiers", fontsize=9, pad=6)
    check = CheckButtons(ax_check, identifiers, [True] * len(identifiers))

    label_fontsize = min(9, max(6, int(200 / max(len(identifiers), 1))))
    for i, ident in enumerate(identifiers):
        check.labels[i].set_color(colors[ident])
        check.labels[i].set_fontsize(label_fontsize)

    # ---- sliders (bottom) ------------------------------------------------
    ax_slider_t1 = fig.add_axes((0.07, 0.07, 0.63, 0.03))
    ax_slider_t2 = fig.add_axes((0.07, 0.02, 0.63, 0.03))
    slider_t1 = Slider(ax_slider_t1, "t1", data_min, data_max,
                        valinit=t1, valfmt="%.3f")
    slider_t2 = Slider(ax_slider_t2, "t2", data_min, data_max,
                        valinit=t2, valfmt="%.3f")

    # ---- viewer state dict -----------------------------------------------
    viewer: dict[str, object] = {
        "t1": t1,
        "t2": t2,
        "visible_ids": set(identifiers),
        "tooltips_enabled": show_tooltips,
    }

    # ---- redraw helper ---------------------------------------------------
    def _redraw() -> None:
        a, b = sorted((float(viewer["t1"]), float(viewer["t2"])))  # type: ignore[arg-type]

        all_n: list[float] = []
        all_u: list[float] = []
        all_e: list[float] = []

        for ident in identifiers:
            visible = ident in viewer["visible_ids"]  # type: ignore[operator]
            sub = df[df["_identifier"] == ident]
            times = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
            n_vals = pd.to_numeric(sub[c_n], errors="coerce").to_numpy(dtype=float)
            u_vals = pd.to_numeric(sub[c_u], errors="coerce").to_numpy(dtype=float)
            e_vals = pd.to_numeric(sub[c_e], errors="coerce").to_numpy(dtype=float)

            if not visible:
                lines_3d[ident].set_data_3d([], [], [])
                lines_nu[ident].set_data([], [])
                lines_ne[ident].set_data([], [])
                lines_eu[ident].set_data([], [])
                continue

            mask = (
                (times >= a) & (times <= b)
                & np.isfinite(times)
                & np.isfinite(n_vals) & np.isfinite(u_vals) & np.isfinite(e_vals)
            )
            n_f, u_f, e_f = n_vals[mask], u_vals[mask], e_vals[mask]

            lines_3d[ident].set_data_3d(n_f, u_f, e_f)
            lines_nu[ident].set_data(n_f, u_f)
            lines_ne[ident].set_data(n_f, e_f)
            lines_eu[ident].set_data(e_f, u_f)

            all_n.extend(n_f.tolist())
            all_u.extend(u_f.tolist())
            all_e.extend(e_f.tolist())

        # autoscale 2-D axes
        for ax in plot_axes_2d:
            ax.relim()
            ax.autoscale_view()

        # autoscale 3-D axes (relim/autoscale_view not supported for 3-D)
        if all_n:
            def _lim(vals: list[float]) -> tuple[float, float]:
                lo, hi = min(vals), max(vals)
                span = (hi - lo) or 1.0
                m = 0.05 * span
                return lo - m, hi + m
            ax_3d.set_xlim3d(*_lim(all_n))
            ax_3d.set_ylim3d(*_lim(all_u))
            ax_3d.set_zlim3d(*_lim(all_e))

        fig.canvas.draw_idle()

    # ---- slider callbacks ------------------------------------------------
    def _on_t1(_val: float) -> None:
        viewer["t1"] = slider_t1.val
        _redraw()

    def _on_t2(_val: float) -> None:
        viewer["t2"] = slider_t2.val
        _redraw()

    slider_t1.on_changed(_on_t1)
    slider_t2.on_changed(_on_t2)

    # ---- checkbox callback -----------------------------------------------
    def _on_check(label: str) -> None:
        vis = viewer["visible_ids"]  # type: ignore[assignment]
        if label in vis:
            vis.discard(label)  # type: ignore[union-attr]
        else:
            vis.add(label)  # type: ignore[union-attr]
        _redraw()

    check.on_clicked(_on_check)

    # ---- hover tooltip (single figure-level text) ------------------------
    tooltip = fig.text(
        0, 0, "",
        fontsize=8,
        visible=False,
        bbox={"boxstyle": "round,pad=0.4", "fc": "lemonchiffon", "alpha": 0.95},
        zorder=999,
    )

    def _find_row(ident: str, **coord_values: float):
        """Return the DataFrame row closest to the given coordinate values."""
        sub = df[df["_identifier"] == ident]
        dist = np.zeros(len(sub))
        for col, val in coord_values.items():
            arr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            dist = dist + (arr - val) ** 2
        return sub.iloc[int(np.argmin(dist))]

    def _build_tip(ident: str, row) -> str:
        parts = [f"[{ident}]", f"{time_col} = {row[time_col]} s"]
        for c in coord_cols:
            parts.append(f"{c} = {row[c]} m")
        return "\n".join(parts)

    def _nearest_2d(event, lines_dict: dict) -> tuple[float, str, float, float] | None:
        ax = event.inaxes
        dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1.0
        dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        best: tuple[float, str, float, float] | None = None
        for ident in identifiers:
            if ident not in viewer["visible_ids"]:  # type: ignore[operator]
                continue
            line = lines_dict[ident]
            xs, ys = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
            if xs.size == 0:
                continue
            d = ((xs - event.xdata) / dx) ** 2 + ((ys - event.ydata) / dy) ** 2
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), ident, float(xs[i]), float(ys[i]))
        return best

    def _nearest_3d(event) -> tuple[float, str, float, float, float] | None:
        from mpl_toolkits.mplot3d import proj3d  # noqa: F811

        best: tuple[float, str, float, float, float] | None = None
        for ident in identifiers:
            if ident not in viewer["visible_ids"]:  # type: ignore[operator]
                continue
            line = lines_3d[ident]
            try:
                xs, ys, zs = line.get_data_3d()
            except AttributeError:  # fallback for older matplotlib
                xs, ys, zs = getattr(line, "_verts3d", ([], [], []))
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            zs = np.asarray(zs, dtype=float)
            if xs.size == 0:
                continue
            # project 3-D → 2-D display coordinates
            x2, y2, _ = proj3d.proj_transform(xs, ys, zs, ax_3d.get_proj())
            disp = ax_3d.transData.transform(np.column_stack([x2, y2]))
            d = (disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), ident, float(xs[i]), float(ys[i]), float(zs[i]))
        return best

    def _on_motion(event: matplotlib.backend_bases.MouseEvent) -> None:
        tooltip.set_visible(False)

        if not viewer["tooltips_enabled"] or event.inaxes is None:
            fig.canvas.draw_idle()
            return

        row = None

        if event.inaxes is ax_3d:
            result = _nearest_3d(event)
            if result is not None and result[0] < 400:  # 20 px radius
                _, ident, nv, uv, ev = result
                row = _find_row(ident, **{c_n: nv, c_u: uv, c_e: ev})
        elif id(event.inaxes) in ax_2d_info:
            x_col, y_col, lines_dict = ax_2d_info[id(event.inaxes)]
            result = _nearest_2d(event, lines_dict)
            if result is not None and result[0] < 0.005:
                _, ident, xv, yv = result
                row = _find_row(ident, **{x_col: xv, y_col: yv})

        if row is None:
            fig.canvas.draw_idle()
            return

        inv = fig.transFigure.inverted()
        fx, fy = inv.transform((event.x + 15, event.y + 15))
        tooltip.set_position((min(fx, 0.92), min(fy, 0.97)))
        tooltip.set_text(_build_tip(ident, row))  # type: ignore[possibly-undefined]
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)

    # ---- initial draw & public helpers -----------------------------------
    _redraw()

    def get_visible_range() -> tuple[float, float]:
        return (float(viewer["t1"]), float(viewer["t2"]))  # type: ignore[arg-type]

    def get_visible_identifiers() -> list[str]:
        return sorted(viewer["visible_ids"])  # type: ignore[arg-type]

    viewer["get_visible_range"] = get_visible_range
    viewer["get_visible_identifiers"] = get_visible_identifiers
    viewer["sliders"] = {"t1": slider_t1, "t2": slider_t2}
    viewer["check"] = check

    if block:
        plt.show()
    return fig, viewer

