"""sweeps

Sweep/metric figures: heatmaps, facets, metric-vs, scan overview.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..results import (
    line_metrics,
    records,
    selection_score,
)
from ._style import (
    COLORS,
    energy_color,
)

# ---- parametric heatmaps + parameter scans -----------------------------------
# Per case field: (axis label, divide-to-display, display unit, value format).
# Lets ANY swept knob be a heatmap/scan axis with sensible labels and units.
_AXIS_SPECS = {
    "tilt_deg": ("polar tilt", 1.0, "deg", "{:g}"),
    "tilt_azim_deg": ("azimuthal tilt", 1.0, "deg", "{:g}"),
    "E0_keV": ("beam energy", 1.0, "keV", "{:g}"),
    "thickness_ang": ("thickness", 1e4, r"$\mu$m", "{:g}"),
    "B_ang2": ("B-factor", 1.0, r"$\AA^2$", "{:g}"),
}

# (metric key, label + units, colormap)
_HEATMAP_QUANTITIES = [
    ("peak_flux", "peak spectral flux  (Phs/eV/s)", "viridis"),
    ("coherent_flux", "integrated coherent flux, all lines  (Phs/s)", "viridis"),
    ("line_flux", "integrated flux under the dominant line  (Phs/s)", "viridis"),
    ("line_eV", "dominant coherent line energy  (eV)", "plasma"),
    ("fwhm_eV", "dominant line FWHM  (eV)", "magma"),
    ("line_frac", "dominant line / total spectral flux", "cividis"),
    ("line_quality", "line-definition quality  (0-1)", "Greens"),
    ("total_flux", "total integrated flux, lines+brem  (Phs/s)", "viridis"),
]

# Metrics that are NOT in the default heatmap set (so a plain plot_scan doesn't
# grow an extra panel) but get a proper label + colormap when asked for by name,
# e.g. plot_scan(..., quantities=["coherent_brem_ratio"]). coherent_brem_ratio is
# ungated (not in _FLUX_GATED): it's the CXR/brem contrast, valid wherever brem>0.
_EXTRA_QUANTITIES = {
    "coherent_brem_ratio": (
        "coherent / incoherent-brem flux ratio  (CXR / brem)",
        "cividis",
    ),
}
_METRIC_LABELS = {key: label for key, label, _ in _HEATMAP_QUANTITIES}
_METRIC_LABELS.update({k: lbl for k, (lbl, _) in _EXTRA_QUANTITIES.items()})


def _resolve_quantity(q):
    """Normalize a quantity spec to a ``(key, label, cmap)`` triple: pass triples
    through, look bare metric keys up in the default + extra registries (cmap
    falls back to viridis for an unknown key)."""
    if not isinstance(q, str):
        return tuple(q)
    if q in _EXTRA_QUANTITIES:
        lbl, cmap = _EXTRA_QUANTITIES[q]
        return (q, lbl, cmap)
    return (q, _METRIC_LABELS.get(q, q), "viridis")


# Line-characterization maps are meaningless where the line is ill-defined --
# either near-zero emission OR a broad ramp / a cluster of comparable peaks
# (low line_quality, see results.line_quality). Gate these by BOTH peak flux
# and line_quality. The always-well-defined maps (peak_flux, coherent_flux,
# total_flux) and the diagnostic line_quality map itself are never gated.
_FLUX_GATED = {"line_eV", "fwhm_eV", "line_frac", "line_flux"}


def _axis_label(key):
    spec = _AXIS_SPECS.get(key)
    return key if spec is None else f"{spec[0]} ({spec[2]})"


def _axis_disp(key, vals):
    """Swept raw values -> display units (e.g. thickness Angstrom -> microns)."""
    div = _AXIS_SPECS.get(key, (None, 1.0))[1]
    return [float(v) / div for v in vals]


def _value_label(key, v):
    """'30 keV' / '17 um' style label for one swept value."""
    _lbl, div, unit, fmt = _AXIS_SPECS.get(key, (key, 1.0, "", "{:g}"))
    return f"{fmt.format(float(v) / div)} {unit}".strip()


def _cell_edges(disp_vals):
    """Cell EDGES for pcolormesh from sorted display values: midpoints between
    neighbours (extrapolated at the ends), so NON-uniform axes (a handful of
    thicknesses or energies) get correct boundaries -- not just linspace tilts."""
    v = np.asarray(sorted(disp_vals), dtype=float)
    if v.size == 1:
        d = abs(v[0]) * 0.1 or 0.5
        return np.array([v[0] - d, v[0] + d])
    mids = 0.5 * (v[:-1] + v[1:])
    return np.concatenate([[v[0] - (mids[0] - v[0])], mids, [v[-1] + (v[-1] - mids[-1])]])


def plot_heatmaps(
    results,
    settings,
    cases=None,
    quantities=None,
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
):
    """Parametric heatmaps over ANY two swept parameters ``x`` x ``y``, one panel
    per value of ``panel``, one figure per quantity.

    For two axes that may or may not both sweep, prefer :func:`plot_scan`, which
    auto-picks this heatmap or a line plot from how many values each axis has;
    this function always draws the map.

    ``x`` / ``y`` / ``panel`` are case-dict keys -- any of "tilt_deg",
    "tilt_azim_deg", "E0_keV", "thickness_ang", "B_ang2", ... The defaults
    reproduce the original map (azimuth x polar tilt, one panel per beam energy).
    Other critical scans are now one call::

        plot_heatmaps(res, s, x="thickness_ang", y="E0_keV", panel="tilt_deg")
        plot_heatmaps(res, s, x="tilt_deg", y="E0_keV", panel="tilt_azim_deg")

    When the sweep varies dimensions OTHER than x/y/panel, several records land in
    one cell; ``select`` (a results.selection_score mode, default
    "quality_peak" = peak flux x line quality) picks the BEST record and the cell
    shows ITS metric -- i.e. "the best achievable here". For the default axes each
    cell is a single case, so the reduction is a no-op (matches the old maps).

    Quantities (see results.line_metrics): peak spectral flux, the integrated
    coherent flux of ALL lines, the integrated flux under the single dominant
    line, that line's energy and FWHM, its share of the total flux, a
    line-definition quality map, and the total integrated flux. Maps that need NO
    peak (peak_flux, coherent_flux, total_flux) and the quality map are valid
    everywhere; the dominant-line maps are gated by BOTH ``min_flux_frac`` (cell
    peak flux below this fraction of the panel max -> no emission) AND
    ``min_line_quality`` (line_quality below this [0,1] -> no well-defined line:
    a broad ramp or a cluster of comparable peaks). Set either to 0 to disable.

    Pass ``cases`` (the current sweep from build_cases) to restrict to THIS
    sweep's configs -- otherwise a checkpoint accumulating several sweeps yields a
    sparse UNION of grids. Returns the list of figs.
    """
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        print("no results yet")
        return []
    quantities = quantities or _HEATMAP_QUANTITIES
    metrics = {id(r): line_metrics(r, settings, rel_prominence, metric=line_metric) for r in recs}
    panel_vals = sorted({r["case"][panel] for r in recs})

    figs = []
    for key, label, cmap in quantities:
        gated = key in _FLUX_GATED
        panels = []  # (panel_value, Z, x_edges, y_edges)
        for pv in panel_vals:
            er = [r for r in recs if r["case"][panel] == pv]
            xs = sorted({r["case"][x] for r in er})
            ys = sorted({r["case"][y] for r in er})
            xi = {v: i for i, v in enumerate(xs)}
            yi = {v: j for j, v in enumerate(ys)}
            fmax = max((metrics[id(r)]["peak_flux"] for r in er), default=0.0)
            floor = min_flux_frac * fmax
            # reduce every (x, y) cell to its single best record (selection_score)
            best = {}  # (xv, yv) -> (score, rec)
            for r in er:
                ck = (r["case"][x], r["case"][y])
                s = selection_score(metrics[id(r)], select)
                if ck not in best or s > best[ck][0]:
                    best[ck] = (s, r)
            Z = np.full((len(ys), len(xs)), np.nan)
            for (xv, yv), (_, r) in best.items():
                m = metrics[id(r)]
                if gated and (m["peak_flux"] < floor or m["line_quality"] < min_line_quality):
                    continue  # near-zero emission / ill-defined line -> blank
                Z[yi[yv], xi[xv]] = m[key]
            panels.append((pv, Z, _cell_edges(_axis_disp(x, xs)), _cell_edges(_axis_disp(y, ys))))
        finite = [Z[np.isfinite(Z)] for _, Z, _, _ in panels]
        finite = np.concatenate(finite) if any(a.size for a in finite) else np.array([0.0, 1.0])
        vmin, vmax = float(finite.min()), float(finite.max())

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(min(3.6 * len(panels) + 1.2, 12.0), 4.2),
            squeeze=False,
            constrained_layout=True,
        )
        im = None
        for ax, (pv, Z, xe, ye) in zip(axes.ravel(), panels, strict=False):
            im = ax.pcolormesh(xe, ye, Z, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{_AXIS_SPECS.get(panel, (panel,))[0]} = {_value_label(panel, pv)}")
            ax.set_xlabel(_axis_label(x))
            ax.set_ylabel(_axis_label(y))
        assert im is not None
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)  # one shared scale
        fig.suptitle(f"{label}    (best per cell: {select})", fontsize=13)
        figs.append(fig)
    return figs


def facet_metric(
    results,
    settings,
    *,
    x="thickness_ang",
    y="peak_flux",
    row=None,
    col=None,
    hue="E0_keV",
    reduce="max",
    logx=False,
    logy=False,
    sharey=True,
    rel_prominence=0.03,
    line_metric="sharpness",
    max_facets=36,
):
    """Small-multiples ("facet grid") view of a many-knob sweep (TODO P3 #9): a
    grid of subplots faceted by the ``row`` and ``col`` knobs, each plotting ``y``
    vs ``x`` with one line per ``hue`` value.

    Unlike :func:`plot_metric_vs` (which reduces every OTHER swept dimension to its
    best geometry), this EXPOSES the chosen knobs as the grid/line axes -- the
    first-class way to eyeball several simultaneously-swept knobs at once::

        facet_metric(res, s, x="thickness_ang", y="line_flux",
                     row="crystal", col="tilt_deg", hue="E0_keV", logx=True)

    ``x``/``y``/``hue`` and the facet knobs are any column of
    :func:`results.results_dataframe` (every case knob + the line metrics). Any
    remaining knobs are reduced per (facet, x, hue) cell by ``reduce``
    (``"max"|"min"|"mean"|"median"``). Returns the Figure (None if empty)."""
    from ..results import results_dataframe

    df = results_dataframe(results, settings, rel_prominence=rel_prominence, metric=line_metric)
    if df.empty:
        print("no results yet")
        return None
    for col_name in (x, y, hue, row, col):
        if col_name is not None and col_name not in df.columns:
            print(f"facet_metric: {col_name!r} is not a column; have {sorted(df.columns)}")
            return None

    row_vals = sorted(df[row].dropna().unique()) if row else [None]
    col_vals = sorted(df[col].dropna().unique()) if col else [None]
    if len(row_vals) * len(col_vals) > max_facets:
        print(
            f"facet_metric: {len(row_vals)}x{len(col_vals)} facets exceeds "
            f"max_facets={max_facets}; truncating (raise max_facets to see all)."
        )
        row_vals = row_vals[: max(1, max_facets // len(col_vals))]

    nrow, ncol = len(row_vals), len(col_vals)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False, sharex=True, sharey=sharey
    )
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            ax = axes[i][j]
            sub = df
            if row is not None:
                sub = sub[sub[row] == rv]
            if col is not None:
                sub = sub[sub[col] == cv]
            hue_groups = sub.groupby(hue) if hue is not None else [(None, sub)]  # type: ignore[reportAttributeAccessIssue]
            for hv, g in hue_groups:
                agg = g.groupby(x)[y].agg(reduce).reset_index().sort_values(x)  # type: ignore[reportAttributeAccessIssue]
                label = None if hue is None else f"{hue}={hv:g}" if _isnum(hv) else f"{hue}={hv}"
                ax.plot(agg[x], agg[y], marker="o", ms=3, lw=1.2, label=label)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            title = " | ".join(
                t
                for t in (
                    None if row is None else f"{row}={rv}",
                    None if col is None else f"{col}={cv}",
                )
                if t
            )
            if title:
                ax.set_title(title, fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel(x)
            if j == 0:
                ax.set_ylabel(y)
    if hue is not None:
        axes[0][0].legend(fontsize=8, title=hue)
    fig.tight_layout()
    return fig


def _isnum(v):
    """True for things f'{v:g}' accepts (numbers, not strings/None)."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def plot_metric_vs(
    results,
    settings,
    x="thickness_ang",
    metric="line_flux",
    hue="E0_keV",
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    logx=False,
    logy=False,
):
    """1-D parameter scan: ``metric`` vs the swept parameter ``x``, one line per
    ``hue`` value, reducing every OTHER swept dimension to its best geometry
    (results.selection_score ``select``). The line-plot companion to the
    heatmaps -- e.g. line flux vs thickness, or peak flux vs beam energy::

        plot_metric_vs(res, s, x="thickness_ang", metric="line_flux", logx=True)
        plot_metric_vs(res, s, x="E0_keV", metric="peak_flux", hue="tilt_deg")

    ``metric`` is any line_metrics key. Generalizes plot_peak_vs_tilt
    (x="tilt_deg", metric="peak_flux", hue="E0_keV")."""
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        print("no results yet")
        return None

    def _ndistinct(field):
        return len({r["case"][field] for r in recs if field in r["case"]})

    # Guard a single-valued x: with only one x value every hue collapses to a
    # vertical stack of points at that x (the "line_flux draws as stacked points"
    # bug -- e.g. x="E0_keV" on a single-energy sweep). Substitute a parameter
    # that actually sweeps so the curve is meaningful, rather than silently
    # connecting points that share an x.
    if _ndistinct(x) < 2:
        alt = next(
            (
                f
                for f in (
                    "tilt_deg",
                    "thickness_ang",
                    "E0_keV",
                    "tilt_azim_deg",
                    "B_ang2",
                )
                if f != x and f != hue and _ndistinct(f) >= 2
            ),
            None,
        )
        if alt is not None:
            print(
                f"plot_metric_vs: x={x!r} has <2 swept values -> using x={alt!r} "
                f"instead (it actually sweeps)."
            )
            x = alt
        else:
            print(
                f"plot_metric_vs: x={x!r} has <2 swept values and nothing else "
                f"sweeps -> single point(s)."
            )

    metrics = {id(r): line_metrics(r, settings, rel_prominence, metric=line_metric) for r in recs}
    hue_vals = sorted({r["case"][hue] for r in recs})
    div_x = _AXIS_SPECS.get(x, (None, 1.0))[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, hv in enumerate(hue_vals):
        hr = [r for r in recs if r["case"][hue] == hv]
        xs = sorted({r["case"][x] for r in hr})
        ys = []
        for xv in xs:
            cell = [r for r in hr if r["case"][x] == xv]
            best = max(cell, key=lambda r: selection_score(metrics[id(r)], select))
            ys.append(metrics[id(best)][metric])
        col = energy_color(hv, hue_vals) if hue == "E0_keV" else COLORS[j % len(COLORS)]
        ax.plot(
            [v / div_x for v in xs],
            ys,
            "o-",
            color=col,
            lw=1.8,
            label=_value_label(hue, hv),
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(_axis_label(x))
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"{_METRIC_LABELS.get(metric, metric)} vs {_axis_label(x)}  (best per point: {select})",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(title=_AXIS_SPECS.get(hue, (hue,))[0], fontsize=9)
    fig.tight_layout()
    return fig


def plot_scan(
    results,
    settings,
    *,
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    hue=None,
    quantities=None,
    heatmap_min=4,
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
    logx=False,
    logy=False,
    force=None,
):
    """ONE parametric-scan entry point that auto-picks a HEATMAP or a LINE plot
    from how many values each axis actually sweeps -- the merge of
    :func:`plot_heatmaps` (2-D maps) and :func:`plot_metric_vs` (1-D line scans),
    so you call this and don't have to choose:

      * BOTH ``x`` and ``y`` sweep >= ``heatmap_min`` values  -> heatmap, one
        panel per ``panel`` value (delegates to plot_heatmaps).
      * otherwise (an axis is fixed, or has only a few values) -> line plot: the
        DENSER axis goes on x, the sparser one becomes the line hue -- so a
        thin-azimuth or few-tilt sweep is a handful of clean lines, not a heatmap
        of horizontal bands (delegates to plot_metric_vs, one call per quantity).

    Either way you get ONE FIGURE PER QUANTITY (same list-of-figs contract as
    plot_heatmaps). ``quantities`` is a list of line_metrics keys (strings) and/or
    ``(key, label, cmap)`` triples; default is the full heatmap set.

    Knobs: ``heatmap_min`` is the per-axis value count below which a heatmap row/
    column is too coarse to be worth it and lines win (so 2-3 tilts -> lines,
    a full grid -> map). ``force`` overrides the choice ("heatmap" | "lines").
    ``hue`` overrides the line-mode grouping (e.g. ``hue="E0_keV"`` for one line
    per beam energy). All other args match plot_heatmaps / plot_metric_vs. Returns
    the list of figures (empty if there are no results)."""
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        print("no results yet")
        return []

    # accept bare metric keys as well as (key, label, cmap) triples
    quantities = quantities or _HEATMAP_QUANTITIES
    quantities = [_resolve_quantity(q) for q in quantities]

    def _ndistinct(field):
        return len({r["case"][field] for r in recs if field in r["case"]})

    nx, ny = _ndistinct(x), _ndistinct(y)
    if force in ("heatmap", "lines"):
        mode = force
    elif nx >= heatmap_min and ny >= heatmap_min:
        mode = "heatmap"
    else:
        mode = "lines"

    if mode == "heatmap":
        print(f"plot_scan: heatmap mode ({x} x {y}, panel per {panel})")
        return plot_heatmaps(
            results,
            settings,
            cases=cases,
            quantities=quantities,
            x=x,
            y=y,
            panel=panel,
            select=select,
            rel_prominence=rel_prominence,
            line_metric=line_metric,
            min_flux_frac=min_flux_frac,
            min_line_quality=min_line_quality,
        )

    # line mode: denser axis -> x, sparser -> hue (unless hue is given)
    line_x, other = (x, y) if nx >= ny else (y, x)
    if hue is None:
        if _ndistinct(other) >= 2:
            hue = other
        elif _ndistinct(panel) >= 2:
            hue = panel
        else:
            hue = other  # nothing else varies -> a single line
    print(f"plot_scan: line mode ({line_x} on x, one line per {hue})")
    return [
        plot_metric_vs(
            results,
            settings,
            x=line_x,
            metric=key,
            hue=hue,
            select=select,
            cases=cases,
            rel_prominence=rel_prominence,
            line_metric=line_metric,
            logx=logx,
            logy=logy,
        )
        for key, _, _ in quantities
    ]
