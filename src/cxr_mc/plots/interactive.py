"""interactive

Interactive / streaming browsers (matplotlib + plotly slider, scan chunks).
"""

import matplotlib.pyplot as plt
import numpy as np

from ..montecarlo import (
    detector_efficiency,
)
from ..results import (
    best_azimuth,
    records,
    show_summary,
)
from ._common import (
    _line_brem,
    _per_tilt_figs,
)
from ._style import (
    energy_color,
)
from .detectors import (
    _draw_eaglexo_charge,
    _draw_eaglexo_detected,
    _draw_timepix_detected,
)
from .spectra import (
    _draw_by_energy,
    _draw_full_spectrum,
)


# ---- interactive viewer ------------------------------------------------------
def browse(results, settings, kind="by_energy", label="polar tilt", static=None, **kw):
    """Page through one figure type BY POLAR TILT instead of printing every tilt
    stacked. ``kind``: "by_energy" | "full" | "chunk" | "timepix" | "eaglexo" |
    "eaglexo_charge".

    A tilt slider + Prev/Next swaps a freshly-drawn figure into an output area --
    reliable on the **inline** backend (recommended; no ``%matplotlib widget`` /
    ipympl needed, and it behaves over SSH). ``static=True`` (or no ipywidgets,
    e.g. nbconvert -> PDF) instead draws every tilt stacked so the export holds
    them all. Extra kwargs pass to the per-tilt drawer (include_brem, floor_frac,
    n_mc).

    For a faster spectral click-through (``kind`` "by_energy"/"full"), see
    :func:`browse_plotly` -- WebGL traces with client-side toggling, no redraw per
    slider move; this matplotlib path remains the one for static/PDF export."""
    # figure sizes kept within an XPS-15 notebook width (~12") so nothing needs
    # horizontal scrolling; single-axis spectra are ~9.5x5.3, the 2-panel chunk
    # is wider but shorter.
    drawers = {
        "by_energy": (_draw_by_energy, (9.5, 5.3)),
        "full": (_draw_full_spectrum, (9.5, 5.3)),
        "chunk": (_draw_chunk, (11.0, 4.8)),
        "timepix": (_draw_timepix_detected, (9.5, 5.3)),
        "eaglexo": (_draw_eaglexo_detected, (9.5, 5.3)),
        "eaglexo_charge": (_draw_eaglexo_charge, (9.5, 5.3)),
    }
    if kind not in drawers:
        raise ValueError(f"kind must be one of {list(drawers)}")
    draw, figsize = drawers[kind]
    recs = records(results)
    if kind == "full":
        recs = [r for r in recs if r.get("brem_wide") is not None]
    if not recs:
        print(
            "no results to browse"
            + (" (need E_grid_brem for kind='full')" if kind == "full" else "")
        )
        return None
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    by_tilt = {t: [r for r in recs if r["case"]["tilt_deg"] == t] for t in tilts}

    if static is None:  # auto: interactive if widgets exist
        try:
            import ipywidgets  # noqa: F401

            static = False
        except ImportError:
            static = True
    if not static:
        _tilt_browser(by_tilt, tilts, settings, draw, figsize, label, **kw)
        return None
    # static: every tilt stacked (PDF export / no ipywidgets)
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=figsize)
        draw(fig, by_tilt[t], settings, **kw)
        figs.append(fig)
    return figs


def _tilt_browser(by_tilt, tilts, settings, draw, figsize, label, **kw):
    """Polar-tilt slider + Prev/Next that renders a FRESH figure per tilt into an
    Output widget (clear + redraw). No dependence on live-canvas redraw, so it
    works reliably on the inline backend and over SSH (ipympl's persistent-figure
    redraw is what tends to get stuck showing one frame).

    This redraws a full matplotlib/Agg figure on EVERY slider move (thousands of
    points x several curves), so the spectral click-through can lag. For a fast
    interactive view of the spectra use :func:`browse_plotly` (WebGL, client-side
    trace toggling -- no Python redraw per click); this matplotlib path stays the
    one for static / nbconvert-PDF export, where there is no client to run JS."""
    import ipywidgets as widgets
    from IPython.display import display

    out = widgets.Output()
    slider = widgets.IntSlider(
        min=0,
        max=len(tilts) - 1,
        value=0,
        description=label,
        continuous_update=False,
        layout=widgets.Layout(width="60%"),
    )
    prev = widgets.Button(description="< Prev", layout=widgets.Layout(width="80px"))
    nxt = widgets.Button(description="Next >", layout=widgets.Layout(width="80px"))

    def render(i):
        fig = plt.figure(figsize=figsize)
        draw(fig, by_tilt[tilts[i]], settings, **kw)
        with out:
            out.clear_output(wait=True)
            display(fig)
        plt.close(fig)  # shown; don't leak or double-display

    prev.on_click(lambda b: setattr(slider, "value", max(0, slider.value - 1)))
    nxt.on_click(lambda b: setattr(slider, "value", min(len(tilts) - 1, slider.value + 1)))
    slider.observe(lambda ch: render(ch["new"]), names="value")
    display(widgets.HBox([prev, slider, nxt]))
    display(out)
    render(0)


def browse_plotly(
    results,
    settings,
    kind="by_energy",
    *,
    include_brem=True,
    collapse_azimuth=True,
    floor_frac=1e-5,
):
    """Fast WebGL spectral browser: ONE Plotly figure holding every (polar tilt,
    beam energy) curve as a ``Scattergl`` trace, with a client-side tilt slider
    that just toggles trace visibility -- no Python/matplotlib redraw per click,
    so paging is instant and 10k-point spectra stay smooth. The interactive
    counterpart to :func:`browse` for the spectral views; the matplotlib ``browse``
    stays the path for static / nbconvert-PDF export (no client-side JS there).

    ``kind``: ``"by_energy"`` (intrinsic lines + brem, linear axes) or ``"full"``
    (sharp lines on the wide brem out to the beam energy, log-log; needs records
    run with a separate ``E_grid_brem``). Beam energy -> colour matches every other
    figure. Needs ``plotly`` installed. Returns the figure (Jupyter renders it; call
    ``.show()`` elsewhere)."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "browse_plotly needs plotly (`uv add plotly` / `pip install plotly`); "
            "use browse(..., kind=...) for the matplotlib path."
        ) from e
    if kind not in ("by_energy", "full"):
        raise ValueError("browse_plotly kind must be 'by_energy' or 'full'")

    recs = records(results)
    if kind == "full":
        recs = [r for r in recs if r.get("brem_wide") is not None]
    if not recs:
        print(
            "no results to browse"
            + (" (need E_grid_brem for kind='full')" if kind == "full" else "")
        )
        return None
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    energies = sorted({r["case"]["E0_keV"] for r in recs})

    fig = go.Figure()
    trace_tilt = []  # parallel to fig.data: which tilt index each trace belongs to
    tilt_case0 = {}  # first record per tilt (for the per-step title)
    # global axis extents for the log-log 'full' view (shared across tilt steps so
    # the axes don't jump while paging); mirrors _draw_full_spectrum's floor logic.
    ymax, ybrem_lo, xmin, xmax = 0.0, np.inf, np.inf, 0.0

    def _collapse(grp):
        if collapse_azimuth and len(grp) > 1:
            return [max(grp, key=lambda r: float(np.max(r["spec"])))]
        return grp

    for ti, t in enumerate(tilts):
        trecs = [r for r in recs if r["case"]["tilt_deg"] == t]
        tilt_case0[ti] = trecs[0]["case"]
        for E0 in energies:
            grp = _collapse([r for r in trecs if r["case"]["E0_keV"] == E0])
            for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
                col = energy_color(E0, energies)
                az = r["case"]["tilt_azim_deg"]
                lbl = f"{E0:g} keV (φ={az:.1f}°)"
                vis = ti == 0
                line_det, brem_det = _line_brem(r, settings, convolve=False)
                if kind == "by_energy":
                    y = (line_det + brem_det) if include_brem else line_det
                    fig.add_trace(
                        go.Scattergl(
                            x=r["E_grid"],
                            y=y * r["scale"],
                            name=lbl,
                            mode="lines",
                            line=dict(color=col, width=1.5),
                            legendgroup=lbl,
                            visible=vis,
                        )
                    )
                    trace_tilt.append(ti)
                    if include_brem:
                        fig.add_trace(
                            go.Scattergl(
                                x=r["E_grid"],
                                y=brem_det * r["scale"],
                                name=lbl + " brem",
                                mode="lines",
                                line=dict(color=col, width=0.8, dash="dash"),
                                legendgroup=lbl,
                                showlegend=False,
                                visible=vis,
                            )
                        )
                        trace_tilt.append(ti)
                else:  # full
                    Eb = np.asarray(r["E_grid_brem"], dtype=float)
                    qe_b = detector_efficiency(Eb) if settings.apply_detector_qe else 1.0
                    brem_wide_det = r["brem_wide"] * qe_b * r["scale"]
                    total_line = (line_det + brem_det) * r["scale"]
                    fig.add_trace(
                        go.Scattergl(
                            x=Eb,
                            y=brem_wide_det,
                            name=lbl + " brem",
                            mode="lines",
                            line=dict(color=col, width=0.8, dash="dash"),
                            legendgroup=lbl,
                            showlegend=False,
                            visible=vis,
                        )
                    )
                    trace_tilt.append(ti)
                    fig.add_trace(
                        go.Scattergl(
                            x=r["E_grid"],
                            y=total_line,
                            name=lbl,
                            mode="lines",
                            line=dict(color=col, width=1.5),
                            legendgroup=lbl,
                            visible=vis,
                        )
                    )
                    trace_tilt.append(ti)
                    xmin = min(xmin, float(Eb[0]))
                    xmax = max(xmax, float(Eb[-1]))
                    ymax = max(ymax, float(np.nanmax(total_line)) if total_line.size else 0.0)
                    ymax = max(
                        ymax,
                        float(np.nanmax(brem_wide_det)) if brem_wide_det.size else 0.0,
                    )
                    bpos = brem_wide_det[np.isfinite(brem_wide_det) & (brem_wide_det > 0)]
                    if bpos.size:
                        ybrem_lo = min(ybrem_lo, float(np.percentile(bpos, 1)))

    tag = "best azimuth/energy" if collapse_azimuth else "all azimuths"

    def _title(ti):
        c = tilt_case0[ti]
        head = f"{c['name'].split()[0]}, {c['thickness_ang'] / 1e4:.1f} µm, tilt={c['tilt_deg']:g}°"
        if kind == "by_energy":
            return f"{head} — intrinsic ({tag})"
        return f"{head} — full measured range, intrinsic (dashed = brem)"

    steps = [
        dict(
            method="update",
            label=f"{t:g}°",
            args=[
                {"visible": [tt == ti for tt in trace_tilt]},
                {"title.text": _title(ti)},
            ],
        )
        for ti, t in enumerate(tilts)
    ]
    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "polar tilt: "}, steps=steps)],
        title=_title(0),
        xaxis_title="Photon energy (eV)",
        yaxis_title="Intensity (Phs/eV/s/nA)",
        template="plotly_white",
        height=560,
        legend=dict(title=("dashed: brem" if include_brem else None)),
    )
    if kind == "full":
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        if xmax > 0:
            lo = max(xmin, 1.0)  # log x can't show 0 (brem grid -> 0)
            fig.update_xaxes(range=[np.log10(lo), np.log10(xmax)])
        if ymax > 0:
            floor = max(ybrem_lo if np.isfinite(ybrem_lo) else 0.0, ymax * floor_frac)
            if floor > 0:
                fig.update_yaxes(range=[np.log10(floor), np.log10(ymax * 2)])
    else:
        fig.update_yaxes(rangemode="tozero")
    return fig


def _draw_chunk(fig, trecs, settings):
    """Render ONE polar tilt's best-geometry INTRINSIC spectra onto ``fig``
    (cleared first): LEFT total (coherent + brem), RIGHT brem-subtracted CXR only.
    The detector view is the separate Eagle XO browser (kind='eaglexo').

    One curve PER BEAM ENERGY: the highest-peak record across azimuth AND any other
    swept dimension (e.g. thickness). best_azimuth() alone keyed on thickness, so a
    thickness sweep dumped one line per thickness -- all with the same energy/azimuth
    label -- onto each axis; collapsing per energy (like _draw_by_energy) is the fix.
    When more than one thickness is in play the label carries the one shown."""
    fig.clear()
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    best = []
    for E0 in energies:
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if grp:
            best.append(max(grp, key=lambda r: float(np.max(r["spec"]))))
    if not best:
        return
    ax_tot, ax_cxr = fig.subplots(1, 2, sharex=True)
    multi_t = len({r["case"]["thickness_ang"] for r in best}) > 1
    for r in best:
        az, E0 = r["case"]["tilt_azim_deg"], r["case"]["E0_keV"]
        c = energy_color(E0, energies)
        lbl = rf"{E0:g} keV ($\phi={az:g}\degree$"
        lbl += rf", {r['case']['thickness_ang'] / 1e4:g} $\mu$m)" if multi_t else ")"
        E = r["E_grid"] / 1e3
        line_raw, brem_raw = _line_brem(r, settings, convolve=False)  # intrinsic
        ax_tot.plot(E, (line_raw + brem_raw) * r["scale"], color=c, lw=1.2, label=lbl)
        ax_tot.plot(E, brem_raw * r["scale"], color=c, ls="--", lw=0.6)
        ax_cxr.plot(E, line_raw * r["scale"], color=c, lw=1.2, label=lbl)
    case = best[0]["case"]
    fig.suptitle(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:g}\degree$ — best azimuth per "
        rf"energy (intrinsic)",
        fontsize=13,
    )
    for ax, title, leg in (
        (ax_tot, "Total X-ray spectrum (coherent + brem)", "dashed = brem bkgnd"),
        (ax_cxr, "Brem-subtracted (CXR only)", None),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.legend(title=leg, fontsize=9)
    fig.tight_layout()


def plot_chunk(results, settings):
    """The best-azimuth intrinsic spectra (total | CXR-only), ONE figure per polar
    tilt. For click-through use ``browse(results, settings, kind="chunk")``."""
    return _per_tilt_figs(records(results), settings, _draw_chunk, (11.0, 4.8))


def stream_chunk(results, names, settings, collapse_azimuth=True, **_):
    """Per-group photon-counting table during a run (live progress). The 2x2
    best-azimuth plots are no longer drawn inline -- page through them after the
    run with ``browse(results, settings, kind="chunk")``. Extra kwargs (e.g. the
    old ``show_plot``) are accepted and ignored."""
    recs = records(results, names)
    if collapse_azimuth:
        recs = best_azimuth(recs)
    show_summary(recs, settings)
