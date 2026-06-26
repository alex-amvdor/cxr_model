"""
plots.py
============

All plotting for the analysis notebook, kept out of the notebook itself.

Intrinsic spectra
  * :func:`plot_by_energy` -- one figure PER POLAR TILT, every beam energy on a
    single axis (each at its best azimuth when ``collapse_azimuth=True``).
    Individual figures, so a wide tilt sweep doesn't run off-screen.
  * :func:`plot_full_spectrum` -- the full measured range: sharp coherent lines
    (fine line grid) on the broad brem evaluated out to the beam energy (wide
    brem grid). Needs a sweep run with a separate ``E_grid_brem``.
  * :func:`plot_peak_vs_tilt` -- a single overview: best-azimuth peak flux vs
    polar tilt, one line per energy. The "which tilt wins" figure for big sweeps.

Timepix3 detector view (forward model in ``timepix_response``)
  * :func:`plot_timepix_efficiency`, :func:`plot_timepix_detected`,
    :func:`plot_timepix_poisson`.

Eagle XO detector view (forward model in ``eaglexo_response``)
  * :func:`plot_eaglexo_efficiency`, :func:`plot_eaglexo_detected` -- the
    direct-detection CCD (solid angle x QE): soft PXR lines pass at ~90% QE while
    the hard brem is crushed by the thin sensor. Browse per tilt with
    ``browse(results, settings, kind="eaglexo")``. A bare CCD integrates charge
    and cannot return a spectrum, so the "what it measures" view is the recorded
    CHARGE: :func:`plot_eaglexo_charge` / :func:`plot_eaglexo_charge_map`.

Everything takes ``results`` + a :class:`results.Settings` explicitly.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import eaglexo_response as eag
from . import timepix_response as tpx
from .montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    convolve_detector,
    detector_efficiency,
    eds_fwhm_eV,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
    simulate_trajectories,
    tilted_geometry,
)
from .results import (
    PER_NA,
    best_azimuth,
    detected_background,
    line_metrics,
    records,
    selection_score,
    show_summary,
)

COLORS = ["r", "y", "g", "b", "m", "c", "k", "orange", "purple", "brown"]

# Beam energy -> colour, CONSISTENT across every figure: a given E0 always gets
# the same colour (keyed to its rank in the sorted energy set, so e.g. 30/45/60
# keV map to the same three colours everywhere), and the palette stays readable
# on white (no low-contrast yellow). Pass the FULL set of energies present in
# the figure so the rank -- hence the colour -- is stable panel to panel.
_ENERGY_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def energy_color(E0, energies):
    """Stable colour for beam energy ``E0`` given the figure's set of energies."""
    order = sorted({float(e) for e in energies})
    try:
        i = order.index(float(E0))
    except ValueError:
        i = 0
    return _ENERGY_PALETTE[i % len(_ENERGY_PALETTE)]


# cache of the (expensive) Timepix efficiency-curve response, keyed by hardware +
# MC settings, so re-running the detector cell with unchanged settings doesn't
# rebuild it. (The per-grid detected-spectra response is already cached inside
# timepix_response.get_response.)
_EFF_CACHE = {}


def _mode(settings):
    return "EDS-convolved" if getattr(settings, "convolve_with_det", False) else "intrinsic"


def _line_brem(r, settings, convolve=None):
    """Detected line and brem densities (per eV, before the unit scale) for one
    record, honoring the QE / brem-source flags. ``convolve`` overrides
    settings.convolve_with_det when given (True/False), so a caller can draw the
    intrinsic (convolve=False) and detector-convolved (convolve=True) spectra
    side by side."""
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    qe = detector_efficiency(r["E_grid"]) if settings.apply_detector_qe else 1.0
    line_in = r["spec"] * qe
    line_det = convolve_detector(r["E_grid"], line_in, r["fwhm"]) if do_conv else line_in
    brem_det = detected_background(r, settings, convolve=do_conv) / r["scale"]
    return line_det, brem_det


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


# ---- per-tilt figure helper --------------------------------------------------
def _per_tilt_figs(recs, settings, draw, figsize, *, empty_msg="no results yet", **kw):
    """Shared body of every ``plot_*`` wrapper: one freshly-drawn figure PER POLAR
    TILT. ``draw(fig, tilt_recs, settings, **kw)`` renders a single tilt onto a
    cleared figure (the same ``_draw_*`` the interactive ``browse`` uses), so the
    wrappers and the slider stay in lockstep. Handles the empty-records check, the
    per-tilt grouping, and collecting the figure list."""
    if not recs:
        print(empty_msg)
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=figsize)
        draw(fig, [r for r in recs if r["case"]["tilt_deg"] == t], settings, **kw)
        figs.append(fig)
    return figs


# ---- intrinsic spectra -------------------------------------------------------
def plot_tilt_panel(ax, group, settings, include_brem=True, collapse_azimuth=False):
    """One panel at fixed (energy, polar tilt): one curve per azimuth, or just
    the best azimuth if ``collapse_azimuth``."""
    if collapse_azimuth and len(group) > 1:
        group = [max(group, key=lambda r: float(np.max(r["spec"])))]
    group = sorted(group, key=lambda r: r["case"]["tilt_azim_deg"])
    for i, r in enumerate(group):
        c = COLORS[i % len(COLORS)]
        az = r["case"]["tilt_azim_deg"]
        line_det, brem_det = _line_brem(r, settings)
        y = (line_det + brem_det) if include_brem else line_det
        ax.plot(
            r["E_grid"] / 1e3,
            y * r["scale"],
            color=c,
            ls="-",
            lw=1.1,
            label=rf"$\phi={az:.1f}\degree$",
        )
        if include_brem:
            ax.plot(r["E_grid"] / 1e3, brem_det * r["scale"], color=c, ls="--", lw=0.7)
    case = group[0]["case"]
    mat = case["name"].split()[0]
    ax.set_title(
        rf"{mat}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, {case['E0_keV']:g} keV, "
        rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:g}\degree$",
        fontsize=11,
    )
    ax.set_xlabel("Photon energy (keV)", fontsize=10)
    ax.set_ylabel("Intensity (Phs/eV/s/nA)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.grid(alpha=0.3)
    ax.legend(title=("dashed: brem" if include_brem else None), fontsize=8)


def _draw_by_energy(fig, trecs, settings, include_brem=True, collapse_azimuth=True):
    """Render ONE polar tilt onto ``fig`` (cleared first): every beam energy
    overlaid, INTRINSIC spectra (the detector view is the separate Eagle XO
    browser, kind='eaglexo'). Single axis -> fits the screen without scrolling."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            az = r["case"]["tilt_azim_deg"]
            lbl = rf"{E0:g} keV ($\phi={az:0.1f}\degree$)"
            line_det, brem_det = _line_brem(r, settings, convolve=False)
            y = (line_det + brem_det) if include_brem else line_det
            ax.plot(r["E_grid"], y * r["scale"], color=c, lw=1.3, label=lbl)
            if include_brem:
                ax.plot(r["E_grid"], brem_det * r["scale"], color=c, ls="--", lw=0.6)
    case = trecs[0]["case"]
    tag = "best azimuth/energy" if collapse_azimuth else "all azimuths"
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} "
        rf"$\mu$m, $\theta_\mathrm{{tilt}}={case['tilt_deg']:0.1f}\degree$ — intrinsic "
        rf"({tag})",
        fontsize=12,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.grid(alpha=0.3)
    ax.legend(title=("dashed: brem" if include_brem else None), fontsize=9)
    fig.tight_layout()


def plot_by_energy(results, settings, include_brem=True, collapse_azimuth=True):
    """One figure PER POLAR TILT, every beam energy overlaid (best azimuth when
    ``collapse_azimuth``); INTRINSIC spectra (detector view = the Eagle XO
    browser). For click-through use ``browse(results, settings, kind="by_energy")``."""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_by_energy,
        (9.5, 5.3),
        include_brem=include_brem,
        collapse_azimuth=collapse_azimuth,
    )


def _draw_full_spectrum(
    fig, trecs, settings, collapse_azimuth=True, logy=True, logx=True, floor_frac=1e-5
):
    """Render ONE polar tilt of the full measured-range view onto ``fig``: sharp
    lines + wide brem out to the beam energy, log-log, INTRINSIC (single axis; the
    detector view is the Eagle XO browser).

    Broad-spectrum view: the y-floor is set from the brem CONTINUUM (its ~1st
    percentile across the full range), not a fixed fraction of the line peak, so
    the whole bremsstrahlung shoulder out to the beam energy stays on-screen
    instead of being clipped under a tall, narrow line. ``floor_frac`` only caps
    the dynamic range (deepest allowed = floor_frac x the peak)."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = 0.0
    ybrem_lo = np.inf  # smallest positive brem value shown -> the broad-spectrum floor
    xmax = 0.0
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0 and r.get("brem_wide") is not None]
        if not grp:
            continue
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        r = grp[0]
        c = energy_color(E0, energies)
        az = r["case"]["tilt_azim_deg"]
        lbl = rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)"
        Eb = r["E_grid_brem"]
        qe_b = detector_efficiency(Eb) if settings.apply_detector_qe else 1.0
        brem_wide_det = r["brem_wide"] * qe_b * r["scale"]
        float(Eb[0])
        xmax = max(xmax, float(Eb[-1]))  # full brem grid -> beam energy
        line_det, brem_det = _line_brem(r, settings, convolve=False)
        total_line = (line_det + brem_det) * r["scale"]
        ax.plot(Eb, brem_wide_det, color=c, ls="--", lw=0.7, alpha=0.85)
        ax.plot(r["E_grid"], total_line, color=c, lw=1.2, label=lbl)
        ymax = max(ymax, float(np.nanmax(total_line)) if total_line.size else 0.0)
        ymax = max(ymax, float(np.nanmax(brem_wide_det)) if brem_wide_det.size else 0.0)
        bpos = brem_wide_det[np.isfinite(brem_wide_det) & (brem_wide_det > 0)]
        if bpos.size:
            ybrem_lo = min(ybrem_lo, float(np.percentile(bpos, 1)))
    case = trecs[0]["case"]
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} "
        rf"$\mu$m, $\theta_\mathrm{{tilt}}={case['tilt_deg']:0.1f}\degree$ — full "
        rf"measured range, intrinsic (dashed = brem)",
        fontsize=12,
    )
    if logy and ymax > 0:
        ax.set_yscale("log")
        # floor driven by the brem continuum so the broad spectrum stays visible,
        # but never deeper than floor_frac x the peak (guards a near-zero edge)
        floor = max(ybrem_lo if np.isfinite(ybrem_lo) else 0.0, ymax * floor_frac)
        ax.set_ylim(floor, ymax * 2)
    else:
        ax.set_ylim(bottom=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    if xmax > 0:
        ax.set_xlim(50.0)  # span the full brem grid (to the beam energy)
    else:
        ax.margins(x=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_full_spectrum(results, settings, collapse_azimuth=True, logy=True, floor_frac=1e-5):
    """Full measured-range view (sharp lines on the wide brem, log-log), ONE figure
    per polar tilt. The x-axis spans the full brem grid (to the beam energy) and
    the y-floor follows the brem continuum, so the broad bremsstrahlung shoulder
    is on-screen instead of clipped under the lines (``floor_frac`` caps the depth
    at floor_frac x the peak). For click-through use
    ``browse(results, settings, kind="full")``. Needs records run with a separate
    ``E_grid_brem`` (``brem_wide`` present)."""
    recs = [r for r in records(results) if r.get("brem_wide") is not None]
    return _per_tilt_figs(
        recs,
        settings,
        _draw_full_spectrum,
        (9.5, 5.3),
        empty_msg="no wide-brem records -- set E_grid_brem in the Sweep and re-run",
        collapse_azimuth=collapse_azimuth,
        logy=logy,
        floor_frac=floor_frac,
    )


def plot_peak_vs_tilt(results, settings):
    """Overview for big sweeps: best-azimuth peak spectral flux vs polar tilt,
    one line per beam energy. Peak = max(spectrum) * scale * beam current."""
    recs = best_azimuth(records(results))
    if not recs:
        print("no results yet")
        return None
    by_E = {}
    for r in recs:
        by_E.setdefault(r["case"]["E0_keV"], []).append(r)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _i, (E0, rs) in enumerate(sorted(by_E.items())):
        rs = sorted(rs, key=lambda r: r["case"]["tilt_deg"])
        tilts = [r["case"]["tilt_deg"] for r in rs]
        peak = [float(np.max(r["spec"])) * r["scale"] * settings.beam_current_na for r in rs]
        ax.plot(tilts, peak, "o-", color=energy_color(E0, by_E), label=f"{E0:g} keV")
    ax.set_xlabel(r"polar tilt $\theta_\mathrm{tilt}$ (deg)")
    ax.set_ylabel("best-azimuth peak (Phs/eV/s)")
    ax.set_title("Peak spectral flux vs polar tilt (best azimuth per point)")
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy")
    fig.tight_layout()
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


# ---- crystal mosaicity (analytic broadening) ---------------------------------
def plot_mosaic_comparison(r, settings, grades_deg=(None, 0.4, 0.8, 3.5), ax=None):
    """Detector-convolved line spectrum of ONE record ``r`` overlaid for several
    crystal mosaic grades -- WITHOUT re-running transport. The analytic mosaic model
    (montecarlo.mosaic_fwhm_eV) only changes the convolution FWHM and leaves the
    intrinsic spectrum fixed, so each grade is just a re-convolution of the same
    ``r["spec"]``. ``grades_deg`` entries are mosaic rocking-curve FWHM in degrees;
    None = perfect crystal (no mosaic term). Handy for HOPG: ZYA 0.4 / ZYB 0.8 /
    ZYH 3.5 deg. The record can come from a mosaic=False run -- the broadening is
    re-derived here per grade. Returns the Figure."""
    case = r["case"]
    E, E_pk = r["E_grid"], r["E_pk"]
    qe = detector_efficiency(E) if settings.apply_detector_qe else 1.0
    line_in = r["spec"] * qe
    base_sq = (
        eds_fwhm_eV(E_pk) ** 2
        + aperture_fwhm_eV(
            E_pk,
            beta_from_keV(case["E0_keV"]),
            case["theta_obs_rad"],
            case["dtheta_obs_rad"],
        )
        ** 2
    )
    psi = mosaic_psi_rad(case, E_pk)

    curves = []  # (label, fwhm, detected)
    for grade in grades_deg:
        if grade is None:
            fwhm = float(np.sqrt(base_sq))
            lbl = "perfect"
        else:
            extra = (
                min(mosaic_fwhm_eV(E_pk, psi, np.deg2rad(grade)), E_pk) if psi is not None else 0.0
            )
            fwhm = float(np.sqrt(base_sq + extra**2))
            lbl = rf"mosaic {grade:g}$\degree$"
        det = convolve_detector(E, line_in, fwhm) * r["scale"]
        curves.append((lbl, fwhm, det))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    else:
        fig = ax.figure
    ymax = 0.0
    for lbl, fwhm, det in curves:
        fin = det[np.isfinite(det)]
        if fin.size:
            ymax = max(ymax, float(fin.max()))
        ax.plot(E, det, lw=1.5, label=f"{lbl} (FWHM {fwhm:.0f} eV)")
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.08)
    # zoom to the line: +-6 of the broadest FWHM about the peak, clamped to the grid
    half = 6.0 * max(f for _, f, _ in curves)
    ax.set_xlim(max(float(E[0]), E_pk - half), min(float(E[-1]), E_pk + half))
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$E_0$={case['E0_keV']:g} keV, $\theta_\mathrm{{tilt}}$="
        rf"{case['tilt_deg']:.1f}$\degree$ — crystal-mosaic broadening",
        fontsize=12,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Phs/eV/s/nA")
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
        title=(rf"$\psi$(v,g)={np.degrees(psi):.1f}$\degree$" if psi is not None else None),
    )
    return fig


# ---- Timepix3 detector view --------------------------------------------------
def _thr_keV():
    return tpx.THRESHOLD_E * tpx.W_EHP_EV / 1e3


def plot_timepix_efficiency(thickness_um=300.0, bias_v=100.0, n_mc=80000, seed=0):
    """Detection efficiency (absorption x counting turn-on) and energy
    resolution / charge-loss bias vs photon energy for the Si quad."""
    E_eff = np.arange(200.0, 60000.0, 25.0)
    key = (thickness_um, bias_v, n_mc, seed)
    resp = _EFF_CACHE.get(key)
    if resp is None:
        resp = tpx.build_response(
            E_eff,
            np.arange(0.0, 60000.0, 100.0),
            n_mc=n_mc,
            seed=seed,
            thickness_um=thickness_um,
            bias_v=bias_v,
        )
        _EFF_CACHE[key] = resp
    E_thr = _thr_keV() * 1e3
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axL.plot(
        E_eff,
        resp["eps_abs"],
        "k--",
        lw=1.2,
        label=r"$\epsilon_\mathrm{abs}$ (Si absorption)",
    )
    axL.plot(
        E_eff,
        resp["P_det"],
        "b-",
        lw=1.8,
        label=r"$P_\mathrm{det}$ (abs $\times$ counting)",
    )
    axL.axvline(E_thr, color="r", ls=":", label=f"threshold = {E_thr:.2f} eV")
    axL.set(
        xlabel="Photon energy (eV)",
        ylabel="efficiency",
        ylim=(0, 1.05),
        title=f"Detection efficiency ({thickness_um:g} $\\mu$m Si, {bias_v:g} V, "
        f"$\\sigma_\\mathrm{{diff}}$={resp['sigma_diff_um']:.1f} $\\mu$m)",
    )
    axL.set_xscale("log")
    axL.set_xlim((min(E_eff), max(E_eff)))
    axL.margins(x=0)
    axL.grid(alpha=0.3)
    axL.legend()
    axR.plot(E_eff, tpx.energy_fwhm_eV(E_eff), "k-", lw=1.5, label="analytic, single-pixel")
    axR.plot(E_eff, resp["fwhm_rec"], "b.", ms=4, label="MC effective (tail + multi-pixel)")
    axR.set(
        xlabel="Photon energy (keV)",
        ylabel="energy FWHM (eV)",
        title="Energy resolution & charge-loss bias",
    )
    axR.margins(x=0)
    axR.grid(alpha=0.3)
    axR.legend(loc="upper left")
    axR2 = axR.twinx()
    axR2.plot(E_eff, 100 * (1 - resp["mean_rec"] / E_eff), "g-", lw=1, alpha=0.5)
    axR2.set_ylabel("mean charge-loss deficit (%)", color="g")
    axR2.tick_params(axis="y", colors="g")  # axR2.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed):
    """Incident and Timepix3-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by tpx.get_response."""
    incident = (r["spec"] + r["brem"]) * r["scale"]
    resp = tpx.get_response(
        r["E_grid"], n_mc=n_mc, seed=seed, thickness_um=thickness_um, bias_v=bias_v
    )
    return incident, resp.apply(incident)


def _draw_timepix_detected(
    fig,
    trecs,
    settings,
    thickness_um=300.0,
    bias_v=100.0,
    collapse_azimuth=True,
    n_mc=80000,
    seed=0,
    floor_frac=1e-3,
):
    """Render ONE polar tilt of the Timepix detected/incident view onto ``fig``
    (cleared first): all energies overlaid, incident dotted / detected solid."""
    fig.clear()
    ax = fig.subplots(1, 1)
    E_thr = _thr_keV()
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = 0.0
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            inc, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
            fin = inc[np.isfinite(inc)]
            if fin.size:
                ymax = max(ymax, float(fin.max()))
            az = r["case"]["tilt_azim_deg"]
            ax.plot(r["E_grid"], inc, color=c, ls=":", lw=1.0, alpha=0.7)
            ax.plot(
                r["E_grid"],
                det,
                color=c,
                ls="-",
                lw=1.2,
                label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)",
            )
    case = trecs[0]["case"]
    ax.axvline(E_thr, color="0.4", ls=":", lw=0.8, label=f"threshold {E_thr:.2f} keV")
    if ymax > 0:
        ax.set_yscale("log")
        ax.set_ylim(ymax * floor_frac, ymax * 2)
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:0.1f}\degree$ — Timepix3 detected "
        rf"(solid) vs incident (dotted)",
        fontsize=12,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Phs/eV/s/nA")
    ax.margins(x=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_timepix_detected(
    results,
    settings,
    thickness_um=300.0,
    bias_v=100.0,
    collapse_azimuth=True,
    n_mc=80000,
    seed=0,
    ncols=5,
    floor_frac=1e-3,
):
    """Incident (dotted) vs Timepix3-detected (solid) spectra, log scale; ONE
    figure per polar tilt, all energies overlaid (best azimuth each). For
    click-through use ``browse(results, settings, kind="timepix")``. (``ncols``
    is accepted for backward compatibility and ignored.)"""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_timepix_detected,
        (9.0, 5.2),
        thickness_um=thickness_um,
        bias_v=bias_v,
        collapse_azimuth=collapse_azimuth,
        n_mc=n_mc,
        seed=seed,
        floor_frac=floor_frac,
    )


def plot_timepix_poisson(
    results,
    settings,
    integration_s=600.0,
    thickness_um=300.0,
    bias_v=100.0,
    n_mc=80000,
    seed=0,
):
    """A Poisson 'measured' realization for the highest-rate config at each
    energy, over ``integration_s`` at the configured beam current."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return None
    rng = np.random.default_rng(seed)
    E_thr = _thr_keV() * 1e3
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    # per-panel ~4.6" with a sensible minimum total width (a single energy was
    # only 3.7" -> too narrow + a clipped suptitle); constrained_layout reserves
    # room for the suptitle instead of tight_layout clipping it.
    n = len(energies)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(max(4.6 * n, 6.8), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, E0 in zip(axes.ravel(), energies, strict=False):
        grp = [r for r in recs if r["case"]["E0_keV"] == E0]
        r = max(grp, key=lambda r: float(np.max(r["spec"])))
        _, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
        counts, expected = tpx.poisson_counts(
            r["E_grid"], det * settings.beam_current_na, integration_s, rng
        )
        ax.step(
            r["E_grid"],
            counts,
            where="mid",
            color="k",
            lw=0.7,
            label=f"measured ({integration_s:g} s @ {settings.beam_current_na:g} nA)",
        )
        ax.plot(r["E_grid"], expected, "r-", lw=1.3, label="expected mean")
        ax.axvline(E_thr, color="b", ls=":", lw=0.8, label="threshold")
        ax.set_title(
            rf"{E0:g} keV, $\theta_\mathrm{{tilt}}={r['case']['tilt_deg']:g}\degree$, "
            rf"$\phi={r['case']['tilt_azim_deg']:g}\degree$  "
            rf"({counts.sum():.0f} cts)",
            fontsize=10,
        )
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("counts / bin")
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Timepix3 Poisson 'measured' spectra ({thickness_um:g} $\\mu$m Si, {bias_v:g} V)",
        fontsize=14,
    )
    return fig


# ---- Eagle XO detector view --------------------------------------------------
SI_K_EDGE_EV = 1839.0  # silicon K absorption edge -> the QE notch the lines cross


def _domega_of(r):
    """The solid angle [sr] actually baked into a record (scale = domega * PER_NA),
    so plot annotations report the geometry the sweep was run with -- not a value
    re-guessed here that might disagree with it."""
    return r["scale"] / PER_NA


def plot_eaglexo_efficiency(sensor="4240", distance_m=None, coating="BN"):
    """The Eagle XO's two knobs vs photon energy. LEFT: quantum efficiency -- the
    measured datasheet curve (BN solid, BEN dashed) plus the thin-Si absorption
    cross-check -- on log energy, with the Si-K notch marked and the soft-line /
    hard-brem regimes shaded. RIGHT: the photon-counting (energy-resolving)
    resolution, Fano + read-noise limited. The title carries the solid angle
    (knob 1) for the chosen sensor + working distance."""
    geo = eag.geometry(sensor, distance_m)
    E = np.geomspace(100.0, 60000.0, 600)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    # -- QE (knob 2) --
    axL.axvspan(E[0], 3000.0, color="g", alpha=0.05)
    axL.axvspan(3000.0, E[-1], color="r", alpha=0.05)
    axL.plot(E, eag.qe(E, "BN"), "b-", lw=1.8, label="QE (BN, no coating)")
    axL.plot(E, eag.qe(E, "BEN"), "g--", lw=1.4, label="QE (BEN, enhanced)")
    axL.plot(
        E,
        eag.qe_absorption_model(E),
        "0.5",
        ls=":",
        lw=1.2,
        label=rf"abs. model ({eag.ACTIVE_SI_UM:g} $\mu$m Si)",
    )
    axL.axvline(SI_K_EDGE_EV, color="0.4", ls="--", lw=0.8)
    axL.text(SI_K_EDGE_EV * 1.05, 0.05, "Si-K", color="0.3", fontsize=8)
    axL.set_xscale("log")
    axL.set_xlim(E[0], E[-1])
    axL.set_ylim(0, 1.0)
    axL.set(
        xlabel="Photon energy (eV)",
        ylabel="quantum efficiency",
        title="QE: soft lines pass (green), hard brem crushed (red)",
    )
    axL.grid(alpha=0.3, which="both")
    axL.legend(fontsize=8, loc="center left")
    # -- photon-counting energy resolution --
    axR.plot(E, eag.energy_fwhm_eV(E), "b-", lw=1.6)
    axR.set_xscale("log")
    axR.set_xlim(E[0], E[-1])
    axR.set(
        xlabel="Photon energy (eV)",
        ylabel="energy FWHM (eV)",
        title=f"Photon-counting resolution (Fano + {eag.READ_NOISE_E:g} e- read)",
    )
    axR.grid(alpha=0.3, which="both")
    fig.suptitle(
        f"Eagle XO {geo['sensor']} @ {geo['distance_m']:g} m  —  "
        rf"$\Omega$ = {geo['domega_sr']:.3e} sr "
        rf"($\Delta\theta$ = {geo['dtheta_obs_deg']:.2f}$\degree$, "
        f"{geo['active_mm'][0]:g}$\\times${geo['active_mm'][1]:g} mm)",
        fontsize=13,
    )
    return fig


def _eag_detected(r, settings, coating="BN", resolve_energy=False):
    """Incident and Eagle-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by eag.get_response."""
    incident = (r["spec"] + r["brem"]) * r["scale"]
    resp = eag.get_response(r["E_grid"], coating=coating, resolve_energy=resolve_energy)
    return incident, resp.apply(incident)


def _draw_eaglexo_detected(
    fig,
    trecs,
    settings,
    coating="BN",
    resolve_energy=False,
    collapse_azimuth=True,
    floor_frac=1e-4,
):
    """Render ONE polar tilt of the Eagle XO detected/incident view onto ``fig``
    (cleared first): all energies overlaid, incident dotted / detected solid, on
    log-log axes so the soft lines and the hard-brem roll-off both show. The wide
    brem (out to the beam energy) is included when present -- that is where the
    thin-sensor QE roll-off visibly crushes the background. A faint QE curve on a
    right axis shows why."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = 0.0
    xlo, xhi = np.inf, 0.0
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            inc, det = _eag_detected(r, settings, coating, resolve_energy)
            fin = inc[np.isfinite(inc)]
            if fin.size:
                ymax = max(ymax, float(fin.max()))
            xlo = min(xlo, float(r["E_grid"][0]))
            xhi = max(xhi, float(r["E_grid"][-1]))
            az = r["case"]["tilt_azim_deg"]
            ax.plot(r["E_grid"], inc, color=c, ls=":", lw=1.0, alpha=0.7)
            ax.plot(
                r["E_grid"],
                det,
                color=c,
                ls="-",
                lw=1.2,
                label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)",
            )
            if r.get("brem_wide") is not None:  # full range -> the brem roll-off
                Eb = np.asarray(r["E_grid_brem"], dtype=float)
                inc_b = r["brem_wide"] * r["scale"]
                det_b = inc_b * eag.qe(Eb, coating)
                xhi = max(xhi, float(Eb[-1]))
                if inc_b.size:  # keep the broad brem in the y-range, not clipped
                    ymax = max(ymax, float(np.nanmax(inc_b)))
                ax.plot(Eb, inc_b, color=c, ls=":", lw=0.8, alpha=0.5)
                ax.plot(Eb, det_b, color=c, ls="-", lw=0.8, alpha=0.9)
    case = trecs[0]["case"]
    ax.axvline(SI_K_EDGE_EV, color="0.4", ls="--", lw=0.8, label="Si-K edge 1.84 keV")
    # faint QE curve on a twin axis -- the "viewed through it" envelope
    axq = ax.twinx()
    Eqe = np.geomspace(max(xlo, 1.0), xhi if xhi > 0 else 6e4, 400)
    axq.plot(Eqe, eag.qe(Eqe, coating), color="0.6", ls="-", lw=1.0, alpha=0.6)
    axq.set_ylim(0, 1.05)
    axq.set_ylabel("QE", color="0.5")
    axq.tick_params(axis="y", colors="0.5")
    if ymax > 0:
        ax.set_yscale("log")
        ax.set_ylim(ymax * floor_frac, ymax * 2)
    ax.set_xscale("log")
    if xhi > 0:
        ax.set_xlim(xlo, xhi)
    blur = ", energy-resolved" if resolve_energy else ""
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:0.1f}\degree$ — Eagle XO "
        rf"detected (solid) vs incident (dotted), {coating}{blur}",
        fontsize=11,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Phs/eV/s/nA")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()


def plot_eaglexo_detected(
    results,
    settings,
    coating="BN",
    resolve_energy=False,
    collapse_azimuth=True,
    floor_frac=1e-4,
):
    """Incident (dotted) vs Eagle-XO-detected (solid) spectra, log-log; ONE figure
    per polar tilt, all energies overlaid (best azimuth each), with a faint QE
    envelope. Shows the camera's signature: soft PXR lines survive at ~90% QE
    while the hard bremsstrahlung is suppressed by the thin back-thinned sensor.
    Uses the wide brem grid when present (run the sweep with ``E_grid_brem``). The
    solid angle is whatever the sweep was run with -- point it at the Eagle with
    ``Sweep(..., **eaglexo_response.sweep_geometry(...))``. For click-through use
    ``browse(results, settings, kind="eaglexo")``."""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_eaglexo_detected,
        (9.0, 5.2),
        coating=coating,
        resolve_energy=resolve_energy,
        collapse_azimuth=collapse_azimuth,
        floor_frac=floor_frac,
    )


# NOTE: there is deliberately no ``plot_eaglexo_measured`` (a "measured spectrum")
# here. A bare Eagle XO is an INTEGRATING CCD -- it accumulates charge and cannot
# resolve individual photons, so it does not return a spectrum at all; presenting a
# Poisson "measured spectrum" misrepresents the instrument. What the camera
# actually reports is the recorded CHARGE -- see plot_eaglexo_charge (where the
# signal comes from) and plot_eaglexo_charge_map (the integrated geometry map).
# The energy-resolving photon-counting mode (eaglexo_response.poisson_counts /
# resolve_energy) is a special low-occupancy extra, not the default readout.


# ---- Eagle XO recorded-charge view (a CCD integrates charge, not photons) -----
def _eag_charge_rate(r, coating="BN"):
    """Total detected charge RATE [e-/s/nA] for one record: the Eagle XO
    integrates every photon that lands -- coherent lines AND bremsstrahlung --
    weighted by E/W_Si, over the full measured range when the wide brem grid is
    present. The scalar 'brightness' the CCD reports for a geometry (no spectrum).
    Coherent lines come from the fine line grid; the brem from the wide grid (or
    the line-grid brem as a fallback) so the two don't double-count their overlap."""
    resp = eag.get_response(r["E_grid"], coating=coating)
    q = resp.integrated_charge(r["spec"] * r["scale"])  # coherent lines [e-/s/nA]
    if r.get("brem_wide") is not None:
        Eb = np.asarray(r["E_grid_brem"], dtype=float)
        inc_b = np.nan_to_num(np.asarray(r["brem_wide"], dtype=float) * r["scale"])
        cd_b = inc_b * eag.qe(Eb, coating) * (Eb / eag.W_EHP_EV)
        q += float(np.trapezoid(cd_b, Eb))
    else:
        q += resp.integrated_charge(r["brem"] * r["scale"])
    return q


def _draw_eaglexo_charge(
    fig, trecs, settings, coating="BN", collapse_azimuth=True, floor_frac=1e-4
):
    """Render ONE polar tilt of the Eagle XO CHARGE spectral density [e-/eV/s]:
    where on the spectrum the CCD's recorded charge comes from, lines (fine grid,
    solid) + wide brem (dashed), log-log. Every photon is weighted by E/W_Si, so
    the hard brem carries far more charge than its photon count -- the legend
    reports each curve's integrated charge rate [e-/s]."""
    fig.clear()
    ax = fig.subplots(1, 1)
    cur = settings.beam_current_na
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax, xlo, xhi = 0.0, np.inf, 0.0
    for E0 in energies:
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            resp = eag.get_response(r["E_grid"], coating=coating)
            E = np.asarray(r["E_grid"], dtype=float)
            cd_line = resp.charge_density((r["spec"] + r["brem"]) * r["scale"]) * cur
            xlo, xhi = min(xlo, float(E[0])), max(xhi, float(E[-1]))
            fin = cd_line[np.isfinite(cd_line)]
            if fin.size:
                ymax = max(ymax, float(fin.max()))
            az = r["case"]["tilt_azim_deg"]
            rate = _eag_charge_rate(r, coating) * cur
            ax.plot(
                E,
                cd_line,
                color=c,
                lw=1.3,
                label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$, {rate:.2g} e$^-$/s)",
            )
            if r.get("brem_wide") is not None:
                Eb = np.asarray(r["E_grid_brem"], dtype=float)
                inc_b = np.nan_to_num(np.asarray(r["brem_wide"], dtype=float) * r["scale"])
                cd_b = inc_b * eag.qe(Eb, coating) * (Eb / eag.W_EHP_EV) * cur
                xhi = max(xhi, float(Eb[-1]))
                fb = cd_b[np.isfinite(cd_b)]
                if fb.size:
                    ymax = max(ymax, float(fb.max()))
                ax.plot(Eb, cd_b, color=c, ls="--", lw=0.8, alpha=0.85)
    case = trecs[0]["case"]
    ax.axvline(SI_K_EDGE_EV, color="0.4", ls="--", lw=0.8, label="Si-K edge 1.84 keV")
    if ymax > 0:
        ax.set_yscale("log")
        ax.set_ylim(ymax * floor_frac, ymax * 2)
    ax.set_xscale("log")
    if xhi > 0:
        ax.set_xlim(max(xlo, 1.0), xhi)
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}$={case['tilt_deg']:0.1f}$\degree$ — Eagle XO recorded "
        rf"charge density ({coating}, dashed = brem)",
        fontsize=11,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("charge density (e$^-$/eV/s)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()


def plot_eaglexo_charge(results, settings, coating="BN", collapse_azimuth=True, floor_frac=1e-4):
    """Eagle XO recorded CHARGE spectral density [e-/eV/s], ONE figure per polar
    tilt, all energies overlaid (best azimuth each), lines + wide brem on log-log.
    A CCD integrates charge rather than counting photons, so each photon is
    weighted by E/W_Si: this shows where the recorded signal actually comes from
    (the hard brem pulls more weight than its photon flux suggests). The companion
    to plot_eaglexo_detected (photon density) and plot_eaglexo_charge_map (the
    integrated geometry map)."""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_eaglexo_charge,
        (9.0, 5.2),
        coating=coating,
        collapse_azimuth=collapse_azimuth,
        floor_frac=floor_frac,
    )


def plot_eaglexo_charge_map(
    results,
    settings,
    cases=None,
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    coating="BN",
    exposure_s=None,
    auto_lines=True,
):
    """Geometry map of the Eagle XO's recorded SIGNAL: the integrated detected
    charge rate [e-/s] (coherent lines + brem, energy-weighted -- see
    _eag_charge_rate) over ``x`` x ``y``, one panel per ``panel`` value, the best
    (max) geometry per cell. This is the "what the CCD actually reports" view the
    QE-only detected-spectrum plots miss: a CCD integrates charge, it cannot
    resolve photons, so its figure of merit is collected charge, not line flux.

    With ``exposure_s`` set, the map shows the WELL-FILL FRACTION (collected e- /
    FULL_WELL_E) for that exposure at ``settings.beam_current_na`` instead -- how
    close the brightest geometry comes to saturating the well. ``cases`` restricts
    to one sweep (cf. plot_heatmaps). Honors the same thin-axis -> line-plot
    fallback as plot_heatmaps (``auto_lines``): a single-valued x or y becomes a
    line plot (signal vs the varying axis, one line per the other)."""
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        print("no results yet")
        return None
    cur = settings.beam_current_na

    def _val(r):
        rate = _eag_charge_rate(r, coating) * cur  # e-/s
        if exposure_s is not None:
            return rate * exposure_s / eag.FULL_WELL_E  # well-fill fraction
        return rate

    label = (
        f"well-fill fraction ({exposure_s:g} s @ {cur:g} nA)"
        if exposure_s is not None
        else "detected charge rate  (e$^-$/s)"
    )

    if auto_lines:
        nx = len({r["case"][x] for r in recs})
        ny = len({r["case"][y] for r in recs})
        if nx < 2 or ny < 2:
            line_x, thin = (x, y) if nx >= ny else (y, x)
            n_panel = len({r["case"][panel] for r in recs})
            hue = panel if n_panel > 1 else thin
            print(
                f"plot_eaglexo_charge_map: axis {thin!r} has <2 values -> line plot "
                f"(signal vs {line_x!r}, one line per {hue!r})."
            )
            hue_vals = sorted({r["case"][hue] for r in recs})
            div_x = _AXIS_SPECS.get(line_x, (None, 1.0))[1]
            fig, ax = plt.subplots(figsize=(8, 5))
            for j, hv in enumerate(hue_vals):
                hr = [r for r in recs if r["case"][hue] == hv]
                xs = sorted({r["case"][line_x] for r in hr})
                ys = [max(_val(r) for r in hr if r["case"][line_x] == xv) for xv in xs]
                col = energy_color(hv, hue_vals) if hue == "E0_keV" else COLORS[j % len(COLORS)]
                ax.plot(
                    [v / div_x for v in xs],
                    ys,
                    "o-",
                    color=col,
                    lw=1.8,
                    label=_value_label(hue, hv),
                )
            ax.set_xlabel(_axis_label(line_x))
            ax.set_ylabel(label)
            ax.set_title(f"Eagle XO recorded signal ({coating}, best per point)", fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(title=_AXIS_SPECS.get(hue, (hue,))[0], fontsize=9)
            fig.tight_layout()
            return fig

    panel_vals = sorted({r["case"][panel] for r in recs})
    panels = []
    for pv in panel_vals:
        er = [r for r in recs if r["case"][panel] == pv]
        xs = sorted({r["case"][x] for r in er})
        ys = sorted({r["case"][y] for r in er})
        xi = {v: i for i, v in enumerate(xs)}
        yi = {v: j for j, v in enumerate(ys)}
        best = {}  # (xv, yv) -> max signal
        for r in er:
            ck = (r["case"][x], r["case"][y])
            v = _val(r)
            if ck not in best or v > best[ck]:
                best[ck] = v
        Z = np.full((len(ys), len(xs)), np.nan)
        for (xv, yv), v in best.items():
            Z[yi[yv], xi[xv]] = v
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
        im = ax.pcolormesh(xe, ye, Z, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"{_AXIS_SPECS.get(panel, (panel,))[0]} = {_value_label(panel, pv)}")
        ax.set_xlabel(_axis_label(x))
        ax.set_ylabel(_axis_label(y))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    fig.suptitle(f"Eagle XO recorded signal: {label}  ({coating}, best per cell)", fontsize=13)
    return fig


# ---- electron trajectory + penetration view ----------------------------------
# Datashader rasterizes the (tens of thousands of) trajectory line-segments into
# ONE image per panel -- fast, and tiny on disk vs a matplotlib LineCollection of
# every segment -- while matplotlib keeps the crisp slab / beam / detector overlay
# and the energy colorbar (so the nbconvert PDF export still works). The segment
# colour is the electron's kinetic energy along the track (turbo); ds.max keeps it
# crisp under the line-width antialiasing (ds.mean would blend track edges low).
C_ANG_PER_FS = 2997.924580  # speed of light [Ang/fs]: age sum(L/beta)[Ang] -> fs
_TRAJ_CMAP = "turbo"


def _case_of(rec_or_case):
    """Accept either a results record (carries 'case') or a raw case dict."""
    return rec_or_case.get("case", rec_or_case)


def _trajectory_cases(cases_or_results):
    """Flatten a build_cases list OR a results store into a list of case dicts."""
    if isinstance(cases_or_results, dict):
        return [r["case"] for r in records(cases_or_results)]
    return [_case_of(c) for c in cases_or_results]


def _turbo_hex(n=256):
    """The turbo colormap as a hex list (the form datashader.shade wants)."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps[_TRAJ_CMAP]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def _beam_detector_basis(beam, n_hat):
    """Orthonormal 2D basis of the BEAM-DETECTOR plane: e1 = beam (-> +x, into the
    slab); e2 = the in-plane part of the detector direction (-> +y, "up"). Working
    in this plane (rather than a fixed x-z slice) keeps the beam horizontal AND the
    detector arrow pointing the right way for ANY polar/azimuthal tilt."""
    e1 = np.asarray(beam, float)
    e1 = e1 / np.linalg.norm(e1)
    nh = np.asarray(n_hat, float)
    nh = nh / np.linalg.norm(nh)
    perp = nh - np.dot(nh, e1) * e1
    if np.linalg.norm(perp) < 1e-9:  # detector ~parallel to beam: any in-plane up
        for ref in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            perp = ref - np.dot(ref, e1) * e1
            if np.linalg.norm(perp) > 1e-9:
                break
    return e1, perp / np.linalg.norm(perp)


def _trajectory_data(case, Ne, seed):
    """Simulate one case and project the cascade into the beam-detector plane.
    Returns 2D segment endpoints (M,2,2) in display units, per-segment energy/age/
    depth, the slab + detector unit vectors in that plane, and the back/through
    fractions."""
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    segs = simulate_trajectories(
        case["E0_keV"],
        Ne,
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=seed,
        beam_dir=beam,
    )
    e1, e2 = _beam_detector_basis(beam, n_hat)
    L, v, r = segs["L_ang"], segs["v_hat"], segs["r_mid"]
    start = r - 0.5 * L[:, None] * v
    u, ulab = (1e4, r"$\mu$m") if case["thickness_ang"] >= 1e4 else (10.0, "nm")

    # Continuous per-electron tracks (not a loose segment cloud): order segments by
    # (electron, age) so each electron's segment START points form a polyline --
    # consecutive starts share an endpoint, so they trace the real zig-zag path --
    # then break with NaN between electrons. This is what makes the tracks read as
    # paths (the old per-segment LineCollection got faint at the cool, slow tail).
    order = np.lexsort((segs["t_ang"], segs["elec_id"]))
    sx = (start @ e1)[order] / u
    sy = (start @ e2)[order] / u
    sE = segs["E_keV"][order]
    brk = np.flatnonzero(np.diff(segs["elec_id"][order]) != 0) + 1
    px = np.insert(sx, brk, np.nan)
    py = np.insert(sy, brk, np.nan)
    pE = np.insert(sE, brk, np.nan)

    z = np.array([0.0, 0.0, 1.0])  # slab normal in the sample frame
    ndet = np.array([n_hat @ e1, n_hat @ e2])
    ndet = ndet / np.linalg.norm(ndet)
    nslab = np.array([z @ e1, z @ e2])
    nn = np.linalg.norm(nslab)
    nslab = nslab / nn if nn > 1e-9 else np.array([1.0, 0.0])
    return dict(
        px=px,
        py=py,
        pE=pE,
        pts=np.column_stack([px, py]),  # for the shared-frame extent
        E=segs["E_keV"],
        t_fs=segs["t_ang"] / C_ANG_PER_FS,
        z_u=r[:, 2] / u,  # penetration depth below the surface, display units
        elec_id=segs["elec_id"],  # emitting electron index, per segment
        L=segs["L_ang"],
        ndet=ndet,
        nslab=nslab,
        u=u,
        ulab=ulab,
        thick=case["thickness_ang"] / u,
        eta=100.0 * segs["n_backscattered"] / segs["Ne"],
        thru=100.0 * segs["n_transmitted"] / segs["Ne"],
        Ne=segs["Ne"],
    )


def _trajectory_frame(pts_list, pct=99.0, pad=0.12, beam_frac=0.16):
    """ONE shared (xlo, xhi, ylo, yhi) for a set of panels, from the robust
    (1st/99th-percentile) extent of all their track vertices, expanded to include
    the origin and padded. Sharing it across tilts is what makes only the slab
    rotate frame-to-frame (the old per-panel autoscale was the "scaling is
    inconsistent" complaint). Symmetric in y (beam axis centred); the left margin
    always clears the beam arrow + label."""
    pts = np.concatenate([np.asarray(s).reshape(-1, 2) for s in pts_list], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    xlo = min(0.0, float(np.percentile(pts[:, 0], 100 - pct)))
    xhi = max(0.0, float(np.percentile(pts[:, 0], pct)))
    ymax = float(np.percentile(np.abs(pts[:, 1]), pct))
    sx = max(xhi - xlo, 1e-6)
    xlo -= pad * sx
    xhi += pad * sx
    yhi = max(ymax * (1.0 + pad), 1e-6)
    aL = beam_frac * (xhi - xlo)
    xlo = min(xlo, -1.5 * aL)  # room for the beam arrow + label
    return (float(xlo), float(xhi), float(-yhi), float(yhi))


def _square_frame(frame):
    """Expand the shorter side of a (xlo, xhi, ylo, yhi) frame symmetrically so it
    is SQUARE -- no data is cropped, the extra room becomes centred margin. With
    set_aspect("equal") this lets square subplot boxes hold the tracks without the
    skinny-strip letterboxing the wide native frame produced (the trajectory-grid
    sizing fix)."""
    xlo, xhi, ylo, yhi = frame
    w, h = xhi - xlo, yhi - ylo
    if w > h:
        pad = 0.5 * (w - h)
        ylo, yhi = ylo - pad, yhi + pad
    elif h > w:
        pad = 0.5 * (h - w)
        xlo, xhi = xlo - pad, xhi + pad
    return (float(xlo), float(xhi), float(ylo), float(yhi))


def _draw_trajectory_panel(
    ax,
    data,
    frame,
    E0,
    *,
    E_cut=5.0,
    px=820,
    spread_px=1,
    cmap=None,
    label=True,
    label_fs=8.5,
):
    """Render ONE penetration cross-section into ``ax`` over the shared ``frame``:
    grey slab, datashader-rasterized energy-coloured tracks, red beam + green
    detector arrows.

    The tracks are aggregated with ``line_width=0`` so each pixel takes the true
    electron energy of the track through it -- antialiased (line_width>0) lines
    instead coverage-weight that value, which paints a bogus radial gradient
    ACROSS the line thickness (hot centre -> cool edges) rather than along the
    path. ``tf.spread`` then thickens the crisp 1-px lines back to visibility
    WITHOUT reintroducing that artifact (it copies each pixel's colour outward)."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    xlo, xhi, ylo, yhi = frame
    nslab, ndet, thick = data["nslab"], data["ndet"], data["thick"]

    # slab polygon: front face through the origin, extending `thick` into +nslab
    tang = np.array([-nslab[1], nslab[0]])
    W = 6.0 * max(xhi - xlo, yhi - ylo)
    slab = np.array([-W * tang, W * tang, W * tang + thick * nslab, -W * tang + thick * nslab])
    ax.fill(slab[:, 0], slab[:, 1], facecolor="0.80", edgecolor="0.55", lw=1.0, zorder=1)

    # continuous NaN-separated per-electron tracks -> datashader raster, colour =
    # electron energy (ds.max keeps it crisp under the line-width antialiasing)
    df = pd.DataFrame({"x": data["px"], "y": data["py"], "E": data["pE"]})
    asp = (yhi - ylo) / (xhi - xlo)
    cvs = ds.Canvas(
        plot_width=px,
        plot_height=max(int(px * asp), 60),
        x_range=(xlo, xhi),
        y_range=(ylo, yhi),
    )
    agg = cvs.line(df, "x", "y", agg=ds.max("E"), line_width=0)  # crisp: true E/pixel
    img = tf.shade(agg, cmap=cmap or _turbo_hex(), span=(E_cut, E0), how="linear")
    if spread_px:
        img = tf.spread(img, px=spread_px, shape="circle")  # thicken, colour kept
    ax.imshow(
        np.asarray(img.to_pil()),
        extent=(xlo, xhi, ylo, yhi),
        origin="upper",
        aspect="equal",
        interpolation="none",
        zorder=2,
    )

    # beam (red) + detector (green) arrows, anchored at the entry point
    aL = 0.16 * (xhi - xlo)
    ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(-aL, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(ndet[0] * aL, ndet[1] * aL),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#119911", lw=2.0),
        zorder=4,
    )
    if label:
        ax.text(
            -aL * 0.5,
            0.03 * (yhi - ylo),
            "beam",
            color="red",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            zorder=5,
        )
        tx = float(np.clip(ndet[0] * aL * 1.1, xlo + 0.06 * (xhi - xlo), xhi - 0.06 * (xhi - xlo)))
        ty = float(np.clip(ndet[1] * aL * 1.1, ylo + 0.06 * (yhi - ylo), yhi - 0.1 * (yhi - ylo)))
        ax.text(
            tx,
            ty,
            "detector",
            color="#0a6a0a",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            zorder=5,
        )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")


def _traj_colorbar(ax, E_cut, E0, label="electron energy (keV)"):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    sm = cm.ScalarMappable(norm=Normalize(E_cut, E0), cmap=_TRAJ_CMAP)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    return cb


def plot_electron_trajectories(
    rec_or_case,
    *,
    Ne=200,
    seed=0,
    frame=None,
    E_cut=5.0,
    colorbar=True,
    spread_px=1,
    label=True,
    ax=None,
):
    """One electron-penetration cross-section in the beam-detector plane: the beam
    enters horizontally at the origin (red), the crystal is the grey slab (which
    rotates with the tilt), the detector direction is the green arrow, and the
    cascade is datashader-rasterized, coloured by electron energy.

    ``frame`` (xlo, xhi, ylo, yhi) fixes the axes so repeated calls at the same
    (material, thickness, energy) share ONE frame and only the slab rotates; None
    auto-fits this case. ``ax`` draws into an existing axis (used by the grid)."""
    case = _case_of(rec_or_case)
    data = _trajectory_data(case, Ne, seed)
    if frame is None:
        frame = _trajectory_frame([data["pts"]])
    if ax is None:
        xlo, xhi, ylo, yhi = frame
        asp = (yhi - ylo) / (xhi - xlo)
        # size the FIGURE to the data aspect so the equal-aspect axes fills it
        # (no floating-title letterbox): reserve ~1.7" width for ylabel+colorbar
        # and ~1.1" height for title+xlabel, then constrained_layout packs it.
        axw = 5.4
        _, ax = plt.subplots(
            figsize=(axw + 1.7, float(np.clip(axw * asp + 1.1, 3.2, 8.4))),
            constrained_layout=True,
        )
    _draw_trajectory_panel(
        ax, data, frame, case["E0_keV"], E_cut=E_cut, spread_px=spread_px, label=label
    )
    ax.set_xlabel(f"distance along beam ({data['ulab']})")
    ax.set_ylabel(f"transverse distance ({data['ulab']})")
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['E0_keV']:g} keV, "
        rf"$\theta$={case.get('tilt_deg', 0.0):g}$\degree$, "
        rf"$\phi$={case.get('tilt_azim_deg', 0.0):g}$\degree$  —  {data['Ne']} e$^-$ "
        rf"({data['eta']:.0f}% back, {data['thru']:.0f}% through)",
        fontsize=10,
    )
    if colorbar:
        _traj_colorbar(ax, E_cut, case["E0_keV"])
    return ax


def plot_trajectory_grid(
    cases_or_results,
    energy=None,
    *,
    Ne=150,
    seed=0,
    E_cut=5.0,
    spread_px=1,
    max_panels=12,
    ncols=None,
    max_width_in=13.0,
    max_height_in=11.0,
    panel_in=3.3,
    min_panel_in=2.5,
):
    """Electron-penetration cross-sections at ONE beam energy, a panel per
    (polar, azimuthal) tilt -- the trajectory analogue of plot_heatmaps. Every
    panel shares ONE (squared) frame, so across the grid ONLY the slab rotates;
    the cascade is datashader-rasterized and energy-coloured with a single shared
    colorbar.

    ``cases_or_results`` is a build_cases list or a results store; ``energy`` picks
    the beam energy (default the lowest). If both polar and azimuthal tilt are
    swept it lays out a polar x azimuth grid (like the heatmaps); otherwise it
    wraps the swept tilt into at most 3 columns (override with ``ncols``). ``Ne``
    (electrons/panel) trades detail for speed -- the electron transport, not the
    drawing, is the cost.

    Panel SIZE: each panel is SQUARE (the shared frame is squared so set_aspect
    "equal" stays exact) and targets ``panel_in`` inches (~3.3", matching
    plot_best_spectra), never shrinking below ``min_panel_in``. For more tilts than
    fit in 3 columns the figure grows TALLER (scrollable) rather than crushing
    panels -- ``max_height_in`` no longer shrinks them. ``max_panels`` caps how
    many (polar, azimuth) combos are drawn (evenly subsampled past the cap); raise
    it (and/or ``panel_in``) for a denser grid, lower it for bigger panels."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    energies = sorted({c["E0_keV"] for c in cases})
    energy = energies[0] if energy is None else min(energies, key=lambda e: abs(e - energy))
    grp = [c for c in cases if c["E0_keV"] == energy]

    polars = sorted({c["tilt_deg"] for c in grp})
    azims = sorted({c["tilt_azim_deg"] for c in grp})
    grid2d = len(polars) > 1 and len(azims) > 1
    # one representative case per (polar, azimuth) combo, in a stable order
    bycombo = {}
    for c in sorted(grp, key=lambda c: (c["tilt_deg"], c["tilt_azim_deg"])):
        bycombo.setdefault((c["tilt_deg"], c["tilt_azim_deg"]), c)
    combos = list(bycombo)
    if len(combos) > max_panels:  # subsample evenly so the grid stays on-screen
        keep = np.unique(np.linspace(0, len(combos) - 1, max_panels).round().astype(int))
        combos = [combos[i] for i in keep]
        grid2d = False
    data = {cb: _trajectory_data(bycombo[cb], Ne, seed) for cb in combos}
    # square the shared frame so the panels read as squares (the native frame is
    # wide -> skinny strips); no data is cropped, set_aspect("equal") stays exact.
    frame = _square_frame(_trajectory_frame([d["pts"] for d in data.values()]))

    if grid2d:
        nrows, ncols = len(polars), len(azims)
        cell = [[(p, a) for a in azims] for p in polars]
    else:
        n = len(combos)
        ncols = ncols or min(3, n)  # cap at 3 columns; wrap the rest into ROWS
        nrows = int(np.ceil(n / ncols))
        cell = [
            [combos[r * ncols + col] if r * ncols + col < n else None for col in range(ncols)]
            for r in range(nrows)
        ]

    # SQUARE panels (the "trajectory plots are tiny / skinny" fix): the frame is
    # square so asp == 1 and pw == ph. Target ``panel_in`` (~3.3", matching
    # plot_best_spectra), clamp to the per-column width budget, never shrink below
    # min_panel_in -- for many tilts the FIGURE grows TALLER (scrollable) rather
    # than crushing every panel. ``max_height_in`` is no longer used to shrink.
    xlo, xhi, ylo, yhi = frame
    asp = (yhi - ylo) / (xhi - xlo)  # == 1 after _square_frame
    pw = min(panel_in, (max_width_in - 1.1) / ncols)
    pw = max(pw, min_panel_in)
    ph = pw * asp
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * pw + 1.1, nrows * ph + 0.7),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for r in range(nrows):
        for col in range(ncols):
            ax = axes[r][col]
            combo = cell[r][col]
            if combo is None or combo not in data:
                ax.axis("off")
                continue
            d = data[combo]
            _draw_trajectory_panel(
                ax, d, frame, energy, E_cut=E_cut, spread_px=spread_px, label=False
            )
            ax.set_title(
                rf"$\theta$={combo[0]:g}$\degree$, $\phi$={combo[1]:g}$\degree$",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
            if r == nrows - 1:
                ax.set_xlabel(f"along beam ({d['ulab']})", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"transverse ({d['ulab']})", fontsize=8)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    mappable = cm.ScalarMappable(norm=Normalize(E_cut, energy), cmap=_TRAJ_CMAP)
    cb = fig.colorbar(mappable, ax=axes, shrink=0.8, aspect=30, pad=0.01)
    cb.set_label("electron energy (keV)")
    case0 = bycombo[combos[0]]
    fig.suptitle(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"{energy:g} keV — electron penetration (red beam, green detector; "
        rf"only the slab rotates)",
        fontsize=11,
    )
    return fig


def plot_penetration_survival(
    cases_or_results,
    *,
    Ne=500,
    seed=0,
    n_bins=80,
    tilt=None,
    depth_frac=True,
):
    """Surviving electron population vs penetration depth -- the fraction of the
    incident electrons (% of N0) that reach AT LEAST a depth z below the entrance
    surface, one monotonically-decreasing curve per beam energy. This is where the
    beam stops: the curve falls from 100% at the surface to 0 at the deepest
    penetration, and a higher-energy beam reaches deeper.

    Per electron the deepest segment it reaches sets its penetration depth (the
    max over its segment midpoints' depth below the slab normal); then
    ``survival(z) = (# electrons reaching depth >= z) / N0``.
    ``tilt`` selects the polar tilt (nearest; default the one closest to normal
    incidence). ``depth_frac`` plots depth as a fraction of the slab thickness (so
    thin and thick slabs overlay); set False for absolute depth."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    tilts = sorted({c["tilt_deg"] for c in cases})
    want = 0.0 if tilt is None else tilt
    t = min(tilts, key=lambda x: abs(x - want))
    energies = sorted({c["E0_keV"] for c in cases})

    fig, ax = plt.subplots(figsize=(8, 5))
    xmax = 1.0
    for E0 in energies:
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is None:
            continue
        d = _trajectory_data(c, Ne, seed)
        # deepest point each electron reaches (max over its segment depths), then
        # clip the tiny negative excursions of backscattered electrons that exit
        # just above the entrance face.
        max_depth = np.full(d["Ne"], -np.inf)
        np.maximum.at(max_depth, d["elec_id"], d["z_u"])
        max_depth = np.clip(max_depth[np.isfinite(max_depth)], 0.0, None)
        thick = d["thick"]
        x = max_depth / thick if depth_frac else max_depth
        xmax = 1.0 if depth_frac else max(xmax, float(thick))
        zs = np.linspace(0.0, 1.0 if depth_frac else float(thick), n_bins)
        surv = 100.0 * np.array([float((x >= z).mean()) for z in zs])
        ax.plot(zs, surv, "-", color=energy_color(E0, energies), lw=1.9, label=f"{E0:g} keV")
    case0 = next(c for c in cases if c["tilt_deg"] == t)
    ulab = r"$\mu$m" if case0["thickness_ang"] >= 1e4 else "nm"
    xlab = "depth / thickness" if depth_frac else f"penetration depth ({ulab})"
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"surviving electrons (% of $N_0$)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy", fontsize=9)
    ax.set_title(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}$={t:g}$\degree$ — electron penetration / survival",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


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
    from .results import results_dataframe

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
            hue_groups = sub.groupby(hue) if hue is not None else [(None, sub)]
            for hv, g in hue_groups:
                agg = g.groupby(x)[y].agg(reduce).reset_index().sort_values(x)
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


def plot_best_spectra(
    results,
    settings,
    top_n=12,
    select="quality_peak",
    include_brem=True,
    cases=None,
    ncols=3,
    rel_prominence=0.03,
    line_metric="sharpness",
):
    """The top-``top_n`` geometries across the WHOLE sweep, ranked by
    results.selection_score(``select``) -- one intrinsic spectrum panel each,
    titled with the geometry, the score components, and the line quality. The
    answer to "thousands of cases, which few do I look at": instead of paging
    every polar tilt, see the best dozen at a glance. ``select`` defaults to
    peak_flux x line_quality (bright AND well-defined), not the raw peak the
    per-tilt browser collapses on -- so spurious tall spikes don't win."""
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        print("no results yet")
        return None
    metrics = {id(r): line_metrics(r, settings, rel_prominence, metric=line_metric) for r in recs}
    ranked = sorted(recs, key=lambda r: selection_score(metrics[id(r)], select), reverse=True)[
        :top_n
    ]
    all_E = {r["case"]["E0_keV"] for r in recs}
    n = len(ranked)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for k, r in enumerate(ranked):
        ax = axes[k // ncols][k % ncols]
        c, m = r["case"], metrics[id(r)]
        line_det, brem_det = _line_brem(r, settings, convolve=False)
        E = r["E_grid"] / 1e3
        col = energy_color(c["E0_keV"], all_E)
        ax.plot(
            E,
            (line_det + brem_det if include_brem else line_det) * r["scale"],
            color=col,
            lw=1.1,
        )
        if include_brem:
            ax.plot(E, brem_det * r["scale"], color=col, ls="--", lw=0.5)
        ax.set_title(
            rf"#{k + 1} {c['name'].split()[0]} {c['E0_keV']:g}keV"
            "\n"
            rf"$\theta$={c['tilt_deg']:g}$\degree$ $\phi$={c['tilt_azim_deg']:g}$\degree$  "
            rf"q={m['line_quality']:.2f}, {m['line_eV']:.0f}eV",
            fontsize=7.5,
        )
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(alpha=0.3)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.supxlabel("Photon energy (keV)", fontsize=9)
    fig.supylabel("Intensity (Phs/eV/s/nA)", fontsize=9)
    fig.suptitle(f"Top {n} geometries by {select} (dashed = brem)", fontsize=12)
    return fig


def plot_material_comparison(
    results_by_material,
    settings,
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
):
    """Cross-material headline: for each material's results store, find the single
    BEST geometry/energy (results.selection_score ``select``) and plot its
    dominant coherent line ENERGY vs its integrated line FLUX -- one point per
    material, coloured by line-definition quality. Answers "which crystal gives
    the brightest well-defined line, and at what energy" for comparison against
    the paper's catalogue.

    ``results_by_material`` : ``{label: results_store}``, e.g. built in the
    notebook with ``{m: load_checkpoint(m) for m in MATERIALS}`` (skip empties)."""
    pts = []  # (label, line_eV, line_flux, quality, case)
    for label, results in results_by_material.items():
        recs = records(results)
        if not recs:
            continue
        metrics = {
            id(r): line_metrics(r, settings, rel_prominence, metric=line_metric) for r in recs
        }
        best = max(recs, key=lambda r: selection_score(metrics[id(r)], select))
        m = metrics[id(best)]
        pts.append((label, m["line_eV"], m["line_flux"], m["line_quality"], best["case"]))
    if not pts:
        print("no results in any material")
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sc = ax.scatter(
        [p[1] / 1e3 for p in pts],
        [p[2] for p in pts],
        c=[p[3] for p in pts],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=110,
        edgecolor="k",
        zorder=3,
    )
    for label, eV, flux, _q, case in pts:
        ax.annotate(
            f"  {label} ({case['E0_keV']:g} keV)",
            (eV / 1e3, flux),
            fontsize=8,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_xlabel("dominant coherent line energy (keV)")
    ax.set_ylabel("integrated line flux at best geometry (Phs/s)")
    ax.set_title(f"Best coherent line per material  (select: {select})", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.12)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("line-definition quality")
    fig.tight_layout()
    return fig
