"""
cxr_plots.py
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
  * :func:`plot_eaglexo_efficiency`, :func:`plot_eaglexo_detected`,
    :func:`plot_eaglexo_measured` -- the direct-detection CCD (solid angle x QE):
    soft PXR lines pass at ~90% QE while the hard brem is crushed by the thin
    sensor. Browse per tilt with ``browse(results, settings, kind="eaglexo")``.

Everything takes ``results`` + a :class:`cxr_results.Settings` explicitly.
"""

import numpy as np
import matplotlib.pyplot as plt

from cxr_montecarlo import (
    convolve_detector,
    detector_efficiency,
    simulate_trajectories,
    tilted_geometry,
)
from cxr_results import (
    detected_background,
    records,
    best_azimuth,
    show_summary,
    line_metrics,
    selection_score,
    PER_NA,
)
import timepix_response as tpx
import eaglexo_response as eag

COLORS = ["r", "y", "g", "b", "m", "c", "k", "orange", "purple", "brown"]

# Beam energy -> colour, CONSISTENT across every figure: a given E0 always gets
# the same colour (keyed to its rank in the sorted energy set, so e.g. 30/45/60
# keV map to the same three colours everywhere), and the palette stays readable
# on white (no low-contrast yellow). Pass the FULL set of energies present in
# the figure so the rank -- hence the colour -- is stable panel to panel.
_ENERGY_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
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
    return (
        "EDS-convolved"
        if getattr(settings, "convolve_with_det", False)
        else "intrinsic"
    )


def _line_brem(r, settings, convolve=None):
    """Detected line and brem densities (per eV, before the unit scale) for one
    record, honoring the QE / brem-source flags. ``convolve`` overrides
    settings.convolve_with_det when given (True/False), so a caller can draw the
    intrinsic (convolve=False) and detector-convolved (convolve=True) spectra
    side by side."""
    do_conv = (
        getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    )
    qe = detector_efficiency(r["E_grid"]) if settings.apply_detector_qe else 1.0
    line_in = r["spec"] * qe
    line_det = (
        convolve_detector(r["E_grid"], line_in, r["fwhm"]) if do_conv else line_in
    )
    brem_det = detected_background(r, settings, convolve=do_conv) / r["scale"]
    return line_det, brem_det


# ---- interactive viewer ------------------------------------------------------
def browse(results, settings, kind="by_energy", label="polar tilt", static=None, **kw):
    """Page through one figure type BY POLAR TILT instead of printing every tilt
    stacked. ``kind``: "by_energy" | "full" | "chunk" | "timepix" | "eaglexo".

    A tilt slider + Prev/Next swaps a freshly-drawn figure into an output area --
    reliable on the **inline** backend (recommended; no ``%matplotlib widget`` /
    ipympl needed, and it behaves over SSH). ``static=True`` (or no ipywidgets,
    e.g. nbconvert -> PDF) instead draws every tilt stacked so the export holds
    them all. Extra kwargs pass to the per-tilt drawer (include_brem, floor_frac,
    n_mc)."""
    # figure sizes kept within an XPS-15 notebook width (~12") so nothing needs
    # horizontal scrolling; single-axis spectra are ~9.5x5.3, the 2-panel chunk
    # is wider but shorter.
    drawers = {
        "by_energy": (_draw_by_energy, (9.5, 5.3)),
        "full": (_draw_full_spectrum, (9.5, 5.3)),
        "chunk": (_draw_chunk, (11.0, 4.8)),
        "timepix": (_draw_timepix_detected, (9.5, 5.3)),
        "eaglexo": (_draw_eaglexo_detected, (9.5, 5.3)),
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

    TODO (a) -- interactive speed: this redraws a full matplotlib/Agg figure on
    EVERY slider move (thousands of points x several curves), which is the slow
    click-through. The fix is a Plotly-based spectral browser: build ONE figure
    with each tilt/energy as a go.Scattergl (WebGL) trace and toggle visibility
    from a client-side slider/dropdown -- no Python redraw per click, so paging is
    instant and 10k+ points stay smooth. Keep this matplotlib path for static /
    nbconvert-PDF export (where there is no client to run the JS); add a
    `browse_plotly(...)` alongside it for interactive use. Adds a `plotly`
    dependency (+ `kaleido` only if you also want Plotly static export)."""
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
    nxt.on_click(
        lambda b: setattr(slider, "value", min(len(tilts) - 1, slider.value + 1))
    )
    slider.observe(lambda ch: render(ch["new"]), names="value")
    display(widgets.HBox([prev, slider, nxt]))
    display(out)
    render(0)


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
    for i, E0 in enumerate(energies):
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
        records(results), settings, _draw_by_energy, (9.5, 5.3),
        include_brem=include_brem, collapse_azimuth=collapse_azimuth,
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
    xmin = 0.0
    for i, E0 in enumerate(energies):
        grp = [
            r
            for r in trecs
            if r["case"]["E0_keV"] == E0 and r.get("brem_wide") is not None
        ]
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
        xmin = float(Eb[0])
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
        lo = max(xmin, 1.0) if logx else xmin  # log x can't show 0 (brem grid -> 0)
        ax.set_xlim(lo, xmax)  # span the full brem grid (to the beam energy)
    else:
        ax.margins(x=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_full_spectrum(
    results, settings, collapse_azimuth=True, logy=True, floor_frac=1e-5
):
    """Full measured-range view (sharp lines on the wide brem, log-log), ONE figure
    per polar tilt. The x-axis spans the full brem grid (to the beam energy) and
    the y-floor follows the brem continuum, so the broad bremsstrahlung shoulder
    is on-screen instead of clipped under the lines (``floor_frac`` caps the depth
    at floor_frac x the peak). For click-through use
    ``browse(results, settings, kind="full")``. Needs records run with a separate
    ``E_grid_brem`` (``brem_wide`` present)."""
    recs = [r for r in records(results) if r.get("brem_wide") is not None]
    return _per_tilt_figs(
        recs, settings, _draw_full_spectrum, (9.5, 5.3),
        empty_msg="no wide-brem records -- set E_grid_brem in the Sweep and re-run",
        collapse_azimuth=collapse_azimuth, logy=logy, floor_frac=floor_frac,
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
    for i, (E0, rs) in enumerate(sorted(by_E.items())):
        rs = sorted(rs, key=lambda r: r["case"]["tilt_deg"])
        tilts = [r["case"]["tilt_deg"] for r in rs]
        peak = [
            float(np.max(r["spec"])) * r["scale"] * settings.beam_current_na for r in rs
        ]
        ax.plot(tilts, peak, "o-", color=energy_color(E0, by_E), label=f"{E0:g} keV")
    ax.set_xlabel(r"polar tilt $\theta_\mathrm{tilt}$ (deg)")
    ax.set_ylabel("best-azimuth peak (Phs/eV/s)")
    ax.set_title("Peak spectral flux vs polar tilt (best azimuth per point)")
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy")
    fig.tight_layout()
    return fig


def _draw_chunk(fig, trecs, settings):
    """Render ONE polar tilt's best-azimuth INTRINSIC spectra onto ``fig`` (cleared
    first): LEFT total (coherent + brem), RIGHT brem-subtracted CXR only. The
    detector view is the separate Eagle XO browser (kind='eaglexo')."""
    fig.clear()
    best = sorted(best_azimuth(trecs), key=lambda r: r["case"]["E0_keV"])
    if not best:
        return
    ax_tot, ax_cxr = fig.subplots(1, 2, sharex=True)
    energies = [r["case"]["E0_keV"] for r in best]
    for r in best:
        az, E0 = r["case"]["tilt_azim_deg"], r["case"]["E0_keV"]
        c = energy_color(E0, energies)
        lbl = rf"{E0:g} keV ($\phi={az:g}\degree$)"
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
    axR.plot(
        E_eff, tpx.energy_fwhm_eV(E_eff), "k-", lw=1.5, label="analytic, single-pixel"
    )
    axR.plot(
        E_eff, resp["fwhm_rec"], "b.", ms=4, label="MC effective (tail + multi-pixel)"
    )
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
    for i, E0 in enumerate(energies):
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
        records(results), settings, _draw_timepix_detected, (9.0, 5.2),
        thickness_um=thickness_um, bias_v=bias_v, collapse_azimuth=collapse_azimuth,
        n_mc=n_mc, seed=seed, floor_frac=floor_frac,
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
    fig, axes = plt.subplots(
        1, len(energies), figsize=(min(3.7 * len(energies), 11.5), 4.4), squeeze=False
    )
    for ax, E0 in zip(axes.ravel(), energies):
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
        f"Timepix3 Poisson 'measured' spectra "
        f"({thickness_um:g} $\\mu$m Si, {bias_v:g} V)",
        fontsize=14,
    )
    fig.tight_layout()
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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
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
    fig.tight_layout()
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
    for i, E0 in enumerate(energies):
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
        records(results), settings, _draw_eaglexo_detected, (9.0, 5.2),
        coating=coating, resolve_energy=resolve_energy,
        collapse_azimuth=collapse_azimuth, floor_frac=floor_frac,
    )


def plot_eaglexo_measured(
    results, settings, integration_s=600.0, coating="BN", resolve_energy=False, seed=0
):
    """A Poisson 'measured' realization (photon-counting mode) for the highest-rate
    config at each energy, over ``integration_s`` at the configured beam current.
    The detected mean is incident x QE; counts are Poisson per bin. The title
    reports the solid angle baked into the records."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return None
    rng = np.random.default_rng(seed)
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    fig, axes = plt.subplots(
        1, len(energies), figsize=(min(3.7 * len(energies), 11.5), 4.4), squeeze=False
    )
    for ax, E0 in zip(axes.ravel(), energies):
        grp = [r for r in recs if r["case"]["E0_keV"] == E0]
        r = max(grp, key=lambda r: float(np.max(r["spec"])))
        _, det = _eag_detected(r, settings, coating, resolve_energy)
        counts, expected = eag.poisson_counts(
            r["E_grid"], det * settings.beam_current_na, integration_s, rng=rng
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
        ax.axvline(SI_K_EDGE_EV, color="b", ls=":", lw=0.8, label="Si-K edge")
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
    dom = _domega_of(recs[0])
    fig.suptitle(
        f"Eagle XO Poisson 'measured' spectra  "
        rf"($\Omega$ = {dom:.3e} sr, {coating})",
        fontsize=14,
    )
    fig.tight_layout()
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
    return rec_or_case["case"] if "case" in rec_or_case else rec_or_case


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


def _draw_trajectory_panel(
    ax, data, frame, E0, *, E_cut=5.0, px=820, spread_px=1, cmap=None, label=True,
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
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf

    xlo, xhi, ylo, yhi = frame
    nslab, ndet, thick = data["nslab"], data["ndet"], data["thick"]

    # slab polygon: front face through the origin, extending `thick` into +nslab
    tang = np.array([-nslab[1], nslab[0]])
    W = 6.0 * max(xhi - xlo, yhi - ylo)
    slab = np.array(
        [-W * tang, W * tang, W * tang + thick * nslab, -W * tang + thick * nslab]
    )
    ax.fill(slab[:, 0], slab[:, 1], facecolor="0.80", edgecolor="0.55", lw=1.0, zorder=1)

    # continuous NaN-separated per-electron tracks -> datashader raster, colour =
    # electron energy (ds.max keeps it crisp under the line-width antialiasing)
    df = pd.DataFrame({"x": data["px"], "y": data["py"], "E": data["pE"]})
    asp = (yhi - ylo) / (xhi - xlo)
    cvs = ds.Canvas(
        plot_width=px, plot_height=max(int(px * asp), 60),
        x_range=(xlo, xhi), y_range=(ylo, yhi),
    )
    agg = cvs.line(df, "x", "y", agg=ds.max("E"), line_width=0)  # crisp: true E/pixel
    img = tf.shade(agg, cmap=cmap or _turbo_hex(), span=(E_cut, E0), how="linear")
    if spread_px:
        img = tf.spread(img, px=spread_px, shape="circle")  # thicken, colour kept
    ax.imshow(
        np.asarray(img.to_pil()), extent=(xlo, xhi, ylo, yhi),
        origin="upper", aspect="equal", interpolation="none", zorder=2,
    )

    # beam (red) + detector (green) arrows, anchored at the entry point
    aL = 0.16 * (xhi - xlo)
    ax.annotate(
        "", xy=(0.0, 0.0), xytext=(-aL, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0), zorder=4,
    )
    ax.annotate(
        "", xy=(ndet[0] * aL, ndet[1] * aL), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#119911", lw=2.0), zorder=4,
    )
    if label:
        ax.text(
            -aL * 0.5, 0.03 * (yhi - ylo), "beam", color="red", fontsize=label_fs,
            ha="center", va="bottom", zorder=5,
        )
        tx = float(np.clip(ndet[0] * aL * 1.1, xlo + 0.06 * (xhi - xlo), xhi - 0.06 * (xhi - xlo)))
        ty = float(np.clip(ndet[1] * aL * 1.1, ylo + 0.06 * (yhi - ylo), yhi - 0.1 * (yhi - ylo)))
        ax.text(
            tx, ty, "detector", color="#0a6a0a", fontsize=label_fs,
            ha="center", va="bottom", zorder=5,
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
    rec_or_case, *, Ne=200, seed=0, frame=None, E_cut=5.0, colorbar=True,
    spread_px=1, label=True, ax=None,
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
        axw = 4.7
        _, ax = plt.subplots(
            figsize=(axw + 1.7, float(np.clip(axw * asp + 1.1, 3.0, 7.6))),
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
    cases_or_results, energy=None, *, Ne=150, seed=0, E_cut=5.0, spread_px=1,
    max_panels=20, ncols=None, max_width_in=12.0, max_height_in=8.5,
):
    """Electron-penetration cross-sections at ONE beam energy, a panel per
    (polar, azimuthal) tilt -- the trajectory analogue of plot_heatmaps. Every
    panel shares ONE frame (computed from the union of all the clouds), so across
    the grid ONLY the slab rotates; the cascade is datashader-rasterized and
    energy-coloured with a single shared colorbar.

    ``cases_or_results`` is a build_cases list or a results store; ``energy`` picks
    the beam energy (default the lowest). If both polar and azimuthal tilt are
    swept it lays out a polar x azimuth grid (like the heatmaps); otherwise it
    wraps the swept tilt into ``ncols`` columns. ``Ne`` (electrons/panel) trades
    detail for speed -- the electron transport, not the drawing, is the cost."""
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
    frame = _trajectory_frame([d["pts"] for d in data.values()])

    if grid2d:
        nrows, ncols = len(polars), len(azims)
        cell = [[(p, a) for a in azims] for p in polars]
    else:
        n = len(combos)
        ncols = ncols or int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        cell = [
            [combos[r * ncols + col] if r * ncols + col < n else None for col in range(ncols)]
            for r in range(nrows)
        ]

    # Size each panel to the shared data aspect so the equal-aspect axes fill
    # their cells (no per-panel letterbox), reserving ~1.1" for the colorbar and
    # ~0.7" for the suptitle; constrained_layout then packs it with no big gaps.
    xlo, xhi, ylo, yhi = frame
    asp = (yhi - ylo) / (xhi - xlo)
    pw = min((max_width_in - 1.1) / ncols, 2.7)
    ph = pw * asp
    if nrows * ph + 0.7 > max_height_in:  # shrink panels so the grid fits on-screen
        pw *= (max_height_in - 0.7) / (nrows * ph)
        ph = pw * asp
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * pw + 1.1, nrows * ph + 0.7), squeeze=False,
        sharex=True, sharey=True, constrained_layout=True,
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


def plot_penetration_profile(
    cases_or_results, *, Ne=500, seed=0, n_bins=40, tilt=None, depth_frac=True,
):
    """TODO #1: mean electron ENERGY vs penetration depth -- the slowing-down /
    penetration profile -- with the mean electron AGE (lifetime, sum L/beta -> fs)
    vs depth alongside, one curve per beam energy.

    Depth is z below the entrance surface (the slab normal), path-length-weighted
    so each bin reflects where the electrons actually spend their track length.
    ``tilt`` selects the polar tilt (nearest; default the one closest to normal
    incidence). ``depth_frac`` plots depth as a fraction of the slab thickness
    (so thin and thick slabs overlay); set False for absolute depth."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    tilts = sorted({c["tilt_deg"] for c in cases})
    want = 0.0 if tilt is None else tilt
    t = min(tilts, key=lambda x: abs(x - want))
    energies = sorted({c["E0_keV"] for c in cases})

    fig, (axE, axT) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for i, E0 in enumerate(energies):
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is None:
            continue
        d = _trajectory_data(c, Ne, seed)
        depth = d["z_u"]
        thick = d["thick"]
        x = depth / thick if depth_frac else depth
        xmax = 1.0 if depth_frac else thick
        edges = np.linspace(0.0, xmax, n_bins + 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        w = d["L"]  # path-length weight
        sw = np.bincount(idx, w, minlength=n_bins)
        good = sw > 0
        swd = np.where(good, sw, 1.0)
        meanE = np.bincount(idx, w * d["E"], minlength=n_bins) / swd
        meanT = np.bincount(idx, w * d["t_fs"], minlength=n_bins) / swd
        col = energy_color(E0, energies)
        axE.plot(ctr[good], meanE[good], "-", color=col, lw=1.9, label=f"{E0:g} keV")
        axT.plot(ctr[good], meanT[good], "-", color=col, lw=1.9, label=f"{E0:g} keV")
    case0 = next(c for c in cases if c["tilt_deg"] == t)
    ulab = r"$\mu$m" if case0["thickness_ang"] >= 1e4 else "nm"
    xlab = "depth / thickness" if depth_frac else f"penetration depth ({ulab})"
    axE.set(xlabel=xlab, ylabel="mean electron energy (keV)", title="Slowing-down profile")
    axT.set(xlabel=xlab, ylabel="mean electron age (fs)", title="Electron lifetime vs depth")
    for ax in (axE, axT):
        ax.set_xlim(0, 1 if depth_frac else None)
        ax.grid(alpha=0.3)
        ax.legend(title="beam energy", fontsize=9)
    fig.suptitle(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}$={t:g}$\degree$ — penetration profiles",
        fontsize=13,
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
_METRIC_LABELS = {key: label for key, label, _ in _HEATMAP_QUANTITIES}

# Line-characterization maps are meaningless where the line is ill-defined --
# either near-zero emission OR a broad ramp / a cluster of comparable peaks
# (low line_quality, see cxr_results.line_quality). Gate these by BOTH peak flux
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
    lbl, div, unit, fmt = _AXIS_SPECS.get(key, (key, 1.0, "", "{:g}"))
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
    return np.concatenate(
        [[v[0] - (mids[0] - v[0])], mids, [v[-1] + (v[-1] - mids[-1])]]
    )


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

    ``x`` / ``y`` / ``panel`` are case-dict keys -- any of "tilt_deg",
    "tilt_azim_deg", "E0_keV", "thickness_ang", "B_ang2", ... The defaults
    reproduce the original map (azimuth x polar tilt, one panel per beam energy).
    Other critical scans are now one call::

        plot_heatmaps(res, s, x="thickness_ang", y="E0_keV", panel="tilt_deg")
        plot_heatmaps(res, s, x="tilt_deg", y="E0_keV", panel="tilt_azim_deg")

    When the sweep varies dimensions OTHER than x/y/panel, several records land in
    one cell; ``select`` (a cxr_results.selection_score mode, default
    "quality_peak" = peak flux x line quality) picks the BEST record and the cell
    shows ITS metric -- i.e. "the best achievable here". For the default axes each
    cell is a single case, so the reduction is a no-op (matches the old maps).

    Quantities (see cxr_results.line_metrics): peak spectral flux, the integrated
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
    metrics = {
        id(r): line_metrics(r, settings, rel_prominence, metric=line_metric)
        for r in recs
    }
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
                if gated and (
                    m["peak_flux"] < floor or m["line_quality"] < min_line_quality
                ):
                    continue  # near-zero emission / ill-defined line -> blank
                Z[yi[yv], xi[xv]] = m[key]
            panels.append(
                (pv, Z, _cell_edges(_axis_disp(x, xs)), _cell_edges(_axis_disp(y, ys)))
            )
        finite = [Z[np.isfinite(Z)] for _, Z, _, _ in panels]
        finite = (
            np.concatenate(finite)
            if any(a.size for a in finite)
            else np.array([0.0, 1.0])
        )
        vmin, vmax = float(finite.min()), float(finite.max())

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(min(3.6 * len(panels) + 1.2, 12.0), 4.2),
            squeeze=False,
            constrained_layout=True,
        )
        im = None
        for ax, (pv, Z, xe, ye) in zip(axes.ravel(), panels):
            im = ax.pcolormesh(xe, ye, Z, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{_AXIS_SPECS.get(panel, (panel,))[0]} = {_value_label(panel, pv)}")
            ax.set_xlabel(_axis_label(x))
            ax.set_ylabel(_axis_label(y))
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)  # one shared scale
        fig.suptitle(f"{label}    (best per cell: {select})", fontsize=13)
        figs.append(fig)
    return figs


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
    (cxr_results.selection_score ``select``). The line-plot companion to the
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
    metrics = {
        id(r): line_metrics(r, settings, rel_prominence, metric=line_metric)
        for r in recs
    }
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
        col = (
            energy_color(hv, hue_vals)
            if hue == "E0_keV"
            else COLORS[j % len(COLORS)]
        )
        ax.plot(
            [v / div_x for v in xs], ys, "o-", color=col, lw=1.8,
            label=_value_label(hue, hv),
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(_axis_label(x))
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"{_METRIC_LABELS.get(metric, metric)} vs {_axis_label(x)}  "
        f"(best per point: {select})",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(title=_AXIS_SPECS.get(hue, (hue,))[0], fontsize=9)
    fig.tight_layout()
    return fig


def plot_best_spectra(
    results,
    settings,
    top_n=12,
    select="quality_peak",
    include_brem=True,
    cases=None,
    ncols=4,
    rel_prominence=0.03,
    line_metric="sharpness",
):
    """The top-``top_n`` geometries across the WHOLE sweep, ranked by
    cxr_results.selection_score(``select``) -- one intrinsic spectrum panel each,
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
    metrics = {
        id(r): line_metrics(r, settings, rel_prominence, metric=line_metric)
        for r in recs
    }
    ranked = sorted(
        recs, key=lambda r: selection_score(metrics[id(r)], select), reverse=True
    )[:top_n]
    all_E = {r["case"]["E0_keV"] for r in recs}
    n = len(ranked)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows), squeeze=False
    )
    for k, r in enumerate(ranked):
        ax = axes[k // ncols][k % ncols]
        c, m = r["case"], metrics[id(r)]
        line_det, brem_det = _line_brem(r, settings, convolve=False)
        E = r["E_grid"] / 1e3
        col = energy_color(c["E0_keV"], all_E)
        ax.plot(
            E, (line_det + brem_det if include_brem else line_det) * r["scale"],
            color=col, lw=1.1,
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
    fig.tight_layout()
    return fig


def plot_material_comparison(
    results_by_material,
    settings,
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
):
    """Cross-material headline: for each material's results store, find the single
    BEST geometry/energy (cxr_results.selection_score ``select``) and plot its
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
            id(r): line_metrics(r, settings, rel_prominence, metric=line_metric)
            for r in recs
        }
        best = max(recs, key=lambda r: selection_score(metrics[id(r)], select))
        m = metrics[id(best)]
        pts.append((label, m["line_eV"], m["line_flux"], m["line_quality"], best["case"]))
    if not pts:
        print("no results in any material")
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sc = ax.scatter(
        [p[1] / 1e3 for p in pts], [p[2] for p in pts], c=[p[3] for p in pts],
        cmap="viridis", vmin=0.0, vmax=1.0, s=110, edgecolor="k", zorder=3,
    )
    for label, eV, flux, q, case in pts:
        ax.annotate(
            f"  {label} ({case['E0_keV']:g} keV)", (eV / 1e3, flux),
            fontsize=8, va="center",
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
