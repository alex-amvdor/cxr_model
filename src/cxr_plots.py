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
    PER_NA,
)
import timepix_response as tpx
import eaglexo_response as eag

COLORS = ["r", "y", "g", "b", "m", "c", "k", "orange", "purple", "brown"]

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
    drawers = {
        "by_energy": (_draw_by_energy, (10.0, 7.5)),
        "full": (_draw_full_spectrum, (10.0, 7.5)),
        "chunk": (_draw_chunk, (10.0, 7.5)),
        "timepix": (_draw_timepix_detected, (10.0, 7.5)),
        "eaglexo": (_draw_eaglexo_detected, (10.0, 7.5)),
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
    redraw is what tends to get stuck showing one frame)."""
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
    overlaid, LEFT intrinsic / RIGHT detector-convolved."""
    fig.clear()
    ax_raw, ax_conv = fig.subplots(1, 2, sharex=True)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    for i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        c = COLORS[i % len(COLORS)]
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            az = r["case"]["tilt_azim_deg"]
            lbl = rf"{E0:g} keV ($\phi={az:0.1f}\degree$)"
            Ee = r["E_grid"]
            for ax, conv in ((ax_raw, False), (ax_conv, True)):
                line_det, brem_det = _line_brem(r, settings, convolve=conv)
                y = (line_det + brem_det) if include_brem else line_det
                ax.plot(Ee, y * r["scale"], color=c, lw=1.3, label=lbl)
                if include_brem:
                    ax.plot(Ee, brem_det * r["scale"], color=c, ls="--", lw=0.6)
    case = trecs[0]["case"]
    tag = "best azimuth/energy" if collapse_azimuth else "all azimuths"
    fig.suptitle(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} "
        rf"$\mu$m, $\theta_\mathrm{{tilt}}={case['tilt_deg']:0.1f}\degree$ ({tag})",
        fontsize=13,
    )
    for ax, sub in ((ax_raw, "intrinsic"), (ax_conv, "detector-convolved")):
        ax.set_title(sub, fontsize=11)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.legend(title=("dashed: brem" if include_brem else None), fontsize=9)
    fig.tight_layout()


def plot_by_energy(results, settings, include_brem=True, collapse_azimuth=True):
    """One figure PER POLAR TILT, every beam energy overlaid (best azimuth when
    ``collapse_azimuth``); LEFT intrinsic, RIGHT detector-convolved. For
    click-through use ``browse(results, settings, kind="by_energy")``."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=(15.0, 5.2))
        _draw_by_energy(
            fig,
            [r for r in recs if r["case"]["tilt_deg"] == t],
            settings,
            include_brem=include_brem,
            collapse_azimuth=collapse_azimuth,
        )
        figs.append(fig)
    return figs


def _draw_full_spectrum(
    fig, trecs, settings, collapse_azimuth=True, logy=True, logx=True, floor_frac=1e-2
):
    """Render ONE polar tilt of the full measured-range view onto ``fig``: sharp
    lines + wide brem out to the beam energy, log y, LEFT intrinsic/RIGHT detector."""
    fig.clear()
    ax_raw, ax_conv = fig.subplots(1, 2, sharex=True)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = {"raw": 0.0, "conv": 0.0}
    xmax = 0.0
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
        c = COLORS[i % len(COLORS)]
        az = r["case"]["tilt_azim_deg"]
        lbl = rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)"
        Eb = r["E_grid_brem"]
        qe_b = detector_efficiency(Eb) if settings.apply_detector_qe else 1.0
        brem_wide_det = r["brem_wide"] * qe_b * r["scale"]
        xmin = float(Eb[0])
        xmax = max(xmax, float(Eb[-1]))  # full brem grid -> beam energy
        for ax, conv, key in ((ax_raw, False, "raw"), (ax_conv, True, "conv")):
            line_det, brem_det = _line_brem(r, settings, convolve=conv)
            total_line = (line_det + brem_det) * r["scale"]
            ax.plot(Eb, brem_wide_det, color=c, ls="--", lw=0.7, alpha=0.85)
            ax.plot(r["E_grid"], total_line, color=c, lw=1.2, label=lbl)
            ymax[key] = max(
                ymax[key], float(np.nanmax(total_line)) if total_line.size else 0.0
            )
    case = trecs[0]["case"]
    fig.suptitle(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} "
        rf"$\mu$m, $\theta_\mathrm{{tilt}}={case['tilt_deg']:0.1f}\degree$ — full "
        rf"measured range (dashed = brem)",
        fontsize=13,
    )
    for ax, sub, key in (
        (ax_raw, "intrinsic", "raw"),
        (ax_conv, "detector-convolved", "conv"),
    ):
        if logy and ymax[key] > 0:
            ax.set_yscale("log")
            ax.set_ylim(ymax[key] * floor_frac, ymax[key] * 2)
        else:
            ax.set_ylim(bottom=0)
        if logx:
            ax.set_xscale("log")
        ax.set_title(sub, fontsize=11)
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        if xmax > 0:
            ax.set_xlim(xmin, xmax)  # span the full brem grid (to the beam energy)
        else:
            ax.margins(x=0)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()


def plot_full_spectrum(
    results, settings, collapse_azimuth=True, logy=True, floor_frac=1e-2
):
    """Full measured-range view (sharp lines on the wide brem, log y), ONE figure
    per polar tilt. The x-axis spans the full brem grid (to the beam energy); the
    log floor is ``floor_frac`` x the peak (~4 decades). For click-through use
    ``browse(results, settings, kind="full")``. Needs records run with a separate
    ``E_grid_brem`` (``brem_wide`` present)."""
    recs = [r for r in records(results) if r.get("brem_wide") is not None]
    if not recs:
        print("no wide-brem records -- set E_grid_brem in the Sweep and re-run")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=(16.0, 5.2))
        _draw_full_spectrum(
            fig,
            [r for r in recs if r["case"]["tilt_deg"] == t],
            settings,
            collapse_azimuth=collapse_azimuth,
            logy=logy,
            floor_frac=floor_frac,
        )
        figs.append(fig)
    return figs


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
        ax.plot(tilts, peak, "o-", color=COLORS[i % len(COLORS)], label=f"{E0:g} keV")
    ax.set_xlabel(r"polar tilt $\theta_\mathrm{tilt}$ (deg)")
    ax.set_ylabel("best-azimuth peak (Phs/eV/s)")
    ax.set_title("Peak spectral flux vs polar tilt (best azimuth per point)")
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy")
    fig.tight_layout()
    return fig


def _draw_chunk(fig, trecs, settings):
    """Render ONE polar tilt's best-azimuth 2x2 (total | CXR-only x intrinsic |
    detector-convolved) onto ``fig`` (cleared first)."""
    fig.clear()
    best = sorted(best_azimuth(trecs), key=lambda r: r["case"]["E0_keV"])
    if not best:
        return
    (tot_raw, tot_conv), (cxr_raw, cxr_conv) = fig.subplots(2, 2, sharex=True)
    for i, r in enumerate(best):
        c = COLORS[i % len(COLORS)]
        az, E0 = r["case"]["tilt_azim_deg"], r["case"]["E0_keV"]
        lbl = rf"{E0:g} keV ($\phi={az:g}\degree$)"
        E = r["E_grid"] / 1e3
        line_raw, brem_raw = _line_brem(r, settings, convolve=False)  # intrinsic
        line_conv, brem_conv = _line_brem(r, settings, convolve=True)  # detector
        tot_raw.plot(E, (line_raw + brem_raw) * r["scale"], color=c, lw=1.2, label=lbl)
        tot_raw.plot(E, brem_raw * r["scale"], color=c, ls="--", lw=0.6)
        tot_conv.plot(
            E, (line_conv + brem_conv) * r["scale"], color=c, lw=1.2, label=lbl
        )
        tot_conv.plot(E, brem_conv * r["scale"], color=c, ls="--", lw=0.6)
        cxr_raw.plot(E, line_raw * r["scale"], color=c, lw=1.2, label=lbl)
        cxr_conv.plot(E, line_conv * r["scale"], color=c, lw=1.2, label=lbl)
    case = best[0]["case"]
    fig.suptitle(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:g}\degree$ — best azimuth per "
        rf"energy  (left: intrinsic   right: detector-convolved)",
        fontsize=14,
    )
    for ax, title, leg in (
        (
            tot_raw,
            "Total X-ray Spectrum (Coherent + Brem) — intrinsic",
            "Dashed=Brem Bkgnd",
        ),
        (
            tot_conv,
            "Total X-ray Spectrum (Coherent + Brem) — detector",
            "Dashed=Brem Bkgnd",
        ),
        (cxr_raw, "Brem-subtracted (CXR Only) — intrinsic", None),
        (cxr_conv, "Brem-subtracted (CXR Only) — detector", None),
    ):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.legend(title=leg, fontsize=9)
    fig.tight_layout()


def plot_chunk(results, settings):
    """The best-azimuth 2x2 spectra (total | CXR-only x intrinsic | detector),
    ONE figure per polar tilt. For click-through use
    ``browse(results, settings, kind="chunk")``."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=(14.0, 9.5))
        _draw_chunk(fig, [r for r in recs if r["case"]["tilt_deg"] == t], settings)
        figs.append(fig)
    return figs


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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
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
        c = COLORS[i % len(COLORS)]
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
    recs = records(results)
    if not recs:
        print("no results yet")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=(9.0, 5.2))
        _draw_timepix_detected(
            fig,
            [r for r in recs if r["case"]["tilt_deg"] == t],
            settings,
            thickness_um=thickness_um,
            bias_v=bias_v,
            collapse_azimuth=collapse_azimuth,
            n_mc=n_mc,
            seed=seed,
            floor_frac=floor_frac,
        )
        figs.append(fig)
    return figs


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
        1, len(energies), figsize=(6 * len(energies), 4.6), squeeze=False
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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
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
    floor_frac=1e-3,
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
        c = COLORS[i % len(COLORS)]
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
    floor_frac=1e-3,
):
    """Incident (dotted) vs Eagle-XO-detected (solid) spectra, log-log; ONE figure
    per polar tilt, all energies overlaid (best azimuth each), with a faint QE
    envelope. Shows the camera's signature: soft PXR lines survive at ~90% QE
    while the hard bremsstrahlung is suppressed by the thin back-thinned sensor.
    Uses the wide brem grid when present (run the sweep with ``E_grid_brem``). The
    solid angle is whatever the sweep was run with -- point it at the Eagle with
    ``Sweep(..., **eaglexo_response.sweep_geometry(...))``. For click-through use
    ``browse(results, settings, kind="eaglexo")``."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=(9.0, 5.2))
        _draw_eaglexo_detected(
            fig,
            [r for r in recs if r["case"]["tilt_deg"] == t],
            settings,
            coating=coating,
            resolve_energy=resolve_energy,
            collapse_azimuth=collapse_azimuth,
            floor_frac=floor_frac,
        )
        figs.append(fig)
    return figs


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
        1, len(energies), figsize=(6 * len(energies), 4.6), squeeze=False
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


# ---- electron trajectory view (penetration) ----------------------------------
def _case_of(rec_or_case):
    """Accept either a results record (carries 'case') or a raw case dict."""
    return rec_or_case["case"] if "case" in rec_or_case else rec_or_case


def plot_electron_trajectories(
    rec_or_case,
    *,
    Ne=200,
    seed=0,
    color_by="energy",
    show_detector=True,
    colorbar=True,
    aspect="equal",
    ax=None,
):
    """
    Cross-section of the electron cascade in a BEAM-ALIGNED frame.

    Coordinates:
        x' : along incident beam (horizontal)
        y' : transverse direction in the beam/slab plane (vertical)

    Slab:
        tilt_deg > 0  -> slab rotated CCW
        tilt_deg < 0  -> slab rotated CW

    Detector:
        n_hat plotted in the same frame.
        0 deg = horizontal right
        +90 deg = vertical up
        positive angles CCW.
    """
    from matplotlib.collections import LineCollection

    case = _case_of(rec_or_case)

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

    # ------------------------------------------------------------------
    # Reconstruct segment endpoints
    # ------------------------------------------------------------------
    r_mid = segs["r_mid"]
    v = segs["v_hat"]
    L = segs["L_ang"]
    Ekv = segs["E_keV"]

    start = r_mid - 0.5 * L[:, None] * v
    end = r_mid + 0.5 * L[:, None] * v

    u, ulab = (1e4, r"$\mu$m") if case["thickness_ang"] >= 1e4 else (10.0, "nm")
    thick = case["thickness_ang"] / u

    # ------------------------------------------------------------------
    # Build beam-aligned coordinate system
    #
    # Xb = along beam
    # Yb = +90° CCW from beam in x-z plane
    # ------------------------------------------------------------------
    beam_xz = np.array([beam[0], beam[2]], dtype=float)
    beam_xz /= np.linalg.norm(beam_xz)

    perp_xz = np.array([-beam_xz[1], beam_xz[0]])

    start2 = np.column_stack(
        [
            start[:, [0, 2]] @ beam_xz,
            -(start[:, [0, 2]] @ perp_xz),
        ]
    )

    end2 = np.column_stack(
        [
            end[:, [0, 2]] @ beam_xz,
            -(end[:, [0, 2]] @ perp_xz),
        ]
    )

    seg2d = np.stack([start2, end2], axis=1) / u

    # ------------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    xs = seg2d[:, :, 0]
    ys = seg2d[:, :, 1]

    xlo = float(xs.min())
    xhi = float(xs.max())
    ylo = float(ys.min())
    yhi = float(ys.max())

    spanx = xhi - xlo
    spany = yhi - ylo
    span = max(spanx, spany, thick)

    pad = 0.08 * span + 0.02 * thick

    xlo -= pad
    xhi += pad
    ylo -= pad
    yhi += pad

    # ------------------------------------------------------------------
    # Slab polygon in beam frame
    #
    # tilt_deg > 0 => CCW rotation
    # tilt_deg < 0 => CW rotation
    # ------------------------------------------------------------------
    tilt = np.deg2rad(case.get("tilt_deg", 0.0))

    n = np.array([np.cos(tilt), np.sin(tilt)])
    t = np.array([-n[1], n[0]])

    W = max(spanx, spany, thick) * 2.5

    slab = np.array(
        [
            -W * t,
            +W * t,
            +W * t + thick * n,
            -W * t + thick * n,
        ]
    )

    ax.fill(
        slab[:, 0],
        slab[:, 1],
        facecolor="0.63",
        edgecolor="0.45",
        lw=1.2,
        zorder=0,
    )

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------
    lc = LineCollection(
        seg2d,
        linewidths=0.5,
        alpha=float(np.clip(60.0 / Ne, 0.12, 0.8)),
    )

    if color_by == "energy":
        lc.set_array(Ekv)
        lc.set_cmap("turbo")
    else:
        lc.set_color("steelblue")

    ax.add_collection(lc)

    # ------------------------------------------------------------------
    # Beam arrow (always horizontal)
    # ------------------------------------------------------------------
    aL = 0.18 * span

    ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(-aL, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
    )

    ax.text(
        -aL,
        0.0,
        "beam ",
        color="red",
        fontsize=9,
        ha="right",
        va="bottom",
    )

    # ------------------------------------------------------------------
    # Detector arrow
    # ------------------------------------------------------------------
    if show_detector:
        n2 = np.array(
            [
                np.dot([n_hat[0], n_hat[2]], beam_xz),
                np.dot([n_hat[0], n_hat[2]], perp_xz),
            ]
        )

        n2 /= np.linalg.norm(n2)

        cx = 0.0
        cy = 0.0

        ax.annotate(
            "",
            xy=(cx + n2[0] * aL, cy - n2[1] * aL),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="green", lw=1.6),
        )

        ax.text(
            cx + n2[0] * aL,
            cy - n2[1] * aL,
            " to detector",
            color="green",
            fontsize=9,
            va="center",
        )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    ax.set_aspect(aspect)

    ax.set_xlabel(f"distance along beam ({ulab})")
    ax.set_ylabel(f"transverse distance ({ulab})")

    eta = 100.0 * segs["n_backscattered"] / segs["Ne"]
    thru = 100.0 * segs["n_transmitted"] / segs["Ne"]

    ax.set_title(
        rf"{case['name'].split()[0]}, "
        rf"{case['E0_keV']:g} keV, "
        rf"$\theta_\mathrm{{tilt}}$={case.get('tilt_deg', 0.0):g}$\degree$"
        rf"  —  {Ne} e$^-$ "
        rf"({eta:.0f}% back, {thru:.0f}% through)",
        fontsize=10,
    )

    if color_by == "energy" and colorbar:
        cb = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("electron energy (keV)")

    return ax


def plot_trajectories(results, *, tilt=None, Ne=200, seed=0, color_by="energy"):
    """Electron-penetration cross-sections for one polar tilt: a row of panels,
    one per beam energy (best azimuth each). ``tilt`` picks the polar tilt (nearest
    match; default the first). Higher beam energy -> deeper teardrop; tilt skews
    the beam arrow. Returns the figure (or None if there are no results)."""
    recs = best_azimuth(records(results))
    if not recs:
        print("no results yet")
        return None
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    t = tilts[0] if tilt is None else min(tilts, key=lambda x: abs(x - tilt))
    grp = sorted(
        [r for r in recs if r["case"]["tilt_deg"] == t],
        key=lambda r: r["case"]["E0_keV"],
    )
    fig, axes = plt.subplots(1, len(grp), figsize=(5.6 * len(grp), 5.2), squeeze=False)
    for ax, r in zip(axes.ravel(), grp):
        plot_electron_trajectories(r, Ne=Ne, seed=seed, color_by=color_by, ax=ax)
    fig.suptitle(
        rf"Electron penetration, $\theta_\mathrm{{tilt}}$={t:g}$\degree$ "
        f"(crystal = shaded slab; beam-detector plane)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


# ---- parametric heatmaps -----------------------------------------------------
def _axis_edges(vals):
    """Cell edges for imshow so the swept values sit at pixel centres (assumes a
    uniform step, as for np.linspace sweeps)."""
    vals = sorted(vals)
    if len(vals) < 2:
        return vals[0] - 0.5, vals[0] + 0.5
    step = vals[1] - vals[0]
    return vals[0] - step / 2, vals[-1] + step / 2


# (metric key, panel-group label + units, colormap)
_HEATMAP_QUANTITIES = [
    ("peak_flux", "peak spectral flux  (Phs/eV/s)", "viridis"),
    ("line_eV", "coherent line energy  (eV)", "plasma"),
    ("fwhm_eV", "line FWHM  (eV)", "magma"),
    ("line_frac", "integrated line / total spectral flux", "cividis"),
    ("line_flux", "integrated coherent line flux  (Phs/s)", "viridis"),
    ("total_flux", "total integrated flux  (Phs/s)", "viridis"),
]

# line-characterization maps are meaningless where there's essentially no
# emission; gate these by peak flux. peak_flux / total_flux are always shown.
_FLUX_GATED = {"line_eV", "fwhm_eV", "line_frac"}


def plot_heatmaps(
    results,
    settings,
    cases=None,
    quantities=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
):
    """Parametric heatmaps over (polar tilt x azimuthal tilt), one panel per beam
    energy, one figure per quantity.

    Quantities (see cxr_results.line_metrics): peak spectral flux, coherent line
    energy, line FWHM, the integrated-line / total-flux ratio, and the total
    absolute integrated flux.

    line_metric : how line_index picks the line -- "sharpness" (default; narrowest
        prominent peak), "prominence" (the dominant/tallest line; use this if the
        line-energy map jumps onto sharp secondary lines), or "max" (global argmax).
    rel_prominence : line-finder prominence floor.
    min_flux_frac : blank the line-characterization maps (line energy, FWHM,
        line/total) wherever a cell's peak flux is below this fraction of that
        energy's max -- those near-zero-emission geometries have no well-defined
        line (peak_widths blows up there). Set 0 to disable. peak/total flux maps
        are never gated.

    Pass ``cases`` (the current sweep from build_cases) to restrict the map to
    THIS sweep's configs -- otherwise a checkpoint that has accumulated several
    sweeps yields the UNION of their grids, which is sparse: blank cells are
    (tilt, azimuth) pairs no single sweep ran together. Needs both tilt and
    azimuth swept to be a 2-D map. Returns the list of figs.
    """
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        print("no results yet")
        return []
    quantities = quantities or _HEATMAP_QUANTITIES
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    metrics = {
        id(r): line_metrics(r, settings, rel_prominence, metric=line_metric)
        for r in recs
    }

    figs = []
    for key, label, cmap in quantities:
        # build each energy's grid first, so all panels can share ONE color scale
        # (spanning every energy's values -> the highest energy sets the top)
        panels = []  # (E0, Z, extent)
        for E0 in energies:
            er = [r for r in recs if r["case"]["E0_keV"] == E0]
            tilts = sorted({r["case"]["tilt_deg"] for r in er})
            azims = sorted({r["case"]["tilt_azim_deg"] for r in er})
            ti = {t: i for i, t in enumerate(tilts)}
            ai = {a: j for j, a in enumerate(azims)}
            fmax = max((metrics[id(r)]["peak_flux"] for r in er), default=0.0)
            floor = min_flux_frac * fmax
            Z = np.full((len(tilts), len(azims)), np.nan)
            for r in er:
                m = metrics[id(r)]
                if key in _FLUX_GATED and m["peak_flux"] < floor:
                    continue  # near-zero emission -> no well-defined line here
                Z[ti[r["case"]["tilt_deg"]], ai[r["case"]["tilt_azim_deg"]]] = m[key]
            x0, x1 = _axis_edges(azims)
            y0, y1 = _axis_edges(tilts)
            panels.append((E0, Z, (x0, x1, y0, y1)))
        finite = [Z[np.isfinite(Z)] for _, Z, _ in panels]
        finite = (
            np.concatenate(finite)
            if any(a.size for a in finite)
            else np.array([0.0, 1.0])
        )
        vmin, vmax = float(finite.min()), float(finite.max())

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5 * len(panels) + 1, 4.3),
            squeeze=False,
            constrained_layout=True,
        )
        im = None
        for ax, (E0, Z, ext) in zip(axes.ravel(), panels):
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                extent=list(ext),
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{E0:g} keV")
            ax.set_xlabel("azimuthal tilt (deg)")
            ax.set_ylabel("polar tilt (deg)")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)  # one shared scale
        fig.suptitle(label, fontsize=14)
        figs.append(fig)
    return figs
