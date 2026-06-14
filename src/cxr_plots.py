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

Everything takes ``results`` + a :class:`cxr_results.Settings` explicitly.
"""
import numpy as np
import matplotlib.pyplot as plt

from cxr_montecarlo import convolve_detector, detector_efficiency
from cxr_results import (
    detected_background, records, best_azimuth, show_summary, line_metrics,
)
import timepix_response as tpx

COLORS = ["r", "y", "g", "b", "m", "c", "k", "orange", "purple", "brown"]

# cache of the (expensive) Timepix efficiency-curve response, keyed by hardware +
# MC settings, so re-running the detector cell with unchanged settings doesn't
# rebuild it. (The per-grid detected-spectra response is already cached inside
# timepix_response.get_response.)
_EFF_CACHE = {}


def _mode(settings):
    return "EDS-convolved" if settings.convolve_with_det else "intrinsic"


def _line_brem(r, settings):
    """Detected line and brem densities (per eV, before the unit scale) for one
    record, honoring the QE / convolution / brem-source flags."""
    qe = detector_efficiency(r["E_grid"]) if settings.apply_detector_qe else 1.0
    line_in = r["spec"] * qe
    line_det = (
        convolve_detector(r["E_grid"], line_in, r["fwhm"])
        if settings.convolve_with_det
        else line_in
    )
    brem_det = detected_background(r, settings) / r["scale"]
    return line_det, brem_det


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


def plot_by_energy(results, settings, include_brem=True, collapse_azimuth=True):
    """One figure PER POLAR TILT, every beam energy overlaid on a single axis
    (each energy at its best azimuth when ``collapse_azimuth``). Individual
    figures rather than one wide grid, so nothing runs off-screen.

    include_brem : plot line+brem (solid) with the brem dashed underneath;
        False = coherent line only. ``collapse_azimuth`` keeps, per energy, the
        azimuth with the highest spectral peak; False overlays every azimuth
        (same color per energy, so busy for a wide azimuth scan). Returns one
        figure per polar tilt."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    figs = []
    for t in tilts:
        trecs = [r for r in recs if r["case"]["tilt_deg"] == t]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for i, E0 in enumerate(energies):
            grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
            if not grp:
                continue
            if collapse_azimuth and len(grp) > 1:
                grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
            c = COLORS[i % len(COLORS)]
            for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
                az = r["case"]["tilt_azim_deg"]
                line_det, brem_det = _line_brem(r, settings)
                y = (line_det + brem_det) if include_brem else line_det
                ax.plot(r["E_grid"] / 1e3, y * r["scale"], color=c, lw=1.3,
                        label=rf"{E0:g} keV ($\phi={az:0.1f}\degree$)")
                if include_brem:
                    ax.plot(r["E_grid"] / 1e3, brem_det * r["scale"],
                            color=c, ls="--", lw=0.6)
        case = trecs[0]["case"]
        tag = "best azimuth/energy" if collapse_azimuth else "all azimuths"
        ax.set_title(rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} "
                     rf"$\mu$m, $\theta_\mathrm{{tilt}}={t:0.1f}\degree$ — "
                     rf"{_mode(settings)} ({tag})", fontsize=12)
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(alpha=0.3)
        ax.legend(title=("dashed: brem" if include_brem else None), fontsize=9)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_full_spectrum(results, settings, collapse_azimuth=True, logy=True):
    """Full measured-range view: the sharp coherent lines (fine line grid) riding
    on the broad bremsstrahlung evaluated out to the beam energy (wide brem grid).
    One figure per polar tilt, every beam energy at its best azimuth. The dashed
    curve is the wide brem; the solid is line+brem on the (narrow) line grid.

    Needs records run with a separate ``E_grid_brem`` (``brem_wide`` present); on
    older single-grid records it says so and returns []."""
    recs = [r for r in records(results) if r.get("brem_wide") is not None]
    if not recs:
        print("no wide-brem records -- set E_grid_brem in the Sweep and re-run")
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    figs = []
    for t in tilts:
        trecs = [r for r in recs if r["case"]["tilt_deg"] == t]
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ymax = 0.0
        for i, E0 in enumerate(energies):
            grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
            if not grp:
                continue
            if collapse_azimuth and len(grp) > 1:
                grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
            r = grp[0]
            c = COLORS[i % len(COLORS)]
            az = r["case"]["tilt_azim_deg"]
            # wide brem in detected units (QE applied on its own grid), full range
            Eb = r["E_grid_brem"]
            qe_b = detector_efficiency(Eb) if settings.apply_detector_qe else 1.0
            brem_wide_det = r["brem_wide"] * qe_b * r["scale"]
            # lines + local brem on the fine line grid
            line_det, brem_det = _line_brem(r, settings)
            total_line = (line_det + brem_det) * r["scale"]
            ax.plot(Eb / 1e3, brem_wide_det, color=c, ls="--", lw=0.7, alpha=0.85)
            ax.plot(r["E_grid"] / 1e3, total_line, color=c, lw=1.2,
                    label=rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)")
            ymax = max(ymax, float(np.nanmax(total_line)) if total_line.size else 0.0)
        case = trecs[0]["case"]
        if logy and ymax > 0:
            ax.set_yscale("log")
            ax.set_ylim(ymax * 1e-7, ymax * 2)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Photon energy (keV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.set_title(rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} "
                     rf"$\mu$m, $\theta_\mathrm{{tilt}}={t:0.1f}\degree$ — full measured "
                     rf"range ({_mode(settings)}; dashed = brem)", fontsize=12)
        ax.margins(x=0)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
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


def plot_chunk(results, names, settings):
    """The best-azimuth spectrum for one streamed group (a single polar tilt):
    one curve per beam energy, each at its winning azimuth. Drawn next to the
    chunk's table during the run."""
    best = sorted(best_azimuth(records(results, names)),
                  key=lambda r: r["case"]["E0_keV"])
    if not best:
        return None
    fig, (total_ax, back_sub_ax) = plt.subplots(2, 1, figsize=(7.5, 9.5))

    for i, r in enumerate(best):
        c = COLORS[i % len(COLORS)]
        line_det, brem_det = _line_brem(r, settings)
        az, E0 = r["case"]["tilt_azim_deg"], r["case"]["E0_keV"]
        total_ax.plot(r["E_grid"] / 1e3, (line_det + brem_det) * r["scale"], color=c, lw=1.2,
                label=rf"{E0:g} keV ($\phi={az:g}\degree$)")
        total_ax.plot(r["E_grid"] / 1e3, brem_det * r["scale"], color=c, ls="--", lw=0.6)
        back_sub_ax.plot(r["E_grid"] / 1e3, line_det * r["scale"], color=c, lw=1.2,
                label=rf"{E0:g} keV ($\phi={az:g}\degree$)")
    case = best[0]["case"]
    fig.suptitle(rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
                 rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:g}\degree$ — best azimuth per energy",
                 fontsize=14)
    total_ax.set_title('Total X-ray Spectrum (Coherent + Brem)', fontsize=12)
    total_ax.set_xlabel("Photon energy (keV)")
    total_ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    total_ax.set_ylim(bottom=0)
    total_ax.margins(x=0)
    total_ax.grid(alpha=0.3)
    total_ax.legend(title='Dashed=Brem Bkgnd', fontsize=9)

    back_sub_ax.set_title('Brem-subtracted (CXR Only)', fontsize=12)
    back_sub_ax.set_xlabel("Photon energy (keV)")
    back_sub_ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    back_sub_ax.set_ylim(bottom=0)
    back_sub_ax.margins(x=0)
    back_sub_ax.grid(alpha=0.3)
    back_sub_ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
    return fig


def stream_chunk(results, names, settings, collapse_azimuth=True, show_plot=True):
    """Per-group feedback during a run: the best-azimuth plot and the
    photon-counting table for the configs that just finished. ``show_plot`` draws
    the best-azimuth spectra (one curve per energy). The table collapses the
    azimuth sweep to the best azimuth per (tilt, energy) when ``collapse_azimuth``
    (so a whole azimuth scan prints one row per beam energy); pass False to list
    every azimuth."""
    if show_plot:
        plot_chunk(results, names, settings)
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
    E_eff = np.arange(200.0, 8000.0, 100.0)
    key = (thickness_um, bias_v, n_mc, seed)
    resp = _EFF_CACHE.get(key)
    if resp is None:
        resp = tpx.build_response(
            E_eff, np.arange(0.0, 9800.0, 25.0), n_mc=n_mc, seed=seed,
            thickness_um=thickness_um, bias_v=bias_v,
        )
        _EFF_CACHE[key] = resp
    E_thr = _thr_keV()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
    axL.plot(E_eff / 1e3, resp["eps_abs"], "k--", lw=1.2,
             label=r"$\epsilon_\mathrm{abs}$ (Si absorption)")
    axL.plot(E_eff / 1e3, resp["P_det"], "b-", lw=1.8,
             label=r"$P_\mathrm{det}$ (abs $\times$ counting)")
    axL.axvline(E_thr, color="r", ls=":", label=f"threshold = {E_thr:.2f} keV")
    axL.set(xlabel="Photon energy (keV)", ylabel="efficiency", ylim=(0, 1.05),
            title=f"Detection efficiency ({thickness_um:g} $\\mu$m Si, {bias_v:g} V, "
                  f"$\\sigma_\\mathrm{{diff}}$={resp['sigma_diff_um']:.1f} $\\mu$m)")
    axL.margins(x=0); axL.grid(alpha=0.3); axL.legend()
    axR.plot(E_eff / 1e3, tpx.energy_fwhm_eV(E_eff), "k-", lw=1.5,
             label="analytic, single-pixel")
    axR.plot(E_eff / 1e3, resp["fwhm_rec"], "b.", ms=4,
             label="MC effective (tail + multi-pixel)")
    axR.set(xlabel="Photon energy (keV)", ylabel="energy FWHM (eV)",
            title="Energy resolution & charge-loss bias")
    axR.margins(x=0); axR.grid(alpha=0.3); axR.legend(loc="upper left")
    axR2 = axR.twinx()
    axR2.plot(E_eff / 1e3, 100 * (1 - resp["mean_rec"] / E_eff), "g-", lw=1, alpha=0.5)
    axR2.set_ylabel("mean charge-loss deficit (%)", color="g")
    axR2.tick_params(axis="y", colors="g"); axR2.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed):
    """Incident and Timepix3-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by tpx.get_response."""
    incident = (r["spec"] + r["brem"]) * r["scale"]
    resp = tpx.get_response(r["E_grid"], n_mc=n_mc, seed=seed,
                            thickness_um=thickness_um, bias_v=bias_v)
    return incident, resp.apply(incident)


def plot_timepix_detected(results, settings, thickness_um=300.0, bias_v=100.0,
                          collapse_azimuth=False, n_mc=80000, seed=0, ncols=5):
    """Incident (dotted) vs Timepix3-detected (solid) spectra, log scale; one
    figure per energy, panels by polar tilt, curves by azimuth (or best)."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return []
    E_thr = _thr_keV()
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for E0 in energies:
        panels = []
        for t in tilts:
            grp = [r for r in recs
                   if r["case"]["E0_keV"] == E0 and r["case"]["tilt_deg"] == t]
            if collapse_azimuth and len(grp) > 1:
                grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
            if grp:
                panels.append(sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]))
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        for ax, grp in zip(axes.ravel(), panels):
            ymax = 0.0
            for i, r in enumerate(grp):
                inc, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
                fin = inc[np.isfinite(inc)]
                if fin.size:
                    ymax = max(ymax, float(fin.max()))
                c = COLORS[i % len(COLORS)]
                ax.plot(r["E_grid"] / 1e3, inc, color=c, ls=":", lw=0.7, alpha=0.45)
                ax.plot(r["E_grid"] / 1e3, det, color=c, ls="-", lw=1.2,
                        label=rf"$\phi={r['case']['tilt_azim_deg']:.1f}\degree$")
            case = grp[0]["case"]
            ax.axvline(E_thr, color="0.4", ls=":", lw=0.8)
            if ymax > 0:
                ax.set_yscale("log")
                ax.set_ylim(ymax * 1e-6, ymax * 2)
            ax.set_title(rf"{case['name'].split()[0]}, {case['E0_keV']:g} keV, "
                         rf"$\theta_\mathrm{{tilt}}={case['tilt_deg']:g}\degree$",
                         fontsize=11)
            ax.set_xlabel("Photon energy (keV)", fontsize=10)
            ax.set_ylabel("Phs/eV/s/nA", fontsize=10)
            ax.margins(x=0); ax.grid(alpha=0.3, which="both")
            ax.legend(title="solid: detected\ndotted: incident", fontsize=8)
        fig.suptitle(f"{E0:g} keV — Timepix3 detected vs incident", fontsize=15)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_timepix_poisson(results, settings, integration_s=600.0, thickness_um=300.0,
                         bias_v=100.0, n_mc=80000, seed=0):
    """A Poisson 'measured' realization for the highest-rate config at each
    energy, over ``integration_s`` at the configured beam current."""
    recs = records(results)
    if not recs:
        print("no results yet")
        return None
    rng = np.random.default_rng(seed)
    E_thr = _thr_keV()
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    fig, axes = plt.subplots(1, len(energies), figsize=(6 * len(energies), 4.6),
                             squeeze=False)
    for ax, E0 in zip(axes.ravel(), energies):
        grp = [r for r in recs if r["case"]["E0_keV"] == E0]
        r = max(grp, key=lambda r: float(np.max(r["spec"])))
        _, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
        counts, expected = tpx.poisson_counts(
            r["E_grid"], det * settings.beam_current_na, integration_s, rng
        )
        ax.step(r["E_grid"] / 1e3, counts, where="mid", color="k", lw=0.7,
                label=f"measured ({integration_s:g} s @ {settings.beam_current_na:g} nA)")
        ax.plot(r["E_grid"] / 1e3, expected, "r-", lw=1.3, label="expected mean")
        ax.axvline(E_thr, color="b", ls=":", lw=0.8, label="threshold")
        ax.set_title(rf"{E0:g} keV, $\theta_\mathrm{{tilt}}={r['case']['tilt_deg']:g}\degree$, "
                     rf"$\phi={r['case']['tilt_azim_deg']:g}\degree$  "
                     rf"({counts.sum():.0f} cts)", fontsize=10)
        ax.set_xlabel("Photon energy (keV)"); ax.set_ylabel("counts / bin")
        ax.margins(x=0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Timepix3 Poisson 'measured' spectra "
                 f"({thickness_um:g} $\\mu$m Si, {bias_v:g} V)", fontsize=14)
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


def plot_heatmaps(results, settings, cases=None, quantities=None, rel_prominence=0.03,
                  line_metric="sharpness", min_flux_frac=0.02):
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
    metrics = {id(r): line_metrics(r, settings, rel_prominence, metric=line_metric)
               for r in recs}

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
        finite = np.concatenate(finite) if any(a.size for a in finite) else np.array([0.0, 1.0])
        vmin, vmax = float(finite.min()), float(finite.max())

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels) + 1, 4.3),
                                 squeeze=False, constrained_layout=True)
        im = None
        for ax, (E0, Z, ext) in zip(axes.ravel(), panels):
            im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                           extent=list(ext), vmin=vmin, vmax=vmax)
            ax.set_title(f"{E0:g} keV")
            ax.set_xlabel("azimuthal tilt (deg)")
            ax.set_ylabel("polar tilt (deg)")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)  # one shared scale
        fig.suptitle(label, fontsize=14)
        figs.append(fig)
    return figs
