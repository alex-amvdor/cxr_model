"""
results.py
==============

Turn finished Monte-Carlo cases into the ``results`` store, reduce an azimuth
sweep to its best geometry, and build the photon-counting statistics table.

``results`` is a plain dict ``{config_name: {E0_keV: record}}``; each record is
the dict produced by :func:`store_result` (spectrum, brem, detector FWHM, unit
scale, and the originating ``case``). Functions here take ``results`` and a
:class:`Settings` explicitly -- no module globals -- so they are easy to reuse
and test.

The azimuth-max reduction (:func:`best_azimuth`) is the key knob for big sweeps:
for each fixed (material, thickness, polar tilt, energy) it keeps only the
azimuth whose spectrum has the highest PEAK value ``max(spectrum)`` (not the
integrated flux), collapsing hundreds of azimuth runs to one row/curve each.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from montecarlo import (
    beta_from_keV,
    eds_fwhm_eV,
    aperture_fwhm_eV,
    convolve_detector,
    detector_efficiency,
    load_external_brem,
)
from sweep import fmt_thickness, MATERIAL_LABELS

PER_NA = 6.2415e9  # electrons/s at 1 nA


@dataclass
class Settings:
    """Analysis / detector / unit knobs shared by post-processing and plots."""

    beam_current_na: float = 5.0
    # Legacy EDS/SDD polymer-window QE. The detector is now the Timepix3 quad or
    # the Eagle XO (each carries its OWN QE in its forward model), so this is OFF
    # by default -- the "intrinsic" spectra are then genuinely what leaves the
    # sample, not silently filtered by an unused SDD window. Leave False unless
    # you specifically want the old polymer-window SDD lens.
    apply_detector_qe: bool = False
    convolve_with_det: bool = False  # Gaussian EDS-resolution convolution
    brem_source: str = "mc"  # "mc" | "external" | "none"
    n_electrons: int = 450  # transport electrons for the lines
    n_electrons_brem: int = 100  # transport electrons for the background


# ---- results store -----------------------------------------------------------
def store_result(results, case, out):
    """Post-process one finished case into ``results[name][E0]`` (in place)."""
    name, E0 = case["name"], case["E0_keV"]
    E_grid = out["E_grid"]
    E_pk = E_grid[np.argmax(out["spec"])]
    fwhm = np.sqrt(
        eds_fwhm_eV(E_pk) ** 2
        + aperture_fwhm_eV(
            E_pk, beta_from_keV(E0), case["theta_obs_rad"], case["dtheta_obs_rad"]
        )
        ** 2
    )
    results.setdefault(name, {})[E0] = dict(
        E_grid=E_grid,
        spec=out["spec"],
        brem=out["brem"],
        E_grid_brem=out.get("E_grid_brem"),  # wide coarse grid (full range)
        brem_wide=out.get("brem_wide"),  # bremsstrahlung out to the beam energy
        E_pk=E_pk,
        fwhm=fwhm,
        eta=out["eta"],
        scale=case["domega_sr"] * PER_NA,  # (per e per sr) -> (per s per nA)
        case=case,
    )


def detected_background(r, settings, convolve=None):
    """Bremsstrahlung background in DETECTED units (Phs/eV/s/nA) on r['E_grid'],
    honoring the brem source + QE flags in ``settings``. ``convolve`` overrides
    settings.convolve_with_det when given (True/False) -- lets a caller draw the
    intrinsic and detector-convolved background side by side."""
    do_conv = (
        getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    )
    E = r["E_grid"]
    if settings.brem_source == "none":
        return np.zeros_like(E)
    if settings.brem_source == "external":
        path = r["case"].get("brem_file")
        return load_external_brem(path, E) if path else np.zeros_like(E)
    qe = detector_efficiency(E) if settings.apply_detector_qe else 1.0
    b = r["brem"] * qe
    if do_conv:
        b = convolve_detector(E, b, r["fwhm"])
    return b * r["scale"]


# ---- record selection --------------------------------------------------------
def records(results, names=None):
    """Flat list of every record in ``results`` (optionally restricted to
    ``names``)."""
    keys = list(results) if names is None else [n for n in names if n in results]
    return [results[n][E0] for n in keys for E0 in results[n]]


def filter_results(results, cases):
    """Subset ``results`` to just the configs in ``cases`` (e.g. the current
    sweep from build_cases), dropping anything left in the checkpoint from
    earlier sweeps. Pass the result to the plot/table functions to get a clean,
    dense grid instead of the sparse UNION of every sweep ever run:

        res = filter_results(results, cases)
        plot_by_energy(res, settings, collapse_azimuth=True)
        plot_heatmaps(res, settings)
    """
    names = {c["name"] for c in cases}
    return {n: results[n] for n in results if n in names}


def _peak(r):
    """The selection metric: the highest spectral flux value, max(spectrum)."""
    return float(np.max(r["spec"]))


def best_azimuth(recs):
    """Collapse an azimuth sweep. Group the records by
    (material, thickness, polar tilt, energy) and, within each group, keep only
    the one whose spectrum has the largest peak ``max(spectrum)``. Returns the
    selected records, sorted by (polar tilt, energy). A no-op shape-wise when
    azimuth is not swept (each group already has one member)."""
    groups = {}
    for r in recs:
        c = r["case"]
        key = (c["crystal"], c["thickness_ang"], c["tilt_deg"], c["E0_keV"])
        groups.setdefault(key, []).append(r)
    best = [max(g, key=_peak) for g in groups.values()]
    return sorted(best, key=lambda r: (r["case"]["tilt_deg"], r["case"]["E0_keV"]))


# ---- statistics table --------------------------------------------------------
_ROUND = {
    "polar [deg]": 1,
    "azimuth [deg]": 1,
    "line [eV]": 0,
    "peak [Phs/eV/s]": 2,
    "peak/bg": 2,
    "line [cts/s]": 1,
    "brem [cts/s]": 1,
    "total [cts/s]": 1,
}
_CONFIG_COLS = {"material", "thickness", "polar [deg]", "azimuth [deg]"}


def summary_table(recs, settings):
    """Photon-counting stats for a list of records. The geometry (material,
    thickness, polar/azimuthal tilt) is broken out of the config name into its
    own columns under a 'config' super-header; the rest are the line peak, the
    EDS-convolved peak height, peak-over-background, and the integrated line /
    brem / total count rates [counts/s] at ``settings.beam_current_na``. Returns
    a DataFrame with a 2-level column index (empty if ``recs`` is empty)."""
    cur = settings.beam_current_na
    rows = []
    for r in sorted(
        recs,
        key=lambda r: (
            r["case"]["tilt_deg"],
            r["case"]["tilt_azim_deg"],
            r["case"]["E0_keV"],
        ),
    ):
        c = r["case"]
        qe = detector_efficiency(r["E_grid"]) if settings.apply_detector_qe else 1.0
        line_det = convolve_detector(r["E_grid"], r["spec"] * qe, r["fwhm"])
        brem_det = detected_background(r, settings) / r["scale"]
        i_pk = np.argmax(line_det)
        line_cts = np.trapezoid(r["spec"], r["E_grid"]) * r["scale"] * cur
        # brem over the FULL measured range (the wide grid) when available, so the
        # total rate reflects the real measurement out to the beam energy; fall
        # back to the line-grid brem if a (stale) wide brem is non-finite
        brem_cts = np.trapezoid(r["brem"], r["E_grid"]) * r["scale"] * cur
        if r.get("brem_wide") is not None:
            wide = np.trapezoid(r["brem_wide"], r["E_grid_brem"]) * r["scale"] * cur
            if np.isfinite(wide):
                brem_cts = wide
        rows.append(
            {
                "material": MATERIAL_LABELS.get(c["crystal"], c["crystal"]),
                "thickness": fmt_thickness(c["thickness_ang"]),
                "polar [deg]": c["tilt_deg"],
                "azimuth [deg]": c["tilt_azim_deg"],
                "Ee [keV]": c["E0_keV"],
                "line [eV]": r["E_grid"][i_pk],
                "peak [Phs/eV/s]": line_det[i_pk] * r["scale"] * cur,
                "peak/bg": (line_det[i_pk] / brem_det[i_pk])
                if brem_det[i_pk]
                else np.inf,
                "line [cts/s]": line_cts,
                "brem [cts/s]": brem_cts,
                "total [cts/s]": line_cts + brem_cts,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.round(_ROUND)
    df.columns = pd.MultiIndex.from_tuples(
        [("config" if c in _CONFIG_COLS else "", c) for c in df.columns]
    )
    return df


def show_summary(recs, settings):
    """Print + render the stats table for a list of records (e.g. one chunk, or
    best_azimuth(records(results)))."""
    from IPython.display import display

    df = summary_table(recs, settings)
    if df.empty:
        return
    print(
        ("window-QE applied, " if settings.apply_detector_qe else "unity QE, ")
        + f"beam current {settings.beam_current_na:g} nA  |  "
        "peak/bg = EDS-convolved peak height / background at the peak"
    )
    display(df)


# ---- per-record scalar metrics (for the parametric heatmaps) ----------------
def _find_peaks_props(spec, rel_prominence):
    """One peak-finding pass shared by line_index/line_quality/line_metrics:
    returns (peaks, props, smax). props carries 'prominences' and 'widths'
    (FWHM in samples). Empty peaks array if the spectrum is flat/zero."""
    spec = np.asarray(spec, dtype=float)
    smax = float(spec.max()) if spec.size else 0.0
    if smax <= 0:
        return np.array([], dtype=int), {}, smax
    peaks, props = find_peaks(spec, prominence=rel_prominence * smax, width=0)
    return peaks, props, smax


def line_index(spec, rel_prominence=0.03, metric="sharpness"):
    """Index of the coherent LINE in a spectrum. Operates on the LINE spectrum
    only (r['spec']); the incoherent brem never enters here.

    ``metric`` chooses which detected peak is "the line":
      "sharpness"  : largest prominence/width -- the narrowest prominent peak, so
                     a sharp PXR line beats a broader coherent feature. Can grab a
                     short, sharp secondary line at transition geometries (default).
      "prominence" : largest prominence -- the DOMINANT line (~ the tallest peak).
                     Smoother heatmaps; only moves where the dominant line truly
                     shifts. Recommended if "sharpness" picks spurious side lines.
      "max"        : the global argmax, no peak finding.
    ``rel_prominence`` is the prominence floor as a fraction of the spectrum's max
    (raise it to ignore more small wiggles). Falls back to argmax if no peak
    clears the floor.

    NOTE: this ALWAYS returns an index, even when the spectrum has no
    well-defined line (a broad ramp, or many comparable peaks) -- it just falls
    back to argmax. Use line_quality() to decide whether that index is
    meaningful before trusting line_eV / fwhm_eV."""
    spec = np.asarray(spec, dtype=float)
    if spec.size == 0:
        return 0
    peaks, props, smax = _find_peaks_props(spec, rel_prominence)
    if smax <= 0 or metric == "max" or peaks.size == 0:
        return int(np.argmax(spec))
    if metric == "prominence":
        score = props["prominences"]
    else:  # "sharpness"
        score = props["prominences"] / np.maximum(props["widths"], 1.0)
    return int(peaks[int(np.argmax(score))])


def line_quality(spec, rel_prominence=0.03, rel_width_max=0.10):
    """How "well-defined" the dominant coherent line is, as a score in [0, 1].
    Designed to flag the geometries where there is NO single meaningful line, so
    the heatmaps can blank them instead of plotting a peak-finder artifact.

    It is the product of three independent factors (all in [0, 1], so every one
    must be good for a high score), each catching a distinct failure mode:

      dominance  = top_prominence / sum(all prominences)
                   -> ~1 for a lone peak, ~1/N for N comparable peaks.
                   Catches the "many small sharp peaks of similar size" case.
      contrast   = top_prominence / max(spec)
                   -> ~1 when the peak rises from baseline, small when it is a
                   wiggle riding on a broad pedestal. Catches the "large slow
                   change with a tiny bump" case.
      narrowness = 1 - (FWHM_samples / size) / rel_width_max, clipped to [0, 1]
                   -> ~1 for a sharp line, 0 once the peak is wider than
                   rel_width_max of the whole grid. Catches the "broad smeared
                   blob" (heavily Doppler-scattered bulk) case.

    Returns 0 when no peak clears the prominence floor at all (flat / zero
    spectrum). rel_width_max is the broadness cutoff as a fraction of the grid
    (0.10 -> a line spanning >10% of the energy window scores 0 on narrowness)."""
    peaks, props, smax = _find_peaks_props(spec, rel_prominence)
    if peaks.size == 0:
        return 0.0
    proms = props["prominences"]
    k = int(np.argmax(proms))
    top = float(proms[k])
    dominance = top / float(proms.sum())
    contrast = top / smax
    relwidth = float(props["widths"][k]) / np.asarray(spec).size
    narrowness = float(np.clip(1.0 - relwidth / rel_width_max, 0.0, 1.0))
    return float(dominance * contrast * narrowness)


def line_metrics(r, settings, rel_prominence=0.03, n_fwhm=3.0, metric="sharpness"):
    """Scalar metrics for one record, used by the heatmaps. Two flux quantities
    are deliberately distinct -- see the note below on which integrates what:

      peak_flux     : max(spec) * scale * current   [Phs/eV/s]. The tallest point
                      of the coherent (line) spectral DENSITY, in absolute
                      detected-rate-per-eV units. No peak finding -- just the max.
      coherent_flux : trapz(spec) over the WHOLE line grid * scale * current
                      [Phs/s]. ALL coherent flux (every line in the window), with
                      no peak finding -- the robust, always-well-defined total.
      line_flux     : trapz(spec) over ONLY the +-n_fwhm/2 window around the
                      single dominant found line * scale * current [Phs/s]. This
                      is one line, not the whole coherent spectrum, so it is only
                      meaningful where line_quality is high.
      line_eV       : energy of that dominant found line [eV] (see line_index).
      fwhm_eV       : spectral FWHM of that line [eV] (peak_widths at half height).
      line_frac     : line_flux / trapz(spec + brem) over the line grid -- the
                      dominant line's share of the total (lines + brem) flux.
      total_flux    : trapz(spec + brem) over the line grid * scale * current
                      [Phs/s]. (brem here is the line-grid brem, not the wide
                      grid -- this is the total IN the line window, not to E0.)
      coherent_brem_ratio : trapz(spec) / trapz(brem) over the line grid -- ALL
                      coherent (CXR) flux relative to the incoherent brem beneath
                      it (a ratio, so scale/current cancel). NaN if there's no brem.
      line_quality  : [0, 1] definition score of the dominant line (line_quality);
                      the heatmaps gate the line-characterization maps on it.

    All line characterization (line_eV, fwhm_eV, line_flux, line_frac) is from
    the LINE spectrum r['spec']; brem only enters total_flux / the line_frac
    denominator. peak_flux / coherent_flux / total_flux need no peak and stay
    valid everywhere; the line_index-based ones are unreliable where
    line_quality is low (broad ramps, or many comparable peaks).
    """
    E = np.asarray(r["E_grid"], dtype=float)
    spec = np.asarray(r["spec"], dtype=float)
    brem = np.asarray(r["brem"], dtype=float)
    dE = float(E[1] - E[0])
    cur, sc = settings.beam_current_na, r["scale"]
    smax = float(spec.max()) if spec.size else 0.0
    idx = line_index(spec, rel_prominence, metric)
    try:
        # a zero-prominence/zero-width peak (argmax fallback, single-sample spike)
        # is expected at ill-defined geometries -- line_quality already gates those,
        # so silence scipy's per-peak PeakPropertyWarning (private class, hence the
        # broad filter on this one call) rather than spamming the scan log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_samp = float(peak_widths(spec, [idx], rel_height=0.5)[0][0])
    except Exception:
        w_samp = 0.0
    half = max(int(round(n_fwhm * w_samp / 2.0)), 1)
    lo, hi = max(idx - half, 0), min(idx + half + 1, spec.size)
    line_int = float(np.trapezoid(spec[lo:hi], E[lo:hi])) if hi > lo else 0.0
    coh_int = float(np.trapezoid(spec, E)) if spec.size else 0.0
    brem_int = float(np.trapezoid(brem, E))
    total_int = coh_int + brem_int
    return {
        "peak_flux": smax * sc * cur,
        "coherent_flux": coh_int * sc * cur,
        "line_eV": float(E[idx]),
        "fwhm_eV": w_samp * dE,
        "line_flux": line_int * sc * cur,
        "line_frac": (line_int / total_int) if total_int > 0 else float("nan"),
        "total_flux": total_int * sc * cur,
        "coherent_brem_ratio": (coh_int / brem_int) if brem_int > 0 else float("nan"),
        "line_quality": line_quality(spec, rel_prominence),
    }


# ---- "best geometry" selection -----------------------------------------------
# Modes for ranking a record by its line_metrics, shared by every "pick the best
# case" path (the heatmap cell reduction, plot_metric_vs, the top-N browser), so
# they all agree on what "best" means.
SELECTION_MODES = ("peak", "line_flux", "coherent_flux", "quality_peak", "quality_line")


def selection_score(m, mode="quality_peak"):
    """Score a line_metrics dict ``m`` for "best geometry" selection. Higher wins.

      "peak"          : peak spectral flux (the legacy best_azimuth criterion --
                        favours the tallest spike, spurious lines included).
      "line_flux"     : integrated flux under the dominant found line.
      "coherent_flux" : integrated flux of ALL coherent lines (no peak finding).
      "quality_peak"  : peak_flux * line_quality (DEFAULT) -- favours geometries
                        that are both bright AND have a well-defined line, so a
                        tall-but-messy spike loses to a clean line.
      "quality_line"  : line_flux * line_quality.

    ``m`` is the dict from line_metrics. Non-finite scores sort to the bottom."""
    q = m.get("line_quality", 1.0)
    val = {
        "peak": m["peak_flux"],
        "line_flux": m["line_flux"],
        "coherent_flux": m["coherent_flux"],
        "quality_peak": m["peak_flux"] * q,
        "quality_line": m["line_flux"] * q,
    }.get(mode)
    if val is None:
        raise ValueError(f"unknown select mode {mode!r}; have {list(SELECTION_MODES)}")
    return val if np.isfinite(val) else -np.inf


# ---- compact ranked table ----------------------------------------------------
def top_geometries(
    results,
    settings,
    top_n=15,
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
    names=None,
):
    """A compact, ranked table of the BEST geometries across a results store --
    the readable alternative to dumping every (tilt, azimuth, energy) row. Ranks
    by results.selection_score(``select``) and returns the top ``top_n`` as a
    best-first DataFrame: material, polar/azimuth tilt, beam energy, dominant line
    energy, line-definition quality, peak spectral flux, integrated coherent flux,
    and the dominant line's share of the total. ``names`` restricts to those
    configs (e.g. one material)."""
    recs = records(results, names)
    if not recs:
        return pd.DataFrame()
    scored = []
    for r in recs:
        m = line_metrics(r, settings, rel_prominence, metric=line_metric)
        scored.append((selection_score(m, select), r, m))
    scored.sort(key=lambda t: -t[0])
    rows = []
    for rank, (_, r, m) in enumerate(scored[:top_n], 1):
        c = r["case"]
        rows.append(
            {
                "rank": rank,
                "material": MATERIAL_LABELS.get(c["crystal"], c["crystal"]),
                "polar": round(c["tilt_deg"], 1),
                "azim": round(c["tilt_azim_deg"], 1),
                "E [keV]": c["E0_keV"],
                "line [eV]": round(m["line_eV"]),
                "quality": round(m["line_quality"], 2),
                "peak [Phs/eV/s]": float(f"{m['peak_flux']:.3g}"),
                "coherent [Phs/s]": float(f"{m['coherent_flux']:.3g}"),
                "line/tot": (
                    round(m["line_frac"], 2) if np.isfinite(m["line_frac"]) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("rank")


def show_top(results, settings, top_n=15, select="quality_peak", **kw):
    """Print + render the compact :func:`top_geometries` table -- a short, sorted
    'here are the best N geometries' view instead of the full per-row dump."""
    from IPython.display import display

    df = top_geometries(results, settings, top_n=top_n, select=select, **kw)
    if df.empty:
        print("no results yet")
        return
    print(
        f"top {len(df)} geometries by '{select}'  (beam {settings.beam_current_na:g} nA; "
        f"peak = intrinsic coherent line density; quality in [0,1])"
    )
    display(df)
