"""
cxr_results.py
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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from cxr_montecarlo import (
    beta_from_keV,
    eds_fwhm_eV,
    aperture_fwhm_eV,
    convolve_detector,
    detector_efficiency,
    load_external_brem,
)
from cxr_sweep import fmt_thickness, MATERIAL_LABELS

PER_NA = 6.2415e9  # electrons/s at 1 nA


@dataclass
class Settings:
    """Analysis / detector / unit knobs shared by post-processing and plots."""

    beam_current_na: float = 5.0
    apply_detector_qe: bool = True  # polymer-window + Al + grid QE (EDS model)
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
        brem_wide=out.get("brem_wide"),      # bremsstrahlung out to the beam energy
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
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
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
                "peak/bg": (line_det[i_pk] / brem_det[i_pk]) if brem_det[i_pk] else np.inf,
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
    clears the floor."""
    spec = np.asarray(spec, dtype=float)
    if spec.size == 0:
        return 0
    smax = float(spec.max())
    if smax <= 0 or metric == "max":
        return int(np.argmax(spec))
    peaks, props = find_peaks(spec, prominence=rel_prominence * smax, width=0)
    if peaks.size == 0:
        return int(np.argmax(spec))
    if metric == "prominence":
        score = props["prominences"]
    else:  # "sharpness"
        score = props["prominences"] / np.maximum(props["widths"], 1.0)
    return int(peaks[int(np.argmax(score))])


def line_metrics(r, settings, rel_prominence=0.03, n_fwhm=3.0, metric="sharpness"):
    """Scalar metrics for one record, used by the heatmaps:

      peak_flux  : max(spec) * scale * current        [Phs/eV/s] (peak height)
      line_eV    : energy of the coherent line        [eV] (peak-found, see line_index)
      fwhm_eV    : spectral FWHM of that line          [eV] (peak_widths at half height)
      line_flux  : integrated coherent line flux       [Phs/s]; the line integrated
                   over +-n_fwhm half-widths about its peak, * scale * current
      line_frac  : line_flux / integrated total (spec+brem) flux over the grid
      total_flux : integrated (spec+brem) * scale * current  [Phs/s] (absolute)

    All line characterization (line_eV, fwhm_eV, line_frac) is from the LINE
    spectrum r['spec']; brem only enters total_flux / the line_frac denominator.
    At near-zero-emission geometries the line is ill-defined (peak_widths blows
    up) -- the heatmaps gate those cells out by peak_flux rather than trusting it.
    """
    E = np.asarray(r["E_grid"], dtype=float)
    spec = np.asarray(r["spec"], dtype=float)
    brem = np.asarray(r["brem"], dtype=float)
    dE = float(E[1] - E[0])
    cur, sc = settings.beam_current_na, r["scale"]
    smax = float(spec.max()) if spec.size else 0.0
    idx = line_index(spec, rel_prominence, metric)
    try:
        w_samp = float(peak_widths(spec, [idx], rel_height=0.5)[0][0])
    except Exception:
        w_samp = 0.0
    half = max(int(round(n_fwhm * w_samp / 2.0)), 1)
    lo, hi = max(idx - half, 0), min(idx + half + 1, spec.size)
    line_int = float(np.trapezoid(spec[lo:hi], E[lo:hi])) if hi > lo else 0.0
    total_int = float(np.trapezoid(spec + brem, E))
    return {
        "peak_flux": smax * sc * cur,
        "line_eV": float(E[idx]),
        "fwhm_eV": w_samp * dE,
        "line_flux": line_int * sc * cur,
        "line_frac": (line_int / total_int) if total_int > 0 else float("nan"),
        "total_flux": total_int * sc * cur,
    }
