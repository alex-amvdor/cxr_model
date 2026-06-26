"""
montecarlo.detector

Detector forward model (Zhai SI S3/S4): soft-X-ray window efficiency, the EDS /
aperture / mosaic line-broadening widths, the representative mosaic geometry
angle, and the Gaussian detector convolution.
"""

import numpy as np

from ..crystallography import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from .geometry import _orientation_R, tilted_geometry
from .materials import _mu_total_inv_ang
from .transport import beta_from_keV


def detector_efficiency(E_eV, polymer_nm=300.0, al_nm=40.0, grid_open=0.78):
    """
    Soft X-ray collection efficiency of a polymer-window SDD (the dominant
    loss below ~2 keV; the Si diode itself is ~fully absorbing there):

        QE(E) = grid_open * T_polymer(E) * T_Al(E)

    computed from Henke f2 attenuation for a polyimide film (C22 H10 N2 O5,
    rho = 1.42 g/cm^3) of thickness polymer_nm, an aluminum light-blocking
    coat of al_nm, and the etched-Si support grid (open-area fraction
    grid_open; the grid bars are opaque at these energies). Defaults model a
    Moxtek AP3.3-class window -- the actual Oxford UltimMax window is
    proprietary, so treat the parameters as tunable. Carries the C, N, O
    edge structure (e.g. the deep notch just above the O-K edge at 532 eV).
    """
    E = np.asarray(E_eV, dtype=float)
    # polyimide C22 H10 N2 O5: formula units per Ang^3 at rho = 1.42 g/cm^3
    n_f = 1.42 / 382.31 * 0.602214076
    mu_poly = sum(
        count * _mu_total_inv_ang([(el, n_f)], E)
        for el, count in (("C", 22), ("H", 10), ("N", 2), ("O", 5))
    )
    n_al = 2.70 / 26.982 * 0.602214076
    mu_al = _mu_total_inv_ang([("Al", n_al)], E)
    # Above the Henke ceiling (~30 keV) the window attenuation is unavailable
    # (NaN, and inf at E=0); the thin polymer/Al window is transparent to hard
    # X-rays, so treat an unavailable mu as zero -> QE -> grid_open rather than
    # NaN (which would clip the detected spectrum on a log plot). NB this
    # window-transmission model assumes the Si diode is fully absorbing, so it
    # OVERestimates QE above ~20 keV where the Si itself turns transparent -- use
    # the Timepix forward model (plot_timepix_detected) for a faithful hard-X-ray
    # detector response.
    mu_poly = np.nan_to_num(mu_poly, nan=0.0, posinf=0.0, neginf=0.0)
    mu_al = np.nan_to_num(mu_al, nan=0.0, posinf=0.0, neginf=0.0)
    return grid_open * np.exp(-mu_poly * polymer_nm * 10.0 - mu_al * al_nm * 10.0)


def eds_fwhm_eV(E_eV):
    """Oxford UltimMax 170 resolution fit, SI Eq. (16)."""
    return np.sqrt(2.52 * E_eV + 988.0)


def aperture_fwhm_eV(E_eV, beta, theta_obs_rad, dtheta_obs_rad):
    """Line broadening from the detector polar-angle span, SI Eq. (14)."""
    dE_dth = E_eV * beta * np.sin(theta_obs_rad) / (1.0 - beta * np.cos(theta_obs_rad))
    return 2.0 * np.sqrt(2.0 * np.log(2.0) / 3.0) * dE_dth * dtheta_obs_rad


def mosaic_fwhm_eV(E_eV, psi_rad, mosaic_fwhm_rad):
    """Line broadening from crystal mosaicity -- the INITIAL ANALYTIC model.

    A mosaic crystal is an incoherent ensemble of crystallites whose orientations
    are Gaussian-spread about the mean with a rocking-curve FWHM `mosaic_fwhm_rad`.
    Tilting a crystallite rotates its reciprocal vector g; only the NUMERATOR v.g of
    the resonance E_res = hbar c (v.g)/(1 - v.n) depends on g, so to first order the
    fractional line shift is dE/E = -tan(psi) dtheta, psi = angle(v, g). A Gaussian
    tilt of FWHM `mosaic_fwhm_rad` (in the longitudinal plane) therefore broadens the
    line by a Gaussian of energy FWHM

        FWHM_mosaic = E * |tan(psi)| * mosaic_fwhm_rad,

    added in quadrature with the EDS and aperture widths (results.store_result) and
    applied via the same convolve_detector pass -- the cheap analytic counterpart of
    a full Monte-Carlo average over crystallite orientations.

    Crude by construction: (i) it captures only the resonance-ENERGY shift, holding
    the amplitudes / lineshape weight fixed across the mosaic cone (good while the
    line stays narrow); (ii) the linearization diverges as psi -> 90 deg (g grazing
    the velocity), so the caller should cap the result (store_result clips it at E).
    The exact treatment is the per-orientation MC sum inside mc_spectrum (future
    work). psi is supplied by mosaic_psi_rad() at the nominal (unscattered) geometry.
    """
    return E_eV * np.abs(np.tan(psi_rad)) * mosaic_fwhm_rad


def mosaic_psi_rad(case, E_pk_eV):
    """psi = angle(beam velocity, g) [rad] of the reflection whose NOMINAL
    (unscattered-beam) resonance energy is nearest E_pk -- the representative
    geometry for the analytic mosaic broadening (mosaic_fwhm_eV). Mirrors how
    aperture_fwhm_eV uses the nominal theta_obs rather than the per-segment scattered
    directions. Returns None if no listed reflection radiates a positive line."""
    info = CRYSTALS[case["crystal"]]
    beam_dir, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    beta = beta_from_keV(case["E0_keV"])
    R = _orientation_R(info["lattice"], case.get("beam_uvw"), case.get("azimuth_rad", 0.0))
    denom = 1.0 - beta * float(beam_dir @ n_hat)  # g-independent (Doppler denominator)
    if denom <= 0.0:
        return None
    best = None
    for hkl in case["hkl_list"]:
        g_vec, g = reciprocal_g_vector(hkl, info["lattice"])
        if R is not None:
            g_vec = R @ g_vec
        bdotg = float(beam_dir @ g_vec)
        E_res = HBARC_EV_ANG * beta * bdotg / denom
        if E_res <= 0.0:
            continue
        psi = np.arccos(np.clip(bdotg / g, -1.0, 1.0))
        d = abs(E_res - E_pk_eV)
        if best is None or d < best[0]:
            best = (d, psi)
    return None if best is None else best[1]


def convolve_detector(E_grid_eV, spec, fwhm_eV):
    """
    Convolve a spectrum with a unit-area Gaussian of the given FWHM [eV].
    Output always matches the input length for ANY fwhm (np.convolve with
    mode="same" returned an OVERSIZED array whenever the kernel outgrew the
    spectrum -- large FWHM on a short grid; truncating the kernel instead
    distorts the lineshape). Edges are zero-padded: counts blurred past the
    grid ends are lost, consistent with a detector band edge. Requires a
    uniform energy grid.
    """
    from scipy.ndimage import gaussian_filter1d

    dE = E_grid_eV[1] - E_grid_eV[0]
    sigma_bins = fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0))) / dE
    return gaussian_filter1d(np.asarray(spec, dtype=float), sigma_bins, mode="constant", cval=0.0)
