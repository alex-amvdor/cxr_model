"""
montecarlo.geometry

Lab/sample-frame geometry shared by transport, spectrum and detector: the
tilted-sample beam/detector directions, the finite-detector-face direction
grid, the crystal-orientation and small-tilt rotations, and the Gauss-Hermite
mosaic-orientation quadrature.
"""

import numpy as np

from ..crystallography import _direct_lattice_vectors, _rotation_between


def tilted_geometry(theta_obs_rad, tilt_polar_rad, tilt_azim_rad=0.0):
    """
    Sample-frame beam and detector directions for a TILTED sample.

    Transport and radiation work in the sample frame (slab normal and crystal
    construction frame along +z, e.g. the HOPG c-axis). In the lab the beam
    is fixed along +z_lab and the detector sits at polar angle theta_obs_rad,
    azimuth 0. Tilting the sample so its normal points along
    (sin tp cos ta, sin tp sin ta, cos tp) in the lab (tp = tilt_polar_rad,
    ta = tilt_azim_rad, cf. Zhai SI Fig. 1) is equivalent to rotating the
    beam and detector into the sample frame.

    Returns (beam_dir, n_hat) to pass to simulate_trajectories(beam_dir=...)
    and mc_spectrum(n_hat=...). For ta = 0 the detector's sample-frame polar
    angle is simply theta_obs - tilt_polar; the lab-frame quantity
    1 - v0.n_hat (hence the zero-scattering line energy denominator) is
    tilt-invariant, while v0.g picks up cos(tilt).
    """
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)
    beam_dir = R.T @ np.array([0.0, 0.0, 1.0])
    n_hat = R.T @ np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    return beam_dir, n_hat


def detector_directions(
    theta_obs_rad,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
    *,
    n_side=1,
    chip_mm=14.0,
    dist_mm=30.0,
    domega_sr,
):
    """
    Grid of detector directions {n_hat_i} (SAMPLE frame) tiling a flat square
    detector chip of side ``chip_mm`` at distance ``dist_mm`` facing the source,
    with their solid-angle weights -- the geometry input to the solid-angle-
    INTEGRATED spectrum (mc_spectrum_solid_angle). This replaces the single-n_hat
    + flat-Omega + analytic aperture_fwhm_eV approximation
    (docs/detector-solid-angle.md) with a first-principles tiling of the face.

    The central cell sits at polar angle ``theta_obs_rad`` (azimuth 0) in the lab
    and is mapped into the sample frame through the SAME tilt rotation as
    tilted_geometry(), so n_side=1 returns exactly that single direction. The chip
    in-plane axes are chosen so one grid axis spreads in the scattering plane (the
    polar / Delta-theta direction) and the other out of plane (azimuth). The
    per-cell weight is the inverse-square + obliquity solid angle
    ``dOmega_i = dA_i cos(psi_i) / r_i^2`` (psi_i to the chip normal = central
    line of sight), then the whole set is rescaled so ``sum_i dOmega_i ==
    domega_sr``: the detector's known total solid angle is conserved and n_side=1
    reproduces today's ``spec * Omega`` exactly.

    Returns (n_hats, weights): n_hats is (N, 3) unit directions in the sample
    frame (N = n_side**2); weights is (N,) and sums to ``domega_sr``.
    """
    if n_side < 1:
        raise ValueError("n_side must be >= 1")
    # tilt rotation (same convention as tilted_geometry): sample normal -> lab
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)

    # central lab line of sight c, and chip in-plane axes (chip face _|_ c):
    #   w lies in the scattering (x-z) plane -> polar (Delta-theta) spread
    #   u is out of plane (+y)               -> azimuthal spread
    c = np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    w = np.array([np.cos(theta_obs_rad), 0.0, -np.sin(theta_obs_rad)])
    u = np.array([0.0, 1.0, 0.0])

    step = chip_mm / n_side
    offs = (np.arange(n_side) - (n_side - 1) / 2.0) * step  # cell centres
    da = step**2  # cell area [mm^2]

    n_hats = np.empty((n_side * n_side, 3))
    weights = np.empty(n_side * n_side)
    i = 0
    for a in offs:  # out-of-plane (azimuth)
        for b in offs:  # in-plane (polar)
            P = dist_mm * c + a * u + b * w  # source -> cell vector [mm]
            r = float(np.linalg.norm(P))
            n_lab = P / r
            cos_psi = float(n_lab @ c)  # obliquity to the chip normal (= c)
            n_hats[i] = R.T @ n_lab
            weights[i] = da * cos_psi / r**2
            i += 1
    weights *= domega_sr / weights.sum()  # conserve the detector's total Omega
    return n_hats, weights


def _orientation_R(lattice, beam_uvw, azimuth_rad):
    """Rotation applied to EVERY reciprocal vector: the minimal rotation taking the
    crystal direct-lattice direction `beam_uvw` onto +z (the slab normal), then a
    roll of `azimuth_rad` about +z. Returns None for the construction-frame default
    (beam_uvw is None and azimuth_rad == 0). Shared by mc_spectrum and mosaic_psi_rad
    so the orientation convention lives in one place."""
    R = None
    if beam_uvw is not None:
        u, v, w = np.asarray(beam_uvw, dtype=float)
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        axis = u * a1 + v * a2 + w * a3
        R = _rotation_between(axis / np.linalg.norm(axis), np.array([0.0, 0.0, 1.0]))
    if azimuth_rad:
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        R = Rz if R is None else Rz @ R
    return R


def _small_tilt_R(dx_rad, dy_rad):
    """Rotation that tilts the slab normal by the rotation vector (dx, dy, 0) [rad]
    via Rodrigues -- exact for any angle, exactly the identity at (0, 0). Used to
    misorient a mosaic crystallite's reciprocal vectors about the mean orientation."""
    ang = float(np.hypot(dx_rad, dy_rad))
    if ang < 1e-15:
        return np.eye(3)
    kx, ky = dx_rad / ang, dy_rad / ang
    K = np.array([[0.0, 0.0, ky], [0.0, 0.0, -kx], [-ky, kx, 0.0]])  # skew of axis
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _mosaic_quadrature(fwhm_rad, nodes):
    """Gauss-Hermite product quadrature over a 2-D Gaussian mosaic tilt of the
    crystallite normal (per-axis sigma = FWHM / 2.3548 -- the rocking curve is the
    1-D projection). Returns a list of (rotation matrix, weight) with weights
    summing to 1, used by mc_spectrum to INCOHERENTLY average a reflection's
    spectrum over crystallite orientations (the exact mosaic route,
    docs/crystal-mosaicity.md (2)).

    Returns None -- the perfect-crystal fast path, today's result bit-for-bit --
    when there is nothing to average: ``fwhm_rad`` falsy, or ``nodes`` <= 1 (the
    single Gauss-Hermite node sits at zero tilt, i.e. the identity, so the loop is
    skipped entirely).

    This is the DETERMINISTIC counterpart to drawing K random orientations: the
    integrand (spectrum vs crystallite tilt) is smooth, so product Gauss-Hermite
    converges in far fewer evaluations than random sampling and needs no RNG
    sub-stream. Cost is K = nodes**2 evaluations of the per-reflection block, a
    direct wall-clock multiplier on the (serial, with CuPy) GPU hot loop."""
    if not fwhm_rad or nodes is None or nodes <= 1:
        return None
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    sigma = float(fwhm_rad) / 2.3548200450309493  # FWHM -> Gaussian sigma
    tilt = np.sqrt(2.0) * sigma * x  # quadrature nodes as tilt angles [rad]
    wt = w / np.sqrt(np.pi)  # per-axis weights, sum to 1
    return [
        (_small_tilt_R(tilt[j], tilt[k]), float(wt[j] * wt[k]))
        for j in range(len(tilt))
        for k in range(len(tilt))
    ]
