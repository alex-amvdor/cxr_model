"""
grating.py

EXPLORATORY forward model for a grazing-incidence soft X-ray diffraction grating
spectrometer (TODO P3 #8): disperse the model's PXR+CBS line spectrum across a
position-sensitive detector (Eagle XO CCD or similar) so a *spatial* image can be
compared to experiment, instead of an energy spectrum at a fixed take-off angle.

Physics -- the reflection grating equation, normal-referenced:

    d (sin alpha + sin beta) = m lambda                      (sin beta = m lambda/d - sin alpha)

with alpha the incidence angle and beta the diffraction angle, both from the
grating NORMAL, d = 1/(groove density) the groove spacing, m the order. m = 0 is
specular (beta = -alpha). Soft X-rays need GRAZING incidence (grazing angle
theta_g = 90 deg - alpha of a few degrees) for usable reflectivity; that is a
property of `alpha`, not a different equation.

This module is standalone (nothing in the sweep/plot pipeline imports it yet);
see docs/grazing-grating.md for the modality and the phased plan. Reflectivity /
groove efficiency are NOT modelled here -- this is dispersion geometry only.
"""

from dataclasses import dataclass

import numpy as np

from .crystallography import HC_EV_ANG  # h*c [eV*Angstrom]


def wavelength_angstrom(E_eV):
    """Photon wavelength [Angstrom] from energy [eV] (lambda = hc / E)."""
    return HC_EV_ANG / np.asarray(E_eV, float)


def groove_spacing_angstrom(groove_density_per_mm):
    """Groove spacing d [Angstrom] from line density [lines/mm] (1 mm = 1e7 A)."""
    return 1.0e7 / float(groove_density_per_mm)


@dataclass(frozen=True)
class Grating:
    """A reflection grating in a fixed mount.

    groove_density_per_mm : ruling density (e.g. 1200 lines/mm).
    alpha_rad : incidence angle from the grating NORMAL (grazing => near pi/2).
    order : diffraction order m (1 typical; 0 is specular).
    """

    groove_density_per_mm: float
    alpha_rad: float
    order: int = 1

    @property
    def d_angstrom(self) -> float:
        return groove_spacing_angstrom(self.groove_density_per_mm)

    @property
    def grazing_angle_rad(self) -> float:
        """Incidence grazing angle theta_g = 90 deg - alpha."""
        return 0.5 * np.pi - self.alpha_rad

    def diffraction_angle_rad(self, E_eV):
        """Diffraction angle beta [rad] from the grating normal for energy E_eV
        (array-safe). sin beta = m lambda/d - sin alpha; energies that would need
        |sin beta| > 1 (no propagating order) return NaN."""
        lam = wavelength_angstrom(E_eV)
        s = self.order * lam / self.d_angstrom - np.sin(self.alpha_rad)
        s = np.where(np.abs(s) <= 1.0, s, np.nan)
        return np.arcsin(s)

    def angular_dispersion_rad_per_angstrom(self, E_eV):
        """d(beta)/d(lambda) = m / (d cos beta) [rad/Angstrom]; the grating's
        intrinsic dispersion before any detector geometry."""
        beta = self.diffraction_angle_rad(E_eV)
        return self.order / (self.d_angstrom * np.cos(beta))


def detector_position_mm(beta_rad, beta_ref_rad, distance_mm):
    """Where a ray diffracted at beta lands on a flat detector a distance
    ``distance_mm`` from the grating, whose face is perpendicular to the
    reference direction ``beta_ref_rad``: x = L tan(beta - beta_ref) [mm]."""
    return distance_mm * np.tan(np.asarray(beta_rad, float) - beta_ref_rad)


def disperse_spectrum(E_grid_eV, spec, grating, distance_mm, *, beta_ref_rad=None):
    """Map an energy spectrum onto detector positions through ``grating``.

    Returns ``(position_mm, intensity_per_mm)`` for the input energies whose order
    propagates (NaN-beta energies dropped). Flux-conserving: the per-eV spectrum
    is reweighted by the Jacobian ``|dE/dx|`` so that integral(I dx) == integral(spec dE).
    ``beta_ref_rad`` (the detector-normal direction) defaults to the mean
    diffraction angle, centring the dispersed band on the detector.
    """
    E = np.asarray(E_grid_eV, float)
    spec = np.asarray(spec, float)
    beta = grating.diffraction_angle_rad(E)
    if beta_ref_rad is None:
        beta_ref_rad = float(np.nanmean(beta))
    x = detector_position_mm(beta, beta_ref_rad, distance_mm)
    ok = np.isfinite(x)
    E, spec, x = E[ok], spec[ok], x[ok]
    if E.size < 2:
        return x, np.zeros_like(x)
    dxdE = np.gradient(x, E)
    inten = np.where(np.abs(dxdE) > 0, spec / np.abs(dxdE), 0.0)
    inten[~np.isfinite(inten)] = 0.0
    return x, inten


def resolving_power(E_eV, grating, distance_mm, pixel_mm, beta_ref_rad=None):
    """Pixel-limited resolving power lambda/dlambda at E_eV: one detector pixel
    subtends dlambda = pixel_mm / (dx/dlambda), with the linear dispersion
    dx/dlambda = distance * d(beta)/d(lambda) / cos^2(beta - beta_ref)."""
    beta = grating.diffraction_angle_rad(E_eV)
    if beta_ref_rad is None:
        beta_ref_rad = beta
    dbeta_dlam = grating.angular_dispersion_rad_per_angstrom(E_eV)
    dx_dlam = distance_mm * dbeta_dlam / np.cos(beta - beta_ref_rad) ** 2  # mm/Angstrom
    dlam = pixel_mm / dx_dlam  # Angstrom per pixel
    lam = wavelength_angstrom(E_eV)
    return lam / dlam
