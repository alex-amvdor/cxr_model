"""Tests for the grazing-incidence grating forward model (P3 #8). Pure numpy."""

import numpy as np
import pytest

from cxr_model.crystallography import HC_EV_ANG
from cxr_model.grating import (
    Grating,
    detector_position_mm,
    disperse_spectrum,
    groove_spacing_angstrom,
    resolving_power,
    wavelength_angstrom,
)


def test_wavelength_and_spacing():
    assert wavelength_angstrom(HC_EV_ANG) == pytest.approx(1.0)  # E = hc -> 1 A
    assert groove_spacing_angstrom(1000.0) == pytest.approx(1.0e4)  # 1000/mm -> 1 um


def test_zero_order_is_specular():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(88.0), order=0)
    beta = g.diffraction_angle_rad(np.array([300.0, 900.0]))
    assert np.allclose(beta, -g.alpha_rad)  # sin beta = -sin alpha


def test_first_order_known_value():
    # alpha = 0, d = 1e4 A (1000/mm), order 1, lambda = 100 A (E = hc/100):
    # sin beta = 1 * 100 / 1e4 = 0.01
    g = Grating(groove_density_per_mm=1000.0, alpha_rad=0.0, order=1)
    E = HC_EV_ANG / 100.0
    assert float(g.diffraction_angle_rad(E)) == pytest.approx(np.arcsin(0.01))


def test_dispersion_monotonic_in_energy():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(85.0), order=1)
    # longer wavelength (lower energy) diffracts to a larger beta
    assert float(g.diffraction_angle_rad(300.0)) > float(g.diffraction_angle_rad(1200.0))


def test_evanescent_orders_are_nan():
    # extreme spacing/order so m*lambda/d > 1 -> no propagating order -> NaN.
    # (Realistic soft X-ray gratings stay well clear of this -- that's why they
    # work; here we just exercise the cutoff branch.) d = 50 A, lambda ~ 124 A.
    g = Grating(groove_density_per_mm=200000.0, alpha_rad=0.0, order=1)
    assert np.isnan(float(g.diffraction_angle_rad(100.0)))


def test_detector_position_sign_and_zero():
    assert detector_position_mm(0.5, 0.5, 100.0) == pytest.approx(0.0)
    assert detector_position_mm(0.6, 0.5, 100.0) > 0.0


def test_disperse_spectrum_conserves_flux_and_localizes_line():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)  # a line at 850 eV
    x, inten = disperse_spectrum(E, spec, g, distance_mm=200.0)
    # flux conserved: integral(I dx) ~ integral(spec dE)
    order = np.argsort(x)
    flux_x = np.trapezoid(inten[order], x[order])
    flux_E = np.trapezoid(spec, E)
    assert flux_x == pytest.approx(flux_E, rel=0.02)
    # the line maps to a localized band (intensity-weighted std is small in mm)
    w = inten[order] / inten[order].sum()
    xc = (w * x[order]).sum()
    spread = np.sqrt((w * (x[order] - xc) ** 2).sum())
    assert spread < 0.05 * (x.max() - x.min())


def test_resolving_power_positive_and_rises_with_density():
    coarse = Grating(groove_density_per_mm=600.0, alpha_rad=np.deg2rad(86.0), order=1)
    fine = Grating(groove_density_per_mm=2400.0, alpha_rad=np.deg2rad(86.0), order=1)
    R_coarse = float(resolving_power(900.0, coarse, distance_mm=200.0, pixel_mm=0.015))
    R_fine = float(resolving_power(900.0, fine, distance_mm=200.0, pixel_mm=0.015))
    assert R_coarse > 0 and R_fine > R_coarse
