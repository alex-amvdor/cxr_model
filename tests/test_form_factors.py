"""Atomic form factors, now sourced from xraydb (Waasmaier-Kirfel f0 + Chantler
f',f''). f0(g=0) == Z is the cheap regression that catches a backend/units slip;
load_henke must still return a finite, physical (E, f1, f2) table per element."""

import numpy as np
import pytest

from cxr_model.atomic_form_factors import (
    Z_TABLE,
    atomic_form_factor,
    cromer_mann_f0,
    henke_dispersion,
    load_henke,
)

# the structure-factor elements the project models (light + edge-prone + heavy)
ELEMENTS = ["C", "Li", "F", "Si", "Ge", "S", "Mo", "Se", "Zr", "Te", "Hf", "W", "Pt"]


@pytest.mark.parametrize("element", ELEMENTS)
def test_f0_at_zero_equals_Z(element):
    # f0(g=0) must reproduce the atomic number (Waasmaier-Kirfel -> Z at s=0)
    assert float(cromer_mann_f0(element, 0.0)) == pytest.approx(Z_TABLE[element], abs=0.1)


def test_f0_decreases_with_g():
    f = cromer_mann_f0("Mo", np.array([0.0, 1.0, 3.0, 6.0]))
    assert np.all(np.diff(f) < 0)


def test_tellurium_registered():
    assert Z_TABLE["Te"] == 52
    assert "Te" in Z_TABLE


def test_unknown_element_raises():
    assert "Xx" not in Z_TABLE
    with pytest.raises(KeyError):
        _ = Z_TABLE["Xx"]


def test_henke_table_loads_and_is_complex_factor():
    E, f1, f2 = load_henke("Te")
    assert E.size > 0 and np.all(np.isfinite(f1)) and np.all(f2 >= 0)
    F = atomic_form_factor("Te", 1.0, 10000.0)  # in-range energy -> finite complex
    assert np.isfinite(F.real) and np.isfinite(F.imag)


def test_out_of_range_energy_is_nan():
    # E below the tabulated grid (and the brem-grid E=0 bin) must read as NaN,
    # not raise, so absorption_length_ang stays index-aligned.
    fp, fpp = henke_dispersion("Te", np.array([0.0, 5000.0]))
    assert np.isnan(fp[0]) and np.isnan(fpp[0])
    assert np.isfinite(fp[1]) and np.isfinite(fpp[1])
