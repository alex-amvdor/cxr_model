"""Atomic form factors: the Cromer-Mann coefficients and Henke loader.

The Sum(a)+c == Z check is the cheap regression that catches a fat-fingered
coefficient when adding a new element (a wrong digit shows up as f0(0) != Z)."""

import numpy as np
import pytest

from atomic_form_factors import (
    Z_TABLE,
    CROMER_MANN,
    cromer_mann_f0,
    atomic_form_factor,
    load_henke,
)


@pytest.mark.parametrize("element", sorted(CROMER_MANN))
def test_cromer_mann_f0_at_zero_equals_Z(element):
    # f0(g=0) = sum(a_i) + c must reproduce the atomic number
    coeffs = CROMER_MANN[element]
    f0_0 = sum(coeffs[0:8:2]) + coeffs[8]
    assert f0_0 == pytest.approx(Z_TABLE[element], abs=0.1)
    assert float(cromer_mann_f0(element, 0.0)) == pytest.approx(Z_TABLE[element], abs=0.1)


def test_f0_decreases_with_g():
    f = cromer_mann_f0("Mo", np.array([0.0, 1.0, 3.0, 6.0]))
    assert np.all(np.diff(f) < 0)


def test_tellurium_registered():
    assert Z_TABLE["Te"] == 52
    assert "Te" in CROMER_MANN


def test_henke_table_loads_and_is_complex_factor():
    E, f1, f2 = load_henke("Te")
    assert E.size > 0 and np.all(np.isfinite(f1)) and np.all(f2 >= 0)
    F = atomic_form_factor("Te", 1.0, 10000.0)  # in-range energy -> finite complex
    assert np.isfinite(F.real) and np.isfinite(F.imag)
