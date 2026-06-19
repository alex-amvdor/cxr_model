"""montecarlo.py: the "no silent default material" guards + the transport table.

Importing montecarlo prints a GPU/CPU banner and is otherwise CPU-only here; no
full sweep is run (that lives in checks/)."""

import numpy as np
import pytest

from montecarlo import (
    simulate_trajectories,
    mc_spectrum,
    _normalize_composition,
    TRANSPORT_ELEMENTS,
)


def test_normalize_requires_material():
    with pytest.raises(ValueError):
        _normalize_composition(None, None, None)


def test_normalize_composition_passthrough():
    assert _normalize_composition(None, None, [("Mo", 0.01)]) == [("Mo", 0.01)]


def test_simulate_trajectories_refuses_silent_default():
    with pytest.raises(ValueError):
        simulate_trajectories(30.0, 4, 1e4)  # no element / composition given


def test_mc_spectrum_requires_crystal_and_hkl():
    with pytest.raises(TypeError):
        mc_spectrum({"E_keV": np.array([30.0])}, np.arange(50.0, 100.0, 1.0))


def test_mc_spectrum_requires_B_ang2():
    # crystal + hkl supplied, but B_ang2 left at its sentinel -> explicit ValueError
    with pytest.raises(ValueError):
        mc_spectrum({}, np.arange(50.0, 100.0, 1.0), crystal="silicon", hkl_list=[(1, 1, 1)])


def test_tellurium_in_transport_table():
    assert TRANSPORT_ELEMENTS["Te"]["Z"] == 52
    assert TRANSPORT_ELEMENTS["Te"]["A"] == pytest.approx(127.6, abs=0.1)
