"""Film-on-substrate (multilayer) first slice: the layered-absorber escape
geometry and the sweep plumbing.

Pure-CPU/numpy tests only -- the project keeps tests/ off the GPU; the full
mc_spectrum / mc_brem_spectrum bit-for-bit + absorption checks live in
checks/multilayer_check.py. These pin the escape-path math and the wiring."""

import numpy as np
import pytest

from montecarlo import _layer_dz, _stack_tau, _mu_total_inv_ang
from sweep import (
    Sweep,
    build_cases,
    substrate_composition,
    film_on_substrate_layers,
)


def test_layer_dz_back_exit():
    # photon exits the back face (n_z>0): escape ray spans depths [z_mid, z_total]
    z = np.array([100.0, 500.0, 900.0])
    assert np.allclose(_layer_dz(z, +1.0, 0.0, 500.0), [400.0, 0.0, 0.0])     # film
    assert np.allclose(_layer_dz(z, +1.0, 500.0, 1000.0), [500.0, 500.0, 100.0])  # sub


def test_layer_dz_front_exit():
    # photon exits the entrance face (n_z<0): escape ray spans depths [0, z_mid]
    z = np.array([100.0, 500.0, 900.0])
    assert np.allclose(_layer_dz(z, -1.0, 0.0, 500.0), [100.0, 500.0, 500.0])  # film
    # the substrate is crossed only by a segment that sits INSIDE it (z=900)
    assert np.allclose(_layer_dz(z, -1.0, 500.0, 1000.0), [0.0, 0.0, 400.0])


def test_substrate_transparent_to_film_segments_on_front_exit():
    # first-slice geometry point: film-region segments (z < t_film) leaving the
    # front face do NOT cross the substrate behind them (so a negative-tilt,
    # high-flux geometry sees no substrate attenuation of the film lines).
    z_film = np.array([10.0, 100.0, 480.0])
    assert np.allclose(_layer_dz(z_film, -1.0, 500.0, 1000.0), 0.0)


def test_stack_tau_single_layer_matches_slab():
    # one layer over [0, T] must equal the single-slab L_esc * mu, both faces
    z = np.array([200.0, 1500.0, 4000.0])
    E = np.full(3, 1500.0)
    comp = [("Si", 0.04994)]
    T = 5000.0
    mu = _mu_total_inv_ang(comp, E)
    for n_z in (-0.7, +0.7):
        tau = _stack_tau([(0.0, T, comp)], z, n_z, E)
        L_esc = z / (-n_z) if n_z < 0 else (T - z) / n_z
        assert np.allclose(tau, L_esc * mu)


def test_stack_tau_substrate_adds_depth_on_back_exit():
    # adding a substrate behind the film increases tau for a back exit
    z = np.array([100.0, 300.0])
    E = np.full(2, 1500.0)
    film = [("Mo", 0.019), ("Se", 0.038)]
    sub = substrate_composition("sio2")
    tau_film = _stack_tau([(0.0, 500.0, film)], z, +0.7, E)
    tau_stack = _stack_tau([(0.0, 500.0, film), (500.0, 1e6, sub)], z, +0.7, E)
    assert np.all(tau_stack > tau_film)


def test_substrate_composition_presets_and_crystal():
    sio2 = dict(substrate_composition("sio2"))
    assert sio2["Si"] == pytest.approx(0.02205)
    assert sio2["O"] == pytest.approx(2 * sio2["Si"], rel=1e-3)  # SiO2 stoichiometry
    si = dict(substrate_composition("silicon"))  # crystalline, from CRYSTALS
    assert 0.04 < si["Si"] < 0.06
    with pytest.raises(ValueError):
        substrate_composition("unobtainium")


def test_film_on_substrate_layers_structure():
    comp = [("Mo", 0.01), ("Se", 0.02)]
    (z0, z1, c_film), (z2, z3, c_sub) = film_on_substrate_layers(comp, 500.0, "sio2", 1e6)
    assert (z0, z1, z2, z3) == (0.0, 500.0, 500.0, 500.0 + 1e6)
    assert c_film == comp
    assert dict(c_sub)["Si"] == pytest.approx(0.02205)


def test_build_cases_attaches_abs_layers_only_with_substrate():
    plain = build_cases(Sweep(material="mose2", tilt_deg=-30.0, energy_keV=30.0))
    assert all(c["abs_layers"] is None for c in plain)

    stacked = build_cases(
        Sweep(material="mose2", tilt_deg=-30.0, energy_keV=30.0,
              substrate="sio2", substrate_thickness_ang=1e6)
    )
    for c in stacked:
        assert c["abs_layers"] is not None and len(c["abs_layers"]) == 2
        (z0, z1, _), (z2, z3, _) = c["abs_layers"]
        assert z0 == 0.0 and z1 == c["thickness_ang"] and z2 == c["thickness_ang"]
        assert "on sio2" in c["name"]
