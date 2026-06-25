"""Film-on-substrate (multilayer) first slice: the layered-absorber escape
geometry and the sweep plumbing.

Pure-CPU/numpy tests only -- the project keeps tests/ off the GPU; the full
mc_spectrum / mc_brem_spectrum bit-for-bit + absorption checks live in
checks/multilayer_check.py. These pin the escape-path math and the wiring."""

import numpy as np
import pytest

from cxr_model.montecarlo import (
    _layer_dz,
    _mu_total_inv_ang,
    _stack_tau,
    simulate_trajectories,
)
from cxr_model.sweep import (
    Sweep,
    build_cases,
    film_on_substrate_layers,
    substrate_composition,
    substrate_radiator,
)


def test_layer_dz_back_exit():
    # photon exits the back face (n_z>0): escape ray spans depths [z_mid, z_total]
    z = np.array([100.0, 500.0, 900.0])
    assert np.allclose(_layer_dz(z, +1.0, 0.0, 500.0), [400.0, 0.0, 0.0])  # film
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


def test_transport_single_layer_explicit_equals_none():
    # an explicit one-layer stack reproduces the None (single-material) transport
    # bit-for-bit (same RNG stream -> identical segments)
    comp = [("Mo", 0.019), ("Se", 0.038)]
    kw = dict(E_cut_keV=5.0, seed=7)
    a = simulate_trajectories(30.0, 150, 1e4, composition=comp, **kw)
    b = simulate_trajectories(30.0, 150, 1e4, layers=[(0.0, 1e4, comp)], **kw)
    assert a["n_backscattered"] == b["n_backscattered"]
    assert a["L_ang"].size == b["L_ang"].size
    assert np.array_equal(a["L_ang"], b["L_ang"])
    assert np.array_equal(a["r_mid"], b["r_mid"])
    assert b["n_layers"] == 1 and set(b["layer"].tolist()) == {0}


def test_transport_backscatter_increases_with_substrate_Z():
    # a high-Z substrate backscatters more electrons into the thin film, raising
    # the film-region (layer 0) electron path vs a low-Z substrate
    film = [("C", 0.176)]
    t_f = 100.0  # 10 nm film
    stack_lo = [(0.0, t_f, film), (t_f, 1e5, [("C", 0.176)])]
    stack_hi = [(0.0, t_f, film), (t_f, 1e5, [("W", 0.0632)])]
    kw = dict(E_cut_keV=5.0, seed=3, elastic_model="sr")  # SR: no Mott table needed
    lo = simulate_trajectories(30.0, 500, 1e5, layers=stack_lo, **kw)
    hi = simulate_trajectories(30.0, 500, 1e5, layers=stack_hi, **kw)
    film_path_lo = lo["L_ang"][lo["layer"] == 0].sum()
    film_path_hi = hi["L_ang"][hi["layer"] == 0].sum()
    assert film_path_hi > film_path_lo
    assert hi["n_backscattered"] > lo["n_backscattered"]
    assert hi["n_layers"] == 2


def test_build_cases_attaches_abs_layers_only_with_substrate():
    plain = build_cases(Sweep(material="mose2", tilt_deg=-30.0, energy_keV=30.0))
    assert all(c["abs_layers"] is None for c in plain)

    stacked = build_cases(
        Sweep(
            material="mose2",
            tilt_deg=-30.0,
            energy_keV=30.0,
            substrate="sio2",
            substrate_thickness_ang=1e6,
        )
    )
    for c in stacked:
        assert c["abs_layers"] is not None and len(c["abs_layers"]) == 2
        (z0, z1, _), (z2, _z3, _) = c["abs_layers"]
        assert z0 == 0.0 and z1 == c["thickness_ang"] and z2 == c["thickness_ang"]
        assert "on sio2" in c["name"]


# ---- slice 3: per-layer coherent radiation -----------------------------------
def test_substrate_radiator_crystalline_vs_amorphous():
    # a crystalline substrate carries its own radiator (crystal params); an
    # amorphous preset radiates no coherent lines (None)
    assert substrate_radiator("sio2") is None
    assert substrate_radiator("al2o3") is None
    si = substrate_radiator("silicon")
    assert si["crystal"] == "silicon"
    assert set(si) == {"crystal", "hkl_list", "B_ang2", "beam_uvw"}
    assert len(si["hkl_list"]) > 0
    with pytest.raises(ValueError):
        substrate_radiator("unobtainium")


def test_build_cases_layer_radiators_match_stack():
    # no substrate -> single slab, no per-layer radiators
    plain = build_cases(Sweep(material="mose2", tilt_deg=-30.0, energy_keV=30.0))
    assert all(c["layer_radiators"] is None for c in plain)

    # amorphous substrate -> [film radiator, None] (substrate adds no lines)
    amorph = build_cases(
        Sweep(material="mose2", tilt_deg=-30.0, energy_keV=30.0, substrate="sio2")
    )[0]
    film, sub = amorph["layer_radiators"]
    assert sub is None
    # the film radiator must match the case's scalar crystal keys exactly
    assert film["crystal"] == amorph["crystal"]
    assert film["hkl_list"] == amorph["hkl_list"]
    assert film["B_ang2"] == amorph["B_ang2"]
    assert film["beam_uvw"] == amorph["beam_uvw"]

    # crystalline substrate -> [film radiator, silicon radiator]
    cryst = build_cases(
        Sweep(material="mose2", tilt_deg=-30.0, energy_keV=30.0, substrate="silicon")
    )[0]
    assert cryst["layer_radiators"][1]["crystal"] == "silicon"
    assert len(cryst["layer_radiators"]) == len(cryst["abs_layers"]) == 2
