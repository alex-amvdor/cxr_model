"""Exact Monte-Carlo crystal-mosaicity average (docs/crystal-mosaicity.md route 2):
the Gauss-Hermite orientation quadrature, the tilt rotation, and the Sweep wiring
that makes the MC and analytic routes mutually exclusive.

These stay synthetic (no Monte-Carlo transport / mc_spectrum) so the suite remains
CPU-only and fast; the mc_spectrum-level validation (perfect-crystal bit-for-bit,
broadening, node convergence, yield change) lives in checks/mosaic_mc_check.py."""

import numpy as np
import pytest

from cxr_model.sweep import Sweep, build_cases
from cxr_model.results import store_result
from cxr_model.montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    eds_fwhm_eV,
    _mosaic_quadrature,
    _small_tilt_R,
)

E_GRID = np.arange(400.0, 1400.0, 1.0)


# ---- the tilt rotation -------------------------------------------------------
def test_small_tilt_identity_at_zero():
    assert np.array_equal(_small_tilt_R(0.0, 0.0), np.eye(3))


def test_small_tilt_is_a_rotation_by_the_right_angle():
    dx, dy = np.deg2rad(3.0), np.deg2rad(-2.0)
    R = _small_tilt_R(dx, dy)
    assert np.allclose(R @ R.T, np.eye(3))  # orthogonal
    assert np.isclose(np.linalg.det(R), 1.0)  # proper
    # tilts +z by the rotation-vector magnitude
    assert np.isclose(np.arccos((R @ [0, 0, 1.0])[2]), np.hypot(dx, dy))


# ---- the quadrature ----------------------------------------------------------
def test_quadrature_is_none_for_perfect_crystal():
    assert _mosaic_quadrature(None, 5) is None  # no spread
    assert _mosaic_quadrature(0.0, 5) is None  # zero spread
    assert _mosaic_quadrature(np.deg2rad(3.5), 1) is None  # single node == identity
    assert _mosaic_quadrature(np.deg2rad(3.5), 0) is None


def test_quadrature_weights_sum_to_one_and_count_is_nodes_squared():
    quad = _mosaic_quadrature(np.deg2rad(3.5), 5)
    assert len(quad) == 25
    assert sum(w for _, w in quad) == pytest.approx(1.0)
    assert all(R.shape == (3, 3) for R, _ in quad)


def test_quadrature_node_zero_carries_the_most_weight():
    # the central (zero-tilt) node of an odd Gauss-Hermite rule has the largest weight
    quad = _mosaic_quadrature(np.deg2rad(3.5), 5)
    weights = [w for _, w in quad]
    assert max(weights) == weights[12]  # (j, k) = (2, 2), the central node of 5x5


def test_quadrature_nodes_scale_with_spread():
    # the off-centre node tilt is proportional to the mosaic FWHM
    q1 = _mosaic_quadrature(np.deg2rad(1.0), 3)
    q2 = _mosaic_quadrature(np.deg2rad(2.0), 3)
    # node 0 is the (-,-) corner; its rotation tilts +z twice as far at 2x the spread
    z1 = np.arccos((q1[0][0] @ [0, 0, 1.0])[2])
    z2 = np.arccos((q2[0][0] @ [0, 0, 1.0])[2])
    assert z2 == pytest.approx(2.0 * z1, rel=1e-6)


# ---- Sweep wiring: the two routes are mutually exclusive ----------------------
def _sweep(**kw):
    return Sweep(
        material="hopg", thickness_ang=1e4, energy_keV=30, tilt_deg=-30.0,
        E_grid_line=E_GRID, E_grid_brem=np.arange(0.0, 1000.0, 100.0), **kw,
    )


def test_route_analytic_sets_analytic_term_only():
    case = build_cases(_sweep(mosaic=True))[0]  # route defaults to "analytic"
    assert case["mosaic_fwhm_rad"] == pytest.approx(np.deg2rad(0.8))
    assert case["mosaic_mc_fwhm_rad"] is None


def test_route_mc_sets_mc_term_and_suppresses_analytic():
    case = build_cases(_sweep(mosaic=True, mosaic_route="mc", mosaic_nodes=7))[0]
    assert case["mosaic_mc_fwhm_rad"] == pytest.approx(np.deg2rad(0.8))
    assert case["mosaic_mc_nodes"] == 7
    assert case["mosaic_fwhm_rad"] is None  # analytic OFF -> no double count


def test_route_mc_is_noop_without_mosaic_data():
    # diamond has no mosaic_fwhm_deg, so even route="mc" leaves it a perfect crystal
    case = build_cases(
        Sweep(material="diamond", thickness_ang=1e4, energy_keV=30, tilt_deg=-30.0,
              E_grid_line=E_GRID, E_grid_brem=np.arange(0.0, 1000.0, 100.0),
              mosaic=True, mosaic_route="mc")
    )[0]
    assert case["mosaic_mc_fwhm_rad"] is None
    assert case["mosaic_fwhm_rad"] is None


def test_route_mc_adds_no_analytic_broadening_in_store_result():
    # store_result must NOT add a mosaic term for an MC-route case (it is already in
    # the spectrum): the reported fwhm is the EDS + aperture detector width only.
    case = build_cases(_sweep(mosaic=True, mosaic_route="mc"))[0]
    out = {
        "E_grid": E_GRID,
        "spec": np.exp(-0.5 * ((E_GRID - 860.0) / 6.0) ** 2),
        "brem": np.full_like(E_GRID, 0.1),
        "eta": 0.1,
    }
    res = {}
    store_result(res, case, out)
    r = res[case["name"]][30.0]
    expected = np.sqrt(
        eds_fwhm_eV(r["E_pk"]) ** 2
        + aperture_fwhm_eV(
            r["E_pk"], beta_from_keV(30.0), case["theta_obs_rad"], case["dtheta_obs_rad"]
        )
        ** 2
    )
    assert r["fwhm"] == pytest.approx(expected)


def test_bad_route_raises():
    with pytest.raises(ValueError, match="mosaic_route"):
        build_cases(_sweep(mosaic=True, mosaic_route="nonsense"))
