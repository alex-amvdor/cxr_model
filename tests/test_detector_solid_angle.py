"""Tests for the detector solid-angle tiling (P2 #4).

detector_directions() is pure geometry (numpy), so it is fully tested here. The
integrated spectrum mc_spectrum_solid_angle() drives the GPU mc_spectrum per
direction, so its physics regression (n_side=1 == single-angle x Omega; the wide-
detector asymmetry) lives in checks/detector_solid_angle_check.py; here we only
check its cheap input validation.
"""

import numpy as np
import pytest

from cxr_mc.montecarlo import (
    detector_directions,
    mc_spectrum_solid_angle,
    tilted_geometry,
)

THETA = np.deg2rad(119.0)
OMEGA = 0.066


def test_nside1_is_single_central_direction():
    """n_side=1 reproduces tilted_geometry's n_hat with the full Omega weight."""
    n_hats, weights = detector_directions(THETA, n_side=1, domega_sr=OMEGA)
    assert n_hats.shape == (1, 3)
    assert weights.shape == (1,)
    _, n_hat = tilted_geometry(THETA, 0.0)
    assert np.allclose(n_hats[0], n_hat)
    assert weights[0] == pytest.approx(OMEGA)


def test_nside1_tracks_tilt():
    """With a tilted sample, the central direction follows tilted_geometry."""
    tp, ta = np.deg2rad(20.0), np.deg2rad(35.0)
    n_hats, _ = detector_directions(THETA, tp, ta, n_side=1, domega_sr=OMEGA)
    _, n_hat = tilted_geometry(THETA, tp, ta)
    assert np.allclose(n_hats[0], n_hat)


@pytest.mark.parametrize("n_side", [1, 2, 3, 5, 8])
def test_weights_sum_to_total_omega(n_side):
    _, weights = detector_directions(THETA, n_side=n_side, domega_sr=OMEGA)
    assert weights.shape == (n_side * n_side,)
    assert weights.sum() == pytest.approx(OMEGA)
    assert np.all(weights > 0)


@pytest.mark.parametrize("n_side", [1, 3, 6])
def test_directions_are_unit_vectors(n_side):
    n_hats, _ = detector_directions(THETA, n_side=n_side, domega_sr=OMEGA)
    assert np.allclose(np.linalg.norm(n_hats, axis=1), 1.0)


def test_grid_is_centred_on_central_direction():
    """The solid-angle-weighted mean direction points along the central n_hat."""
    n_hats, weights = detector_directions(THETA, n_side=6, domega_sr=OMEGA)
    mean = (weights[:, None] * n_hats).sum(axis=0)
    mean /= np.linalg.norm(mean)
    c = np.array([np.sin(THETA), 0.0, np.cos(THETA)])
    assert np.allclose(mean, c, atol=2e-3)


def test_centre_cell_has_largest_weight():
    """Inverse-square + obliquity make the central (nearest, head-on) cell heaviest."""
    n_side = 5
    _, weights = detector_directions(THETA, n_side=n_side, domega_sr=OMEGA)
    centre = (n_side // 2) * n_side + (n_side // 2)
    assert np.argmax(weights) == centre


def test_wider_chip_spreads_polar_angles():
    """A larger chip (same distance) tiles a wider angular range."""

    def polar_spread(chip_mm):
        n_hats, _ = detector_directions(
            THETA, n_side=5, chip_mm=chip_mm, dist_mm=30.0, domega_sr=OMEGA
        )
        polar = np.arccos(np.clip(n_hats[:, 2], -1.0, 1.0))
        return polar.max() - polar.min()

    assert polar_spread(30.0) > polar_spread(5.0)


def test_nside_must_be_positive():
    with pytest.raises(ValueError):
        detector_directions(THETA, n_side=0, domega_sr=OMEGA)


def test_solid_angle_spectrum_validates_shapes():
    """mc_spectrum_solid_angle rejects malformed grids before any GPU work."""
    segs = {"E_keV": np.array([25.0]), "Ne": 1, "thickness_ang": 1e7}
    with pytest.raises(ValueError):
        mc_spectrum_solid_angle(
            segs,
            np.arange(500.0, 600.0),
            "hopg",
            [(0, 0, 2)],
            n_hats=np.zeros((3, 2)),
            weights=np.ones(3),
            B_ang2=0.8,
        )
    with pytest.raises(ValueError):
        mc_spectrum_solid_angle(
            segs,
            np.arange(500.0, 600.0),
            "hopg",
            [(0, 0, 2)],
            n_hats=np.zeros((3, 3)),
            weights=np.ones(4),
            B_ang2=0.8,
        )
