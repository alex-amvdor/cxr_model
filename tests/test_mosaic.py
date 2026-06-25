"""Analytic crystal-mosaicity broadening: the formula, the per-crystal optional
data, the on/off switch, and the store_result quadrature.

All synthetic (a Gaussian line, no Monte-Carlo transport) so it stays fast."""

import numpy as np
import pytest

from cxr_mc.crystallography import CRYSTALS
from cxr_mc.montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    eds_fwhm_eV,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
)
from cxr_mc.results import store_result
from cxr_mc.sweep import Sweep, build_cases

LINE_GRID = np.arange(50.0, 300.0, 5.0)
BREM_GRID = np.arange(0.0, 1000.0, 100.0)


def _sweep(material, **kw):
    return Sweep(
        material=material,
        thickness_ang=1e4,
        energy_keV=30,
        tilt_deg=-30.0,
        E_grid_line=LINE_GRID,
        E_grid_brem=BREM_GRID,
        **kw,
    )


def _synthetic_out(E_pk=120.0):
    spec = np.exp(-0.5 * ((LINE_GRID - E_pk) / 4.0) ** 2)  # one clean line
    return {
        "E_grid": LINE_GRID,
        "spec": spec,
        "brem": np.full_like(LINE_GRID, 0.1),
        "eta": 0.1,
    }


# ---- the formula -------------------------------------------------------------
def test_mosaic_fwhm_zero_when_g_parallel_v():
    # psi = 0 (g along v): only a second-order shift, so the analytic FWHM is 0.
    assert mosaic_fwhm_eV(1000.0, 0.0, np.deg2rad(0.8)) == 0.0


def test_mosaic_fwhm_linear_in_spread_and_tan_psi():
    E, psi = 1500.0, np.deg2rad(35.0)
    one = mosaic_fwhm_eV(E, psi, np.deg2rad(0.4))
    two = mosaic_fwhm_eV(E, psi, np.deg2rad(0.8))
    assert two == pytest.approx(2.0 * one)  # linear in the mosaic spread
    assert one == pytest.approx(E * np.tan(psi) * np.deg2rad(0.4))  # E |tan psi| eta


# ---- per-crystal optional data ----------------------------------------------
def test_hopg_has_mosaic_data_perfect_crystals_dont():
    assert CRYSTALS["hopg"]["mosaic_fwhm_deg"] == pytest.approx(0.8)
    assert CRYSTALS["diamond"]["mosaic_fwhm_deg"] is None
    assert CRYSTALS["silicon"]["mosaic_fwhm_deg"] is None


# ---- the on/off switch in build_cases ---------------------------------------
def test_switch_off_is_no_mosaic():
    case = build_cases(_sweep("hopg"))[0]  # mosaic defaults to False
    assert case["mosaic_fwhm_rad"] is None


def test_switch_on_uses_per_crystal_value_for_hopg():
    case = build_cases(_sweep("hopg", mosaic=True))[0]
    assert case["mosaic_fwhm_rad"] == pytest.approx(np.deg2rad(0.8))


def test_switch_on_leaves_perfect_crystal_perfect():
    # diamond has no mosaic data, so mosaic=True is still a no-op for it.
    case = build_cases(_sweep("diamond", mosaic=True))[0]
    assert case["mosaic_fwhm_rad"] is None


def test_override_applies_even_to_a_perfect_crystal():
    case = build_cases(_sweep("diamond", mosaic=True, mosaic_fwhm_deg=0.4))[0]
    assert case["mosaic_fwhm_rad"] == pytest.approx(np.deg2rad(0.4))


def test_override_beats_per_crystal_value():
    case = build_cases(_sweep("hopg", mosaic=True, mosaic_fwhm_deg=3.5))[0]  # ZYH
    assert case["mosaic_fwhm_rad"] == pytest.approx(np.deg2rad(3.5))


# ---- geometry helper ---------------------------------------------------------
def test_psi_is_a_sane_angle_for_tilted_hopg():
    case = build_cases(_sweep("hopg", mosaic=True))[0]
    psi = mosaic_psi_rad(case, E_pk_eV=120.0)
    assert psi is not None
    assert 0.0 < psi < np.pi / 2  # tilted -> g neither along nor perpendicular to v


# ---- store_result quadrature -------------------------------------------------
def test_store_result_off_matches_eds_plus_aperture_only():
    case = build_cases(_sweep("hopg"))[0]  # mosaic OFF
    res = {}
    store_result(res, case, _synthetic_out())
    r = res[case["name"]][30.0]
    expected = np.sqrt(
        eds_fwhm_eV(r["E_pk"]) ** 2
        + aperture_fwhm_eV(
            r["E_pk"],
            beta_from_keV(30.0),
            case["theta_obs_rad"],
            case["dtheta_obs_rad"],
        )
        ** 2
    )
    assert r["fwhm"] == pytest.approx(expected)


def test_store_result_on_broadens_strictly():
    out = _synthetic_out()
    res_off, res_on = {}, {}
    case_off = build_cases(_sweep("hopg"))[0]
    case_on = build_cases(_sweep("hopg", mosaic=True))[0]
    store_result(res_off, case_off, out)
    store_result(res_on, case_on, out)
    fwhm_off = res_off[case_off["name"]][30.0]["fwhm"]
    fwhm_on = res_on[case_on["name"]][30.0]["fwhm"]
    assert fwhm_on > fwhm_off

    # and it is exactly the mosaic term added in quadrature
    E_pk = res_off[case_off["name"]][30.0]["E_pk"]
    psi = mosaic_psi_rad(case_on, E_pk)
    extra = min(mosaic_fwhm_eV(E_pk, psi, np.deg2rad(0.8)), E_pk)
    assert fwhm_on == pytest.approx(np.sqrt(fwhm_off**2 + extra**2))


# ---- the comparison plot -----------------------------------------------------
def test_plot_mosaic_comparison_runs_and_broadens():
    import matplotlib

    matplotlib.use("Agg")
    from cxr_mc.config import default_settings
    from cxr_mc.plots import plot_mosaic_comparison

    case = build_cases(_sweep("hopg"))[0]  # mosaic OFF; the plot re-derives grades
    res = {}
    store_result(res, case, _synthetic_out())
    r = res[case["name"]][30.0]

    grades = (None, 0.4, 0.8, 3.5)
    fig = plot_mosaic_comparison(r, default_settings(), grades_deg=grades)
    ax = fig.axes[0]
    assert len(ax.lines) == len(grades)  # one curve per grade
    # broader mosaic -> wider convolution -> lower peak (integral preserved)
    peaks = [float(np.nanmax(line.get_ydata())) for line in ax.lines]
    assert peaks == sorted(peaks, reverse=True)
    assert peaks[0] > peaks[-1]  # perfect strictly taller than ZYH 3.5 deg

    import matplotlib.pyplot as plt

    plt.close(fig)
