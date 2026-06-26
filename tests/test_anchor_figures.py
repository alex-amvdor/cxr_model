"""Fast unit tests for checks/anchor_figures.py (the P1 #2 validation figures).

The heavy MC figure run lives in checks/; here we test only the cheap, pure
pieces -- theory anchors, the reference-CSV loader, series matching, the
single-segment lineshape-normalization anchor, and that the figure builders
return Figures on synthetic data -- so the suite stays CPU-only and fast.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless; no display in CI

# anchor_figures lives in checks/, not on the package path -- add it.
_CHECKS = Path(__file__).resolve().parent.parent / "checks"
if str(_CHECKS) not in sys.path:
    sys.path.insert(0, str(_CHECKS))

import anchor_figures as af  # noqa: E402


@pytest.fixture(scope="module")
def anchor():
    return af.ZhaiAnchor()


def test_line_energy_matches_dispersion(anchor):
    """line_energy_eV reproduces E = hbar c beta g / (1 - beta cos theta)."""
    from cxr_mc.crystallography import (
        CRYSTALS,
        HBARC_EV_ANG,
        reciprocal_g_vector,
    )
    from cxr_mc.montecarlo import beta_from_keV

    info = CRYSTALS[anchor.crystal]
    _, g = reciprocal_g_vector(anchor.hkl, info["lattice"])
    for E0 in anchor.energies_keV:
        beta = beta_from_keV(E0)
        expect = HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(anchor.theta_obs_rad))
        assert af.line_energy_eV(anchor, E0) == pytest.approx(expect, rel=1e-12)


def test_line_energies_monotonic_and_in_window(anchor):
    lines = af.theory_line_energies(anchor)
    assert set(lines) == set(anchor.energies_keV)
    vals = [lines[E0] for E0 in anchor.energies_keV]
    assert vals == sorted(vals)  # energy rises with beam energy
    assert all(anchor.e_min_eV < v < anchor.e_max_eV for v in vals)


def test_reference_curve_absent_returns_none(anchor, tmp_path):
    assert af.reference_curve(anchor, path=tmp_path / "nope.csv") is None
    # the default location ships no real zhai_fig1c.csv, only the .example.csv
    assert af.reference_curve(anchor) is None


def test_reference_curve_loads_and_groups(tmp_path):
    csv = tmp_path / "ref.csv"
    csv.write_text(
        "# a comment\nseries,energy_eV,intensity\n25keV,900,0.1\n25keV,940,1.0\n17.5keV,690,0.9\n"
    )
    ref = af.reference_curve(path=csv)
    assert set(ref) == {"25keV", "17.5keV"}
    e, inten = ref["25keV"]
    assert np.allclose(e, [900, 940])
    assert np.allclose(inten, [0.1, 1.0])


def test_example_csv_is_valid_schema():
    """The shipped template parses under the documented schema."""
    example = _CHECKS / "reference_data" / "zhai_fig1c.example.csv"
    ref = af.reference_curve(path=example)
    assert ref is not None and len(ref) == 4


def test_match_series_tolerant():
    ref = {"17.5keV": None, "25.0 keV": None, "20": None}
    assert af._match_series(ref, 17.5) == "17.5keV"
    assert af._match_series(ref, 25.0) == "25.0 keV"
    assert af._match_series(ref, 20.0) == "20"
    assert af._match_series(ref, 99.0) is None


# NB: single_segment_anchor() and model_spectra() call mc_spectrum, which runs on
# the GPU array module (xp=cupy on a CUDA box). Per repo convention the fast suite
# stays CPU-only, so the lineshape-normalization anchor (the single-segment MC vs
# Eq.(12) ratio) is exercised by the CPU-forced check run -- it is a column in
# validation_table() printed by anchor_figures.main() -- not here.


def _synthetic_model(anchor):
    """A cheap stand-in for model_spectra() output, for figure smoke tests."""
    E = anchor.E_grid
    lines = af.theory_line_energies(anchor)
    model = {}
    for E0 in anchor.energies_keV:
        peak = lines[E0]
        spec = np.exp(-0.5 * ((E - peak) / 8.0) ** 2)
        brem = 0.02 * np.ones_like(E)
        model[E0] = {
            "spec": spec,
            "spec_det": spec,
            "brem": brem,
            "brem_det": brem,
            "E_peak": float(peak),
            "fwhm": 30.0,
            "line_flux_per_e": float(np.trapezoid(spec, E) * anchor.domega_sr),
            "backscatter": 0.1,
        }
    E0 = anchor.energies_keV[-1]
    film = 0.3 * np.exp(-0.5 * ((E - lines[E0]) / 8.0) ** 2)
    model["film"] = {
        "E0_keV": E0,
        "spec": film,
        "spec_det": film,
        "line_flux_per_e": float(np.trapezoid(film, E) * anchor.domega_sr),
        "n_transmitted": 5,
    }
    return model


def test_figure_spectra_smoke(anchor):
    from matplotlib.figure import Figure

    fig = af.figure_spectra(anchor, _synthetic_model(anchor))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_figure_spectra_with_reference_overlay(anchor):
    model = _synthetic_model(anchor)
    ref = af.reference_curve(path=_CHECKS / "reference_data" / "zhai_fig1c.example.csv")
    fig = af.figure_spectra(anchor, model, reference=ref)
    # detector panel gains scatter collections from the overlay
    assert len(fig.axes[1].collections) >= 1


def test_figure_enhancement_smoke(anchor):
    from matplotlib.figure import Figure

    fig = af.figure_enhancement(anchor, _synthetic_model(anchor))
    assert isinstance(fig, Figure)
