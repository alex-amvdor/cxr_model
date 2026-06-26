"""Guard tests for the Altair spectrum renderer (cxr_mc.plots.altair_spectra).

The renderer shares its physics/data prep with the matplotlib path; these tests
exercise only the NEW rendering layer, on synthetic records (no GPU, no
checkpoint), keeping the data small enough to stay under Vega-Lite's default
5000-row cap so ``chart.to_dict()`` serializes without touching global config.
"""

from types import SimpleNamespace

import altair as alt
import numpy as np

from cxr_mc.plots.altair_spectra import spectrum_chart, spectrum_frame


def _settings():
    # apply_detector_qe=False -> no QE table; convolve_with_det=False -> no
    # convolution; brem_source="mc" -> brem reads straight from r["brem"]. This
    # keeps _line_brem/detected_background on the pure-numpy path.
    return SimpleNamespace(
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",
    )


def _record(E0_keV, tilt_deg, azim_deg, n=200):
    E = np.linspace(1000.0, 5000.0, n)  # eV
    peak = np.exp(-(((E - 2500.0) / 50.0) ** 2))  # a single sharp line
    brem = np.linspace(1.0, 0.2, n)  # a smooth falling continuum
    return {
        "E_grid": E,
        "spec": peak,
        "brem": brem,
        "scale": 2.0,
        "fwhm": 30.0,
        "case": {
            "name": "HOPG bulk",
            "E0_keV": E0_keV,
            "tilt_deg": tilt_deg,
            "tilt_azim_deg": azim_deg,
            "thickness_ang": 5.0e4,
        },
    }


def _store():
    # {name: {E0: record}} -- one polar tilt, two beam energies.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0),
            60.0: _record(60.0, -20.0, 0.0),
        }
    }


def test_spectrum_frame_shapes_and_components():
    df = spectrum_frame([_record(30.0, -20.0, 0.0), _record(60.0, -20.0, 0.0)], _settings())
    assert list(df.columns) == ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]
    # 2 energies x 2 components x 200 grid points
    assert len(df) == 2 * 2 * 200
    assert set(df["component"]) == {"total", "brem"}
    assert sorted(df["E0_keV"].unique()) == [30.0, 60.0]
    # total = line + brem >= brem everywhere (line is non-negative)
    assert (
        df[df.component == "total"]["intensity"].to_numpy()
        >= df[df.component == "brem"]["intensity"].to_numpy() - 1e-9
    ).all()


def test_spectrum_frame_excludes_brem_when_disabled():
    df = spectrum_frame([_record(30.0, -20.0, 0.0)], _settings(), include_brem=False)
    assert set(df["component"]) == {"total"}


def test_spectrum_chart_builds_valid_spec():
    chart = spectrum_chart(_store(), _settings())
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()  # raises if the spec is malformed / over the row cap
    enc = spec["layer"][0]["encoding"]
    assert enc["x"]["field"] == "energy_eV"
    assert enc["y"]["field"] == "intensity"
    assert enc["color"]["field"] == "E0_keV"
    # total + brem layers
    assert len(spec["layer"]) == 2


def test_spectrum_chart_single_layer_without_brem():
    chart = spectrum_chart(_store(), _settings(), include_brem=False)
    assert len(chart.to_dict()["layer"]) == 1


def test_spectrum_chart_none_on_empty():
    assert spectrum_chart({}, _settings()) is None
