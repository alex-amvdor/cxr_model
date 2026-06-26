"""Tests for the many-knob sweep faceting (P3 #9): results.results_dataframe +
plots.facet_metric. Synthetic results, CPU-only (no mc_spectrum)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from cxr_mc.results import Settings, results_dataframe  # noqa: E402


def _rec(crystal, thick, tilt, E0):
    E = np.linspace(500.0, 1200.0, 200)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2) * (thick / 1e4)  # amp ~ thickness
    case = {
        "name": f"{crystal}_{thick:g}_{tilt:g}",
        "crystal": crystal,
        "thickness_ang": thick,
        "tilt_deg": tilt,
        "E0_keV": E0,
        "theta_obs_rad": np.deg2rad(119.0),
        "dtheta_obs_rad": np.deg2rad(16.6),
    }
    return dict(
        E_grid=E,
        spec=spec,
        brem=np.full_like(E, 0.01),
        E_grid_brem=None,
        brem_wide=None,
        E_pk=850.0,
        fwhm=30.0,
        eta=2.0,
        scale=1.0,
        case=case,
    )


def _results():
    res = {}
    for crystal in ("hopg", "silicon"):
        for thick in (1e3, 1e4):
            for tilt in (0.0, 30.0):
                rec = _rec(crystal, thick, tilt, 30.0)
                res[rec["case"]["name"]] = {30.0: rec}
    return res


def test_dataframe_has_knobs_and_metrics_one_row_per_record():
    df = results_dataframe(_results(), Settings())
    assert len(df) == 8
    needed = {
        "name",
        "crystal",
        "thickness_ang",
        "tilt_deg",
        "E0_keV",
        "E_pk",
        "eta",
        "peak_flux",
        "coherent_flux",
        "line_flux",
        "line_quality",
    }
    assert needed <= set(df.columns)
    # peak_flux tracks the amplitude we scaled with thickness
    thick_rows = df[df["thickness_ang"] == 1e4]
    thin_rows = df[df["thickness_ang"] == 1e3]
    assert thick_rows["peak_flux"].mean() > thin_rows["peak_flux"].mean()


def test_dataframe_metrics_false_skips_peak_finding():
    df = results_dataframe(_results(), metrics=False)
    assert "peak_flux" not in df.columns
    assert {"crystal", "thickness_ang", "E_pk", "eta"} <= set(df.columns)
    assert len(df) == 8


def test_facet_metric_grid_shape():
    from matplotlib.figure import Figure

    from cxr_mc.plots import facet_metric

    fig = facet_metric(
        _results(),
        Settings(),
        x="thickness_ang",
        y="peak_flux",
        row="crystal",
        col="tilt_deg",
        hue="E0_keV",
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2 * 2  # 2 crystals x 2 tilts


def test_facet_metric_no_hue_and_single_facet():
    from matplotlib.figure import Figure

    from cxr_mc.plots import facet_metric

    fig = facet_metric(_results(), Settings(), x="thickness_ang", y="line_flux", hue=None)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_facet_metric_bad_column_returns_none():
    from cxr_mc.plots import facet_metric

    assert facet_metric(_results(), Settings(), x="not_a_knob") is None
