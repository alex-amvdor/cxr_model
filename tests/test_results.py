"""results.line_metrics: the coherent_brem_ratio metric (formula + edge case).

Uses a synthetic record (a Gaussian line on a flat brem) so it stays fast and
needs no Monte-Carlo run."""

import numpy as np
import pytest

from cxr_model.results import line_metrics
from cxr_model.config import default_settings


def _record(E, spec, brem):
    # the only fields line_metrics reads
    return {"E_grid": E, "spec": spec, "brem": brem, "scale": 1.0}


def test_coherent_brem_ratio_matches_integral_ratio():
    E = np.arange(50.0, 150.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)  # a clean line
    brem = np.full_like(E, 0.1)                       # flat background
    m = line_metrics(_record(E, spec, brem), default_settings())
    expected = float(np.trapezoid(spec, E)) / float(np.trapezoid(brem, E))
    assert m["coherent_brem_ratio"] == pytest.approx(expected, rel=1e-9)
    # ratio is scale/current independent -> equals coherent_flux/brem_flux too
    assert m["coherent_brem_ratio"] > 0


def test_coherent_brem_ratio_nan_without_brem():
    E = np.arange(50.0, 150.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)
    m = line_metrics(_record(E, spec, np.zeros_like(E)), default_settings())
    assert np.isnan(m["coherent_brem_ratio"])
