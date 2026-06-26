"""Tests for dev/elsepa_tables.py (P2 #6). Exercises only the pure, Docker-free
pieces -- the NIST-CSV writer/reader interop, the comparison, the defensive
tcstable extraction, and the gating error -- never a real ELSEPA run."""

import importlib.util

import elsepa_tables as et
import numpy as np
import pytest

from cxr_mc.montecarlo import _load_mott_transport


def test_read_mirror_matches_the_model_loader():
    """read_mott_transport_csv must parse a real NIST table identically to
    montecarlo._load_mott_transport (the drop-in contract)."""
    from cxr_mc.montecarlo import MOTT_DIR

    E_model, sig_model = _load_mott_transport("C")
    E_mine, sig_mine = et.read_mott_transport_csv(f"{MOTT_DIR}/DisplayCalcTCSTableForC.csv")
    assert np.array_equal(E_model, E_mine)
    assert np.allclose(sig_model, sig_mine, rtol=0, atol=0)


def test_write_then_load_with_the_model_loader(tmp_path, monkeypatch):
    """A table written by write_mott_transport_csv loads back through the actual
    montecarlo._load_mott_transport, values preserved to the written precision."""
    import cxr_mc.montecarlo as mc
    from cxr_mc.montecarlo import transport

    E = et.default_energy_grid_eV(n=40)
    sigma_cm2 = 1e-17 * (E / 50.0) ** -1.3  # a plausible falling sigma_tr(E)
    # _load_mott_transport reads MOTT_DIR from the transport submodule (montecarlo
    # is a package now), and is @cache'd -- patch the real global and clear the cache.
    monkeypatch.setattr(transport, "MOTT_DIR", str(tmp_path))
    mc._load_mott_transport.cache_clear()
    et.write_mott_transport_csv(str(tmp_path / "DisplayCalcTCSTableForXx.csv"), 6, E, sigma_cm2)
    E_back, sig_back = mc._load_mott_transport("Xx")
    assert np.allclose(E_back, E, rtol=1e-5)
    assert np.allclose(sig_back, sigma_cm2, rtol=1e-5)


def test_write_rejects_mismatched_shapes(tmp_path):
    with pytest.raises(ValueError):
        et.write_mott_transport_csv(str(tmp_path / "x.csv"), 6, np.arange(5.0), np.arange(4.0))


def test_compare_tables_zero_and_offset():
    E = et.default_energy_grid_eV(n=50)
    s = 1e-17 * (E / 50.0) ** -1.3
    assert et.compare_tables(E, s, E, s)[0] == pytest.approx(0.0, abs=1e-12)
    max_rel, _ = et.compare_tables(E, 1.10 * s, E, s)
    assert max_rel == pytest.approx(0.10, rel=1e-6)


def test_extract_sigma_tr1_from_structured_array():
    """Defensive extraction: structured array, a0**2 fallback (no Pint units)."""
    n = 6
    E = np.linspace(50.0, 3e5, n)
    tr1_a0 = np.linspace(10.0, 0.01, n)
    arr = np.zeros(n, dtype=[("E", float), ("total", float), ("tr1", float), ("tr2", float)])
    arr["E"] = E
    arr["tr1"] = tr1_a0
    E_out, sig_out = et._extract_sigma_tr1_cm2(arr)  # col=2 -> "tr1"
    assert np.allclose(E_out, E)
    assert np.allclose(sig_out, tr1_a0 * et.A0_SQ_CM2)


def test_extract_sigma_tr1_from_plain_2d_array():
    cols = np.array([[100.0, 5.0, 3.0, 1.0], [1000.0, 2.0, 1.0, 0.5]])
    E_out, sig_out = et._extract_sigma_tr1_cm2(cols, col=2)
    assert np.allclose(E_out, [100.0, 1000.0])
    assert np.allclose(sig_out, np.array([3.0, 1.0]) * et.A0_SQ_CM2)


@pytest.mark.skipif(
    importlib.util.find_spec("elsepa") is not None,
    reason="pyelsepa is importable here; the gating path can't be asserted deterministically",
)
def test_driver_is_gated_without_pyelsepa():
    """Without pyelsepa, the driver raises a clear, actionable RuntimeError rather
    than failing obscurely -- the model never depends on it."""
    with pytest.raises(RuntimeError, match="pyelsepa"):
        et.elsepa_transport_cross_section(6, [100.0, 1000.0])
