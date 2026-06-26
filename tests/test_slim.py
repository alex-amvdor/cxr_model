"""Tests for the checkpoint slim-export (P2 #5): results.slim_results +
slim.slim_checkpoint. All synthetic + pure-numpy, so CPU-only and fast."""

import pickle

import numpy as np

from cxr_mc.results import slim_results
from cxr_mc.slim import slim_checkpoint


def _record(tilt_deg, E0):
    E = np.linspace(500.0, 1200.0, 200)
    Eb = np.linspace(0.0, 30000.0, 400)  # the wide-brem grid: the largest field
    return dict(
        E_grid=E,
        spec=np.exp(-((E - 900.0) ** 2) / 50.0),
        brem=np.full_like(E, 0.01),
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, 0.001),
        E_pk=900.0,
        fwhm=30.0,
        eta=2.0,
        scale=1.0,
        case={"name": f"t{tilt_deg}", "E0_keV": E0, "tilt_deg": tilt_deg, "crystal": "hopg"},
    )


def _results():
    return {
        f"t{tilt}": {25.0: _record(tilt, 25.0), 30.0: _record(tilt, 30.0)} for tilt in (0.0, 30.0)
    }


def test_drop_wide_brem_removes_largest_fields_only():
    res = _results()
    slim = slim_results(res, drop_wide_brem=True)
    for by_E in slim.values():
        for rec in by_E.values():
            assert "brem_wide" not in rec and "E_grid_brem" not in rec
            assert {"spec", "brem", "E_grid", "case"} <= set(rec)
    assert "brem_wide" in res["t0.0"][25.0]  # original untouched


def test_downcast_to_float32_preserves_values():
    res = _results()
    slim = slim_results(res, downcast=True)
    rec = slim["t0.0"][25.0]
    assert rec["spec"].dtype == np.float32 and rec["brem_wide"].dtype == np.float32
    assert res["t0.0"][25.0]["spec"].dtype == np.float64  # original untouched
    assert np.allclose(rec["spec"], res["t0.0"][25.0]["spec"], rtol=1e-5)


def test_constraint_filters_records():
    res = _results()
    assert set(slim_results(res, tilt_deg=0.0)) == {"t0.0"}
    by_energy = slim_results(res, E0_keV=25.0)
    for by_E in by_energy.values():
        assert set(by_E) == {25.0}


def test_fields_allowlist_keeps_only_requested_plus_case():
    res = _results()
    rec = slim_results(res, fields=["spec"])["t0.0"][25.0]
    assert set(rec) == {"spec", "case"}


def test_no_args_keeps_structure_but_copies_records():
    res = _results()
    slim = slim_results(res)
    assert set(slim) == set(res)
    assert all(set(slim[n]) == set(res[n]) for n in res)
    assert slim["t0.0"][25.0] is not res["t0.0"][25.0]  # fresh dict, safe to mutate


def test_slim_checkpoint_roundtrip_is_smaller(tmp_path):
    res = _results()
    src = tmp_path / "hopg.pkl"
    with open(src, "wb") as f:
        pickle.dump(res, f)
    out = tmp_path / "hopg.slim.pkl"
    slim_checkpoint(str(src), str(out), drop_wide_brem=True, downcast=True)
    assert out.exists() and out.stat().st_size < src.stat().st_size
    with open(out, "rb") as f:
        reloaded = pickle.load(f)
    assert set(reloaded) == set(res)
    assert "brem_wide" not in reloaded["t0.0"][25.0]
    assert reloaded["t0.0"][25.0]["spec"].dtype == np.float32


def test_slim_checkpoint_default_out_path(tmp_path):
    src = tmp_path / "hopg.pkl"
    with open(src, "wb") as f:
        pickle.dump(_results(), f)
    slim_checkpoint(str(src))
    assert (tmp_path / "hopg.slim.pkl").exists()
