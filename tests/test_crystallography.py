"""Crystallography primitives: DB load, reciprocal geometry, structure factor."""

import numpy as np
import pytest

from cxr_mc.crystallography import (
    CRYSTALS,
    chi_g,
    debye_waller,
    dominant_reflections,
    reciprocal_g_vector,
    structure_factor,
)

EXPECTED = {
    "diamond",
    "silicon",
    "lif",
    "hopg",
    "mose2",
    "wse2",
    "mote2",
    "mos2",
    "ws2",
    "ptse2",
    "hfse2",
    "zrse2",
}


def test_catalog_loads():
    assert set(CRYSTALS) >= EXPECTED

    for name in EXPECTED:
        info = CRYSTALS[name]

        assert info["V_cell"] > 0 and len(info["basis"]) > 0


def test_silicon_111_dspacing():
    # cubic Si a=5.4309 -> d(111) = a/sqrt(3); |g| = 2 pi / d

    _, g = reciprocal_g_vector([1, 1, 1], CRYSTALS["silicon"]["lattice"])

    assert 2 * np.pi / g == pytest.approx(5.4309 / np.sqrt(3), abs=1e-3)


def test_structure_factor_finite_nonzero():
    S, g = structure_factor("silicon", (1, 1, 1), 8000.0)

    assert g > 0 and np.isfinite(abs(S)) and abs(S) > 0


def test_chi_g_finite():
    chi = chi_g("mose2", (0, 0, 2), 1000.0, B_ang2=0.6)

    assert np.isfinite(abs(chi)) and abs(chi) > 0


@pytest.mark.parametrize("g", [0.0, 1.0, 5.0])
def test_debye_waller_in_unit_interval(g):
    w = debye_waller(g, 0.5)

    assert 0.0 < w <= 1.0


def test_dominant_reflections_nonempty_triples():
    refl = dominant_reflections("mose2", n_families=4, B_ang2=0.6)

    assert len(refl) > 0 and all(len(h) == 3 for h in refl)


def test_mote2_structure_sane():
    # 2H-MoTe2 a=3.519, c=13.964 hexagonal: V = (sqrt(3)/2) a^2 c ~ 149.7 A^3,

    # 2 f.u. (2 Mo + 4 Te) per cell.

    info = CRYSTALS["mote2"]

    assert info["V_cell"] == pytest.approx(149.75, abs=1.0)

    assert len(info["basis"]) == 6

    assert sum(1 for el, _ in info["basis"] if el == "Te") == 4
