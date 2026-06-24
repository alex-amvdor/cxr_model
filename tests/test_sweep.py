"""Sweep / build_cases: the Cartesian expansion and the required-material guard."""

import numpy as np
import pytest

from cxr_model.config import MATERIALS
from cxr_model.sweep import MATERIAL_LABELS, Sweep, build_cases, crystal_params

ALL = [
    "mose2",
    "wse2",
    "mote2",
    "mos2",
    "ws2",
    "ptse2",
    "hfse2",
    "zrse2",
    "diamond",
    "silicon",
    "hopg",
]


def test_sweep_requires_material():
    with pytest.raises(TypeError):
        Sweep(thickness_ang=1e4)  # material has no default


@pytest.mark.parametrize("material", ALL)
def test_crystal_params_complete(material):
    cp = crystal_params(material)

    for key in ("crystal", "composition", "hkl_list", "beam_uvw", "B_ang2"):
        assert key in cp

    assert cp["composition"] and all(n > 0 for _, n in cp["composition"])


def test_crystal_params_unknown_raises():
    with pytest.raises(ValueError):
        crystal_params("unobtanium")


def test_build_cases_is_cartesian_product():
    sw = Sweep(
        material="mose2",
        thickness_ang=1e4,
        energy_keV=[30, 45],
        tilt_deg=[-30, -10],
        tilt_azim_deg=[-45],
        E_grid_line=np.arange(50.0, 100.0, 5.0),
        E_grid_brem=np.arange(0.0, 1000.0, 100.0),
    )

    cases = build_cases(sw)

    assert len(cases) == 2 * 2 * 1  # energies x polar tilts x azimuths

    required = {
        "crystal",
        "composition",
        "hkl_list",
        "B_ang2",
        "E0_keV",
        "thickness_ang",
        "theta_obs_rad",
        "tilt_deg",
    }

    assert required <= set(cases[0])


def test_mote2_registered():
    assert MATERIAL_LABELS["mote2"] == "MoTe2"

    assert "mote2" in MATERIALS
