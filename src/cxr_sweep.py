"""
cxr_sweep.py
============

Define a parameter sweep and expand it into the per-case dicts that
``cxr_montecarlo.run_case`` consumes.

The driver notebook sets ONE :class:`Sweep`. Every physical parameter accepts
**either a single value (fixed) or a sequence/array (swept)**; :func:`build_cases`
takes the Cartesian product over whatever is swept. So

    Sweep(tilt_deg=-30.0,              tilt_azim_deg=0.0)          # one case geometry
    Sweep(tilt_deg=np.linspace(-36,-1,14), tilt_azim_deg=[-9,0,9])  # 14*3 geometries

are both valid and need no other code changes.

Crystallography (composition, dominant reflections, zone axis, B-factor, default
energy grid) is looked up per material; the detector geometry defaults to the
2x2 Timepix3 quad. Only ``cxr_feranchuk_spence`` is imported here (no GPU), so
this module is cheap to import and test.
"""
from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Union

import numpy as np

from cxr_feranchuk_spence import CRYSTALS, dominant_reflections

# ---- Timepix3 quad geometry (fixed hardware) --------------------------------
TIMEPIX3_PIXEL_PITCH_M = 55e-6
TIMEPIX3_CHIP_WIDTH_M = 256 * TIMEPIX3_PIXEL_PITCH_M
TIMEPIX3_DISTANCE_M = 0.4
TIMEPIX3_DTHETA_OBS_DEG = float(
    np.degrees(2 * np.arctan((TIMEPIX3_CHIP_WIDTH_M / 2) / TIMEPIX3_DISTANCE_M))
)
TIMEPIX3_DOMEGA_SR = float(TIMEPIX3_CHIP_WIDTH_M**2 / TIMEPIX3_DISTANCE_M**2)

# pretty labels for the material column / config names
MATERIAL_LABELS = {
    "mose2": "MoSe2",
    "wse2": "WSe2",
    "graphite": "HOPG",
    "diamond": "diamond",
    "silicon": "silicon",
}

ScalarOrSeq = Union[float, Sequence[float], np.ndarray]


def fmt_thickness(t_ang):
    """Compact human thickness label from Angstroms: 316A / 31.6nm / 17um / 1mm."""
    if t_ang < 1e2:
        return f"{t_ang:g}A"
    if t_ang < 1e4:
        return f"{t_ang / 10:g}nm"
    if t_ang < 1e7:
        return f"{t_ang / 1e4:g}um"
    return f"{t_ang / 1e7:g}mm"


def n_of(crystal, element):
    """Number density [1/Ang^3] of one element in a crystal's unit cell."""
    info = CRYSTALS[crystal]
    count = sum(1 for el, _ in info["basis"] if el == element)
    return count / info["V_cell"]


def pm(*hkls):
    """A list of reflections together with their negatives."""
    out = []
    for h in hkls:
        out += [tuple(h), tuple(-x for x in h)]
    return out


def crystal_params(material, n_families=4):
    """Fixed crystallography for a material: composition, the dominant
    reflections, the beam zone axis [uvw], the (isotropic) B-factor, and a
    sensible default photon-energy grid. Override the grid via Sweep.e_grid_eV."""
    if material == "mose2":
        return dict(
            crystal="mose2",
            composition=[("Mo", n_of("mose2", "Mo")), ("Se", n_of("mose2", "Se"))],
            hkl_list=dominant_reflections("mose2", n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 2),
            B_ang2=0.6,
            E_grid=np.arange(350.0, 1750.0, 3.0),
        )
    if material == "wse2":
        # isostructural with MoSe2; W has no NIST Mott table so transport falls
        # back to analytic screened-Rutherford screening for W (see cxr_montecarlo).
        return dict(
            crystal="wse2",
            composition=[("W", n_of("wse2", "W")), ("Se", n_of("wse2", "Se"))],
            hkl_list=dominant_reflections("wse2", n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 2),
            B_ang2=0.6,
            E_grid=np.arange(350.0, 2500.0, 3.0),
        )
    if material == "diamond":
        return dict(
            crystal="diamond",
            composition=[("C", n_of("diamond", "C"))],
            hkl_list=dominant_reflections("diamond", n_families=n_families, B_ang2=0.21),
            beam_uvw=(4, 0, 0),
            B_ang2=0.21,
            E_grid=np.arange(100.0, 5000.0, 2.0),
        )
    if material == "silicon":
        return dict(
            crystal="silicon",
            composition=[("Si", n_of("silicon", "Si"))],
            hkl_list=dominant_reflections("silicon", n_families=n_families, B_ang2=0.46),
            beam_uvw=(4, 4, 0),
            B_ang2=0.46,
            E_grid=np.arange(100.0, 5000.0, 3.0),
        )
    if material == "graphite":
        # HOPG is fiber-textured: only the (00l) c-axis reflections are coherent.
        return dict(
            crystal="graphite",
            composition=[("C", n_of("graphite", "C"))],
            hkl_list=pm((0, 0, 2), (0, 0, 4)),
            beam_uvw=(0, 0, 1),
            B_ang2=0.8,
            E_grid=np.arange(100.0, 5000.0, 3.0),
        )
    raise ValueError(f"unknown material {material!r} (have {list(MATERIAL_LABELS)})")


@dataclass
class Sweep:
    """One parameter sweep.

    Each of ``thickness_ang``, ``energy_keV``, ``tilt_deg`` and
    ``tilt_azim_deg`` is either a single number (fixed) or a sequence/array
    (swept); build_cases() takes the product. The remaining fields are fixed
    setup that rarely changes per run.
    """

    material: str = "mose2"
    thickness_ang: ScalarOrSeq = 2e4
    energy_keV: ScalarOrSeq = (30.0, 45.0, 60.0)
    tilt_deg: ScalarOrSeq = -30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    # fixed setup (single values) ------------------------------------------
    theta_obs_deg: float = 90.0
    n_families: int = 4
    e_grid_eV: Optional[np.ndarray] = None  # None -> per-material default
    dtheta_obs_deg: Optional[float] = None  # None -> Timepix3 default
    domega_sr: Optional[float] = None  # None -> Timepix3 default
    beam_uvw: Optional[tuple] = None  # None -> per-material default


def _seq(x):
    """Normalize a scalar-or-sequence into a 1-D float array, order preserved."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def build_cases(sweep: Sweep, n_electrons=450, n_electrons_brem=100):
    """Expand a :class:`Sweep` into a list of run_case dicts (the Cartesian
    product over the swept thickness / tilt / azimuth, each crossed with every
    beam energy). Returns the ``cases`` list; preview it with
    :func:`geometry_table`."""
    cp = crystal_params(sweep.material, sweep.n_families)
    E_grid = cp["E_grid"] if sweep.e_grid_eV is None else np.asarray(sweep.e_grid_eV, float)
    dtheta = TIMEPIX3_DTHETA_OBS_DEG if sweep.dtheta_obs_deg is None else sweep.dtheta_obs_deg
    domega = TIMEPIX3_DOMEGA_SR if sweep.domega_sr is None else sweep.domega_sr
    beam_uvw = cp["beam_uvw"] if sweep.beam_uvw is None else sweep.beam_uvw
    label = MATERIAL_LABELS.get(sweep.material, sweep.material)

    E_start, E_step = float(E_grid[0]), float(E_grid[1] - E_grid[0])
    E_stop = float(E_grid[-1]) + E_step
    energies = _seq(sweep.energy_keV)

    cases = []
    for i_c, (thickness, tilt, azim) in enumerate(
        product(_seq(sweep.thickness_ang), _seq(sweep.tilt_deg), _seq(sweep.tilt_azim_deg))
    ):
        name = f"{label} {fmt_thickness(thickness)} pol={tilt:g} az={azim:g}"
        for i_e, E0 in enumerate(energies):
            cases.append(
                dict(
                    name=name,
                    crystal=cp["crystal"],
                    composition=cp["composition"],
                    hkl_list=cp["hkl_list"],
                    B_ang2=cp["B_ang2"],
                    E0_keV=float(E0),
                    thickness_ang=float(thickness),
                    E_grid=(E_start, E_stop, E_step),
                    theta_obs_rad=np.deg2rad(sweep.theta_obs_deg),
                    tilt_deg=float(tilt),
                    tilt_azim_deg=float(azim),
                    beam_uvw=beam_uvw,
                    brem_file=None,
                    Ne=n_electrons,
                    Ne_brem=n_electrons_brem,
                    seed=1000 * i_c + 10 * i_e + 1,
                    # used downstream (detector model / unit scaling); ignored by run_case:
                    dtheta_obs_rad=np.deg2rad(dtheta),
                    domega_sr=domega,
                )
            )
    return cases


def geometry_table(cases):
    """A one-row-per-config DataFrame summarizing the geometry of a case list,
    for a quick sanity check before running."""
    import pandas as pd

    rows, seen = [], set()
    for c in cases:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        same = [k for k in cases if k["name"] == c["name"]]
        rows.append(
            {
                "config": c["name"],
                "refl": len(c["hkl_list"]),
                "t [um]": c["thickness_ang"] / 1e4,
                "polar [deg]": round(c["tilt_deg"], 2),
                "azim [deg]": round(c["tilt_azim_deg"], 2),
                "energies [keV]": [k["E0_keV"] for k in same],
                "theta_obs [deg]": round(np.degrees(c["theta_obs_rad"]), 1),
                "dOmega [sr]": c["domega_sr"],
            }
        )
    return pd.DataFrame(rows)
