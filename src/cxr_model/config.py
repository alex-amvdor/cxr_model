"""
config.py
=============

Shared run configuration for the CXR pipeline, imported by BOTH notebooks so the
scan-runner (``scan.ipynb``) and the visualization driver
(``analysis.ipynb``) can never drift apart: they build the SAME
:class:`results.Settings` and the SAME per-material :class:`sweep.Sweep`,
so the viz notebook is guaranteed to be looking at the checkpoint the runner
wrote. Edit a material's grid (or the detector/analysis knobs) here ONCE and both
notebooks pick it up.

  * :func:`default_settings` -- beam current, electron counts, detector flags.
  * :func:`material_sweep`   -- the full parametric scan for a material (thickness,
    beam energies, polar/azimuthal tilt sweeps, the line/brem energy grids).
  * :func:`trajectory_sweep` -- a small, dedicated geometry sweep for the electron
    -penetration figures (a handful of polar tilts at normal azimuth, 2 energies).
  * :data:`COLLAPSE_AZIMUTH` -- keep only the best azimuth per (tilt, energy).
"""

import numpy as np

from .results import Settings
from .sweep import Sweep

# When the azimuth is swept, collapse it: for each (polar tilt, energy) keep only
# the azimuth with the highest spectral peak. False -> show every azimuth.
COLLAPSE_AZIMUTH = True


def default_settings():
    """The analysis / detector / unit knobs shared by the runner and the plots.
    The runner uses n_electrons*; every notebook that plots must use the SAME
    detector flags (apply_detector_qe / brem_source) so the displayed spectra
    match what was simulated."""
    return Settings(
        beam_current_na=5.0,
        n_electrons=300,  # transport electrons per line spectrum
        n_electrons_brem=150,  # transport electrons per background
        # OFF: the intrinsic spectra stay intrinsic (no legacy SDD polymer-window
        # QE). The Timepix3 / Eagle XO views apply their own QE downstream.
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",  # "mc" | "external" | "none"
    )


# ---- per-material parametric scan grids --------------------------------------
# Each entry is the geometry + energy grids for one material's full sweep. Keep
# the line grid fine + narrow (the expensive coherent lines top out at a few keV)
# and the brem grid coarse + WIDE (out to the beam energy) -- see sweep.
_MATERIAL_GRIDS = {
    # Thickness study: total flux + CXR/brem ratio vs thickness at a few key
    # tilts (negative = entrance-toward-detector = high flux). Single azimuth
    # (pitch plane) and single energy so plot_metric_vs(x="thickness_ang",
    # hue="tilt_deg") has nothing to silently collapse -- one clean curve per tilt.
    # For an energy comparison instead, add 40 to energy_keV and use hue="E0_keV".
    "hopg": dict(
        thickness_ang=np.logspace(np.log10(0.1e4), np.log10(30e4), 40, endpoint=True),  # 0.1-30 um
        energy_keV=[30],
        tilt_deg=np.linspace(-89.9, -5, 10, endpoint=True),
        tilt_azim_deg=0.0,
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 30.0),
    ),
    "diamond": dict(
        thickness_ang=10e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89.9, 89.9, 60, endpoint=True),
        tilt_azim_deg=np.linspace(-89.9, -0.1, 30, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 30.0),
    ),
    "silicon": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 40, endpoint=True),
        tilt_azim_deg=np.linspace(-89, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),  # (was a stray 1-tuple in the nb)
        E_grid_brem=np.arange(0.0, 60000.0, 30.0),
    ),
    "mose2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 40, endpoint=True),
        tilt_azim_deg=np.linspace(-89, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 30.0),
    ),
    "wse2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 30, endpoint=True),
        tilt_azim_deg=np.linspace(-85, -0.1, 15),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 25.0),
    ),
    "mote2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 40, endpoint=True),
        tilt_azim_deg=np.linspace(-89, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 25.0),
    ),
    "ptse2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 40, endpoint=True),
        tilt_azim_deg=np.linspace(-89, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 25.0),
    ),
    "hfse2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 40, endpoint=True),
        tilt_azim_deg=np.linspace(-89, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 30.0),
    ),
    "zrse2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 50, endpoint=True),
        tilt_azim_deg=np.linspace(-89, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 25.0),
    ),
    "ws2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 30, endpoint=True),
        tilt_azim_deg=np.linspace(-85, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 3500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 25.0),
    ),
    "mos2": dict(
        thickness_ang=1e4,
        energy_keV=[30, 45, 60],
        tilt_deg=np.linspace(-89, 89, 60, endpoint=True),
        tilt_azim_deg=np.linspace(-85, -0.1, 15, endpoint=True),
        E_grid_line=np.arange(50.0, 4500.0, 1.0),
        E_grid_brem=np.arange(0.0, 60000.0, 25.0),
    ),
}

MATERIALS = tuple(_MATERIAL_GRIDS)


def material_grid(material):
    """The raw per-material grid dict (thickness, energies, tilt sweeps, grids)."""
    if material not in _MATERIAL_GRIDS:
        raise ValueError(f"unknown material {material!r} (have {list(_MATERIAL_GRIDS)})")
    return _MATERIAL_GRIDS[material]


def material_sweep(material, *, theta_obs_deg=90.0, **overrides):
    """The full parametric :class:`sweep.Sweep` for ``material`` (the geometry
    the runner scans and the viz notebook reduces). ``overrides`` replace any grid
    field, e.g. ``material_sweep("ptse2", thickness_ang=2e4)``."""
    p = dict(material_grid(material))
    p.update(overrides)
    return Sweep(material=material, theta_obs_deg=theta_obs_deg, **p)


def trajectory_sweep(material, *, n_tilts=9, energies=(30, 60), tilt_span=80.0):
    """A small dedicated geometry sweep for the electron-penetration figures: a
    handful of polar tilts at normal azimuth, two beam energies (transport only,
    so the energy grids are irrelevant -- kept for build_cases). ``n_tilts`` panels
    span +-``tilt_span`` degrees."""
    p = material_grid(material)
    return Sweep(
        material=material,
        thickness_ang=p["thickness_ang"],
        energy_keV=list(energies),
        tilt_deg=np.linspace(-tilt_span, tilt_span, n_tilts, endpoint=True),
        tilt_azim_deg=0.0,
        theta_obs_deg=90.0,
        E_grid_line=p["E_grid_line"],
        E_grid_brem=p["E_grid_brem"],
    )
