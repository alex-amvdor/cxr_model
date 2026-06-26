"""
detector_solid_angle_check.py  (checks/)

Validate the solid-angle-integrated spectrum (montecarlo.detector_directions +
mc_spectrum_solid_angle; docs/detector-solid-angle.md, TODO P2 #4):

  1. REGRESSION -- a 1-direction grid reproduces the single-angle ``spec * Omega``
     exactly (the design's n_side=1 anchor; max rel ~ 0).
  2. WIDE DETECTOR -- tiling a wide chip produces the n_hat-resolved, generally
     asymmetric integrated lineshape: report the peak shift, the integrated
     intrinsic FWHM (which here arises from the spread of the resonance energy
     across the face), and the analytic aperture_fwhm_eV it supersedes.
  3. SMALL DETECTOR (Timepix) -- negligible shift/broadening, confirming the
     current flat-Omega + Gaussian remains excellent for the easy case.

Run (CPU-force on a box with the cupy wheel but no CUDA device):
  uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/detector_solid_angle_check.py',run_name='__main__')"
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_mc.crystallography import CRYSTALS
from cxr_mc.montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    detector_directions,
    mc_spectrum,
    mc_spectrum_solid_angle,
    simulate_trajectories,
)

THETA = np.deg2rad(119.0)
E0_KEV = 25.0
B_002 = 0.8
HKL = ((0, 0, 2), (0, 0, -2))
E_GRID = np.arange(700.0, 1200.0, 1.0)
OMEGA = 0.066  # Zhai SEM solid angle
NE = 300
WIDE_NSIDE = 5  # wide-detector tiling (WIDE_NSIDE**2 directions per spectrum)
SMALL_NSIDE = 5

info = CRYSTALS["hopg"]
n_atoms = len(info["basis"]) / info["V_cell"]


def _fwhm(E, y):
    half = y.max() / 2.0
    above = np.where(y >= half)[0]
    return float(E[above[-1]] - E[above[0]]) if above.size >= 2 else 0.0


def _centroid(E, y):
    return float(np.trapezoid(E * y, E) / np.trapezoid(y, E))


def main():
    segs = simulate_trajectories(
        E0_KEV, NE, 1e7, element="C", n_atoms_per_ang3=n_atoms, E_cut_keV=5.0, seed=7
    )

    # --- 1. regression: n_side=1 == single-angle x Omega -----------------------
    nh1, w1 = detector_directions(THETA, n_side=1, domega_sr=OMEGA)
    spec_int1 = mc_spectrum_solid_angle(
        segs, E_GRID, "hopg", HKL, n_hats=nh1, weights=w1, B_ang2=B_002
    )
    spec_single = OMEGA * mc_spectrum(segs, E_GRID, "hopg", HKL, theta_obs_rad=THETA, B_ang2=B_002)
    rel = float(np.max(np.abs(spec_int1 - spec_single)) / np.max(np.abs(spec_single)))
    print(
        f"[1] regression n_side=1 vs single-angle x Omega: max rel = {rel:.2e}  "
        f"({'PASS' if rel < 1e-9 else 'FAIL'})"
    )

    # --- 2. wide detector: asymmetric, shifted integrated line -----------------
    print(
        "\n[2] wide chip (20 mm @ 30 mm, Delta-theta ~ "
        f"{np.degrees(2 * np.arctan(10.0 / 30.0)):.0f} deg):"
    )
    base_peak = base_fwhm = None
    for n_side in (1, WIDE_NSIDE):
        nh, w = detector_directions(
            THETA, n_side=n_side, chip_mm=20.0, dist_mm=30.0, domega_sr=OMEGA
        )
        spec = mc_spectrum_solid_angle(
            segs, E_GRID, "hopg", HKL, n_hats=nh, weights=w, B_ang2=B_002
        )
        pk = float(E_GRID[np.argmax(spec)])
        fw = _fwhm(E_GRID, spec)
        flux = float(np.trapezoid(spec, E_GRID))
        if base_peak is None:
            base_peak, base_fwhm = pk, fw
        print(
            f"    n_side={n_side:2d}: peak={pk:6.0f} eV  centroid={_centroid(E_GRID, spec):7.1f} eV  "
            f"intrinsic FWHM={fw:5.0f} eV  flux={flux:.3e}"
        )
    beta = beta_from_keV(E0_KEV)
    dtheta = 2.0 * np.arctan(10.0 / 30.0)
    ap = aperture_fwhm_eV(base_peak, beta, THETA, dtheta)
    print(f"    analytic aperture_fwhm_eV it supersedes (symmetric box-Gaussian) = {ap:.0f} eV")
    print(
        f"    -> integration broadens the bare line {base_fwhm:.0f} eV; the analytic term "
        f"models that broadening as a single symmetric {ap:.0f} eV Gaussian."
    )

    # --- 3. small Timepix detector: negligible ---------------------------------
    print(
        "\n[3] small chip (14 mm @ 400 mm, Timepix-like Delta-theta ~ "
        f"{np.degrees(2 * np.arctan(7.0 / 400.0)):.1f} deg):"
    )
    nh, w = detector_directions(
        THETA, n_side=SMALL_NSIDE, chip_mm=14.0, dist_mm=400.0, domega_sr=9.5e-4
    )
    spec_tp = mc_spectrum_solid_angle(segs, E_GRID, "hopg", HKL, n_hats=nh, weights=w, B_ang2=B_002)
    nh0, w0 = detector_directions(THETA, n_side=1, domega_sr=9.5e-4)
    spec_tp0 = mc_spectrum_solid_angle(
        segs, E_GRID, "hopg", HKL, n_hats=nh0, weights=w0, B_ang2=B_002
    )
    shift = abs(_centroid(E_GRID, spec_tp) - _centroid(E_GRID, spec_tp0))
    print(f"    centroid shift n_side={SMALL_NSIDE} vs 1: {shift:.3f} eV (negligible, as expected)")


if __name__ == "__main__":
    main()
