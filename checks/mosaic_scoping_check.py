"""Mosaicity SCOPING: is the exact Monte-Carlo mosaic average worth building?

The shipped analytic mosaic model (montecarlo.mosaic_fwhm_eV) only broadens the
detector convolution by FWHM = E*|tan psi|*eta and is energy-shift only. The exact
per-orientation MC route (docs/crystal-mosaicity.md route 2) is ~5-7 engineer-days
and multiplies the GPU hot loop by K, so the design doc says: "Run a thin-vs-bulk
linewidth comparison before committing." This is that comparison.

For HOPG (the only crystal carrying a mosaic spread, and the broad-line case the
exact route targets) it measures, thin -> bulk and for a span of tilts:

  * the INTRINSIC mc_spectrum line FWHM -- the electron multiple-scattering
    Doppler width that is already in the simulation (no mosaic);
  * the ANALYTIC mosaic broadening for the three HOPG grades ZYA 0.4 / ZYB 0.8 /
    ZYH 3.5 deg;
  * their ratio.

Decision rule (from the design doc's "When it is worth it"):
  * mosaic << intrinsic  -> the Doppler skirt dominates; the analytic Gaussian is
    already a sub-dominant correction and the exact route buys little.
  * mosaic >~ intrinsic   -> the line is mosaic-broad; the energy-shift-only model
    breaks (amplitudes vary across the cone, lineshape goes asymmetric) and the
    exact route is the physically correct upgrade.
  * psi -> 90 deg         -> the analytic tan(psi) diverges (capped at E); the
    exact route has no such divergence.

Run CPU-forced (the viz laptop has no CUDA toolkit):
    uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/mosaic_scoping_check.py',run_name='__main__')"
or simply, on a CPU box:  uv run python checks/mosaic_scoping_check.py
"""

import os
import sys

import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_mc.montecarlo import (
    mc_spectrum,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
    simulate_trajectories,
    tilted_geometry,
)
from cxr_mc.sweep import crystal_params

# ---- setup (HOPG, the mosaic case; Timepix3 theta_obs = 90 deg) ---------------
E0_KEV = 30.0
THETA_OBS = np.deg2rad(90.0)
NE = 300
SEED = 7
GRADES_DEG = (("ZYA", 0.4), ("ZYB", 0.8), ("ZYH", 3.5))  # HOPG rocking-curve FWHM
THICKNESSES = (("29 nm", 290.0), ("1 um", 1.0e4), ("10 um", 1.0e5))
TILTS_DEG = (-20.0, -45.0, -75.0)  # moderate -> steep (psi -> 90 as tilt -> grazing)

cp = crystal_params("hopg")
COMP = cp["composition"]


def _line_fwhm_eV(E, y):
    """(peak energy, FWHM) of the tallest peak via half-max crossings with linear
    interpolation. FWHM is NaN if the line never falls to half-max inside the grid."""
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    pk = y[i]
    if not np.isfinite(pk) or pk <= 0.0:
        return float(E[i]), np.nan

    def _cross(e0, e1, y0, y1, h):  # linear interpolation to the half-max level
        return e0 + (h - y0) * (e1 - e0) / (y1 - y0)

    h = 0.5 * pk
    L = i
    while L > 0 and y[L] > h:
        L -= 1
    eL = float(E[0]) if y[L] > h else _cross(E[L], E[L + 1], y[L], y[L + 1], h)
    R = i
    while len(y) - 1 > R and y[R] > h:
        R += 1
    eR = float(E[-1]) if y[R] > h else _cross(E[R - 1], E[R], y[R - 1], y[R], h)
    return float(E[i]), eR - eL


def _intrinsic_line(tilt_deg, thickness_ang):
    """Transport once, then measure the intrinsic (no-mosaic) line: a coarse pass
    to locate the brightest line, a fine 0.5 eV pass around it for the FWHM."""
    beam, n_hat = tilted_geometry(THETA_OBS, np.deg2rad(tilt_deg))
    segs = simulate_trajectories(
        E0_KEV,
        NE,
        thickness_ang,
        composition=COMP,
        E_cut_keV=5.0,
        seed=SEED,
        beam_dir=beam,
    )

    def _spec(E_grid):
        return mc_spectrum(
            segs,
            E_grid,
            crystal="hopg",
            hkl_list=cp["hkl_list"],
            n_hat=n_hat,
            B_ang2=cp["B_ang2"],
            composition=COMP,
            beam_uvw=cp["beam_uvw"],
        )

    E_coarse = np.arange(60.0, 4500.0, 2.0)
    E_pk0, _ = _line_fwhm_eV(E_coarse, _spec(E_coarse))
    E_fine = np.arange(max(60.0, E_pk0 - 250.0), E_pk0 + 250.0, 0.5)
    E_pk, fwhm = _line_fwhm_eV(E_fine, _spec(E_fine))
    return n_hat, E_pk, fwhm, int(segs["L_ang"].size)


def _case(tilt_deg):
    """Minimal case dict for mosaic_psi_rad (it reads only these keys)."""
    return dict(
        crystal="hopg",
        theta_obs_rad=THETA_OBS,
        tilt_deg=tilt_deg,
        tilt_azim_deg=0.0,
        E0_keV=E0_KEV,
        hkl_list=cp["hkl_list"],
        beam_uvw=cp["beam_uvw"],
        azimuth_rad=0.0,
    )


print(
    tabulate(
        [
            ["crystal / reflections", "HOPG (00,+/-2),(00,+/-4), beam || c"],
            ["beam energy", f"{E0_KEV:.0f} keV"],
            ["detector", f"theta_obs = {np.degrees(THETA_OBS):.0f} deg"],
            ["transport electrons", f"{NE} (line), E_cut 5 keV"],
            ["mosaic grades", ", ".join(f"{n} {d:g} deg" for n, d in GRADES_DEG)],
        ],
        headers=["setup", "value"],
        tablefmt="github",
    )
)
print()

rows = []
verdicts = []  # (thickness, tilt, grade) flagged where the exact route matters
for t_lbl, t_ang in THICKNESSES:
    for tilt in TILTS_DEG:
        n_hat, E_pk, intrinsic, n_seg = _intrinsic_line(tilt, t_ang)
        psi = mosaic_psi_rad(_case(tilt), E_pk)
        psi_deg = np.nan if psi is None else np.degrees(psi)
        row = [
            t_lbl,
            f"{tilt:g}",
            f"{E_pk:.0f}",
            f"{intrinsic:.1f}",
            f"{psi_deg:.1f}",
            n_seg,
        ]
        for name, grade in GRADES_DEG:
            if psi is None:
                row.append("-")
                continue
            mos = mosaic_fwhm_eV(E_pk, psi, np.deg2rad(grade))
            ratio = mos / intrinsic if intrinsic > 0 else np.inf
            row.append(f"{mos:.0f} ({ratio:.1f}x)")
            if ratio >= 0.5:  # mosaic comparable to or wider than the Doppler width
                verdicts.append((t_lbl, tilt, name, ratio, psi_deg))
        rows.append(row)

print(
    tabulate(
        rows,
        headers=[
            "thick",
            "tilt",
            "E_pk\n[eV]",
            "intrinsic\nFWHM [eV]",
            "psi\n[deg]",
            "segs",
            "ZYA 0.4\n[eV] (xint)",
            "ZYB 0.8\n[eV] (xint)",
            "ZYH 3.5\n[eV] (xint)",
        ],
        tablefmt="github",
    )
)
print()
print(
    "intrinsic FWHM = electron multiple-scattering Doppler width already in the MC\n"
    "(no mosaic). 'NN (R.Rx)' = analytic mosaic FWHM and its ratio to that width.\n"
)

if verdicts:
    print("EXACT ROUTE ADDS VALUE where analytic mosaic >~ 0.5x the Doppler width:")
    for t_lbl, tilt, name, ratio, psi_deg in verdicts:
        flag = "  (psi near grazing -> analytic tan(psi) unreliable)" if psi_deg > 75 else ""
        print(f"  - {t_lbl:>6}, tilt {tilt:>5g} deg, {name}: {ratio:.1f}x intrinsic{flag}")
else:
    print("Analytic mosaic stays << the Doppler width everywhere tested: exact route LOW value.")
