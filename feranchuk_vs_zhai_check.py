"""
Head-to-head comparison of the two flux pipelines on an IDENTICAL setup,
in identical units (Phs/eV/s/nA spectra and integrated detector counts):

  A. "Feranchuk" idealized model: single straight trajectory at the full beam
     energy, absorption-limited effective length (photons_per_electron, now
     including the exact 1/(1 - beta n.v) delta-function Jacobian). The line
     is spread with the detector Gaussian for plotting.
  B. "Zhai" Monte Carlo model: scattered trajectories, finite-segment
     lineshapes, per-segment escape absorption (cxr_montecarlo).

Setup: HOPG graphite (002), 25 keV electrons, beam || c-axis, detector at
theta_obs = 119 deg into 0.066 sr (the Zhai experimental geometry). Compared
for a 29 nm film (where the idealized model's assumptions hold) and for bulk
(where they don't -- the Eq. 6 audit in kinematic_validity_check.py marks
the idealized bulk extrapolation as out of validity; shown for scale only).

NO calibration factors anywhere: both pipelines are absolute. (The notebook's
CAL = 1.385e-2 LiF anchor is NOT applied -- see printout.)

Also prints the estimated detector COUNT RATE (unity quantum efficiency):
line + bremsstrahlung integrated over the detector band and solid angle.

Run:  python feranchuk_vs_zhai_check.py
"""

import numpy as np

from cxr_feranchuk_spence import (
    CRYSTALS, HBARC_EV_ANG, photons_per_electron, absorption_length_ang,
    reciprocal_g_vector,
)
from cxr_montecarlo import (
    simulate_trajectories, mc_spectrum, mc_brem_spectrum, beta_from_keV,
    eds_fwhm_eV, aperture_fwhm_eV, convolve_detector,
)

# ---- shared setup (Zhai experimental geometry) -------------------------------
E0_KEV = 25.0
THETA_OBS = np.deg2rad(119.0)
DTHETA_OBS = np.deg2rad(16.6)
DOMEGA_SR = 0.066
PER_NA = 6.2415e9                       # electrons/s at 1 nA
BAND = (500.0, 1250.0)                  # detector energy window [eV]
E_GRID = np.arange(BAND[0], BAND[1], 1.0)
HKL = (0, 0, 2)
B_002 = 0.8
NE = 800

info = CRYSTALS["graphite"]
n_atoms = len(info["basis"]) / info["V_cell"]
beta = beta_from_keV(E0_KEV)
_, g = reciprocal_g_vector(HKL, info["lattice"])

# zero-scattering line energy, Eq. (10) with beam || g:
#   E = hbar c * beta g / (1 - beta cos(theta_obs))
E_line = HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(THETA_OBS))
L_abs = absorption_length_ang("C", E_line, n_atoms)
fwhm = np.sqrt(eds_fwhm_eV(E_line) ** 2
               + aperture_fwhm_eV(E_line, beta, THETA_OBS, DTHETA_OBS) ** 2)
print(f"setup: graphite (002), {E0_KEV:.0f} keV, theta_obs = 119 deg, "
      f"{DOMEGA_SR} sr | line at {E_line:.0f} eV, L_abs = {L_abs/1e4:.2f} um, "
      f"detector FWHM = {fwhm:.0f} eV\n")


def gaussian_line(E_grid, E0, fwhm_eV, area):
    """Unit-normalized Gaussian line of given integrated area [counts/eV]."""
    sig = fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return area * np.exp(-0.5 * ((E_grid - E0) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def report(name, line_counts, brem_counts, peak_height):
    print(f"{name:46s} line = {line_counts:9.1f}  brem = {brem_counts:9.1f}  "
          f"total = {line_counts + brem_counts:9.1f} cts/s/nA   "
          f"peak = {peak_height:6.3f} Phs/eV/s/nA")


for thickness, label in ((290.0, "29 nm film"), (1e7, "1 mm bulk")):
    # ---- A. idealized Feranchuk pipeline -------------------------------------
    # geometry="lif" is exactly the HOPG case: g || beam, observe at theta_obs.
    # For bulk, L_z >> L_abs saturates L_eff -> L_abs ("escape-limited"); the
    # Eq. (6) audit shows this extrapolation is OUTSIDE kinematic validity.
    N_line_A = photons_per_electron(
        "graphite", HKL, E_line, THETA_OBS, beta,
        L_z_ang=thickness, L_abs_ang=L_abs, dOmega_sr=DOMEGA_SR,
        polarization="both", B_ang2=B_002, use_henke=True, geometry="lif",
    ) * PER_NA                                          # counts/s/nA in the line
    spec_A = gaussian_line(E_GRID, E_line, fwhm, N_line_A)
    note = "" if thickness < 1e4 else "  [outside Eq.(6) validity]"
    report(f"A idealized Feranchuk, {label}{note}", N_line_A, 0.0, spec_A.max())

    # ---- B. Monte Carlo Zhai pipeline -----------------------------------------
    segs = simulate_trajectories(E0_KEV, NE, thickness, element="C",
                                 n_atoms_per_ang3=n_atoms, E_cut_keV=5.0, seed=5)
    spec_line = mc_spectrum(segs, E_GRID, hkl_list=(HKL, (0, 0, -2)),
                            theta_obs_rad=THETA_OBS, B_ang2=B_002)
    segs_b = simulate_trajectories(E0_KEV, max(NE // 2, 200), thickness,
                                   element="C", n_atoms_per_ang3=n_atoms,
                                   E_cut_keV=1.0, seed=6)
    spec_brem = mc_brem_spectrum(segs_b, E_GRID, element="C",
                                 n_atoms_per_ang3=n_atoms,
                                 theta_obs_rad=THETA_OBS)
    scale = DOMEGA_SR * PER_NA                          # /sr -> counts, /e -> /s/nA
    line_cts = np.trapezoid(spec_line, E_GRID) * scale
    brem_cts = np.trapezoid(spec_brem, E_GRID) * scale
    peak = convolve_detector(E_GRID, spec_line, fwhm).max() * scale
    report(f"B Monte Carlo (Zhai),  {label}", line_cts, brem_cts, peak)

    if thickness < 1e4:
        print(f"{'':46s} ratio A/B (line counts) = "
              f"{N_line_A / line_cts:.2f}  <- should be ~1 for the film\n")
    else:
        print(f"{'':46s} ratio A/B (line counts) = "
              f"{N_line_A / line_cts:.2f}  <- idealized bulk extrapolation "
              f"overshoots\n")

print("Detector count estimate (unity QE, band "
      f"{BAND[0]:.0f}-{BAND[1]:.0f} eV, {DOMEGA_SR} sr, 1 nA): see "
      "'total' column above.")
