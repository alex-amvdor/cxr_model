"""

Head-to-head comparison of the two flux pipelines on an IDENTICAL setup,

in identical units (Phs/eV/s/nA spectra and integrated detector counts):



  A. "Feranchuk" idealized model: single straight trajectory at the full beam

     energy, absorption-limited effective length (photons_per_electron, now

     including the exact 1/(1 - beta n.v) delta-function Jacobian). The line

     is spread with the detector Gaussian for plotting.

  B. "Zhai" Monte Carlo model: scattered trajectories, finite-segment

     lineshapes, per-segment escape absorption (montecarlo).



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

# the core modules now live in ../src; put it on the path regardless of CWD
import os
import sys

import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


from feranchuk_spence import photons_per_electron

from cxr_mc.crystallography import (
    CRYSTALS,
    HBARC_EV_ANG,
    absorption_length_ang,
    reciprocal_g_vector,
)
from cxr_mc.montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    convolve_detector,
    eds_fwhm_eV,
    mc_brem_spectrum,
    mc_spectrum,
    simulate_trajectories,
)

# ---- shared setup (Zhai experimental geometry) -------------------------------

E0_KEV = 25.0

THETA_OBS = np.deg2rad(119.0)

DTHETA_OBS = np.deg2rad(16.6)

DOMEGA_SR = 0.066

PER_NA = 6.2415e9  # electrons/s at 1 nA

BAND = (500.0, 1250.0)  # detector energy window [eV]

E_GRID = np.arange(BAND[0], BAND[1], 1.0)

HKL = (0, 0, 2)

B_002 = 0.8

NE = 800


info = CRYSTALS["hopg"]

n_atoms = len(info["basis"]) / info["V_cell"]

beta = beta_from_keV(E0_KEV)

_, g = reciprocal_g_vector(HKL, info["lattice"])


# zero-scattering line energy, Eq. (10) with beam || g:

#   E = hbar c * beta g / (1 - beta cos(theta_obs))

E_line = HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(THETA_OBS))

L_abs = absorption_length_ang("C", E_line, n_atoms)

fwhm = np.sqrt(
    eds_fwhm_eV(E_line) ** 2 + aperture_fwhm_eV(E_line, beta, THETA_OBS, DTHETA_OBS) ** 2
)

print(
    tabulate(
        [
            ["crystal / reflection", "graphite (002), beam || c-axis"],
            ["beam energy", f"{E0_KEV:.1f} keV"],
            [
                "detector",
                f"theta_obs = {np.degrees(THETA_OBS):.0f} deg, "
                f"{DOMEGA_SR} sr, band {BAND[0]:.0f}-{BAND[1]:.0f} eV",
            ],
            ["line energy (Eq. 10)", f"{E_line:.0f} eV"],
            ["absorption length", f"{L_abs / 1e4:.2f} um"],
            ["detector FWHM at line", f"{fwhm:.0f} eV"],
        ],
        headers=["setup", "value"],
        tablefmt="github",
    )
)

print()


def gaussian_line(E_grid, E0, fwhm_eV, area):
    """Unit-normalized Gaussian line of given integrated area [counts/eV]."""

    sig = fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    return area * np.exp(-0.5 * ((E_grid - E0) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


# accumulated by the loop below; one row per (pipeline, target)

rows = []  # model, target, line cts, brem cts, total cts, peak, note

ratios = []  # target, A/B line-count ratio, interpretation


for thickness, label in ((290.0, "29 nm film"), (1e7, "1 mm bulk")):
    # ---- A. idealized Feranchuk pipeline -------------------------------------

    # geometry="lif" is exactly the HOPG case: g || beam, observe at theta_obs.

    # For bulk, L_z >> L_abs saturates L_eff -> L_abs ("escape-limited"); the

    # Eq. (6) audit shows this extrapolation is OUTSIDE kinematic validity.

    N_line_A = (
        photons_per_electron(
            "hopg",
            HKL,
            E_line,
            THETA_OBS,
            beta,
            L_z_ang=thickness,
            L_abs_ang=L_abs,
            dOmega_sr=DOMEGA_SR,
            polarization="both",
            B_ang2=B_002,
            use_henke=True,
            geometry="lif",
        )
        * PER_NA
    )  # counts/s/nA in the line

    spec_A = gaussian_line(E_GRID, E_line, fwhm, N_line_A)

    note = "" if thickness < 1e4 else "outside Eq.(6) validity"

    rows.append(["A idealized Feranchuk", label, N_line_A, 0.0, N_line_A, spec_A.max(), note])

    # ---- B. Monte Carlo Zhai pipeline -----------------------------------------

    segs = simulate_trajectories(
        E0_KEV,
        NE,
        thickness,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=5.0,
        seed=5,
    )

    spec_line = mc_spectrum(
        segs,
        E_GRID,
        crystal="hopg",
        hkl_list=(HKL, (0, 0, -2)),
        theta_obs_rad=THETA_OBS,
        B_ang2=B_002,
    )

    segs_b = simulate_trajectories(
        E0_KEV,
        max(NE // 2, 200),
        thickness,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=1.0,
        seed=6,
    )

    spec_brem = mc_brem_spectrum(
        segs_b, E_GRID, element="C", n_atoms_per_ang3=n_atoms, theta_obs_rad=THETA_OBS
    )

    scale = DOMEGA_SR * PER_NA  # /sr -> counts, /e -> /s/nA

    line_cts = np.trapezoid(spec_line, E_GRID) * scale

    brem_cts = np.trapezoid(spec_brem, E_GRID) * scale

    peak = convolve_detector(E_GRID, spec_line, fwhm).max() * scale

    rows.append(
        [
            "B Monte Carlo (Zhai)",
            label,
            line_cts,
            brem_cts,
            line_cts + brem_cts,
            peak,
            "",
        ]
    )

    ratios.append(
        [
            label,
            N_line_A / line_cts,
            "should be ~1 (idealized model valid here)"
            if thickness < 1e4
            else "idealized bulk extrapolation overshoots",
        ]
    )


print(
    tabulate(
        rows,
        headers=[
            "model",
            "target",
            "line\n[cts/s/nA]",
            "brem\n[cts/s/nA]",
            "total\n[cts/s/nA]",
            "peak\n[Phs/eV/s/nA]",
            "note",
        ],
        tablefmt="github",
        floatfmt=(None, None, ".1f", ".1f", ".1f", ".3f"),
    )
)

print()

print(
    tabulate(
        ratios,
        headers=["target", "A/B line counts", "interpretation"],
        tablefmt="github",
        floatfmt=(None, ".2f"),
    )
)

print()

print(
    f"Count rates are unity-QE detector estimates over {BAND[0]:.0f}-"
    f"{BAND[1]:.0f} eV into {DOMEGA_SR} sr at 1 nA; no calibration factors."
)
