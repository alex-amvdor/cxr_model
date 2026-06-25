---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python (sci)
  language: python
  name: sci
---

```{code-cell} ipython3
# Core library modules live in src/; put it on the import path. Run from repo root.
import sys

sys.path.insert(0, "src")
```

```{code-cell} ipython3
"""
Reproduce Zhai et al., Nat. Commun. 16, 11218 (2025), Fig. 1c:
tunable X-ray (PXR+CBS) spectra from a 1 mm HOPG bulk crystal under
17.5 / 20 / 22.5 / 25 keV electrons, observed at theta_obs = 119 deg
(EDS take-off geometry) into 0.066 sr, plus the bulk vs 29 nm thin-film
enhancement at 25 keV.

Run:  python zhai_fig1c_check.py
Writes zhai_fig1c_analog.png and prints validation + peak tables.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from cxr_mc.crystallography import (
    HBARC_EV_ANG,
    ALPHA_FS,
    CRYSTALS,
    reciprocal_g_vector,
)
from feranchuk_spence import amplitudes_PXR_CBS_both, bremsstrahlung_background
from cxr_mc.montecarlo import (
    simulate_trajectories,
    mc_spectrum,
    mc_brem_spectrum,
    beta_from_keV,
    eds_fwhm_eV,
    aperture_fwhm_eV,
    convolve_detector,
)

# ---- experimental conditions (Zhai SI S3) -----------------------------------
THETA_OBS = np.deg2rad(119.0)
DTHETA_OBS = np.deg2rad(16.6)
DOMEGA_SR = 0.066
PER_NA = 6.2415e9  # electrons per second per nA
ENERGIES_KEV = [17.5, 20.0, 22.5, 25.0]
THICK_BULK = 1e7  # 1 mm in Angstrom
THICK_FILM = 290.0  # 29 nm
B_002 = 0.8  # graphite c-axis B-factor [Ang^2], approximate
HKL_LIST = ((0, 0, 2), (0, 0, -2))  # HOPG: only (00l) coherent
NE = 500
NE_BREM = 200  # background is smooth; fewer electrons fine
E_GRID = np.arange(500.0, 1250.0, 1.0)

info = CRYSTALS["hopg"]
n_atoms = len(info["basis"]) / info["V_cell"]

# ---- validation 1: single straight segment vs closed-form Eq. (12) ----------
beta0 = beta_from_keV(25.0)
L_seg = 290.0
fake = {
    "r_mid": np.array([[0.0, 0.0, 0.0]]),
    "v_hat": np.array([[0.0, 0.0, 1.0]]),
    "L_ang": np.array([L_seg]),
    "E_keV": np.array([25.0]),
    "Ne": 1,
    "thickness_ang": THICK_BULK,
}
spec1 = mc_spectrum(
    fake,
    E_GRID,
    crystal="hopg",
    hkl_list=((0, 0, 2),),
    theta_obs_rad=THETA_OBS,
    B_ang2=B_002,
)
num = np.trapezoid(spec1, E_GRID)

_, g002 = reciprocal_g_vector([0, 0, 2], info["lattice"])
denom = 1.0 - beta0 * np.cos(THETA_OBS)
E_res = HBARC_EV_ANG * beta0 * g002 / denom
amps, omega, _ = amplitudes_PXR_CBS_both(
    "hopg",
    [0, 0, 2],
    E_res,
    beta0,
    0.0,
    B_002,
    True,
    geometry="fixed",
    theta_obs=THETA_OBS,
)
A2 = sum(abs(a + b) ** 2 for a, b in amps.values())
ana = ALPHA_FS / (2.0 * np.pi) * omega * (L_seg / beta0) * A2 / denom
print(
    f"single-segment check: integral = {num:.4e}, closed form = {ana:.4e}, "
    f"ratio = {num / ana:.4f}  (E_res = {E_res:.1f} eV)"
)

# ---- transport + spectra ------------------------------------------------------
results = {}
t0 = time.perf_counter()
for E0 in ENERGIES_KEV:
    t1 = time.perf_counter()
    segs = simulate_trajectories(
        E0,
        NE,
        THICK_BULK,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=5.0,
        seed=int(E0 * 10),
    )
    eta = segs["n_backscattered"] / segs["Ne"]
    depth = segs["r_mid"][:, 2]
    spec = mc_spectrum(
        segs,
        E_GRID,
        crystal="hopg",
        hkl_list=HKL_LIST,
        theta_obs_rad=THETA_OBS,
        B_ang2=B_002,
    )
    # detector model at the nominal line energy
    b = beta_from_keV(E0)
    E_pk = E_GRID[np.argmax(spec)]
    fwhm = np.sqrt(
        eds_fwhm_eV(E_pk) ** 2 + aperture_fwhm_eV(E_pk, b, THETA_OBS, DTHETA_OBS) ** 2
    )
    spec_det = convolve_detector(E_GRID, spec, fwhm)

    # bremsstrahlung background from a dedicated low-cutoff transport run
    # (electrons below the CXR cutoff still radiate in this window)
    segs_b = simulate_trajectories(
        E0,
        NE_BREM,
        THICK_BULK,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=1.0,
        seed=int(E0 * 10) + 1,
    )
    brem = mc_brem_spectrum(
        segs_b, E_GRID, element="C", n_atoms_per_ang3=n_atoms, theta_obs_rad=THETA_OBS
    )
    brem_det = convolve_detector(E_GRID, brem, fwhm)

    results[E0] = (spec, spec_det, E_pk, fwhm, brem, brem_det)
    band = np.abs(E_GRID - E_pk) <= 5.0  # 10 eV band, their metric
    r10 = np.trapezoid(spec[band], E_GRID[band]) / np.trapezoid(
        brem[band], E_GRID[band]
    )
    r_eds = spec_det.max() / brem_det[np.argmax(spec_det)]
    print(
        f"E0 = {E0:5.1f} keV: {segs['L_ang'].size:6d} segments, "
        f"backscatter = {eta:.3f}, max depth = {depth.max() / 1e4:.2f} um, "
        f"line at {E_pk:.0f} eV, FWHM = {fwhm:.0f} eV | "
        f"tunx/brem = {r10:.1f} (10 eV band), {r_eds:.1f} (EDS) "
        f"({time.perf_counter() - t1:.1f} s)"
    )

# ---- bulk vs 29 nm thin film at 25 keV ---------------------------------------
segs_f = simulate_trajectories(
    25.0, NE, THICK_FILM, element="C", n_atoms_per_ang3=n_atoms, E_cut_keV=5.0, seed=7
)
spec_f = mc_spectrum(
    segs_f,
    E_GRID,
    crystal="hopg",
    hkl_list=HKL_LIST,
    theta_obs_rad=THETA_OBS,
    B_ang2=B_002,
)
spec_f_det = convolve_detector(E_GRID, spec_f, results[25.0][3])

# context: the Feranchuk Eq. (17) ultra-relativistic estimate (the notebook's
# old yardstick) vs the MC background, both per electron/sr/eV. Eq. (17) is
# evaluated with an effective path of ~the electron range (5 um) since it has
# no transport; expect it to undershoot by ~10x (missing the 1/T factor).
E_chk = results[25.0][2]
eq17 = bremsstrahlung_background(E_chk, 6, n_atoms, 5.0e4, 1.0)
brem_chk = results[25.0][4][np.argmin(np.abs(E_GRID - E_chk))]
print(
    f"\nBS at {E_chk:.0f} eV (25 keV beam): MC = {brem_chk:.3e}, "
    f"Eq.(17) w/ L=range = {eq17:.3e} per e/sr/eV "
    f"(ratio {brem_chk / eq17:.0f}x)"
)

scale = DOMEGA_SR * PER_NA  # -> photons / eV / s / nA
pk_bulk = results[25.0][1].max() * scale
pk_film = spec_f_det.max() * scale
print(
    f"\n25 keV peak (EDS-convolved): bulk = {pk_bulk:.3f}, "
    f"29 nm film = {pk_film:.3f} Phs/eV/s/nA  -> enhancement x{pk_bulk / pk_film:.1f}"
)
print(f"transmitted through film: {segs_f['n_transmitted']}/{segs_f['Ne']}")

# ---- plot ---------------------------------------------------------------------
fig, (ax_raw, ax_det) = plt.subplots(1, 2, figsize=(13, 5))
for i, E0 in enumerate(ENERGIES_KEV):
    spec, spec_det, E_pk, _, brem, brem_det = results[E0]
    ax_raw.plot(E_GRID, spec * scale, color=f"C{i}", label=f"{E0:g} keV")
    ax_det.plot(
        E_GRID, (spec_det + brem_det) * scale, color=f"C{i}", label=f"{E0:g} keV (bulk)"
    )
    ax_det.plot(E_GRID, brem_det * scale, color=f"C{i}", ls="--", lw=1.0)
ax_det.plot(E_GRID, spec_f_det * scale, "k-", lw=1.0, label="25 keV, 29 nm film")

ax_raw.set_title("Intrinsic spectra (1 mm HOPG), no background")
ax_det.set_title("EDS-convolved, peaks + bremsstrahlung (dashed) -- cf. Zhai Fig. 3b")
for ax in (ax_raw, ax_det):
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(
    r"PXR+CBS from HOPG, $\theta_{obs}$=119$\degree$, 0.066 sr "
    f"(Ne = {NE} electrons/energy)"
)
fig.tight_layout()
plt.show()
print(f"\nTotal time: {time.perf_counter() - t0:.1f} s")
```
