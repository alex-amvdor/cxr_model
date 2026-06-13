"""
Kinematic-approximation validity audit for the PXR/CBS machinery.

For each (crystal, reflection, beam energy) -- evaluated in the bulk
normal-incidence geometry (beam parallel to g, detector at 119 deg, as in the
Zhai setup) -- four independent validity parameters are printed:

1. DYN  = omega^2 |chi_g| / |k_g^2 - omega^2| : photon-side dynamical
   diffraction parameter. The kinematic (first-order) eigenmode drops terms
   of relative size DYN; for nonrelativistic electrons the detuning stays
   ~g^2 so this is tiny ("k is always far from the Ewald sphere").
   Also printed: the two-beam extinction scale L_ext = 2/(omega |chi_g|)
   vs the absorption length -- if L_abs << L_ext, absorption terminates the
   photon before dynamical exchange could build even AT a Bragg condition.

2. K_e  = (e U_g / m c^2) * omega * L : electron-side first-order condition,
   Feranchuk Eq. (6) (trajectory perturbation small over the coherent
   emission length L). Printed for two lengths:
     - L = lambda_el (elastic mean free path): the Monte Carlo case, where
       scattering interrupts coherence at every collision;
     - L = L_abs (saturated absorption-limited length): the idealized
       straight-trajectory model of the notebook.
   Multiple scattering PROTECTS the kinematic approximation in bulk.

3. XI   = electron two-beam extinction length xi_e = pi gamma beta hbar c /
   |e U_g| vs lambda_el: if xi_e <~ lambda_el at a coherent transverse
   reflection, electron CHANNELING / dynamical electron diffraction (ignored
   by the amorphous transport) can matter at zone-axis orientations.
   HOPG is exempt for in-plane g (fiber texture destroys transverse
   coherence); single crystals at exact zone axes are not.

4. REC  = E_photon / E_kinetic : quantum recoil (classical-trajectory)
   parameter; the paper keeps this < 5%.

Run: python kinematic_validity_check.py
"""

import numpy as np

# the core modules now live in ../src; put it on the path regardless of CWD
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_feranchuk_spence import (
    CRYSTALS, Z_TABLE, HBARC_EV_ANG, M_E_EV,
    chi_g, U_g, absorption_length_ang, reciprocal_g_vector,
)
from cxr_montecarlo import beta_from_keV, _sigma_browning_cm2

THETA_OBS = np.deg2rad(119.0)
CASES = [
    ("graphite", (0, 0, 2)),
    ("mose2",    (0, 0, 2)),
    ("diamond",  (1, 1, 1)),
    ("diamond",  (4, 0, 0)),
    ("silicon",  (1, 1, 1)),
    ("silicon",  (4, 4, 0)),
    ("lif",      (2, 0, 0)),
]
ENERGIES_KEV = (25.0, 120.0)

DYN_OK = 1e-2     # photon dynamical parameter
KE_OK = 0.1       # Feranchuk Eq. (6)
REC_OK = 0.05     # quantum recoil


def composition(info):
    """[(element, number density 1/Ang^3)] and Z_eff = sum n_i Z_i."""
    counts = {}
    for el, _ in info["basis"]:
        counts[el] = counts.get(el, 0) + 1
    comp = [(el, c / info["V_cell"]) for el, c in counts.items()]
    z_eff = sum(n * Z_TABLE[el] for el, n in comp)
    return comp, z_eff


def mean_free_path_ang(comp, E_keV):
    inv = sum(n * 1e24 * _sigma_browning_cm2(Z_TABLE[el], E_keV)
              for el, n in comp)                       # 1/cm
    return 1e8 / inv


def absorption_length_total_ang(comp, E_eV):
    mu = sum(1.0 / absorption_length_ang(el, E_eV, n) for el, n in comp)
    return 1.0 / mu


def flag(value, limit):
    return "ok  " if value < limit else "WARN"


print(f"{'crystal':9s} {'hkl':6s} {'Ee':>5s} | {'E_p':>6s} {'|chi_g|':>8s} "
      f"{'DYN':>8s}      {'L_ext':>7s} {'L_abs':>7s} | {'K_e(mfp)':>8s}      "
      f"{'K_e(Labs)':>9s}      {'xi_e':>6s} {'mfp':>6s} | {'REC':>5s}")
print(f"{'':9s} {'':6s} {'keV':>5s} | {'eV':>6s} {'':>8s} {'':>8s}      "
      f"{'um':>7s} {'um':>7s} | {'':>8s}      {'':>9s}      "
      f"{'nm':>6s} {'nm':>6s} | {'':>5s}")
print("-" * 132)

for crystal, hkl in CASES:
    info = CRYSTALS[crystal]
    comp, z_eff = composition(info)
    _, g = reciprocal_g_vector(hkl, info["lattice"])

    for Ee in ENERGIES_KEV:
        beta = beta_from_keV(Ee)
        gamma = 1.0 + Ee / 510.99895
        # bulk geometry: beam || g, detector at theta_obs
        E_p = HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(THETA_OBS))
        omega = E_p / HBARC_EV_ANG

        chi = abs(chi_g(crystal, hkl, E_p, 0.0, True))
        eU = abs(U_g(crystal, hkl, E_p, 0.0, True))

        # photon side
        detune = abs(g**2 + 2.0 * omega * g * np.cos(THETA_OBS))
        dyn = omega**2 * chi / detune
        L_ext = 2.0 / (omega * chi)                    # Ang
        L_abs = absorption_length_total_ang(comp, E_p)

        # electron side (Eq. 6)
        mfp = mean_free_path_ang(comp, Ee)
        K_mfp = eU / M_E_EV * omega * mfp
        K_sat = eU / M_E_EV * omega * L_abs

        # electron extinction (channeling scale)
        xi_e = np.pi * gamma * beta * HBARC_EV_ANG / eU

        rec = E_p / (Ee * 1e3)

        print(f"{crystal:9s} {''.join(map(str, hkl)):6s} {Ee:5.0f} | "
              f"{E_p:6.0f} {chi:8.1e} {dyn:8.1e} {flag(dyn, DYN_OK)} "
              f"{L_ext / 1e4:7.2f} {L_abs / 1e4:7.2f} | "
              f"{K_mfp:8.1e} {flag(K_mfp, KE_OK)} "
              f"{K_sat:9.1e} {flag(K_sat, KE_OK)} "
              f"{xi_e / 10:6.0f} {mfp / 10:6.0f} | "
              f"{rec:5.1%} {flag(rec, REC_OK)}")

print("""
Legend: DYN  photon dynamical-diffraction parameter (kinematic needs << 1)
        L_ext photon two-beam extinction scale vs L_abs (absorption first?)
        K_e  Feranchuk Eq. (6) over one mean free path (Monte Carlo) and over
             the saturated absorption length (idealized straight-line model)
        xi_e electron two-beam extinction length vs elastic mean free path:
             xi_e <~ few * mfp => channeling possible at zone axes (HOPG
             exempt for in-plane g; relevant for single crystals)
        REC  quantum recoil E_p/E_kin (classical trajectory needs < ~5%)
""")

# ---- vdW enhancement anatomy: the paper's merit-parameter ingredients --------
print("vdW enhancement anatomy at 25 keV (per crystal, strongest listed g):")
print(f"{'crystal':9s} {'hkl':6s} {'E_p':>6s} {'|chi|^2':>9s} {'Z_eff':>7s} "
      f"{'mfp':>7s} {'L_abs':>7s} {'Ep|chi|^2/(Zeff/Ep)':>20s}")
print(f"{'':9s} {'':6s} {'eV':>6s} {'':>9s} {'1/A^3':>7s} {'nm':>7s} {'um':>7s}")
print("-" * 78)
seen = set()
for crystal, hkl in CASES:
    if crystal in seen:
        continue
    seen.add(crystal)
    info = CRYSTALS[crystal]
    comp, z_eff = composition(info)
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    beta = beta_from_keV(25.0)
    E_p = HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(THETA_OBS))
    chi = abs(chi_g(crystal, hkl, E_p, 0.0, True))
    mfp = mean_free_path_ang(comp, 25.0)
    L_abs = absorption_length_total_ang(comp, E_p)
    merit = E_p * chi**2 / (z_eff / E_p)
    print(f"{crystal:9s} {''.join(map(str, hkl)):6s} {E_p:6.0f} {chi**2:9.1e} "
          f"{z_eff:7.3f} {mfp / 10:7.0f} {L_abs / 1e4:7.2f} {merit:20.2e}")
