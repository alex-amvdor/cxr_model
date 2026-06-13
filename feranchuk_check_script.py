"""
Anchor the model's absolute flux against the LiF (200) experiment quoted in
Feranchuk-Spence Eq. (22) / Table I (Ref. [11]) and report the correction
factor between model and paper.
"""

import numpy as np

from cxr_feranchuk_spence import (
    CRYSTALS, Z_TABLE, amplitudes_PXR_CBS, beta_from_Ee, chi_g, delta_g,
    flux_per_second, absorption_length_ang,
)

for el in ("Li", "F"):
    if el not in Z_TABLE:
        raise RuntimeError(f"Add '{el}' to atomic_form_factors (Z, Cromer-Mann, "
                           f"and {el}.csv Henke file) before anchoring.")

# --- paper's experimental conditions (Eq. 22, Ref. [11] / Table I) ----------
Ee_eV     = 63e3           # 63 keV electrons
current_A = 1e-3           # 1 mA
L_z_ang   = 100 * 10       # 100 nm film -> 1000 Angstrom (Table I)
dOmega_sr = 1e-3           # detector solid angle ~10^-3 sr (Eq. 22)
E_paper   = 3890.0         # eV, the (200) fundamental line they quote
N_paper   = 5.1e2          # photons/sec, their result
theta_obs = np.deg2rad(67.5)

beta = beta_from_Ee(Ee_eV)

# --- LiF geometry: v0 || [100], observation theta = 67.5 deg, theta_B = 0 ---
# Their (200) reflection. With v0 along [100] and theta_B (to plane normal) ~ 0,
# this is a grazing/forward-ish geometry, NOT the symmetric Omega=2theta_B case.
# Our omega_n() assumes the symmetric reflex, so for LiF we instead pin the
# line energy to their quoted 3.89 keV and feed that energy in directly,
# rather than trusting omega_n for this asymmetric geometry.
hkl_LiF = [2, 0, 0]
E_line  = E_paper          # use their quoted line energy directly

# --- absorption length at the line energy (LiF, F dominates absorption) -----
# number densities [1/Angstrom^3]: 4 formula units per cell
n_formula = 4.0 / CRYSTALS["lif"]["V_cell"]
# crude: use F (Z=9) as the dominant absorber near 3.9 keV; refine if needed
L_abs = absorption_length_ang("F", E_line, n_formula)
print(f"LiF L_abs at {E_line:.0f} eV: {L_abs:.1f} Angstrom "
      f"(film L_z = {L_z_ang:.0f} Angstrom)")

# --- compute flux with current prefactors -----------------------------------
N_model = flux_per_second(
    crystal="lif", hkl=hkl_LiF, photon_E_eV=E_line,
    theta_B_normal=theta_obs, beta=beta,
    L_z_ang=L_z_ang, L_abs_ang=L_abs, dOmega_sr=dOmega_sr, current_A=current_A,
    polarization="both", use_henke=True, geometry="lif",   # <-- the two that matter
)
print(f"{N_model:.3e} photons/s")

# --- the anchor -------------------------------------------------------------
correction = N_paper / N_model
print(f"\nmodel flux : {N_model:.3e} photons/s")
print(f"paper flux : {N_paper:.3e} photons/s (Eq. 22)")
print(f"correction factor (multiply model by): {correction:.3e}")
print(f"log10 discrepancy: {np.log10(correction):+.2f} decades")

# sanity: report the coupling and ratio used
chi = chi_g("lif", hkl_LiF, E_line, use_henke=True)
dg  = delta_g("lif", hkl_LiF, E_line, use_henke=True)
print(f"\n|chi_g(LiF 200)| = {abs(chi):.3e}")
print(f"delta_g (A_PXR/A_CBS) = {dg:.3f}")

A_PXR, A_CBS, omega, g = amplitudes_PXR_CBS(
    "lif", hkl_LiF, E_line, beta,
    theta_obs, polarization="pi", use_henke=True, geometry="lif")
print(f"omega [1/Ang] = {omega:.4f}")
print(f"g     [1/Ang] = {g:.4f}")
print(f"|A_PXR| = {abs(A_PXR):.3e}")
print(f"|A_CBS| = {abs(A_CBS):.3e}")
print(f"detuning kg^2 - omega^2 = {abs(omega**2 - g**2):.3e}")
print(f"ratio |A_PXR/A_CBS| = {abs(A_PXR/A_CBS):.3e}")
