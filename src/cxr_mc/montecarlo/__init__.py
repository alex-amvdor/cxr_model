"""
montecarlo (package)

Monte Carlo CXR (PXR + CBS) from scattered electrons in thick crystals,
following Zhai et al., Nat. Commun. 16, 11218 (2025), SI Sections S1-S5:

  1. Electron transport (SI S2): single-scattering Monte Carlo with
     Joy-Luo continuous slowing-down. Elastic scattering (default "mott"):
     free paths from the Browning fit to the Mott total cross sections, and
     scattering angles from a screened-Rutherford form whose screening
     parameter alpha(E) is calibrated, per element, to reproduce the NIST
     SRD 64 relativistic Mott TRANSPORT cross sections
     (mott_transport_cross_sections/) -- so both the collision rate and the
     momentum-transfer rate match Mott data. Same architecture as CASINO.
     A purely analytic screened-Rutherford fallback is kept ("sr").
  2. Radiation (SI S1): each straight trajectory segment between elastic
     collisions radiates independently (incoherent across segments, coherent
     across reciprocal vectors within a segment) with the finite-interaction-
     time factor |Q|^2 = t_L^2 sinc^2(P t_L), P = [w - v.(k+g)]/2, replacing
     the absorption-limited delta-function limit of Feranchuk Eq. (9).
     Amplitudes are the same Eqs. (13)/(14) as checks/feranchuk_spence.py, with
     arbitrary segment velocity direction.
  3. Self-absorption (SI S5): Beer-Lambert along the observation direction
     from each segment midpoint (slab geometry).
  4. Detector (SI S3/S4): Gaussian convolution with
     FWHM^2 = FWHM_EDS^2 + FWHM_dtheta^2.

Conventions: lab frame with the incident beam along +z; the sample is a slab
occupying 0 <= z <= thickness (surface normal -z toward vacuum), surface
perpendicular to the beam (no tilt). The crystal construction frame is used
as-is, so for graphite the c-axis (g(00l)) lies along the beam -- the HOPG
fiber-texture geometry of the paper. Only (00l)-type reflections are coherent
in HOPG (random in-plane grain azimuths), so pass (00l) reflections only.

Nonrelativistic amplitudes (no gamma corrections): fine at <~30 keV; the
paper's gamma factors matter toward 100-300 keV.

Lengths in Angstrom, energies in eV (electron energies in keV where noted).

This module was split from a single montecarlo.py into a package; every public
and internal name remains importable as ``from cxr_mc.montecarlo import X`` for
backward compatibility. The submodules are:
  _backend  -- GPU/CPU array backend (xp, cp, REAL, _to_cpu, _GPU)
  materials -- composition normalization + X-ray self-absorption
  transport -- electron transport, scattering, stopping power
  geometry  -- tilted-sample / detector / orientation rotations
  spectrum  -- CXR line spectrum, solid-angle integral, bremsstrahlung
  detector  -- detector efficiency, line widths, convolution
  runner    -- per-case driver and the pipelined run_cases sweep
"""

from ._backend import _GPU, REAL, _to_cpu, cp, xp
from .detector import (
    aperture_fwhm_eV,
    convolve_detector,
    detector_efficiency,
    eds_fwhm_eV,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
)
from .geometry import (
    _mosaic_quadrature,
    _orientation_R,
    _small_tilt_R,
    detector_directions,
    tilted_geometry,
)
from .materials import (
    _layer_dz,
    _mu_total_inv_ang,
    _normalize_composition,
    _stack_tau,
)
from .runner import (
    _spectrum_case,
    _transport_case,
    _worker_init,
    run_case,
    run_cases,
)
from .spectrum import (
    _SEG_ARRAYS,
    R_E_CM2,
    _brem_dsigma_dk,
    _polarization_pair,
    _segments_in_layer,
    load_external_brem,
    mc_brem_spectrum,
    mc_spectrum,
    mc_spectrum_solid_angle,
)
from .transport import (
    _NO_MOTT,
    A0_SQ_CM2,
    MOTT_DIR,
    TRANSPORT_ELEMENTS,
    _alpha_from_first_moment,
    _alpha_sr_joy,
    _dEds_compound,
    _dEds_keV_per_ang,
    _load_mott_transport,
    _mott_alpha_table,
    _rotate_directions,
    _sample_cos_theta,
    _sigma_browning_cm2,
    beta_from_keV,
    simulate_trajectories,
)

__all__ = [
    # backend
    "xp",
    "cp",
    "REAL",
    "_to_cpu",
    "_GPU",
    # materials
    "_normalize_composition",
    "_mu_total_inv_ang",
    "_layer_dz",
    "_stack_tau",
    # transport
    "TRANSPORT_ELEMENTS",
    "MOTT_DIR",
    "A0_SQ_CM2",
    "beta_from_keV",
    "_sigma_browning_cm2",
    "_alpha_sr_joy",
    "_alpha_from_first_moment",
    "_load_mott_transport",
    "_mott_alpha_table",
    "_NO_MOTT",
    "_sample_cos_theta",
    "_dEds_keV_per_ang",
    "_dEds_compound",
    "_rotate_directions",
    "simulate_trajectories",
    # geometry
    "tilted_geometry",
    "detector_directions",
    "_orientation_R",
    "_small_tilt_R",
    "_mosaic_quadrature",
    # spectrum
    "_SEG_ARRAYS",
    "_segments_in_layer",
    "_polarization_pair",
    "mc_spectrum",
    "mc_spectrum_solid_angle",
    "R_E_CM2",
    "_brem_dsigma_dk",
    "mc_brem_spectrum",
    "load_external_brem",
    # detector
    "detector_efficiency",
    "eds_fwhm_eV",
    "aperture_fwhm_eV",
    "mosaic_fwhm_eV",
    "mosaic_psi_rad",
    "convolve_detector",
    # runner
    "run_case",
    "_transport_case",
    "_spectrum_case",
    "_worker_init",
    "run_cases",
]
