"""
montecarlo.py

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
"""

import os
import warnings
from functools import cache

import numpy as np

try:
    # cupy-cuda* imports cleanly even with no usable CUDA runtime (e.g. on the
    # viz laptop, which has the wheel but no GPU/driver), but on such a machine it
    # emits a UserWarning at import ("CUDA path could not be detected ..."). We
    # fall back to CPU below anyway, so silence just that import-time warning to
    # keep the notebook output clean. Importing is NOT proof the GPU path works --
    # a later cp.float32 / kernel call would blow up with AttributeError or a
    # CUDARuntimeError. Probe for a real device and fall back to CPU on ANY
    # failure, so the laptop runs the notebook on numpy.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
        import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device")
    _GPU = True
    xp = cp
    print("Using GPU")
except Exception:
    _GPU = False
    xp = np
    print("No GPU found, or cupy not installed!\nFalling back to CPU execution.")

# On-GPU spectrum precision. Consumer GPUs run fp64 at 1/32-1/64 of their fp32
# rate, so the big sinc/brem matmuls dominate -- single precision ~halves their
# cost and device-memory traffic. The one cancellation-sensitive spot, the
# (E_grid - E_res) subtraction, stays >3 orders below the line width, so fp32 is
# safe here; the complex couplings fall to complex64 automatically. Set
# CXR_FP64=1 to force double precision (reference/validation runs). The CPU
# fallback, where fp32 buys no speed, always stays double.
REAL = xp.float32 if (_GPU and os.environ.get("CXR_FP64") != "1") else xp.float64

from . import DATA_DIR
from .crystallography import (
    ALPHA_FS,
    CRYSTALS,
    HBARC_EV_ANG,
    M_E_EV,
    U_g,
    _direct_lattice_vectors,
    _rotation_between,
    absorption_length_ang,
    chi_g,
    reciprocal_g_vector,
)


def _to_cpu(a):
    """Move array to CPU (numpy). No-op if already numpy."""
    if _GPU and isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)


MOTT_DIR = str(DATA_DIR / "mott_transport_cross_sections")
A0_SQ_CM2 = 2.8002852e-17  # Bohr radius squared [cm^2] (NIST SRD 64 unit)

# ---- element data for transport ---------------------------------------------
# A [g/mol], J = mean ionization potential [keV] (Berger-Seltzer values)
TRANSPORT_ELEMENTS = {
    "C": {"Z": 6, "A": 12.011, "J_keV": 0.078},
    "Si": {"Z": 14, "A": 28.085, "J_keV": 0.173},
    "Ge": {"Z": 32, "A": 72.630, "J_keV": 0.350},
    "Se": {"Z": 34, "A": 78.971, "J_keV": 0.348},
    "Te": {"Z": 52, "A": 127.60, "J_keV": 0.485},
    "S": {"Z": 16, "A": 32.06, "J_keV": 0.180},
    "Mo": {"Z": 42, "A": 95.95, "J_keV": 0.424},
    "W": {"Z": 74, "A": 183.84, "J_keV": 0.727},
    "Zr": {"Z": 40, "A": 91.224, "J_keV": 0.393},
    "Hf": {"Z": 72, "A": 178.49, "J_keV": 0.705},
    "Pt": {"Z": 78, "A": 195.08, "J_keV": 0.790},
    # substrate elements (SiO2 / Al2O3); J = ICRU 37 mean excitation energy
    "O": {"Z": 8, "A": 15.999, "J_keV": 0.095},
    "Al": {"Z": 13, "A": 26.982, "J_keV": 0.166},
}


def beta_from_keV(E_keV):
    g = 1.0 + E_keV / 510.99895
    return np.sqrt(1.0 - 1.0 / g**2)


# ---- elastic scattering models ------------------------------------------------
def _sigma_browning_cm2(Z, E_keV):
    """
    Browning et al., J. Appl. Phys. 76, 2016 (1994): empirical fit to the
    tabulated Mott TOTAL elastic cross sections [cm^2], valid 0.1-30 keV,
    Z <= 92.
    """
    return (
        3.0e-18
        * Z**1.7
        / (E_keV + 0.005 * Z**1.7 * np.sqrt(E_keV) + 0.0007 * Z**2 / np.sqrt(E_keV))
    )


def _alpha_sr_joy(Z, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    return 3.4e-3 * Z**0.67 / E_keV


def _alpha_from_first_moment(target):
    """
    Invert <1-cos(theta)> = 2 a [(1+a) ln(1+1/a) - 1] for the screened-
    Rutherford screening parameter a (monotonic; vectorized bisection in
    log10 a). target must lie in (0, 1); values outside are clipped.
    """
    t = np.clip(np.asarray(target, dtype=float), 1e-12, 0.999)
    lo = np.full(t.shape, -12.0)
    hi = np.full(t.shape, 4.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = 10.0**mid
        val = 2.0 * a * ((1.0 + a) * np.log1p(1.0 / a) - 1.0)
        small = val < t
        lo = np.where(small, mid, lo)
        hi = np.where(small, hi, mid)
    return 10.0 ** (0.5 * (lo + hi))


@cache
def _load_mott_transport(element):
    """
    NIST SRD 64 relativistic Mott TRANSPORT cross sections
    sigma_tr = integral (1-cos theta) dsigma, from
    mott_transport_cross_sections/DisplayCalcTCSTableFor<El>.csv
    (50 eV - 300 keV, 401 points). Returns (E_eV, sigma_tr_cm2).
    """
    path = os.path.join(MOTT_DIR, f"DisplayCalcTCSTableFor{element}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No NIST Mott transport table for '{element}' ({path}). "
            f"Download from https://srdata.nist.gov/srd64/ or use "
            f"elastic_model='sr'."
        )
    E, sig = [], []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                E.append(float(parts[1]))
                sig.append(float(parts[2]))
    return np.array(E), np.array(sig) * A0_SQ_CM2


@cache
def _mott_alpha_table(element, Z):
    """
    Screening parameter alpha(E) calibrated so the screened-Rutherford
    angular distribution reproduces the NIST Mott transport cross section:
        sigma_tr / sigma_el = <1-cos theta>(alpha)
    with sigma_el from the Browning fit to the Mott totals. Returns
    (log10_E_eV_grid, log10_alpha_grid) for interpolation.
    """
    E_eV, sig_tr = _load_mott_transport(element)
    sig_el = _sigma_browning_cm2(Z, E_eV / 1e3)
    alpha = _alpha_from_first_moment(sig_tr / sig_el)
    return np.log10(E_eV), np.log10(alpha)


_NO_MOTT = set()  # elements with no NIST Mott table -> screened-Rutherford


def _sample_cos_theta(Z, E_keV, rng, elastic_model, element):
    """Polar scattering angle from the screened-Rutherford inversion, with the
    screening parameter from the chosen model. If elastic_model="mott" but no
    NIST Mott transport table exists for `element` (e.g. W), fall back to the
    analytic screened-Rutherford screening for that element. The miss is cached
    in _NO_MOTT so we don't re-stat the filesystem every transport step
    (lru_cache doesn't cache the FileNotFoundError); warned once."""
    if elastic_model == "mott" and element not in _NO_MOTT:
        try:
            logE, logA = _mott_alpha_table(element, Z)
            alpha = 10.0 ** np.interp(np.log10(E_keV * 1e3), logE, logA)
        except FileNotFoundError:
            print(
                f"transport: no Mott transport table for {element!r}; using "
                f"the analytic screened-Rutherford screening for it instead."
            )
            _NO_MOTT.add(element)
            alpha = _alpha_sr_joy(Z, E_keV)
    else:
        alpha = _alpha_sr_joy(Z, E_keV)
    R = rng.random(E_keV.shape)
    return 1.0 - 2.0 * alpha * R / (1.0 + alpha - R)


def _dEds_keV_per_ang(Z, A, J_keV, rho_g_cm3, E_keV):
    """Joy-Luo modified Bethe stopping power [keV/Angstrom] (negative)."""
    k = 0.731 + 0.0688 * np.log10(Z)
    # 78500 keV/cm -> 7.85e-4 keV/Angstrom prefactor
    return -7.85e-4 * rho_g_cm3 * Z / (A * E_keV) * np.log(1.166 * (E_keV + k * J_keV) / J_keV)


def _normalize_composition(element, n_atoms_per_ang3, composition):
    """
    Accept either the single-element API (element=, n_atoms_per_ang3=) or a
    compound composition=[(element, number_density_1_per_Ang3), ...];
    return the latter form.
    """
    if composition is not None:
        return [(el, float(n)) for el, n in composition]
    if element is None or n_atoms_per_ang3 is None:
        raise ValueError(
            "specify the material: pass composition=[(element, n_per_Ang3), ...] "
            "(or both element= and n_atoms_per_ang3=). Refusing to fall back to a "
            "default element so a material can't be silently mis-loaded."
        )
    return [(element, float(n_atoms_per_ang3))]


def _dEds_compound(comp, E_keV):
    """
    Joy-Luo stopping power [keV/Angstrom] for a compound, additive over
    elements: dE/ds = -7.85e-4 / E * sum_i (n_i/N_A') Z_i ln(1.166(E+k J)/J)
    with n_i in atoms/Ang^3 and N_A' = 0.602214 (Avogadro in mol/Ang^3*g
    bookkeeping units; equals the single-element rho*Z/A form).
    """
    total = 0.0
    for el, n_i in comp:
        p = TRANSPORT_ELEMENTS[el]
        k = 0.731 + 0.0688 * np.log10(p["Z"])
        total = total + (n_i / 0.602214076) * p["Z"] * np.log(
            1.166 * (E_keV + k * p["J_keV"]) / p["J_keV"]
        )
    return -7.85e-4 / E_keV * total


def _mu_total_inv_ang(comp, E_eV):
    """Total linear attenuation 1/L_abs [1/Angstrom] summed over elements.

    absorption_length_ang (from crystallography) is CPU-only, so the sum
    is always computed on the CPU. The result is returned on the SAME device as
    E_eV: a GPU array if the caller passed one (mc_spectrum, mixing it with
    on-device factors), a numpy array otherwise (detector_efficiency, whose
    output is multiplied into the host-side spectra in the notebook). Keying off
    the input device -- not the global _GPU flag -- keeps the CPU post-processing
    path numpy even when a GPU is present."""
    E_cpu = _to_cpu(E_eV)
    mu = 0.0
    for el, n_i in comp:
        mu = mu + 1.0 / absorption_length_ang(el, E_cpu, n_i)
    if _GPU and isinstance(E_eV, cp.ndarray):
        return cp.asarray(mu, dtype=REAL)
    return mu


# ---- layered (film-on-substrate) self-absorption ----------------------------
def _layer_dz(z_mid, n_z, z_top, z_bot):
    """z-extent of the layer [z_top, z_bot] that a photon leaving depth z_mid
    along n_hat (z-component n_z) crosses on its way out: toward z=0 when n_z<0
    (the entrance face) or the back face when n_z>0. numpy ufuncs are used so the
    same code serves numpy or cupy z_mid. Returns an array shaped like z_mid."""
    if n_z < 0:  # escape ray spans depths [0, z_mid]
        return np.maximum(np.minimum(z_mid, z_bot) - z_top, 0.0)
    return np.maximum(z_bot - np.maximum(z_mid, z_top), 0.0)  # spans [z_mid, z_total]


def _stack_tau(layers, z_mid, n_z, E):
    """Beer-Lambert optical depth for a photon leaving each segment midpoint
    (depth z_mid) along n_hat through a LAYERED absorber stack:
        tau = (1/|n_z|) * sum_i mu_i(E) * dz_i
    layers = [(z_top, z_bot, composition), ...] top (entrance) first, contiguous,
    the deepest z_bot being the total stack thickness. z_mid and E are per-segment
    arrays (E the resonance energy); the result matches their device. A single
    layer over [0, total_thickness] reproduces the single-slab escape exactly,
    so passing layers=None elsewhere stays bit-for-bit identical."""
    inv = 1.0 / max(abs(float(n_z)), 1e-12)
    tau = 0.0
    for z_top, z_bot, comp in layers:
        dz = _layer_dz(z_mid, n_z, float(z_top), float(z_bot))
        tau = tau + _mu_total_inv_ang(comp, E) * dz * inv
    return tau


def _rotate_directions(d, cos_t, phi):
    """Rotate unit vectors d (N,3) by polar angle theta, azimuth phi."""
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    ref = np.zeros_like(d)
    use_x = np.abs(d[:, 0]) < 0.9
    ref[use_x, 0] = 1.0
    ref[~use_x, 1] = 1.0
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    w = np.cross(d, u)
    out = (
        cos_t[:, None] * d + (sin_t * np.cos(phi))[:, None] * u + (sin_t * np.sin(phi))[:, None] * w
    )
    return out / np.linalg.norm(out, axis=1)[:, None]


def tilted_geometry(theta_obs_rad, tilt_polar_rad, tilt_azim_rad=0.0):
    """
    Sample-frame beam and detector directions for a TILTED sample.

    Transport and radiation work in the sample frame (slab normal and crystal
    construction frame along +z, e.g. the HOPG c-axis). In the lab the beam
    is fixed along +z_lab and the detector sits at polar angle theta_obs_rad,
    azimuth 0. Tilting the sample so its normal points along
    (sin tp cos ta, sin tp sin ta, cos tp) in the lab (tp = tilt_polar_rad,
    ta = tilt_azim_rad, cf. Zhai SI Fig. 1) is equivalent to rotating the
    beam and detector into the sample frame.

    Returns (beam_dir, n_hat) to pass to simulate_trajectories(beam_dir=...)
    and mc_spectrum(n_hat=...). For ta = 0 the detector's sample-frame polar
    angle is simply theta_obs - tilt_polar; the lab-frame quantity
    1 - v0.n_hat (hence the zero-scattering line energy denominator) is
    tilt-invariant, while v0.g picks up cos(tilt).
    """
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)
    beam_dir = R.T @ np.array([0.0, 0.0, 1.0])
    n_hat = R.T @ np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    return beam_dir, n_hat


def detector_directions(
    theta_obs_rad,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
    *,
    n_side=1,
    chip_mm=14.0,
    dist_mm=30.0,
    domega_sr,
):
    """
    Grid of detector directions {n_hat_i} (SAMPLE frame) tiling a flat square
    detector chip of side ``chip_mm`` at distance ``dist_mm`` facing the source,
    with their solid-angle weights -- the geometry input to the solid-angle-
    INTEGRATED spectrum (mc_spectrum_solid_angle). This replaces the single-n_hat
    + flat-Omega + analytic aperture_fwhm_eV approximation
    (docs/detector-solid-angle.md) with a first-principles tiling of the face.

    The central cell sits at polar angle ``theta_obs_rad`` (azimuth 0) in the lab
    and is mapped into the sample frame through the SAME tilt rotation as
    tilted_geometry(), so n_side=1 returns exactly that single direction. The chip
    in-plane axes are chosen so one grid axis spreads in the scattering plane (the
    polar / Delta-theta direction) and the other out of plane (azimuth). The
    per-cell weight is the inverse-square + obliquity solid angle
    ``dOmega_i = dA_i cos(psi_i) / r_i^2`` (psi_i to the chip normal = central
    line of sight), then the whole set is rescaled so ``sum_i dOmega_i ==
    domega_sr``: the detector's known total solid angle is conserved and n_side=1
    reproduces today's ``spec * Omega`` exactly.

    Returns (n_hats, weights): n_hats is (N, 3) unit directions in the sample
    frame (N = n_side**2); weights is (N,) and sums to ``domega_sr``.
    """
    if n_side < 1:
        raise ValueError("n_side must be >= 1")
    # tilt rotation (same convention as tilted_geometry): sample normal -> lab
    st, ct = np.sin(tilt_polar_rad), np.cos(tilt_polar_rad)
    normal_lab = np.array([st * np.cos(tilt_azim_rad), st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)

    # central lab line of sight c, and chip in-plane axes (chip face _|_ c):
    #   w lies in the scattering (x-z) plane -> polar (Delta-theta) spread
    #   u is out of plane (+y)               -> azimuthal spread
    c = np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    w = np.array([np.cos(theta_obs_rad), 0.0, -np.sin(theta_obs_rad)])
    u = np.array([0.0, 1.0, 0.0])

    step = chip_mm / n_side
    offs = (np.arange(n_side) - (n_side - 1) / 2.0) * step  # cell centres
    da = step**2  # cell area [mm^2]

    n_hats = np.empty((n_side * n_side, 3))
    weights = np.empty(n_side * n_side)
    i = 0
    for a in offs:  # out-of-plane (azimuth)
        for b in offs:  # in-plane (polar)
            P = dist_mm * c + a * u + b * w  # source -> cell vector [mm]
            r = float(np.linalg.norm(P))
            n_lab = P / r
            cos_psi = float(n_lab @ c)  # obliquity to the chip normal (= c)
            n_hats[i] = R.T @ n_lab
            weights[i] = da * cos_psi / r**2
            i += 1
    weights *= domega_sr / weights.sum()  # conserve the detector's total Omega
    return n_hats, weights


def _orientation_R(lattice, beam_uvw, azimuth_rad):
    """Rotation applied to EVERY reciprocal vector: the minimal rotation taking the
    crystal direct-lattice direction `beam_uvw` onto +z (the slab normal), then a
    roll of `azimuth_rad` about +z. Returns None for the construction-frame default
    (beam_uvw is None and azimuth_rad == 0). Shared by mc_spectrum and mosaic_psi_rad
    so the orientation convention lives in one place."""
    R = None
    if beam_uvw is not None:
        u, v, w = np.asarray(beam_uvw, dtype=float)
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        axis = u * a1 + v * a2 + w * a3
        R = _rotation_between(axis / np.linalg.norm(axis), np.array([0.0, 0.0, 1.0]))
    if azimuth_rad:
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        R = Rz if R is None else Rz @ R
    return R


def _small_tilt_R(dx_rad, dy_rad):
    """Rotation that tilts the slab normal by the rotation vector (dx, dy, 0) [rad]
    via Rodrigues -- exact for any angle, exactly the identity at (0, 0). Used to
    misorient a mosaic crystallite's reciprocal vectors about the mean orientation."""
    ang = float(np.hypot(dx_rad, dy_rad))
    if ang < 1e-15:
        return np.eye(3)
    kx, ky = dx_rad / ang, dy_rad / ang
    K = np.array([[0.0, 0.0, ky], [0.0, 0.0, -kx], [-ky, kx, 0.0]])  # skew of axis
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _mosaic_quadrature(fwhm_rad, nodes):
    """Gauss-Hermite product quadrature over a 2-D Gaussian mosaic tilt of the
    crystallite normal (per-axis sigma = FWHM / 2.3548 -- the rocking curve is the
    1-D projection). Returns a list of (rotation matrix, weight) with weights
    summing to 1, used by mc_spectrum to INCOHERENTLY average a reflection's
    spectrum over crystallite orientations (the exact mosaic route,
    docs/crystal-mosaicity.md (2)).

    Returns None -- the perfect-crystal fast path, today's result bit-for-bit --
    when there is nothing to average: ``fwhm_rad`` falsy, or ``nodes`` <= 1 (the
    single Gauss-Hermite node sits at zero tilt, i.e. the identity, so the loop is
    skipped entirely).

    This is the DETERMINISTIC counterpart to drawing K random orientations: the
    integrand (spectrum vs crystallite tilt) is smooth, so product Gauss-Hermite
    converges in far fewer evaluations than random sampling and needs no RNG
    sub-stream. Cost is K = nodes**2 evaluations of the per-reflection block, a
    direct wall-clock multiplier on the (serial, with CuPy) GPU hot loop."""
    if not fwhm_rad or nodes is None or nodes <= 1:
        return None
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    sigma = float(fwhm_rad) / 2.3548200450309493  # FWHM -> Gaussian sigma
    tilt = np.sqrt(2.0) * sigma * x  # quadrature nodes as tilt angles [rad]
    wt = w / np.sqrt(np.pi)  # per-axis weights, sum to 1
    return [
        (_small_tilt_R(tilt[j], tilt[k]), float(wt[j] * wt[k]))
        for j in range(len(tilt))
        for k in range(len(tilt))
    ]


def simulate_trajectories(
    E0_keV,
    Ne,
    thickness_ang,
    element=None,
    n_atoms_per_ang3=None,
    E_cut_keV=5.0,
    seed=0,
    max_steps=20000,
    elastic_model="mott",
    beam_dir=None,
    composition=None,
    layers=None,
):
    """
    Transport Ne electrons of energy E0_keV [keV] into a slab 0<=z<=thickness.
    Beam enters at the origin along +z. Electrons terminate when they exit
    either surface or drop below E_cut_keV (segments below the cutoff don't
    radiate in the spectral window of interest anyway).

    elastic_model:
      "mott" (default) -- Browning fit to the Mott TOTAL cross sections for
          the free path + screening parameter alpha(E) calibrated per element
          to reproduce the NIST SRD 64 relativistic Mott TRANSPORT cross
          section (so both the collision rate and the momentum-transfer rate
          match Mott data). Requires the NIST table in
          mott_transport_cross_sections/.
      "sr" -- classic analytic screened-Rutherford model (Joy), no data files.

    beam_dir: initial electron direction in the SLAB frame (default +z,
    i.e. normal incidence). For a tilted sample use tilted_geometry().

    composition: for COMPOUNDS, [(element, number_density_1_per_Ang3), ...]
    overriding element/n_atoms_per_ang3. Free paths and stopping are
    additive over elements; the scattering element at each collision is
    chosen with probability n_i sigma_i / sum.

    layers: optional film-on-substrate stack
    [(z_top, z_bot, composition), ...] (top/entrance first, contiguous, deepest
    z_bot = total thickness). Each electron's free path / stopping / scattering
    element switch by the layer it is currently in; a flight is truncated at an
    internal boundary (no collision -- the electron continues into the neighbor),
    so the substrate's higher-Z backscatter feeds electron path back into the
    film. None -> a single layer over [0, thickness_ang] (the old single-material
    transport, BIT-FOR-BIT). When given, thickness_ang is superseded by the
    stack's total thickness.

    Returns dict of per-segment arrays:
      "r_mid" (M,3) [Ang], "v_hat" (M,3), "L_ang" (M,), "E_keV" (M,),
      "t_ang" (M,), "elec_id" (M,), "layer" (M,) [emitting layer index]
    and diagnostics: "n_backscattered", "n_transmitted", "n_stopped", "n_layers".
    """
    # Build the layer stack: explicit `layers` (film-on-substrate) overrides;
    # else a single layer spanning the slab (bit-for-bit the old transport).
    if layers is None:
        layers = [
            (
                0.0,
                float(thickness_ang),
                _normalize_composition(element, n_atoms_per_ang3, composition),
            )
        ]
    z_total = float(layers[-1][1])
    n_layers = len(layers)
    L_comp = [[(el, float(n)) for el, n in lc] for (_, _, lc) in layers]
    L_Zs = [[TRANSPORT_ELEMENTS[el]["Z"] for el, _ in lc] for lc in L_comp]
    L_ncm3 = [[n * 1e24 for _, n in lc] for lc in L_comp]
    L_top = [float(a) for (a, _, _) in layers]
    L_bot = [float(b) for (_, b, _) in layers]
    internal_bounds = np.array(L_bot[:-1], dtype=float)  # between consecutive layers
    EPS = 1e-6  # nudge across an internal boundary so the layer lookup is unambiguous

    def _scatter_rates(Ea, Zs, n_cm3s):
        """Per-element elastic scattering rates [1/cm] at energies Ea (one layer)."""
        rates = []
        for Z_i, n_i in zip(Zs, n_cm3s, strict=False):
            if elastic_model == "mott":
                sig_i = _sigma_browning_cm2(Z_i, Ea)
            elif elastic_model == "sr":
                a = _alpha_sr_joy(Z_i, Ea)
                sig_i = (
                    5.21e-21
                    * Z_i**2
                    / Ea**2
                    * 4.0
                    * np.pi
                    / (a * (1.0 + a))
                    * ((Ea + 511.0) / (Ea + 1024.0)) ** 2
                )
            else:
                raise ValueError("elastic_model must be 'mott' or 'sr'")
            rates.append(n_i * sig_i)
        return np.array(rates)  # (n_elements, m)

    rng = np.random.default_rng(seed)
    pos = np.zeros((Ne, 3))
    if beam_dir is None:
        beam_dir = np.array([0.0, 0.0, 1.0])
    beam_dir = np.asarray(beam_dir, dtype=float)
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    if beam_dir[2] <= 1e-6:
        raise ValueError("beam_dir must point into the slab (z component > 0)")
    dirs = np.tile(beam_dir, (Ne, 1))
    E = np.full(Ne, float(E0_keV))
    alive = np.ones(Ne, dtype=bool)
    n_back = n_trans = 0
    # per-electron clock: cumulative flight "time" sum(L/beta) [Ang, c=1], the
    # same unit as the radiation interaction time t_L. Recorded at each segment's
    # START so a segment carries (depth, energy, age) -- consumed by the
    # penetration / electron-lifetime plots (-> fs via c = 2997.92 Ang/fs).
    clock = np.zeros(Ne)

    seg_mid, seg_dir, seg_len, seg_E, seg_t0, seg_id, seg_lay = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Event-driven loop: every iteration is one FREE FLIGHT + one ELASTIC
    # COLLISION for every still-alive electron, executed in lockstep (pure
    # vectorization -- electrons are independent, so synchronizing them is
    # exact). There is no time step and no discretization parameter: flight
    # lengths are sampled from the physical free-path distribution. With a
    # multilayer stack the alive electrons are grouped by their CURRENT layer
    # each iteration so each uses that layer's free path / stopping / scattering;
    # a single layer is one group, so that path stays bit-for-bit identical.
    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)  # indices of electrons still in play
        if n_layers == 1:
            lay_all = None  # everyone is in layer 0
        else:
            lay_all = np.clip(
                np.searchsorted(internal_bounds, pos[idx, 2], side="right"),
                0,
                n_layers - 1,
            )
        for L in range(n_layers):
            grp = idx if lay_all is None else idx[lay_all == L]
            if grp.size == 0:
                continue
            comp = L_comp[L]
            Zs = L_Zs[L]
            n_cm3s = L_ncm3[L]
            z_top_L, z_bot_L = L_top[L], L_bot[L]
            Ea = E[grp]  # kinetic energies [keV]

            # -- 1. distance to the next elastic collision (this layer) ---------
            # Exponential free path P(s)=exp(-s/lambda)/lambda, total rate
            # additive over the layer's elements: 1/lambda = sum_i n_i sigma_i(E).
            rates = _scatter_rates(Ea, Zs, n_cm3s)  # (n_elements, m) [1/cm]
            lam_ang = 1e8 / rates.sum(axis=0)  # mean free path [Ang]
            step = -lam_ang * np.log(rng.random(grp.size))  # sampled flight [Ang]

            d = dirs[grp]  # current unit direction of each electron
            p = pos[grp]  # current position [Ang]
            dz = d[:, 2]
            pz = p[:, 2]

            # -- 2. truncate at THIS layer's faces ------------------------------
            # The entrance face (z_top==0) and back face (z_bot==z_total) are
            # exits (vacuum -> no re-entry); an INTERNAL boundary instead hands
            # the electron to the neighbor layer with NO collision (it continues
            # straight and re-samples its free path in that layer next iteration).
            cross_up = (dz < 0) & (pz + step * dz < z_top_L)
            cross_dn = (dz > 0) & (pz + step * dz > z_bot_L)
            s_up = np.where(dz < 0, (pz - z_top_L) / (-dz + 1e-300), np.inf)
            s_dn = np.where(dz > 0, (z_bot_L - pz) / (dz + 1e-300), np.inf)
            step = np.where(cross_up, s_up, step)
            step = np.where(cross_dn, s_dn, step)
            exit_top = cross_up & (z_top_L <= 0.0)  # exited entrance (backscatter)
            exit_bot = cross_dn & (z_bot_L >= z_total)  # exited back (transmit)

            # -- 3. record the segment (the radiation source list) --------------
            # midpoint -> escape-absorption path; direction -> v.g, v.n in the
            # amplitudes; length -> interaction time t_L; START energy -> beta.
            seg_mid.append(p + 0.5 * step[:, None] * d)
            seg_dir.append(d.copy())
            seg_len.append(step)
            seg_E.append(Ea.copy())
            seg_t0.append(clock[grp].copy())  # age at segment START [Ang, c=1]
            seg_id.append(grp.copy())  # which electron emitted this segment
            seg_lay.append(np.full(grp.size, L, dtype=np.int16))  # emitting layer

            # -- 4. advance: straight line + continuous slowing-down ------------
            # Energy drains deterministically along the flight (CSDA: Joy-Luo
            # modified Bethe for THIS layer's composition; no straggling). The
            # clock advances by L/beta at the segment's start speed.
            pos[grp] = p + step[:, None] * d
            E[grp] = Ea + _dEds_compound(comp, Ea) * step
            clock[grp] += step / beta_from_keV(Ea)

            # -- 5. kill exited / exhausted; pass internal crossers on ----------
            died = exit_top | exit_bot | (E[grp] < E_cut_keV)
            n_back += int(exit_top.sum())  # exited the entrance face
            n_trans += int(exit_bot.sum())  # punched through the back face
            alive[grp[died]] = False
            crossed_internal = (cross_up | cross_dn) & ~died  # reached a layer seam
            if crossed_internal.any():
                ci = grp[crossed_internal]
                pos[ci, 2] += np.sign(dirs[ci, 2]) * EPS  # nudge just into neighbor

            # -- 6. elastic collision: scatter the FULL-FLIGHT survivors --------
            # (truncated flights did not collide). The scattering ELEMENT is
            # chosen with probability n_i sigma_i / sum; the polar angle from that
            # element's screened-Rutherford inversion; azimuth uniform; E unchanged.
            full = ~(cross_up | cross_dn) & ~died
            srv = grp[full]
            if srv.size:
                cos_t = np.empty(srv.size)
                if len(comp) == 1:
                    cos_t = _sample_cos_theta(Zs[0], E[srv], rng, elastic_model, comp[0][0])
                else:
                    p_el = rates[:, full] / rates[:, full].sum(axis=0)
                    u = rng.random(srv.size)
                    cum = np.cumsum(p_el, axis=0)
                    which = (u[None, :] > cum).sum(axis=0)  # element index
                    for i_el, (el_i, _) in enumerate(comp):
                        m = which == i_el
                        if m.any():
                            cos_t[m] = _sample_cos_theta(
                                Zs[i_el], E[srv][m], rng, elastic_model, el_i
                            )
                phi = 2.0 * np.pi * rng.random(srv.size)
                dirs[srv] = _rotate_directions(dirs[srv], cos_t, phi)

    return {
        "r_mid": np.concatenate(seg_mid),
        "v_hat": np.concatenate(seg_dir),
        "L_ang": np.concatenate(seg_len),
        "E_keV": np.concatenate(seg_E),
        "t_ang": np.concatenate(seg_t0),  # segment-start age sum(L/beta) [Ang, c=1]
        "elec_id": np.concatenate(seg_id),  # emitting electron index in [0, Ne)
        "layer": np.concatenate(seg_lay),  # emitting layer index in [0, n_layers)
        "n_backscattered": n_back,
        "n_transmitted": n_trans,
        "n_stopped": int(Ne - n_back - n_trans),
        "Ne": Ne,
        "thickness_ang": z_total,
        "n_layers": n_layers,
    }


# ---- segment-sum CXR spectrum ------------------------------------------------
_SEG_ARRAYS = ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer")


def _segments_in_layer(segments, L):
    """A view of `segments` restricted to those emitted in layer index L, keeping
    the scalar fields (Ne, thickness_ang, ...) so the per-electron normalization
    and geometry are unchanged. For a single-layer stack, layer 0 returns all
    segments (same values), so the single-material path is unaffected."""
    mask = segments["layer"] == L
    out = dict(segments)
    for k in _SEG_ARRAYS:
        if k in out:
            out[k] = out[k][mask]
    return out


def _polarization_pair(k_hat, g_vec):
    n_plane = np.cross(k_hat, g_vec)
    npl = np.linalg.norm(n_plane)
    if npl < 1e-12:
        tmp = np.array([0.0, 1.0, 0.0])
        n_plane = np.cross(k_hat, tmp)
        npl = np.linalg.norm(n_plane)
    e_s = n_plane / npl
    e_p = np.cross(e_s, k_hat)
    return e_s, e_p / np.linalg.norm(e_p)


def mc_spectrum(
    segments,
    E_grid_eV,
    crystal,
    hkl_list,
    theta_obs_rad=np.deg2rad(119.0),
    B_ang2=None,
    use_henke=True,
    absorber_element="C",
    chunk=40000,
    n_hat=None,
    composition=None,
    beam_uvw=None,
    azimuth_rad=0.0,
    sinc_cutoff=None,
    components=False,
    layers=None,
    mosaic_fwhm_rad=None,
    mosaic_nodes=1,
):
    """
    Per-electron CXR spectrum d2N/dE dOmega [photons / eV / sr / electron]
    on E_grid_eV, summed incoherently over the trajectory segments and the
    listed reflections (their resonances are spectrally separated, so
    cross-g coherence is negligible).

    Per segment and reflection (Zhai SI Eqs. 5-7, nonrelativistic):
      omega_res = beta v_hat.g / (1 - beta v_hat.n)         [Eq. 10 resonance]
      d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |A|^2 t_L^2
                      sinc^2[(1 - beta v.n)(omega-omega_res) t_L / 2] T_abs
    with A = A_PXR + A_CBS per polarization (Feranchuk Eqs. 13/14 evaluated
    at omega_res with the segment's velocity vector), t_L = L_seg/beta, and
    T_abs the Beer-Lambert escape factor from the segment midpoint.

    n_hat: detector direction in the SAMPLE frame; overrides theta_obs_rad
    when given (use tilted_geometry() for a tilted sample).
    composition: [(element, n_per_Ang3), ...] for compound self-absorption;
    defaults to the single absorber_element at the crystal's total atom
    density (exact for elemental crystals).
    layers: optional film-on-substrate absorber stack
    [(z_top, z_bot, composition), ...] (top/entrance first). When given, the
    escape attenuation is the piecewise mu_i*dz_i sum across the whole stack
    rather than the single slab; the RADIATION still comes from crystal/hkl_list
    (the film). None -> single slab (bit-for-bit unchanged).

    beam_uvw: CRYSTAL AXIS along the slab normal (+z). Default None keeps the
    construction-frame convention, i.e. [001] (the c-axis for hexagonal
    crystals, a cube edge for cubic ones). Passing e.g. (1, 1, 1) cuts the
    slab perpendicular to the [111] direct-lattice direction: every g in
    hkl_list is rotated by the MINIMAL rotation taking [uvw] -> +z, then by
    azimuth_rad about +z (the in-plane setting of the crystal relative to
    the detector azimuth -- it matters for individual family members).

    sinc_cutoff: None (default) evaluates every segment's lineshape over the
    FULL grid (exact). A number C truncates each lineshape at |P t_L| > C,
    i.e. |E - E_res| > C/a_width -- segments are processed in resonance-
    sorted blocks against only the relevant grid window, which is several
    times faster on wide grids. Tail loss is ~1/(pi C) of each line's
    integral (0.3% at C = 100); peak heights are unaffected. Requires a
    UNIFORM E_grid.

    mosaic_fwhm_rad / mosaic_nodes: the EXACT crystal-mosaicity route
    (docs/crystal-mosaicity.md (2)). None / nodes<=1 (the default) is a perfect
    crystal -- today's single-orientation result bit-for-bit. Otherwise the
    spectrum is incoherently averaged over crystallite orientations drawn from a
    Gaussian mosaic of rocking-curve FWHM ``mosaic_fwhm_rad`` [rad], via a 2-D
    Gauss-Hermite product quadrature of ``mosaic_nodes`` nodes per tilt axis (so
    K = mosaic_nodes**2 evaluations of the per-reflection block). Unlike the
    analytic mosaic_fwhm_eV (energy-shift only, applied at detector convolution),
    this broadens BOTH PXR and CBS, captures the amplitude/polarization variation
    across the cone, and yields the correct (generally asymmetric) lineshape and
    integrated yield. Do NOT also apply the analytic term to the result (double
    count); build_cases handles that mutual exclusion.
    """
    if B_ang2 is None:
        raise ValueError(
            "mc_spectrum: B_ang2 (Debye-Waller B-factor [Ang^2]) is required; "
            "pass the material's value (no silent default)."
        )
    info = CRYSTALS[crystal]
    n_atoms = len(info["basis"]) / info["V_cell"]
    abs_comp = _normalize_composition(absorber_element, n_atoms, composition)

    # crystal orientation: rotation applied to all reciprocal vectors
    R_orient = _orientation_R(info["lattice"], beam_uvw, azimuth_rad)
    thickness = segments["thickness_ang"]
    Ne = segments["Ne"]

    if n_hat is None:
        n_hat = np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    else:
        n_hat = np.asarray(n_hat, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
    E_grid = xp.asarray(E_grid_eV, dtype=REAL)
    spec = xp.zeros(E_grid.size, dtype=REAL)
    spec_pxr = xp.zeros(E_grid.size, dtype=REAL)
    spec_cbs = xp.zeros(E_grid.size, dtype=REAL)

    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)
    seg_v = xp.asarray(segments["v_hat"], dtype=REAL)
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)
    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)
    beta_all = beta_from_keV(seg_E)  # speed/c per segment
    v_all = beta_all[:, None] * seg_v  # velocity vectors (c=1)

    # chi_g / U_g are smooth in energy AWAY from absorption edges, so evaluate
    # them on a tabulation grid and interpolate at the per-segment resonance
    # energies ON THE GPU (step 3) -- a few ms over ~10^3 grid points instead of
    # ~0.25 s/case over ~10^5 segments, and E_res stays on the device (no
    # per-reflection GPU->CPU->GPU round-trip; that structure-factor CPU cost was
    # what capped GPU utilisation on a fast card). The grid is a 1 eV mesh UNION
    # the basis elements' native Henke energies, which densely sample the edges --
    # a plain uniform mesh mis-resolves the edge jumps (tens of % at e.g. the
    # C K-edge). Window matches the keep mask below.
    from .atomic_form_factors import load_henke

    _pad = 0.2 * (float(E_grid_eV[-1]) - float(E_grid_eV[0]))
    _lo, _hi = float(E_grid_eV[0]) - _pad, float(E_grid_eV[-1]) + _pad
    _grids = [np.arange(_lo, _hi + 1.0, 1.0)]
    for _el in {el for el, _ in info["basis"]}:
        try:
            _Eh = load_henke(_el)[0]
            _grids.append(_Eh[(_Eh >= _lo) & (_Eh <= _hi)])
        except Exception:
            pass
    E_tab = np.unique(np.concatenate(_grids))
    E_tab_g = xp.asarray(E_tab, dtype=REAL)

    n_hat_d = xp.asarray(n_hat, dtype=REAL)  # detector dir is g-independent: hoist
    # mosaic crystallite-orientation quadrature: None -> perfect crystal (default;
    # today's single-orientation result bit-for-bit). Otherwise a list of
    # (rotation, weight) tilting g across the Gaussian mosaic cone, summed
    # incoherently below (docs/crystal-mosaicity.md route 2).
    mosaic_quad = _mosaic_quadrature(mosaic_fwhm_rad, mosaic_nodes)

    def _accumulate(g_vec, chi_re, chi_im, u_re, u_im, wm):
        """Add one reflection's contribution for crystallite reciprocal vector
        ``g_vec``, scaled by the mosaic-quadrature weight ``wm``, into spec /
        spec_pxr / spec_cbs in place. The structure-factor tabulations (chi/u on
        E_tab_g) are precomputed per reflection -- they depend on hkl and energy,
        NOT on the mosaic orientation; everything else depends on g and so is
        recomputed per orientation. wm = 1.0 for the perfect-crystal path."""
        # sigma/pi polarization unit vectors: fixed once n_hat and g are fixed
        e_s, e_p = _polarization_pair(n_hat, g_vec)
        g_vec_d = xp.asarray(g_vec, dtype=REAL)

        # -- 1. per-segment resonance energy (Eq. 10) ---------------------------
        #   omega_res = v.g / (1 - v.n)   [1/Ang]   (>0 required to radiate)
        v_dot_g = v_all @ g_vec_d
        denom = 1.0 - v_all @ n_hat_d  # the Doppler-like denominator
        omega_res = v_dot_g / denom
        E_res = HBARC_EV_ANG * omega_res  # -> eV

        # -- 2. drop segments whose line misses the spectral window -------------
        # (pad by 20% so sinc tails that reach into the window still count)
        pad = 0.2 * (E_grid[-1] - E_grid[0])
        keep = (E_res > float(E_grid[0] - pad)) & (E_res > 10.0) & (E_res < E_grid[-1] + pad)
        if not keep.any():
            return
        idx = xp.flatnonzero(keep)

        E_r = E_res[idx]  # line energy per kept segment [eV]
        om = omega_res[idx]  # same in 1/Ang
        v = v_all[idx]  # velocity vectors
        beta = beta_all[idx]
        t_L = seg_L[idx] / beta  # interaction time [Ang] (c=1)
        dnm = denom[idx]
        vdg = v_dot_g[idx]

        # -- 3. couplings AT each segment's resonance energy --------------------
        # (amplitudes vary slowly across the narrow line; freezing them at E_res
        # is accurate to the linewidth/E_res level). Interpolated at E_res ON THE
        # GPU from the per-reflection tabulation; chi_g/U_g are complex, so the
        # real and imaginary parts are interpolated separately.
        chi = xp.interp(E_r, E_tab_g, chi_re) + 1j * xp.interp(E_r, E_tab_g, chi_im)
        eUg_over_m = (xp.interp(E_r, E_tab_g, u_re) + 1j * xp.interp(E_r, E_tab_g, u_im)) / M_E_EV

        # -- 4. photon kinematics per segment ------------------------------------
        k_vec = om[:, None] * n_hat_d  # photon wavevector omega * n_hat
        kg_vec = k_vec + g_vec_d  # diffracted wavevector k + g
        kg2 = xp.einsum("ij,ij->i", kg_vec, kg_vec)
        detuning = kg2 - om**2  # PXR denominator (~g^2, never small)
        k_dot_g = k_vec @ g_vec_d

        # -- 5. Eq. (13) + relativistic Eq. (14) amplitudes, per segment ----------
        # CBS braced product {a;b} = a.b - (a.v)(b.v) and 1/gamma prefactor
        # (Zhai SI Eq. 6); v here is the SEGMENT velocity, so k.v = omega(1-dnm)
        gamma = 1.0 / xp.sqrt(1.0 - beta**2)
        k_dot_v = om * (1.0 - dnm)
        A2 = xp.zeros(idx.size, dtype=REAL)
        A2_pxr = xp.zeros(idx.size, dtype=REAL)
        A2_cbs = xp.zeros(idx.size, dtype=REAL)
        for e in (e_s, e_p):  # sum |A|^2 over both polarizations
            e_d = xp.asarray(e, dtype=REAL)
            g_dot_e = g_vec_d @ e_d  # scalar (e fixed per reflection)
            v_dot_e = v @ e_d
            v_dot_kg = xp.einsum("ij,ij->i", v, kg_vec)
            A_PXR = chi / detuning * (v_dot_kg * g_dot_e - om**2 * v_dot_e)
            braced_ge = g_dot_e - vdg * v_dot_e
            braced_kg = k_dot_g - k_dot_v * vdg
            A_CBS = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
            A2 += xp.abs(A_PXR + A_CBS) ** 2
            A2_pxr += xp.abs(A_PXR) ** 2
            A2_cbs += xp.abs(A_CBS) ** 2

        # -- 6. Beer-Lambert escape factor from the segment midpoint -------------
        # straight path along n_hat to whichever face the photon exits. With a
        # LAYERED absorber (layers) the optical depth sums mu_i*dz_i across the
        # film-on-substrate stack; otherwise it's the single-slab path. The
        # geometric path is mosaic-independent; the optical depth uses E_r (the
        # orientation-shifted line energy), so it is recomputed per orientation.
        z_mid = seg_r[idx, 2]
        if layers is None:
            if n_hat[2] < 0:
                L_esc = z_mid / (-n_hat[2])  # out the entrance face
            else:
                L_esc = (thickness - z_mid) / n_hat[2]  # out the back face
            tau = L_esc * _mu_total_inv_ang(abs_comp, E_r)
        else:
            tau = _stack_tau(layers, z_mid, n_hat[2], E_r)
        T_abs = xp.exp(-tau)

        # -- 7. accumulate the finite-segment lineshape ---------------------------
        # d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |A|^2 t_L^2
        #                  * sinc^2[(1 - v.n)(omega - omega_res) t_L / 2] * T_abs
        # weight = everything except the sinc^2 (times the mosaic weight wm);
        # a_width converts (E - E_res) to the sinc argument: P t_L = a_width(E - E_res).
        pref = ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * t_L**2 * T_abs
        weight = pref * A2 * wm
        targets = [(weight, spec)]
        if components:
            targets += [(pref * A2_pxr * wm, spec_pxr), (pref * A2_cbs * wm, spec_cbs)]
        a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
        good = xp.isfinite(weight) & (weight > 0)

        if sinc_cutoff is None:
            for j0 in range(0, idx.size, chunk):
                sl = slice(j0, min(j0 + chunk, idx.size))
                m = good[sl]
                if not m.any():
                    continue
                x = a_width[sl][m, None] * (E_grid[None, :] - E_r[sl][m, None]) / xp.pi
                S = xp.sinc(x) ** 2
                for w, tgt in targets:
                    tgt += w[sl][m] @ S
        else:
            dE = E_grid[1] - E_grid[0]
            order = xp.argsort(E_r)
            blk = 8192
            for j0 in range(0, order.size, blk):
                sel = order[j0 : j0 + blk]
                sel = sel[good[sel]]
                if sel.size == 0:
                    continue
                half = sinc_cutoff / a_width[sel]
                lo = float(_to_cpu((E_r[sel] - half).min()))
                hi = float(_to_cpu((E_r[sel] + half).max()))
                i0 = max(int((lo - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))), 0)
                i1 = min(
                    int((hi - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))) + 2,
                    E_grid.size,
                )
                if i1 <= i0:
                    continue
                x = a_width[sel][:, None] * (E_grid[None, i0:i1] - E_r[sel][:, None]) / xp.pi
                S = xp.sinc(x) ** 2
                for w, tgt in targets:
                    tgt[i0:i1] += w[sel] @ S

    for hkl in hkl_list:
        # reciprocal vector in the sample frame: construction frame by default
        # ([001] along the slab normal), rotated if beam_uvw given
        g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
        if R_orient is not None:
            g_vec = R_orient @ g_vec
        # structure-factor couplings depend on hkl + tabulation energy only (NOT on
        # the mosaic orientation), so tabulate once per reflection (CPU, a few ms)
        # and reuse across the orientation quadrature; push real/imag to the device.
        chi_tab = np.asarray(chi_g(crystal, hkl, E_tab, B_ang2, use_henke))
        u_tab = np.asarray(U_g(crystal, hkl, E_tab, B_ang2, use_henke))
        chi_re = xp.asarray(chi_tab.real, dtype=REAL)
        chi_im = xp.asarray(chi_tab.imag, dtype=REAL)
        u_re = xp.asarray(u_tab.real, dtype=REAL)
        u_im = xp.asarray(u_tab.imag, dtype=REAL)

        if mosaic_quad is None:  # perfect crystal: one orientation, weight 1
            _accumulate(g_vec, chi_re, chi_im, u_re, u_im, 1.0)
        else:  # incoherent average over the mosaic crystallite orientations
            for R_m, wm in mosaic_quad:
                _accumulate(R_m @ g_vec, chi_re, chi_im, u_re, u_im, wm)

    if components:
        return _to_cpu(spec / Ne), _to_cpu(spec_pxr / Ne), _to_cpu(spec_cbs / Ne)
    return _to_cpu(spec / Ne)


def mc_spectrum_solid_angle(
    segments,
    E_grid_eV,
    crystal,
    hkl_list,
    *,
    n_hats,
    weights,
    **kwargs,
):
    """
    Solid-angle-INTEGRATED per-electron line spectrum dN/dE [photons / eV /
    electron], already x Omega: ``sum_i weights_i * mc_spectrum(n_hat=n_hats_i)``.

    The finite detector face is tiled by detector_directions() into directions
    ``n_hats`` (sample frame) carrying solid-angle ``weights``. Because the
    resonance energy AND the amplitudes depend on n_hat, summing per-direction
    spectra yields the true, generally ASYMMETRIC integrated lineshape and the
    across-face intensity gradient -- the first-principles replacement for the
    flat-Omega + analytic aperture_fwhm_eV pair (docs/detector-solid-angle.md). It
    reuses the validated single-angle mc_spectrum, so a 1-direction grid
    reproduces ``spec * Omega`` exactly (the regression anchor).

    Units: the result ALREADY includes the solid angle (the weights carry
    dOmega). When consuming it do NOT multiply by domega_sr again, and drop the
    aperture_fwhm_eV term from the detector convolution (keep the EDS-resolution
    term). This is an opt-in tool; it does not change the checkpoint pipeline's
    single-n_hat unit convention. ``**kwargs`` forward to mc_spectrum (B_ang2,
    use_henke, layers, composition, beam_uvw, azimuth_rad, mosaic_*, ...).
    """
    n_hats = np.asarray(n_hats, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if n_hats.ndim != 2 or n_hats.shape[1] != 3:
        raise ValueError("n_hats must be (N, 3)")
    if weights.shape != (n_hats.shape[0],):
        raise ValueError("weights must be (N,) matching n_hats")
    total = None
    for n_i, w_i in zip(n_hats, weights, strict=True):
        spec_i = np.asarray(
            mc_spectrum(segments, E_grid_eV, crystal, hkl_list, n_hat=n_i, **kwargs)
        )
        contrib = float(w_i) * spec_i
        total = contrib if total is None else total + contrib
    return total


# ---- bremsstrahlung background -------------------------------------------------
R_E_CM2 = 7.9407877e-26  # classical electron radius squared [cm^2]


def _brem_dsigma_dk(Z, T_keV, k_eV):
    """
    Bremsstrahlung cross section differential in photon energy,
    dsigma/dk [cm^2/eV]: nonrelativistic Bethe-Heitler in Born approximation
    with the Elwert Coulomb correction (cf. Koch & Motz, Rev. Mod. Phys. 31,
    920 (1959)), evaluated with relativistic electron momenta:

        dsigma/dk = (16/3) alpha r_e^2 Z^2 (1/k) (1/p_i^2)
                    ln[(p_i+p_f)/(p_i-p_f)] * f_Elwert,
        f_Elwert  = (beta_i/beta_f) (1-exp(-2 pi Z alpha/beta_i))
                                  / (1-exp(-2 pi Z alpha/beta_f)),

    with p in units of m_e c. Broadcasts T_keV (segments) against k_eV
    (spectral grid); zero where k >= T. Adequate for Z <~ 30 and
    T <~ 100 keV; swap in Seltzer-Berger tables for better accuracy.
    """
    mc2 = 510.99895  # keV
    T_i = xp.asarray(T_keV, dtype=REAL)[:, None]
    k = xp.asarray(k_eV, dtype=REAL)[None, :] / 1e3  # keV
    T_f = T_i - k
    ok = (T_f > 1e-6) & (k > 0.0)  # k>0: no photon (and no 1/k blowup) at k=0
    T_f = xp.where(ok, T_f, 1e-6)

    p_i = xp.sqrt(T_i * (T_i + 2.0 * mc2)) / mc2
    p_f = xp.sqrt(T_f * (T_f + 2.0 * mc2)) / mc2
    beta_i = p_i / (1.0 + T_i / mc2)
    beta_f = p_f / (1.0 + T_f / mc2)

    born_log = xp.log((p_i + p_f) / xp.maximum(p_i - p_f, 1e-30))
    elwert = (
        beta_i
        / beta_f
        * (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_i))
        / (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_f))
    )

    dsig = (
        16.0
        / 3.0
        * ALPHA_FS
        * R_E_CM2
        * Z**2
        / xp.maximum(k * 1e3, 1e-30)
        / p_i**2
        * born_log
        * elwert
    )  # per eV
    return xp.where(ok, dsig, 0.0)


def mc_brem_spectrum(
    segments,
    E_grid_eV,
    element=None,
    n_atoms_per_ang3=None,
    theta_obs_rad=np.deg2rad(119.0),
    n_hat=None,
    chunk=20000,
    composition=None,
    layers=None,
):
    """
    Incoherent bremsstrahlung background d2N/dE dOmega
    [photons / eV / sr / electron] from the same Monte Carlo segments as
    mc_spectrum: each segment radiates n * dsigma/dk * L_seg photons/eV at
    its (start) kinetic energy, attenuated by the Beer-Lambert escape factor
    from the segment midpoint along the observation direction.

    Approximations: emission taken isotropic (1/4pi) -- the standard
    assumption at weakly relativistic energies once electron directions are
    scattering-randomized (and the one Zhai et al. adopt for their
    estimates); the tiny coherent fraction of the continuum (which is what
    forms the CBS lines) is not subtracted.

    NOTE: run the transport with a LOW E_cut_keV (~1 keV) for backgrounds --
    electrons below the CXR cutoff still radiate in the soft X-ray window.

    composition: [(element, n_per_Ang3), ...] for compounds; the emission is
    additive over elements (each weighted by its own Z^2 cross section), and
    the self-absorption uses the summed attenuation.
    """
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
    thickness = segments["thickness_ang"]
    Ne = segments["Ne"]

    if n_hat is None:
        n_hat = np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    else:
        n_hat = np.asarray(n_hat, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)

    E_grid = xp.asarray(E_grid_eV, dtype=REAL)
    mu = _mu_total_inv_ang(comp, E_grid)  # (NE,) [1/Ang], single-slab fallback
    # The Henke absorption tables span ~20 eV - 30 keV; outside that the wide
    # brem grid gets NaN (above 30 keV) or inf (at E=0), and a single bad bin
    # makes brem_wide -- and its integrated count rate -- NaN. Hard X-rays escape
    # essentially unattenuated, so treat an unavailable mu as zero (transparent).
    mu = xp.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    # layered (film-on-substrate) absorber: precompute each layer's mu(E_grid);
    # the per-segment z-path dz folds in inside the chunk loop. None -> single slab.
    if layers is not None:
        inv_nz = 1.0 / max(abs(float(n_hat[2])), 1e-12)
        layer_mu = [
            xp.nan_to_num(_mu_total_inv_ang(c, E_grid), nan=0.0, posinf=0.0, neginf=0.0)
            for (_, _, c) in layers
        ]

    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)
    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)
    z_mid = seg_r[:, 2]
    L_esc = z_mid / -n_hat[2] if n_hat[2] < 0 else (thickness - z_mid) / n_hat[2]

    spec = xp.zeros(E_grid.size, dtype=REAL)
    M = seg_E.size
    for j0 in range(0, M, chunk):
        sl = slice(j0, min(j0 + chunk, M))
        if layers is None:
            T_abs = xp.exp(-L_esc[sl][:, None] * mu[None, :])
        else:
            tau = 0.0
            for (z_top, z_bot, _), mu_i in zip(layers, layer_mu, strict=False):
                dz = _layer_dz(z_mid[sl], n_hat[2], float(z_top), float(z_bot))
                tau = tau + (dz * inv_nz)[:, None] * mu_i[None, :]
            T_abs = xp.exp(-tau)
        path_cm = seg_L[sl] * 1e-8
        for el_i, n_i in comp:
            Z_i = TRANSPORT_ELEMENTS[el_i]["Z"]
            dsig = _brem_dsigma_dk(Z_i, seg_E[sl], E_grid)
            spec += (n_i * 1e24 * path_cm) @ (dsig * T_abs)
    return _to_cpu(spec / (4.0 * xp.pi) / Ne)


# ---- parallel case runner -------------------------------------------------------
def run_case(case):
    """
    Worker for one (crystal, beam energy) Monte Carlo case: transport + line
    spectrum + bremsstrahlung. Module-level so it can be pickled into worker
    processes on Windows (notebook-defined functions cannot).

    case: a plain dict --
        required: crystal, composition, hkl_list, B_ang2, E0_keV, thickness_ang,
                theta_obs_rad, Ne, Ne_brem, seed, and EITHER a single
                E_grid = (start_eV, stop_eV, step_eV) OR the decoupled pair
                E_grid_line / E_grid_brem (each a (start, stop, step) tuple):
                the lines are evaluated on the fine NARROW E_grid_line, the
                smooth bremsstrahlung on the coarse WIDE E_grid_brem (extend the
                latter to the beam energy for the full measured spectrum, without
                paying the line cost up there -- the lines top out at a few keV).
        optional: tilt_deg (0), tilt_azim_deg (0), beam_uvw (None),
                azimuth_rad (0), E_cut_lines_keV (5), E_cut_brem_keV (1),
                spec_chunk (40000) / brem_chunk (20000): segments per GPU matmul
                -- lower these to cap peak GPU memory on a busy/shared device,
                sinc_cutoff (None = exact lineshapes; windowing buys nothing
                for bulk targets, where scattering Doppler-spreads the lines
                across the whole grid),
                mosaic_mc_fwhm_rad (None) / mosaic_mc_nodes (1): the exact
                Monte-Carlo crystal-mosaicity average (mc_spectrum); None/1 ->
                perfect crystal,
                brem_step_eV (10; legacy single-E_grid fallback only)

    Returns dict(E_grid, spec, brem [on E_grid], E_grid_brem, brem_wide [the
                full-range background], eta, n_segments) plus crystal/E0.
    """
    return _spectrum_case(case, _transport_case(case))


def _transport_case(case):
    """CPU-only phase of run_case: the line + brem trajectory transport (pure
    numpy, never touches the GPU). Returns the segments + geometry + grids the
    spectrum phase consumes. run_cases farms this out to a worker pool so the
    transport of upcoming cases overlaps the GPU work on the current one."""
    if "E_grid_line" in case:
        E_grid = np.arange(*case["E_grid_line"])
        E_brem = np.arange(*case["E_grid_brem"])
    else:
        E_grid = np.arange(*case["E_grid"])
        step_b = case.get("brem_step_eV", 10.0)
        E_brem = np.arange(E_grid[0], E_grid[-1] + step_b, step_b)
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    # film-on-substrate stack drives multilayer transport too (substrate
    # backscatter / substrate brem); None -> single-material slab (unchanged).
    layers = case.get("abs_layers")
    segs = simulate_trajectories(
        case["E0_keV"],
        case["Ne"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=case["seed"],
        beam_dir=beam,
        layers=layers,
    )
    segs_b = simulate_trajectories(
        case["E0_keV"],
        case["Ne_brem"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        seed=case["seed"] + 1,
        beam_dir=beam,
        layers=layers,
    )
    return dict(E_grid=E_grid, E_brem=E_brem, n_hat=n_hat, segs=segs, segs_b=segs_b)


def _spectrum_case(case, tp):
    """GPU phase of run_case: line spectrum + brem from the already-transported
    segments ``tp`` (from _transport_case). Runs in the main process, so only one
    CUDA context ever touches the device."""
    E_grid, E_brem, n_hat = tp["E_grid"], tp["E_brem"], tp["n_hat"]
    segs, segs_b = tp["segs"], tp["segs_b"]
    # optional film-on-substrate stack (None -> single slab, unchanged)
    abs_layers = case.get("abs_layers")
    n_lay = int(segs.get("n_layers", 1))

    # LINES: each CRYSTALLINE layer radiates its own PXR/CBS lines, summed
    # INCOHERENTLY (separate crystals -> no cross-layer coherence); every line
    # self-absorbs through the WHOLE stack (layers=abs_layers). `layer_radiators`
    # is a per-layer list aligned with the stack -- a dict of crystal params for a
    # crystalline layer (film or crystalline substrate), None for an amorphous one
    # (no coherent lines). layer_radiators absent -> single slab: the film radiates
    # from ALL its segments via the case's scalar crystal keys (bit-for-bit the
    # pre-multilayer path). See docs/multilayer-materials.md (per-layer radiation).
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),  # None -> perfect crystal
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = case.get("spec_chunk") or 40000
    if radiators is None:
        spec = mc_spectrum(
            segs,
            E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case.get("beam_uvw"),
            azimuth_rad=case.get("azimuth_rad", 0.0),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            **mosaic_kw,
        )
    else:
        spec = np.zeros(E_grid.shape, dtype=float)
        for L, rad in enumerate(radiators):
            if rad is None:  # amorphous layer -> no coherent lines
                continue
            sL = _segments_in_layer(segs, L)
            if sL["L_ang"].size == 0:
                continue
            spec = spec + mc_spectrum(
                sL,
                E_grid,
                crystal=rad["crystal"],
                hkl_list=rad["hkl_list"],
                n_hat=n_hat,
                B_ang2=rad["B_ang2"],
                composition=abs_layers[L][2],
                beam_uvw=rad.get("beam_uvw"),
                azimuth_rad=case.get("azimuth_rad", 0.0),
                sinc_cutoff=case.get("sinc_cutoff"),
                chunk=spec_chunk,
                layers=abs_layers,
                **mosaic_kw,
            )

    # BREM: EVERY layer radiates with its OWN composition (each Z^2 cross
    # section); each layer's brem self-absorbs through the whole stack. Summed
    # over layers; a single layer is exactly the old single-material brem.
    brem_chunk = case.get("brem_chunk") or 20000
    if n_lay == 1:
        brem_wide = mc_brem_spectrum(
            segs_b,
            E_brem,
            composition=case["composition"],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
        )
    else:
        brem_wide = np.zeros(E_brem.shape, dtype=float)
        for L in range(n_lay):
            sL = _segments_in_layer(segs_b, L)
            if sL["L_ang"].size == 0:
                continue
            brem_wide = brem_wide + mc_brem_spectrum(
                sL,
                E_brem,
                composition=abs_layers[L][2],
                n_hat=n_hat,
                chunk=brem_chunk,
                layers=abs_layers,
            )
    brem = np.interp(E_grid, E_brem, brem_wide)  # brem under the lines (line grid)
    # Hand this case's GPU scratch back to the OS so the CuPy memory pool can't
    # accumulate (and fragment) across a long sweep until it fills the card.
    if _GPU:
        cp.get_default_memory_pool().free_all_blocks()
    return dict(
        E_grid=E_grid,
        spec=spec,
        brem=brem,
        E_grid_brem=E_brem,
        brem_wide=brem_wide,
        eta=segs["n_backscattered"] / segs["Ne"],
        n_segments=int(segs["L_ang"].size),
        crystal=case["crystal"],
        E0_keV=case["E0_keV"],
    )


def _worker_init():
    """
    Runs once in each worker process: drop to BELOW_NORMAL priority so the
    desktop stays responsive. Workers still use idle CPU at full speed; the
    OS just schedules interactive applications first.
    """
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # typed signatures matter: the untyped pseudo-handle (-1) gets
        # truncated on 64-bit and the call silently fails
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        try:
            os.nice(10)  # POSIX fallback
        except Exception:
            pass


def run_cases(cases, max_workers=None, progress=True, callback=None):
    """
    Run a list of case dicts through run_case, results in input order.

    GPU present (the usual path): the CPU transport is PIPELINED across a worker
    pool while THIS process drives the spectrum/brem serially on the single CUDA
    context -- the ~40% transport idle overlaps the GPU work, with no device
    contention (multiple CUDA contexts are what crawled the old max_workers>1).
    Workers run ONLY transport (pure CPU/numpy), never the GPU. Callbacks fire in
    input order as each case's GPU phase finishes.

    No GPU: cases run through a worker pool (or serially), completion order.

    max_workers: None -> sized automatically (a few transport workers when a GPU
        is present; ~3/4 of the CPUs otherwise). An integer pins the count; 0
        runs everything serially in this process (debugging / safe fallback).
    progress: tqdm bar over completed cases.
    callback: callable(i, case, out) invoked in THIS process as each case
        finishes; stream/checkpoint/plot without waiting for the batch.
        Exceptions propagate and abort the run.

    Crawl protections: workers run BELOW_NORMAL priority (_worker_init) and get
    single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, inherited) -- N workers
    x M BLAS threads is the classic oversubscription freeze.
    """

    def _maybe_bar(iterable):
        if not progress:
            return iterable
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, total=len(cases), desc="cases")
        except ImportError:
            # tqdm.auto picks the widget bar inside Jupyter, and that bar
            # raises ImportError AT CONSTRUCTION if ipywidgets is missing --
            # fall back to the plain-text console bar before giving up.
            try:
                from tqdm import tqdm

                return tqdm(iterable, total=len(cases), desc="cases")
            except ImportError:
                return iterable

    n = len(cases)
    results = [None] * n
    if n == 0:
        return results

    def _serial():
        for i in _maybe_bar(range(n)):
            out = run_case(cases[i])
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
        return results

    def _single_thread_blas():
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = "1"

    # ---- GPU: pipeline CPU transport (worker pool) behind the serial GPU ------
    if _GPU:
        if max_workers == 0:
            return _serial()
        if max_workers is None:
            ncpu = os.process_cpu_count() or os.cpu_count() or 8
            nw = max(2, min(n, ncpu // 2))  # ~physical cores; transport is the tail
        else:
            nw = min(max_workers, n)
        if nw < 2:
            return _serial()
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor

        prefetch = nw + 2  # keep the transport pool ahead
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as ex:
            inflight = {i: ex.submit(_transport_case, cases[i]) for i in range(min(prefetch, n))}
            for i in _maybe_bar(range(n)):
                tp = inflight.pop(i).result()  # transport (already overlapped)
                j = i + prefetch
                if j < n:
                    inflight[j] = ex.submit(_transport_case, cases[j])
                out = _spectrum_case(cases[i], tp)  # GPU, THIS process only
                results[i] = out
                if callback is not None:
                    callback(i, cases[i], out)
        return results

    # ---- no GPU: serial in-process, or a full-case worker pool ---------------
    if max_workers is None:
        ncpu = os.process_cpu_count() or os.cpu_count() or 8
        max_workers = max(1, min(n, ncpu * 3 // 4))
    if max_workers == 0:
        return _serial()
    _single_thread_blas()
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
        futures = {ex.submit(run_case, c): i for i, c in enumerate(cases)}
        for fut in _maybe_bar(as_completed(futures)):
            i = futures[fut]
            out = fut.result()
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
    return results


def load_external_brem(path, E_grid_eV):
    """
    Interpolate an EXTERNAL bremsstrahlung background onto the spectral grid
    -- e.g. a NIST DTSA-II simulation, which is what Zhai et al. use both
    for their simulated backgrounds (refs 96-100) and, with a PIXE-style
    numerical fit, for their experimental subtraction (SI S3).

    File format: two columns (energy [eV], intensity), whitespace- or
    comma-separated; '#' comment lines and non-numeric headers are skipped.
    The intensity must already be in DETECTED units matching your plots
    (e.g. Phs/eV/s/nA: from a DTSA-II counts export, divide counts/channel
    by channel width [eV] x live time [s] x beam current [nA]). It is
    treated as an as-detected spectrum: window efficiency and detector
    resolution are NOT re-applied. Energies outside the file's range
    interpolate to zero.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2 or parts[0].startswith(("#", "//")):
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # header / text line
    if not rows:
        raise ValueError(f"no numeric (E, intensity) rows found in {path}")
    arr = np.array(sorted(rows))
    return np.interp(np.asarray(E_grid_eV, dtype=float), arr[:, 0], arr[:, 1], left=0.0, right=0.0)


# ---- detector model (SI S3/S4) -----------------------------------------------
def detector_efficiency(E_eV, polymer_nm=300.0, al_nm=40.0, grid_open=0.78):
    """
    Soft X-ray collection efficiency of a polymer-window SDD (the dominant
    loss below ~2 keV; the Si diode itself is ~fully absorbing there):

        QE(E) = grid_open * T_polymer(E) * T_Al(E)

    computed from Henke f2 attenuation for a polyimide film (C22 H10 N2 O5,
    rho = 1.42 g/cm^3) of thickness polymer_nm, an aluminum light-blocking
    coat of al_nm, and the etched-Si support grid (open-area fraction
    grid_open; the grid bars are opaque at these energies). Defaults model a
    Moxtek AP3.3-class window -- the actual Oxford UltimMax window is
    proprietary, so treat the parameters as tunable. Carries the C, N, O
    edge structure (e.g. the deep notch just above the O-K edge at 532 eV).
    """
    E = np.asarray(E_eV, dtype=float)
    # polyimide C22 H10 N2 O5: formula units per Ang^3 at rho = 1.42 g/cm^3
    n_f = 1.42 / 382.31 * 0.602214076
    mu_poly = sum(
        count * _mu_total_inv_ang([(el, n_f)], E)
        for el, count in (("C", 22), ("H", 10), ("N", 2), ("O", 5))
    )
    n_al = 2.70 / 26.982 * 0.602214076
    mu_al = _mu_total_inv_ang([("Al", n_al)], E)
    # Above the Henke ceiling (~30 keV) the window attenuation is unavailable
    # (NaN, and inf at E=0); the thin polymer/Al window is transparent to hard
    # X-rays, so treat an unavailable mu as zero -> QE -> grid_open rather than
    # NaN (which would clip the detected spectrum on a log plot). NB this
    # window-transmission model assumes the Si diode is fully absorbing, so it
    # OVERestimates QE above ~20 keV where the Si itself turns transparent -- use
    # the Timepix forward model (plot_timepix_detected) for a faithful hard-X-ray
    # detector response.
    mu_poly = np.nan_to_num(mu_poly, nan=0.0, posinf=0.0, neginf=0.0)
    mu_al = np.nan_to_num(mu_al, nan=0.0, posinf=0.0, neginf=0.0)
    return grid_open * np.exp(-mu_poly * polymer_nm * 10.0 - mu_al * al_nm * 10.0)


def eds_fwhm_eV(E_eV):
    """Oxford UltimMax 170 resolution fit, SI Eq. (16)."""
    return np.sqrt(2.52 * E_eV + 988.0)


def aperture_fwhm_eV(E_eV, beta, theta_obs_rad, dtheta_obs_rad):
    """Line broadening from the detector polar-angle span, SI Eq. (14)."""
    dE_dth = E_eV * beta * np.sin(theta_obs_rad) / (1.0 - beta * np.cos(theta_obs_rad))
    return 2.0 * np.sqrt(2.0 * np.log(2.0) / 3.0) * dE_dth * dtheta_obs_rad


def mosaic_fwhm_eV(E_eV, psi_rad, mosaic_fwhm_rad):
    """Line broadening from crystal mosaicity -- the INITIAL ANALYTIC model.

    A mosaic crystal is an incoherent ensemble of crystallites whose orientations
    are Gaussian-spread about the mean with a rocking-curve FWHM `mosaic_fwhm_rad`.
    Tilting a crystallite rotates its reciprocal vector g; only the NUMERATOR v.g of
    the resonance E_res = hbar c (v.g)/(1 - v.n) depends on g, so to first order the
    fractional line shift is dE/E = -tan(psi) dtheta, psi = angle(v, g). A Gaussian
    tilt of FWHM `mosaic_fwhm_rad` (in the longitudinal plane) therefore broadens the
    line by a Gaussian of energy FWHM

        FWHM_mosaic = E * |tan(psi)| * mosaic_fwhm_rad,

    added in quadrature with the EDS and aperture widths (results.store_result) and
    applied via the same convolve_detector pass -- the cheap analytic counterpart of
    a full Monte-Carlo average over crystallite orientations.

    Crude by construction: (i) it captures only the resonance-ENERGY shift, holding
    the amplitudes / lineshape weight fixed across the mosaic cone (good while the
    line stays narrow); (ii) the linearization diverges as psi -> 90 deg (g grazing
    the velocity), so the caller should cap the result (store_result clips it at E).
    The exact treatment is the per-orientation MC sum inside mc_spectrum (future
    work). psi is supplied by mosaic_psi_rad() at the nominal (unscattered) geometry.
    """
    return E_eV * np.abs(np.tan(psi_rad)) * mosaic_fwhm_rad


def mosaic_psi_rad(case, E_pk_eV):
    """psi = angle(beam velocity, g) [rad] of the reflection whose NOMINAL
    (unscattered-beam) resonance energy is nearest E_pk -- the representative
    geometry for the analytic mosaic broadening (mosaic_fwhm_eV). Mirrors how
    aperture_fwhm_eV uses the nominal theta_obs rather than the per-segment scattered
    directions. Returns None if no listed reflection radiates a positive line."""
    info = CRYSTALS[case["crystal"]]
    beam_dir, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    beta = beta_from_keV(case["E0_keV"])
    R = _orientation_R(info["lattice"], case.get("beam_uvw"), case.get("azimuth_rad", 0.0))
    denom = 1.0 - beta * float(beam_dir @ n_hat)  # g-independent (Doppler denominator)
    if denom <= 0.0:
        return None
    best = None
    for hkl in case["hkl_list"]:
        g_vec, g = reciprocal_g_vector(hkl, info["lattice"])
        if R is not None:
            g_vec = R @ g_vec
        bdotg = float(beam_dir @ g_vec)
        E_res = HBARC_EV_ANG * beta * bdotg / denom
        if E_res <= 0.0:
            continue
        psi = np.arccos(np.clip(bdotg / g, -1.0, 1.0))
        d = abs(E_res - E_pk_eV)
        if best is None or d < best[0]:
            best = (d, psi)
    return None if best is None else best[1]


def convolve_detector(E_grid_eV, spec, fwhm_eV):
    """
    Convolve a spectrum with a unit-area Gaussian of the given FWHM [eV].
    Output always matches the input length for ANY fwhm (np.convolve with
    mode="same" returned an OVERSIZED array whenever the kernel outgrew the
    spectrum -- large FWHM on a short grid; truncating the kernel instead
    distorts the lineshape). Edges are zero-padded: counts blurred past the
    grid ends are lost, consistent with a detector band edge. Requires a
    uniform energy grid.
    """
    from scipy.ndimage import gaussian_filter1d

    dE = E_grid_eV[1] - E_grid_eV[0]
    sigma_bins = fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0))) / dE
    return gaussian_filter1d(np.asarray(spec, dtype=float), sigma_bins, mode="constant", cval=0.0)
