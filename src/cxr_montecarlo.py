"""
cxr_montecarlo.py

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
     Amplitudes are the same Eqs. (13)/(14) as cxr_feranchuk_spence, with
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
from functools import lru_cache

import numpy as np

try:
    import cupy as cp
    _GPU = True
    xp = cp
    print("Using GPU")
except ImportError:
    _GPU = False
    xp = np
    print("No GPU found, or cupy not installed!\nFalling back to CPU execution.")

from cxr_feranchuk_spence import (
    ALPHA_FS, HBARC_EV_ANG, M_E_EV, CRYSTALS,
    chi_g, U_g, absorption_length_ang, reciprocal_g_vector,
    _rotation_between, _direct_lattice_vectors,
)


def _to_cpu(a):
    """Move array to CPU (numpy). No-op if already numpy."""
    if _GPU and isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)

MOTT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "mott_transport_cross_sections")
A0_SQ_CM2 = 2.8002852e-17     # Bohr radius squared [cm^2] (NIST SRD 64 unit)

# ---- element data for transport ---------------------------------------------
# A [g/mol], J = mean ionization potential [keV] (Berger-Seltzer values)
TRANSPORT_ELEMENTS = {
    "C":  {"Z": 6,  "A": 12.011, "J_keV": 0.078},
    "Si": {"Z": 14, "A": 28.085, "J_keV": 0.173},
    "Ge": {"Z": 32, "A": 72.630, "J_keV": 0.350},
    "Se": {"Z": 34, "A": 78.971, "J_keV": 0.348},
    "S":  {"Z": 16, "A": 32.06,  "J_keV": 0.180},
    "Mo": {"Z": 42, "A": 95.95,  "J_keV": 0.424},
    "W":  {"Z": 74, "A": 183.84, "J_keV": 0.727},
    "Zr": {"Z": 40, "A": 91.224, "J_keV": 0.393},
    "Hf": {"Z": 72, "A": 178.49, "J_keV": 0.705},
    "Pt": {"Z": 78, "A": 195.08, "J_keV": 0.790},
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
    return (3.0e-18 * Z**1.7
            / (E_keV + 0.005 * Z**1.7 * np.sqrt(E_keV)
               + 0.0007 * Z**2 / np.sqrt(E_keV)))


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
    return 10.0**(0.5 * (lo + hi))


@lru_cache(maxsize=None)
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
            f"elastic_model='sr'.")
    E, sig = [], []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                E.append(float(parts[1]))
                sig.append(float(parts[2]))
    return np.array(E), np.array(sig) * A0_SQ_CM2


@lru_cache(maxsize=None)
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
            alpha = 10.0**np.interp(np.log10(E_keV * 1e3), logE, logA)
        except FileNotFoundError:
            print(f"transport: no Mott transport table for {element!r}; using "
                  f"the analytic screened-Rutherford screening for it instead.")
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
    return (-7.85e-4 * rho_g_cm3 * Z / (A * E_keV)
            * np.log(1.166 * (E_keV + k * J_keV) / J_keV))


def _normalize_composition(element, n_atoms_per_ang3, composition):
    """
    Accept either the single-element API (element=, n_atoms_per_ang3=) or a
    compound composition=[(element, number_density_1_per_Ang3), ...];
    return the latter form.
    """
    if composition is not None:
        return [(el, float(n)) for el, n in composition]
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
            1.166 * (E_keV + k * p["J_keV"]) / p["J_keV"])
    return -7.85e-4 / E_keV * total


def _mu_total_inv_ang(comp, E_eV):
    """Total linear attenuation 1/L_abs [1/Angstrom] summed over elements.

    absorption_length_ang (from cxr_feranchuk_spence) is CPU-only, so the sum
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
        return cp.asarray(mu)
    return mu


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
    out = (cos_t[:, None] * d
           + (sin_t * np.cos(phi))[:, None] * u
           + (sin_t * np.sin(phi))[:, None] * w)
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
    normal_lab = np.array([st * np.cos(tilt_azim_rad),
                           st * np.sin(tilt_azim_rad), ct])
    R = _rotation_between(np.array([0.0, 0.0, 1.0]), normal_lab)
    beam_dir = R.T @ np.array([0.0, 0.0, 1.0])
    n_hat = R.T @ np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    return beam_dir, n_hat


def simulate_trajectories(E0_keV, Ne, thickness_ang, element="C",
                          n_atoms_per_ang3=None, E_cut_keV=5.0,
                          seed=0, max_steps=20000, elastic_model="mott",
                          beam_dir=None, composition=None):
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

    Returns dict of per-segment arrays:
      "r_mid" (M,3) [Ang], "v_hat" (M,3), "L_ang" (M,), "E_keV" (M,)
    and diagnostics: "n_backscattered", "n_transmitted", "n_stopped".
    """
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
    Zs = [TRANSPORT_ELEMENTS[el]["Z"] for el, _ in comp]
    n_cm3s = [n_i * 1e24 for _, n_i in comp]

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

    seg_mid, seg_dir, seg_len, seg_E = [], [], [], []

    # Event-driven loop: every iteration is one FREE FLIGHT + one ELASTIC
    # COLLISION for every still-alive electron, executed in lockstep (pure
    # vectorization -- electrons are independent, so synchronizing them is
    # exact). There is no time step and no discretization parameter: flight
    # lengths are sampled from the physical free-path distribution.
    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)     # indices of electrons still in play
        Ea = E[idx]                     # their kinetic energies [keV]

        # -- 1. distance to the next elastic collision -------------------------
        # Exponential free path: P(s) = exp(-s/lambda)/lambda, with the total
        # rate additive over elements: 1/lambda = sum_i n_i sigma_i(E).
        rates = []                       # per-element scattering rates [1/cm]
        for (el_i, _), Z_i, n_i in zip(comp, Zs, n_cm3s):
            if elastic_model == "mott":
                sig_i = _sigma_browning_cm2(Z_i, Ea)   # Browning fit to Mott
            elif elastic_model == "sr":
                a = _alpha_sr_joy(Z_i, Ea)
                sig_i = (5.21e-21 * Z_i**2 / Ea**2 * 4.0 * np.pi / (a * (1.0 + a))
                         * ((Ea + 511.0) / (Ea + 1024.0))**2)
            else:
                raise ValueError("elastic_model must be 'mott' or 'sr'")
            rates.append(n_i * sig_i)
        rates = np.array(rates)                       # (n_elements, m)
        lam_ang = 1e8 / rates.sum(axis=0)             # mean free path [Ang]
        step = -lam_ang * np.log(rng.random(idx.size))   # sampled flight [Ang]

        d = dirs[idx]                   # current unit direction of each electron
        p = pos[idx]                    # current position [Ang]

        # -- 2. slab-boundary truncation ----------------------------------------
        # The flight is the ray r(s') = p + s' d. It crosses the entrance face
        # z=0 at s'_top = p_z / (-d_z)   (only reachable if d_z < 0), and the
        # back face z=thickness at s'_bot = (t - p_z)/d_z (only if d_z > 0).
        # If the sampled flight overshoots a face, truncate it there and flag
        # the electron as exited (vacuum outside -> no re-entry).
        exit_top = (d[:, 2] < 0) & (p[:, 2] + step * d[:, 2] < 0.0)
        exit_bot = (d[:, 2] > 0) & (p[:, 2] + step * d[:, 2] > thickness_ang)
        s_top = np.where(d[:, 2] < 0, p[:, 2] / (-d[:, 2] + 1e-300), np.inf)
        s_bot = np.where(d[:, 2] > 0, (thickness_ang - p[:, 2]) / (d[:, 2] + 1e-300), np.inf)
        step = np.where(exit_top, s_top, step)
        step = np.where(exit_bot, s_bot, step)

        # -- 3. record the segment (the radiation source list) ------------------
        # midpoint -> escape-absorption path; direction -> v.g, v.n in the
        # amplitudes; length -> interaction time t_L; START energy -> beta
        # (biases beta high by at most the per-segment loss, <0.5%).
        seg_mid.append(p + 0.5 * step[:, None] * d)
        seg_dir.append(d.copy())
        seg_len.append(step)
        seg_E.append(Ea.copy())

        # -- 4. advance: straight line + continuous slowing-down ----------------
        # Positions move the full flight; energy drains deterministically along
        # it (CSDA: Joy-Luo modified Bethe, additive over elements; no
        # straggling, no fast secondaries).
        pos[idx] = p + step[:, None] * d
        E[idx] = Ea + _dEds_compound(comp, Ea) * step

        # -- 5. kill exited / exhausted electrons --------------------------------
        died = exit_top | exit_bot | (E[idx] < E_cut_keV)
        n_back += int(exit_top.sum())   # exited the entrance face (backscattered)
        n_trans += int(exit_bot.sum())  # punched through the back face
        alive[idx[died]] = False

        # -- 6. elastic collision: new direction for the survivors ---------------
        # The scattering ELEMENT is chosen with probability n_i sigma_i / sum
        # (energy-dependent, evaluated at the pre-flight energy); then the
        # polar angle comes from that element's screened-Rutherford inversion
        # (alpha(E) Mott-calibrated or analytic). Azimuth uniform; energy
        # unchanged (elastic).
        srv_mask = ~died
        srv = idx[srv_mask]
        if srv.size:
            cos_t = np.empty(srv.size)
            if len(comp) == 1:
                cos_t = _sample_cos_theta(Zs[0], E[srv], rng,
                                          elastic_model, comp[0][0])
            else:
                p_el = rates[:, srv_mask] / rates[:, srv_mask].sum(axis=0)
                u = rng.random(srv.size)
                cum = np.cumsum(p_el, axis=0)
                which = (u[None, :] > cum).sum(axis=0)    # element index
                for i_el, (el_i, _) in enumerate(comp):
                    m = which == i_el
                    if m.any():
                        cos_t[m] = _sample_cos_theta(Zs[i_el], E[srv][m], rng,
                                                     elastic_model, el_i)
            phi = 2.0 * np.pi * rng.random(srv.size)
            dirs[srv] = _rotate_directions(dirs[srv], cos_t, phi)

    return {
        "r_mid": np.concatenate(seg_mid),
        "v_hat": np.concatenate(seg_dir),
        "L_ang": np.concatenate(seg_len),
        "E_keV": np.concatenate(seg_E),
        "n_backscattered": n_back,
        "n_transmitted": n_trans,
        "n_stopped": int(Ne - n_back - n_trans),
        "Ne": Ne,
        "thickness_ang": thickness_ang,
    }


# ---- segment-sum CXR spectrum ------------------------------------------------
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


def mc_spectrum(segments, E_grid_eV, crystal="graphite",
                hkl_list=((0, 0, 2), (0, 0, -2)),
                theta_obs_rad=np.deg2rad(119.0),
                B_ang2=0.8, use_henke=True,
                absorber_element="C", chunk=40000, n_hat=None,
                composition=None, beam_uvw=None, azimuth_rad=0.0,
                sinc_cutoff=None, components=False):
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
    """
    info = CRYSTALS[crystal]
    n_atoms = len(info["basis"]) / info["V_cell"]
    abs_comp = _normalize_composition(absorber_element, n_atoms, composition)

    # crystal orientation: rotation applied to all reciprocal vectors
    R_orient = None
    if beam_uvw is not None:
        u, v, w = np.asarray(beam_uvw, dtype=float)
        a1, a2, a3 = _direct_lattice_vectors(info["lattice"])
        axis = u * a1 + v * a2 + w * a3
        R_orient = _rotation_between(axis / np.linalg.norm(axis),
                                     np.array([0.0, 0.0, 1.0]))
    if azimuth_rad:
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        R_orient = Rz if R_orient is None else Rz @ R_orient
    thickness = segments["thickness_ang"]
    Ne = segments["Ne"]

    if n_hat is None:
        n_hat = np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    else:
        n_hat = np.asarray(n_hat, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
    E_grid = xp.asarray(E_grid_eV, dtype=float)
    spec = xp.zeros(E_grid.size)
    spec_pxr = xp.zeros(E_grid.size)
    spec_cbs = xp.zeros(E_grid.size)

    seg_E = xp.asarray(segments["E_keV"])
    seg_v = xp.asarray(segments["v_hat"])
    seg_L = xp.asarray(segments["L_ang"])
    seg_r = xp.asarray(segments["r_mid"])
    beta_all = beta_from_keV(seg_E)                   # speed/c per segment
    v_all = beta_all[:, None] * seg_v                 # velocity vectors (c=1)

    for hkl in hkl_list:
        # reciprocal vector in the sample frame: construction frame by
        # default ([001] along the slab normal), rotated if beam_uvw given
        g_vec, g = reciprocal_g_vector(hkl, info["lattice"])
        if R_orient is not None:
            g_vec = R_orient @ g_vec
        # sigma/pi polarization unit vectors: constants per reflection, since
        # both the detector direction n_hat and g are fixed (only v varies)
        e_s, e_p = _polarization_pair(n_hat, g_vec)
        g_vec_d = xp.asarray(g_vec)
        n_hat_d = xp.asarray(n_hat)

        # -- 1. per-segment resonance energy (Eq. 10) ---------------------------
        #   omega_res = v.g / (1 - v.n)   [1/Ang]   (>0 required to radiate)
        v_dot_g = v_all @ g_vec_d
        denom = 1.0 - v_all @ n_hat_d     # the Doppler-like denominator
        omega_res = v_dot_g / denom
        E_res = HBARC_EV_ANG * omega_res  # -> eV

        # -- 2. drop segments whose line misses the spectral window -------------
        # (pad by 20% so sinc tails that reach into the window still count)
        pad = 0.2 * (E_grid[-1] - E_grid[0])
        keep = (E_res > float(E_grid[0] - pad)) & (E_res > 10.0) & (E_res < E_grid[-1] + pad)
        if not keep.any():
            continue
        idx = xp.flatnonzero(keep)

        E_r = E_res[idx]                  # line energy per kept segment [eV]
        om = omega_res[idx]               # same in 1/Ang
        v = v_all[idx]                    # velocity vectors
        beta = beta_all[idx]
        t_L = seg_L[idx] / beta           # interaction time [Ang] (c=1)
        dnm = denom[idx]
        vdg = v_dot_g[idx]

        # -- 3. couplings evaluated AT each segment's resonance energy ----------
        # (amplitudes vary slowly across the narrow line; freezing them at
        # E_res is accurate to the linewidth/E_res level)
        E_r_cpu = _to_cpu(E_r)
        chi = xp.asarray(chi_g(crystal, hkl, E_r_cpu, B_ang2, use_henke))
        eUg_over_m = xp.asarray(U_g(crystal, hkl, E_r_cpu, B_ang2, use_henke)) / M_E_EV

        # -- 4. photon kinematics per segment ------------------------------------
        k_vec = om[:, None] * n_hat_d     # photon wavevector omega * n_hat
        kg_vec = k_vec + g_vec_d          # diffracted wavevector k + g
        kg2 = xp.einsum("ij,ij->i", kg_vec, kg_vec)
        detuning = kg2 - om**2            # PXR denominator (~g^2, never small)
        k_dot_g = k_vec @ g_vec_d

        # -- 5. Eq. (13) + relativistic Eq. (14) amplitudes, per segment ----------
        # CBS braced product {a;b} = a.b - (a.v)(b.v) and 1/gamma prefactor
        # (Zhai SI Eq. 6); v here is the SEGMENT velocity, so k.v = omega(1-dnm)
        gamma = 1.0 / xp.sqrt(1.0 - beta**2)
        k_dot_v = om * (1.0 - dnm)
        A2 = xp.zeros(idx.size)
        A2_pxr = xp.zeros(idx.size)
        A2_cbs = xp.zeros(idx.size)
        for e in (e_s, e_p):              # sum |A|^2 over both polarizations
            e_d = xp.asarray(e)
            g_dot_e = g_vec_d @ e_d       # scalar (e fixed per reflection)
            v_dot_e = v @ e_d
            v_dot_kg = xp.einsum("ij,ij->i", v, kg_vec)
            A_PXR = chi / detuning * (v_dot_kg * g_dot_e - om**2 * v_dot_e)
            braced_ge = g_dot_e - vdg * v_dot_e
            braced_kg = k_dot_g - k_dot_v * vdg
            A_CBS = (-eUg_over_m / (gamma * vdg)
                     * (braced_ge + v_dot_e * braced_kg / vdg))
            A2 += xp.abs(A_PXR + A_CBS) ** 2
            A2_pxr += xp.abs(A_PXR) ** 2
            A2_cbs += xp.abs(A_CBS) ** 2

        # -- 6. Beer-Lambert escape factor from the segment midpoint -------------
        # straight path along n_hat to whichever slab face the photon exits
        z_mid = seg_r[idx, 2]
        if n_hat[2] < 0:
            L_esc = z_mid / (-n_hat[2])               # out the entrance face
        else:
            L_esc = (thickness - z_mid) / n_hat[2]    # out the back face
        T_abs = xp.exp(-L_esc * _mu_total_inv_ang(abs_comp, E_r))

        # -- 7. accumulate the finite-segment lineshape ---------------------------
        # d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |A|^2 t_L^2
        #                  * sinc^2[(1 - v.n)(omega - omega_res) t_L / 2] * T_abs
        # weight = everything except the sinc^2; a_width converts (E - E_res)
        # to the sinc argument: P t_L = a_width * (E - E_res).
        pref = (ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG)
                * t_L**2 * T_abs)
        weight = pref * A2
        targets = [(weight, spec)]
        if components:
            targets += [(pref * A2_pxr, spec_pxr), (pref * A2_cbs, spec_cbs)]
        a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
        good = xp.isfinite(weight) & (weight > 0)

        if sinc_cutoff is None:
            for j0 in range(0, idx.size, chunk):
                sl = slice(j0, min(j0 + chunk, idx.size))
                m = good[sl]
                if not m.any():
                    continue
                x = (a_width[sl][m, None] * (E_grid[None, :] - E_r[sl][m, None])
                     / xp.pi)
                S = xp.sinc(x) ** 2
                for w, tgt in targets:
                    tgt += w[sl][m] @ S
        else:
            dE = E_grid[1] - E_grid[0]
            order = xp.argsort(E_r)
            blk = 8192
            for j0 in range(0, order.size, blk):
                sel = order[j0:j0 + blk]
                sel = sel[good[sel]]
                if sel.size == 0:
                    continue
                half = sinc_cutoff / a_width[sel]
                lo = float(_to_cpu((E_r[sel] - half).min()))
                hi = float(_to_cpu((E_r[sel] + half).max()))
                i0 = max(int((lo - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))), 0)
                i1 = min(int((hi - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))) + 2, E_grid.size)
                if i1 <= i0:
                    continue
                x = (a_width[sel][:, None]
                     * (E_grid[None, i0:i1] - E_r[sel][:, None]) / xp.pi)
                S = xp.sinc(x) ** 2
                for w, tgt in targets:
                    tgt[i0:i1] += w[sel] @ S

    if components:
        return _to_cpu(spec / Ne), _to_cpu(spec_pxr / Ne), _to_cpu(spec_cbs / Ne)
    return _to_cpu(spec / Ne)


# ---- bremsstrahlung background -------------------------------------------------
R_E_CM2 = 7.9407877e-26       # classical electron radius squared [cm^2]


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
    mc2 = 510.99895                                    # keV
    T_i = xp.asarray(T_keV, dtype=float)[:, None]
    k = xp.asarray(k_eV, dtype=float)[None, :] / 1e3   # keV
    T_f = T_i - k
    ok = (T_f > 1e-6) & (k > 0.0)   # k>0: no photon (and no 1/k blowup) at k=0
    T_f = xp.where(ok, T_f, 1e-6)

    p_i = xp.sqrt(T_i * (T_i + 2.0 * mc2)) / mc2
    p_f = xp.sqrt(T_f * (T_f + 2.0 * mc2)) / mc2
    beta_i = p_i / (1.0 + T_i / mc2)
    beta_f = p_f / (1.0 + T_f / mc2)

    born_log = xp.log((p_i + p_f) / xp.maximum(p_i - p_f, 1e-30))
    elwert = (beta_i / beta_f
              * (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_i))
              / (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_f)))

    dsig = (16.0 / 3.0 * ALPHA_FS * R_E_CM2 * Z**2
            / xp.maximum(k * 1e3, 1e-30) / p_i**2 * born_log * elwert)   # per eV
    return xp.where(ok, dsig, 0.0)


def mc_brem_spectrum(
    segments, E_grid_eV, element="C", n_atoms_per_ang3=None,
    theta_obs_rad=np.deg2rad(119.0), n_hat=None, chunk=20000,
    composition=None
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

    E_grid = xp.asarray(E_grid_eV, dtype=float)
    mu = _mu_total_inv_ang(comp, E_grid)               # (NE,) [1/Ang]
    # The Henke absorption tables span ~20 eV - 30 keV; outside that the wide
    # brem grid gets NaN (above 30 keV) or inf (at E=0), and a single bad bin
    # makes brem_wide -- and its integrated count rate -- NaN. Hard X-rays escape
    # essentially unattenuated, so treat an unavailable mu as zero (transparent).
    mu = xp.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    seg_r = xp.asarray(segments["r_mid"])
    seg_L = xp.asarray(segments["L_ang"])
    seg_E = xp.asarray(segments["E_keV"])
    z_mid = seg_r[:, 2]
    if n_hat[2] < 0:
        L_esc = z_mid / (-n_hat[2])
    else:
        L_esc = (thickness - z_mid) / n_hat[2]

    spec = xp.zeros(E_grid.size)
    M = seg_E.size
    for j0 in range(0, M, chunk):
        sl = slice(j0, min(j0 + chunk, M))
        T_abs = xp.exp(-L_esc[sl][:, None] * mu[None, :])
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
        np.deg2rad(case.get("tilt_azim_deg", 0.0)))
    segs = simulate_trajectories(
        case["E0_keV"], case["Ne"], case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=case["seed"], beam_dir=beam)
    segs_b = simulate_trajectories(
        case["E0_keV"], case["Ne_brem"], case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        seed=case["seed"] + 1, beam_dir=beam)
    return dict(E_grid=E_grid, E_brem=E_brem, n_hat=n_hat, segs=segs, segs_b=segs_b)


def _spectrum_case(case, tp):
    """GPU phase of run_case: line spectrum + brem from the already-transported
    segments ``tp`` (from _transport_case). Runs in the main process, so only one
    CUDA context ever touches the device."""
    E_grid, E_brem, n_hat = tp["E_grid"], tp["E_brem"], tp["n_hat"]
    segs, segs_b = tp["segs"], tp["segs_b"]
    spec = mc_spectrum(
        segs, E_grid, crystal=case["crystal"], hkl_list=case["hkl_list"],
        n_hat=n_hat, B_ang2=case["B_ang2"], composition=case["composition"],
        beam_uvw=case.get("beam_uvw"), azimuth_rad=case.get("azimuth_rad", 0.0),
        sinc_cutoff=case.get("sinc_cutoff"), chunk=case.get("spec_chunk") or 40000)
    brem_wide = mc_brem_spectrum(segs_b, E_brem, composition=case["composition"],
                                 n_hat=n_hat, chunk=case.get("brem_chunk") or 20000)
    brem = np.interp(E_grid, E_brem, brem_wide)   # brem under the lines (line grid)
    # Hand this case's GPU scratch back to the OS so the CuPy memory pool can't
    # accumulate (and fragment) across a long sweep until it fills the card.
    if _GPU:
        cp.get_default_memory_pool().free_all_blocks()
    return dict(E_grid=E_grid, spec=spec, brem=brem,
                E_grid_brem=E_brem, brem_wide=brem_wide,
                eta=segs["n_backscattered"] / segs["Ne"],
                n_segments=int(segs["L_ang"].size),
                crystal=case["crystal"], E0_keV=case["E0_keV"])


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
            os.nice(10)                       # POSIX fallback
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
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"

    # ---- GPU: pipeline CPU transport (worker pool) behind the serial GPU ------
    if _GPU:
        if max_workers == 0:
            return _serial()
        if max_workers is None:
            ncpu = os.process_cpu_count() or os.cpu_count() or 8
            nw = max(2, min(n, ncpu // 4, 8))
        else:
            nw = min(max_workers, n)
        if nw < 2:
            return _serial()
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor
        prefetch = nw + 2                          # keep the transport pool ahead
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as ex:
            inflight = {i: ex.submit(_transport_case, cases[i])
                        for i in range(min(prefetch, n))}
            for i in _maybe_bar(range(n)):
                tp = inflight.pop(i).result()         # transport (already overlapped)
                j = i + prefetch
                if j < n:
                    inflight[j] = ex.submit(_transport_case, cases[j])
                out = _spectrum_case(cases[i], tp)    # GPU, THIS process only
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
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_worker_init) as ex:
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
                continue                      # header / text line
    if not rows:
        raise ValueError(f"no numeric (E, intensity) rows found in {path}")
    arr = np.array(sorted(rows))
    return np.interp(np.asarray(E_grid_eV, dtype=float), arr[:, 0], arr[:, 1],
                     left=0.0, right=0.0)


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
    mu_poly = sum(count * _mu_total_inv_ang([(el, n_f)], E)
                  for el, count in (("C", 22), ("H", 10), ("N", 2), ("O", 5)))
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
    return grid_open * np.exp(-mu_poly * polymer_nm * 10.0
                              - mu_al * al_nm * 10.0)


def eds_fwhm_eV(E_eV):
    """Oxford UltimMax 170 resolution fit, SI Eq. (16)."""
    return np.sqrt(2.52 * E_eV + 988.0)


def aperture_fwhm_eV(E_eV, beta, theta_obs_rad, dtheta_obs_rad):
    """Line broadening from the detector polar-angle span, SI Eq. (14)."""
    dE_dth = E_eV * beta * np.sin(theta_obs_rad) / (1.0 - beta * np.cos(theta_obs_rad))
    return 2.0 * np.sqrt(2.0 * np.log(2.0) / 3.0) * dE_dth * dtheta_obs_rad


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
    return gaussian_filter1d(np.asarray(spec, dtype=float), sigma_bins,
                             mode="constant", cval=0.0)
