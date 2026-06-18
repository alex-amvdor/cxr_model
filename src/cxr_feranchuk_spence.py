"""
cxr_feranchuk_spence.py

Coherent X-ray radiation (PXR + coherent bremsstrahlung) from NONRELATIVISTIC
electrons in a thin/absorption-limited crystal.

Implements Feranchuk, Ulyanenkov, Harada & Spence, Phys. Rev. E 62, 4225 (2000):
  - Eq. (3):  chi_g  (non-resonant polarizability, real F(g))
  - Eq. (4):  U_g    (crystal potential, CBS coupling, (Z-F)/g^2)
  - Eq. (10): resonant frequencies omega_n(theta)
  - Eqs. (13)/(14): PXR and CBS amplitudes
  - Eq. (16): delta_g = A_PXR / A_CBS  (depends only on charge distribution)
  - Eq. (12)+(9): absolute photons/electron, absorption-limited
  - Eq. (18): CXR-to-bremsstrahlung-background ratio

Conventions:
  theta_B  = angle between v0 and the PLANE NORMAL (Feranchuk-Spence Eq. 10),
             NOT the beam-to-plane Bragg angle. (theta_B^here = 90deg - theta_Bragg)
  Energies eV, lengths Angstrom, angles radians.
  hbar = c = 1 is used in the paper; we restore units at the boundaries.

Crystal structures are loaded from crystal_structures.toml (same directory)
into the CRYSTALS dict; see that file for the format.

Depends on atomic_form_factors.py (cromer_mann_f0, atomic_form_factor, Z_TABLE).

CAVEATS:
  * Eq. (3) is the NON-RESONANT (far-from-edge) approximation: F(g)=f0(g), drops
    f' and f''. Fine for carbon at 1-4 keV (K-edge 284 eV). For Si (K 1.84 keV),
    Ge (L 1.2-1.4 keV), Mo/Se (L 1.4-2.9 keV) with a line near an edge, use
    use_henke=True so chi_g and the delta_g ratio carry f'(omega).
  * Thin/absorption-limited regime (L_z <= L_abs), perturbative in U_g (Eq. 6).
    For bulk-stopping targets, set L_z = L_abs (escape-depth-limited; the
    saturated limit of Eq. 9). Slowing-down of the electron is NOT integrated.
  * Nonrelativistic isotropic-angular limit: the weak angular dependence of the
    polarization terms is neglected (paper, p.4). No sharp PXR lobes here.
"""

import tomllib
from pathlib import Path

import numpy as np

from atomic_form_factors import (
    cromer_mann_f0,
    atomic_form_factor,
    henke_dispersion,
    Z_TABLE,
)

# ---- constants --------------------------------------------------------------
HC_EV_ANG = 12398.4198  # h c [eV*Angstrom]
HBARC_EV_ANG = 1973.269804  # hbar c [eV*Angstrom]
ALPHA_FS = 1.0 / 137.035999
M_E_EV = 510998.95  # electron rest energy [eV]
R_E_ANG = 2.8179403e-5  # classical electron radius [Angstrom]
E2_EV_ANG = (
    ALPHA_FS * HBARC_EV_ANG
)  # e^2 (Gaussian) = alpha hbar c = 14.3996 [eV*Angstrom]

# elements whose edges fall in the soft-x-ray band -> force Henke correction
_EDGE_PRONE = {"Si", "Ge", "Mo", "Se"}


# ---- lattice geometry --------------------------------------------------------
def _direct_lattice_vectors(lattice):
    """
    Direct lattice vectors (a1, a2, a3) [Angstrom] from a lattice dict:
        {"system": "cubic",       "a": a}
        {"system": "tetragonal",  "a": a, "c": c}
        {"system": "orthorhombic","a": a, "b": b, "c": c}
        {"system": "hexagonal",   "a": a, "c": c}
        {"system": "general",     "a": a, "b": b, "c": c,
         "alpha": alpha, "beta": beta, "gamma": gamma}   # angles in DEGREES
    """
    sysname = lattice["system"]
    if sysname == "cubic":
        a = lattice["a"]
        return (
            np.array([a, 0.0, 0.0]),
            np.array([0.0, a, 0.0]),
            np.array([0.0, 0.0, a]),
        )
    if sysname == "tetragonal":
        a, c = lattice["a"], lattice["c"]
        return (
            np.array([a, 0.0, 0.0]),
            np.array([0.0, a, 0.0]),
            np.array([0.0, 0.0, c]),
        )
    if sysname == "orthorhombic":
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        return (
            np.array([a, 0.0, 0.0]),
            np.array([0.0, b, 0.0]),
            np.array([0.0, 0.0, c]),
        )
    if sysname == "hexagonal":
        a, c = lattice["a"], lattice["c"]
        # standard hexagonal: gamma = 120 deg between a1 and a2
        return (
            np.array([a, 0.0, 0.0]),
            np.array([-a / 2.0, a * np.sqrt(3) / 2.0, 0.0]),
            np.array([0.0, 0.0, c]),
        )
    if sysname == "general":
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        al = np.radians(lattice["alpha"])
        be = np.radians(lattice["beta"])
        ga = np.radians(lattice["gamma"])
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
        return a1, a2, np.array([cx, cy, cz])
    raise ValueError(f"unknown crystal system '{sysname}'")


def _cross3(u, v):
    """Cross product for plain 3-vectors (much faster than np.cross)."""
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


_RECIP_BASIS_CACHE = {}


def _reciprocal_basis(lattice):
    """Rows are b1, b2, b3 [1/Angstrom]; cached per lattice (hot path)."""
    key = tuple(sorted(lattice.items()))
    B = _RECIP_BASIS_CACHE.get(key)
    if B is None:
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        V = np.dot(a1, _cross3(a2, a3))
        B = (
            2.0
            * np.pi
            * np.array([_cross3(a2, a3), _cross3(a3, a1), _cross3(a1, a2)])
            / V
        )
        _RECIP_BASIS_CACHE[key] = B
    return B


def reciprocal_g_vector(hkl, lattice):
    """
    Reciprocal lattice vector g = h b1 + k b2 + l b3 for any crystal system.
    Returns (g_vec [1/Angstrom, 3-vector], g_mag [1/Angstrom]).
    Convention: |g| = 2*pi/d_hkl, matching the rest of the module.
    """
    g_vec = np.asarray(hkl, dtype=float) @ _reciprocal_basis(lattice)
    return g_vec, np.linalg.norm(g_vec)


# ---- crystal database (crystal_structures.toml) ------------------------------
def load_crystals(
    path=Path(__file__).parent.parent / "data" / "crystal_structures.toml",
):
    """
    Load the crystal database. Each entry becomes
        {"lattice": {...}, "basis": [(element, frac_pos), ...], "V_cell": float}
    with V_cell [Angstrom^3] computed from the lattice vectors.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    crystals = {}
    for name, spec in raw.items():
        lattice = {key: val for key, val in spec.items() if key != "basis"}
        basis = [
            (atom["element"], np.array(atom["pos"], dtype=float))
            for atom in spec["basis"]
        ]
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        crystals[name] = {
            "lattice": lattice,
            "basis": basis,
            "V_cell": float(np.dot(a1, np.cross(a2, a3))),
        }
    return crystals


CRYSTALS = load_crystals()


# ---- kinematics -------------------------------------------------------------
def beta_from_Ee(Ee_eV):
    g = 1.0 + Ee_eV / M_E_EV
    return np.sqrt(1.0 - 1.0 / g**2)


def g_mag(d_ang):
    """|g| = 2 pi / d  [1/Angstrom]."""
    return 2.0 * np.pi / d_ang


def omega_n(d_ang, beta, theta_B_normal, n=1, eps0=1.0):
    """
    Resonant photon energy [eV], Feranchuk-Spence Eq. (10):
        omega_n = 2 pi v0 cos(theta_B) n / [ d (1 - v0 cos(theta)) ]
    Here theta_B is the angle to the PLANE NORMAL and theta is the observation
    angle to v0. At the n-th harmonic. We write it in energy units via hc/d.
    NOTE: paper uses theta in the denominator (observation) and theta_B
    (to normal) in the numerator; for the symmetric reflex these are linked by
    the Bragg condition. Pass theta_B_normal = 90deg - theta_Bragg_from_planes.
    """
    # Compact symmetric form: omega = (hc/d) * beta sin(theta_Bragg)
    #                                 / (1 - sqrt(eps0) beta cos Omega)
    # with theta_Bragg = 90deg - theta_B_normal, Omega = 2 theta_Bragg.
    theta_bragg = 0.5 * np.pi - theta_B_normal
    Omega = 2.0 * theta_bragg
    return (
        (HC_EV_ANG / d_ang)
        * n
        * beta
        * np.sin(theta_bragg)
        / (1.0 - np.sqrt(eps0) * beta * np.cos(Omega))
    )


# ---- structure & Debye-Waller ----------------------------------------------
def debye_waller(g_invang, B_ang2):
    """Amplitude Debye-Waller factor exp(-W), W = B (sin(theta)/lambda)^2
    = B (g/4pi)^2, for a tabulated B-factor [Angstrom^2] (B = 8 pi^2 <u_x^2>).
    Intensities carry exp(-2W) = the square of this."""
    s = g_invang / (4.0 * np.pi)
    return np.exp(-B_ang2 * s**2)


def _atom_F(element, g, photon_E_eV, use_henke):
    """
    Per-atom form factor policy: complex Henke-corrected f0+f'+if'' for
    edge-prone elements (or when use_henke is set), else non-resonant
    Cromer-Mann f0 per Eq. (3). Always returns complex.
    """
    if use_henke or element in _EDGE_PRONE:
        return atomic_form_factor(element, g, photon_E_eV)
    return cromer_mann_f0(element, g) + 0.0j


def _basis_F(basis, g, photon_E_eV, use_henke):
    """Form factor per basis atom, computed once per unique element."""
    cache = {}
    for el, _ in basis:
        if el not in cache:
            cache[el] = _atom_F(el, g, photon_E_eV, use_henke)
    return [cache[el] for el, _ in basis]


def structure_factor(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """
    S(g) = sum_i F_i(g) exp(i g . R_i) exp(-W_i), Eq. (3).
    Returns complex S(g) and |g| [1/Angstrom].
    F_i is f0 (non-resonant) unless use_henke, then f0+f' (+ i f'').
    """
    info = CRYSTALS[crystal]
    hkl = np.asarray(hkl, dtype=float)
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    dwf = debye_waller(g, B_ang2)
    S = 0.0 + 0.0j
    for (el, R), F in zip(
        info["basis"], _basis_F(info["basis"], g, photon_E_eV, use_henke)
    ):
        phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))
        S += F * phase * dwf
    return S, g


# ---- couplings: chi_g (PXR) and U_g (CBS) ----------------------------------
def chi_g(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """
    Polarizability Fourier component, Eq. (3):
        chi_g = -(4 pi e^2 / m omega^2) * S(g)/V * exp(-W)
    In the direct electron-density form with classical electron radius r_e:
        chi_g = - r_e lambda^2 / (pi V_cell) * S(g)     [dimensionless]
    """
    S, _ = structure_factor(crystal, hkl, photon_E_eV, B_ang2, use_henke)
    lam = HC_EV_ANG / photon_E_eV  # wavelength [Angstrom]
    return -R_E_ANG * lam**2 / (np.pi * CRYSTALS[crystal]["V_cell"]) * S


def U_g(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """
    Crystal-potential Fourier component (CBS coupling), Eq. (4), folded with
    the 1/V of Eq. (14) and one electron charge:
        e U_g / V = 4 pi e^2 sum_i exp(i g Ri) (Z_i - F_i)/g^2 exp(-W) / V
    with e^2 = alpha hbar c = 14.3996 eV*Angstrom (Gaussian). Returned in eV;
    divide by m_e c^2 to get the dimensionless e U_g / (m V) of Eq. (14).
    """
    info = CRYSTALS[crystal]
    hkl = np.asarray(hkl, dtype=float)
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    dwf = debye_waller(g, B_ang2)
    acc = 0.0 + 0.0j
    for (el, R), F in zip(
        info["basis"], _basis_F(info["basis"], g, photon_E_eV, use_henke)
    ):
        phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))
        acc += phase * (Z_TABLE[el] - F.real) / g**2 * dwf
    return 4.0 * np.pi * E2_EV_ANG * acc / info["V_cell"]


def delta_g(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """
    A_PXR / A_CBS, Feranchuk-Spence Eq. (16):
        delta_g = sum_i F_i exp(i g Ri) / sum_i (Z_i - F_i) exp(i g Ri)
    Depends only on the charge distribution (energy-independent far from edges).
    NOTE: Eq. (16) is the nonrelativistic limit. The exact amplitude ratio from
    Eqs. (13)/(14) is delta_g times a geometry factor X(beta, theta, omega)
    with X -> 1 as beta -> 0; at beta ~ 0.5, X can be ~0.2-0.5.
    """
    info = CRYSTALS[crystal]
    hkl = np.asarray(hkl, dtype=float)
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    num = 0.0 + 0.0j
    den = 0.0 + 0.0j
    for (el, R), F in zip(
        info["basis"], _basis_F(info["basis"], g, photon_E_eV, use_henke)
    ):
        phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))
        num += F.real * phase
        den += (Z_TABLE[el] - F.real) * phase
    return num / den


# ---- absorption length ------------------------------------------------------
def absorption_length_ang(element, photon_E_eV, number_density_per_ang3):
    """
    L_abs [Angstrom] from Henke f2:  mu = 2 (omega/c) * (n_atoms r_e lambda^2 f2 / ... )
    Practical form: 1/L_abs = 2 k beta_index, beta_index = (r_e lambda^2 / 2pi) n f2.
    Returns L_abs in Angstrom. (For compounds, sum n_i f2_i.)

    TODO (c) -- benign RuntimeWarning "divide by zero ... lam = HC_EV_ANG /
    photon_E_eV": the wide brem grid starts at 0 eV (cxr_config E_grid_brem =
    np.arange(0.0, ...)), so this is called with E=0; lam->inf, mu->0, L_abs->inf,
    which the brem path already swallows via nan_to_num. Just noise. To silence it,
    EITHER start the brem grid at a small positive eV (e.g. arange(10.0, ...) in
    cxr_config / the build_cases default) OR guard here, e.g.
    `with np.errstate(divide="ignore"): ...` and return inf at E<=0. Left as-is to
    keep this physics-core function untouched.
    """
    _, f2 = henke_dispersion(element, photon_E_eV)
    lam = HC_EV_ANG / photon_E_eV
    beta_idx = R_E_ANG * lam**2 / (2.0 * np.pi) * number_density_per_ang3 * f2
    k = 2.0 * np.pi / lam
    mu = 2.0 * k * beta_idx  # 1/Angstrom
    return 1.0 / mu


# ---- PXR / CBS amplitudes and flux -------------------------------------------
def _polarization_vectors(k_hat, g_vec):
    """
    Build sigma (out-of-plane, e1) and pi (in-plane, e2) unit polarization
    vectors for a photon direction k_hat, given the reaction plane spanned by
    k_hat and g_vec. Feranchuk-Spence e_1g = [k g]/|...|, e_2 = [k e1]/k.

    Returns (e_sigma, e_pi), each a 3-vector unit polarization.
    """
    k_hat = k_hat / np.linalg.norm(k_hat)
    # sigma: perpendicular to the reaction plane (plane of k and g)
    n_plane = _cross3(k_hat, g_vec)
    npl = np.linalg.norm(n_plane)
    if npl < 1e-12:
        # k parallel to g: degenerate; pick arbitrary perpendicular
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(k_hat @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        n_plane = _cross3(k_hat, tmp)
        npl = np.linalg.norm(n_plane)
    e_sigma = n_plane / npl
    # pi: in the reaction plane, perpendicular to k_hat
    e_pi = _cross3(e_sigma, k_hat)
    e_pi /= np.linalg.norm(e_pi)
    return e_sigma, e_pi


def amplitudes_PXR_CBS_both(
    crystal,
    hkl,
    photon_E_eV,
    beta,
    theta_B_normal,
    B_ang2=0.0,
    use_henke=False,
    geometry="symmetric",
    theta_obs=None,
):
    """
    Full A_PXR and A_CBS from Feranchuk-Spence Eqs. (13)/(14), for BOTH
    polarizations at once (the couplings chi_g and U_g are polarization-
    independent, so this costs half of two single-polarization calls).

        A_PXR = chi_g / (k_g^2 - omega^2) * [ (v0.k_g)(g.e) - omega^2 (v0.e) ]
        A_CBS = -(e U_g)/(m V (g.v0)) * [ g.e + (v0.e)(k.g)/(v0 g) ]

    Units: hbar = c = 1 internally. omega, |k|, |g| all in [1/Angstrom]
    (omega -> omega/c). Velocities in units of c (so v0 = beta).

    Returns (amps, omega_per_ang, g) where
        amps = {"sigma": (A_PXR, A_CBS), "pi": (A_PXR, A_CBS)}.

    geometry: "symmetric" -> Omega = 2*theta_Bragg, k at angle Omega to v0
                             (crystal co-rotated so k is always specular).
              "fixed"     -> crystal FIXED: g at angle theta_B_normal from the
                             beam (g.v0 > 0); detector at the independent
                             angle theta_obs (same reaction plane). symmetric
                             is the special case theta_obs = pi - 2*theta_B_normal.
              "lif"       -> v0 || plane normal (theta_B_normal ~ 0), observe
                             at the asymmetric angle; pass theta_B_normal as the
                             OBSERVATION angle theta_0 in that case.
    """
    info = CRYSTALS[crystal]

    omega = photon_E_eV / HBARC_EV_ANG  # photon wavenumber omega/c [1/Angstrom]
    k0 = omega  # |k| = omega/c (eps0 ~ 1)

    # geometry: electron along +z
    v0_hat = np.array([0.0, 0.0, 1.0])
    v0 = beta  # speed in units of c

    g_vec, g = reciprocal_g_vector(hkl, info["lattice"])

    if geometry == "symmetric":
        theta_bragg = 0.5 * np.pi - theta_B_normal
        Omega = 2.0 * theta_bragg
        # photon direction in x-z plane at angle Omega from beam (+z)
        k_hat = np.array([np.sin(Omega), 0.0, np.cos(Omega)])
        # g along the plane normal, oriented so g.v0 > 0: the resonance
        # delta[omega - v0.(k+g)] (Eq. 9) requires v0.g = omega(1 - v0 cos
        # theta) > 0. k_hat is then the mirror direction of v0 in the planes.
        g_vec = g * np.array([-np.cos(theta_bragg), 0.0, np.sin(theta_bragg)])
    elif geometry == "fixed":
        if theta_obs is None:
            raise ValueError("geometry='fixed' requires theta_obs")
        k_hat = np.array([np.sin(theta_obs), 0.0, np.cos(theta_obs)])
        # g at polar angle theta_B_normal from the beam, opposite azimuth side
        # from the detector (g.v0 = beta g cos(theta_B_normal) > 0)
        g_vec = g * np.array([-np.sin(theta_B_normal), 0.0, np.cos(theta_B_normal)])
    elif geometry == "lif":
        # v0 || [100] plane normal; theta_B_normal carries the OBSERVATION angle
        theta_obs = theta_B_normal
        k_hat = np.array([np.sin(theta_obs), 0.0, np.cos(theta_obs)])
        # g along the beam direction (g.v0 > 0), magnitude g
        g_vec = g * v0_hat
    else:
        raise ValueError("geometry must be 'symmetric', 'fixed' or 'lif'")

    k_vec = k0 * k_hat  # photon wavevector, |k| = omega/c [1/Ang]
    kg_vec = k_vec + g_vec  # diffracted wavevector k_g = k + g
    kg2 = kg_vec @ kg_vec  # |k_g|^2

    # sigma (out of the k-g plane) and pi (in-plane) unit polarization vectors
    e_sigma, e_pi = _polarization_vectors(k_hat, g_vec)

    # couplings (polarization-independent; computed once for both):
    #   chi  = chi_g(omega), Eq. (3)  -- dimensionless, complex
    #   eUg  = e*U_g/V,      Eq. (4)  -- eV; /m_e c^2 below makes it the
    #                                    dimensionless CBS strength of Eq. (14)
    chi = chi_g(crystal, hkl, photon_E_eV, B_ang2, use_henke)
    eUg = U_g(crystal, hkl, photon_E_eV, B_ang2, use_henke)

    detuning = kg2 - omega**2  # PXR denominator; ~g^2(1-beta...) > 0
    v0_vec = v0 * v0_hat  # electron velocity vector (units of c)
    eUg_over_m = eUg / M_E_EV  # dimensionless (m c^2 = 511 keV in eV)
    g_dot_v0 = g_vec @ v0_vec  # = omega(1 - v.n) at resonance, > 0

    amps = {}
    for pol, e in (("sigma", e_sigma), ("pi", e_pi)):
        # --- A_PXR, Eq. (13):  chi/(k_g^2-w^2) [(v.k_g)(g.e) - w^2 (v.e)] ---
        bracket_pxr = (v0_vec @ kg_vec) * (g_vec @ e) - omega**2 * (v0_vec @ e)
        A_PXR = chi / detuning * bracket_pxr

        # --- A_CBS with relativistic corrections (Zhai SI Eq. 6): ---
        #   A_CBS = -(e U_g)/(gamma m V (g.v)) [{g;e} + (v.e){k;g}/(v.g)]
        # braced product {a;b} = a.b - (a.v)(b.v) (v in units of c): the
        # electron's longitudinal response to the lattice force is gamma^2-
        # suppressed. Reduces exactly to Feranchuk Eq. (14) as beta -> 0.
        if abs(g_dot_v0) < 1e-12:
            # g perpendicular to v: formal 1/(g.v) divergence, but the
            # resonance energy -> 0 there, so the line is unobservable anyway
            A_CBS = 0.0 + 0.0j
        else:
            gamma = 1.0 / np.sqrt(1.0 - v0**2)
            braced_ge = (g_vec @ e) - g_dot_v0 * (v0_vec @ e)
            braced_kg = (k_vec @ g_vec) - (k_vec @ v0_vec) * g_dot_v0
            bracket_cbs = braced_ge + (v0_vec @ e) * braced_kg / g_dot_v0
            A_CBS = -eUg_over_m / (gamma * g_dot_v0) * bracket_cbs
        amps[pol] = (A_PXR, A_CBS)

    return amps, omega, g


def amplitudes_PXR_CBS(
    crystal,
    hkl,
    photon_E_eV,
    beta,
    theta_B_normal,
    polarization="pi",
    B_ang2=0.0,
    use_henke=False,
    geometry="symmetric",
    theta_obs=None,
):
    """
    Single-polarization wrapper around amplitudes_PXR_CBS_both.
    Returns (A_PXR, A_CBS, omega_per_ang, g). If you need both polarizations,
    call amplitudes_PXR_CBS_both directly -- it is twice as fast.
    """
    amps, omega, g = amplitudes_PXR_CBS_both(
        crystal,
        hkl,
        photon_E_eV,
        beta,
        theta_B_normal,
        B_ang2,
        use_henke,
        geometry,
        theta_obs,
    )
    A_PXR, A_CBS = amps[polarization]
    return A_PXR, A_CBS, omega, g


def amplitudes_PXR_CBS_sweep(
    crystal,
    hkl,
    photon_E_eV,
    beta,
    theta_B_normal,
    B_ang2=0.0,
    use_henke=False,
    geometry="symmetric",
    theta_obs=None,
):
    """
    Vectorized amplitudes_PXR_CBS_both over arrays of photon energy and angle:
    photon_E_eV and theta_B_normal are broadcast-compatible 1-D arrays (one
    geometry point per entry). Identical physics to the scalar version, but
    the whole sweep runs in numpy instead of a Python loop per point.

    Returns (amps, omega_per_ang, g) where each entry of
        amps = {"sigma": (A_PXR, A_CBS), "pi": (A_PXR, A_CBS)}
    is a complex array shaped like photon_E_eV.
    """
    info = CRYSTALS[crystal]
    E = np.atleast_1d(np.asarray(photon_E_eV, dtype=float))
    th = np.broadcast_to(np.asarray(theta_B_normal, dtype=float), E.shape)
    n = E.size

    omega = E / HBARC_EV_ANG  # (n,) photon wavenumber [1/Ang]
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    zeros = np.zeros(n)

    if geometry == "symmetric":
        theta_bragg = 0.5 * np.pi - th
        Omega = 2.0 * theta_bragg
        k_hat = np.stack([np.sin(Omega), zeros, np.cos(Omega)], axis=1)
        # g.v0 > 0 orientation required by the Eq. (9) resonance; see
        # amplitudes_PXR_CBS_both
        g_vec = g * np.stack([-np.cos(theta_bragg), zeros, np.sin(theta_bragg)], axis=1)
    elif geometry == "fixed":
        if theta_obs is None:
            raise ValueError("geometry='fixed' requires theta_obs")
        to = np.broadcast_to(np.asarray(theta_obs, dtype=float), E.shape)
        k_hat = np.stack([np.sin(to), zeros, np.cos(to)], axis=1)
        g_vec = g * np.stack([-np.sin(th), zeros, np.cos(th)], axis=1)
    elif geometry == "lif":
        k_hat = np.stack([np.sin(th), zeros, np.cos(th)], axis=1)
        g_vec = np.broadcast_to(np.array([0.0, 0.0, g]), (n, 3)).copy()
    else:
        raise ValueError("geometry must be 'symmetric', 'fixed' or 'lif'")

    k_vec = omega[:, None] * k_hat
    kg_vec = k_vec + g_vec
    kg2 = np.einsum("ij,ij->i", kg_vec, kg_vec)

    # polarization vectors, row-wise (with the same k || g degeneracy fallback)
    n_plane = np.cross(k_hat, g_vec)
    npl = np.linalg.norm(n_plane, axis=1)
    bad = npl < 1e-12
    if np.any(bad):
        tmp = np.where(
            np.abs(k_hat[bad, 0:1]) > 0.9,
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        n_plane[bad] = np.cross(k_hat[bad], tmp)
        npl[bad] = np.linalg.norm(n_plane[bad], axis=1)
    e_sigma = n_plane / npl[:, None]
    e_pi = np.cross(e_sigma, k_hat)
    e_pi /= np.linalg.norm(e_pi, axis=1)[:, None]

    # couplings: chi_g / U_g already broadcast over array photon energies
    chi = chi_g(crystal, hkl, E, B_ang2, use_henke)
    eUg = U_g(crystal, hkl, E, B_ang2, use_henke)  # e*U_g/V [eV]

    detuning = kg2 - omega**2
    v0_vec = beta * np.array([0.0, 0.0, 1.0])
    eUg_over_m = eUg / M_E_EV
    g_dot_v0 = g_vec @ v0_vec
    safe_g_dot_v0 = np.where(np.abs(g_dot_v0) < 1e-12, 1.0, g_dot_v0)

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    k_dot_v0 = k_vec @ v0_vec  # (n,) since v0_vec is constant

    amps = {}
    for pol, e in (("sigma", e_sigma), ("pi", e_pi)):
        v0_dot_e = e @ v0_vec
        g_dot_e = np.einsum("ij,ij->i", g_vec, e)
        bracket_pxr = (
            np.einsum("ij,ij->i", kg_vec, np.broadcast_to(v0_vec, (n, 3))) * g_dot_e
            - omega**2 * v0_dot_e
        )
        A_PXR = chi / detuning * bracket_pxr

        # relativistic A_CBS (Zhai SI Eq. 6): braced {a;b} = a.b - (a.v)(b.v),
        # 1/gamma prefactor; reduces to Feranchuk Eq. (14) at beta -> 0
        braced_ge = g_dot_e - g_dot_v0 * v0_dot_e
        braced_kg = np.einsum("ij,ij->i", k_vec, g_vec) - k_dot_v0 * g_dot_v0
        bracket_cbs = braced_ge + v0_dot_e * braced_kg / safe_g_dot_v0
        A_CBS = np.where(
            np.abs(g_dot_v0) < 1e-12,
            0.0 + 0.0j,
            -eUg_over_m / (gamma * safe_g_dot_v0) * bracket_cbs,
        )
        amps[pol] = (A_PXR, A_CBS)

    return amps, omega, g


def _rotation_between(u_hat, t_hat):
    """Minimal rotation matrix R such that R @ u_hat = t_hat (unit vectors)."""
    c = float(np.dot(u_hat, t_hat))
    axis = _cross3(u_hat, t_hat)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: 180 deg about any axis perpendicular to u_hat
        tmp = (
            np.array([1.0, 0.0, 0.0])
            if abs(u_hat[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = _cross3(u_hat, tmp)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * K @ K
    axis = axis / s
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def cxr_lines_fixed(
    crystal,
    beta,
    theta_obs,
    orient_hkl=None,
    theta_B_normal=None,
    E_min_eV=300.0,
    E_max_eV=15000.0,
    B_ang2=0.0,
    use_henke=False,
    g_max_invang=16.0,
    beam_uvw=None,
    azimuth_rad=0.0,
):
    """
    ALL CXR lines a detector at polar angle theta_obs sees from a crystal at
    FIXED orientation: every reciprocal lattice vector g with g.v0 > 0 emits a
    line at omega = g.v0 / (1 - v0 cos theta_obs) (Eq. 10 resonance); the
    amplitudes are Eqs. (13)/(14) with the full 3-D g (out-of-plane
    reflections radiate in both polarizations).

    Orientation (pick ONE):
      * orient_hkl + theta_B_normal: that reflection's g is placed along the
        geometry="fixed" direction (-sin(theta_B), 0, cos(theta_B)).
      * beam_uvw: the DIRECT-lattice direction [u v w] is placed along the
        beam (+z), i.e. "electrons parallel to <uvw>" as in paper Fig. 2.
    Both use the MINIMAL rotation from the crystal's construction frame; the
    remaining azimuthal freedom about the rotated axis is fixed by that
    choice and can be explored with azimuth_rad (extra rotation about +z
    applied to the crystal). Line ENERGIES along a zone axis are azimuth-
    independent (E depends only on g_z); individual line intensities at the
    detector are not.

    Returns a dict of arrays sorted by energy:
      "E_eV", "omega" [1/Ang], "A2" (|A_PXR+A_CBS|^2 summed over sigma+pi),
      "A2_PXR", "A2_CBS" (separate |A|^2 sums), "hkl" (N,3 int),
      "g_mag" [1/Ang].
    Flux per line: dN/dOmega = ALPHA_FS/(2 pi) * omega * (L_eff/beta) * A2
    (Eq. 12), absorption length and L_eff left to the caller.
    """
    info = CRYSTALS[crystal]
    lattice = info["lattice"]
    B = _reciprocal_basis(lattice)  # rows b1, b2, b3
    a_vecs = _direct_lattice_vectors(lattice)

    # --- orientation
    if beam_uvw is not None:
        u, v, w = np.asarray(beam_uvw, dtype=float)
        axis = u * a_vecs[0] + v * a_vecs[1] + w * a_vecs[2]
        R = _rotation_between(axis / np.linalg.norm(axis), np.array([0.0, 0.0, 1.0]))
    elif orient_hkl is not None and theta_B_normal is not None:
        g0 = np.asarray(orient_hkl, dtype=float) @ B
        t_hat = np.array([-np.sin(theta_B_normal), 0.0, np.cos(theta_B_normal)])
        R = _rotation_between(g0 / np.linalg.norm(g0), t_hat)
    else:
        raise ValueError("give either beam_uvw or (orient_hkl, theta_B_normal)")
    if azimuth_rad:
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]]) @ R
    B_lab = B @ R.T  # g_lab = hkl @ B_lab

    # --- enumerate candidate hkl (exact bound: |h_i| <= g_max |a_i| / 2 pi)
    nmax = [
        int(np.floor(g_max_invang * np.linalg.norm(a) / (2.0 * np.pi))) for a in a_vecs
    ]
    grids = np.meshgrid(*(np.arange(-n, n + 1) for n in nmax), indexing="ij")
    hkl = np.column_stack([G.ravel() for G in grids]).astype(float)
    g_vec = hkl @ B_lab
    g_mag = np.linalg.norm(g_vec, axis=1)
    keep = (g_mag > 1e-9) & (g_mag <= g_max_invang) & (g_vec[:, 2] > 1e-9)
    hkl, g_vec, g_mag = hkl[keep], g_vec[keep], g_mag[keep]

    # --- line energies at this observation angle (Eq. 10)
    E = HBARC_EV_ANG * beta * g_vec[:, 2] / (1.0 - beta * np.cos(theta_obs))
    keep = (E >= E_min_eV) & (E <= E_max_eV)
    hkl, g_vec, g_mag, E = hkl[keep], g_vec[keep], g_mag[keep], E[keep]
    omega = E / HBARC_EV_ANG
    n_lines = E.size

    # --- couplings per line (vectorized over lines; F once per element)
    dwf = debye_waller(g_mag, B_ang2)
    F_el = {}
    for el in {el for el, _ in info["basis"]}:
        if use_henke or el in _EDGE_PRONE:
            F_el[el] = atomic_form_factor(el, g_mag, E)
        else:
            F_el[el] = cromer_mann_f0(el, g_mag) + 0.0j
    S = np.zeros(n_lines, dtype=complex)
    ZmF = np.zeros(n_lines, dtype=complex)
    for el, R_frac in info["basis"]:
        phase = np.exp(2j * np.pi * (hkl @ R_frac))
        S += F_el[el] * phase * dwf
        ZmF += (Z_TABLE[el] - F_el[el].real) * phase * dwf
    lam = HC_EV_ANG / E
    chi = -R_E_ANG * lam**2 / (np.pi * info["V_cell"]) * S  # Eq. (3)
    eUg = 4.0 * np.pi * E2_EV_ANG * ZmF / g_mag**2 / info["V_cell"]  # Eq. (4)/(14)

    # --- geometry & amplitudes, Eqs. (13)/(14)
    k_hat = np.array([np.sin(theta_obs), 0.0, np.cos(theta_obs)])
    k_hat_rows = np.broadcast_to(k_hat, (n_lines, 3))
    k_vec = omega[:, None] * k_hat
    kg_vec = k_vec + g_vec
    kg2 = np.einsum("ij,ij->i", kg_vec, kg_vec)

    n_plane = np.cross(k_hat_rows, g_vec)
    npl = np.linalg.norm(n_plane, axis=1)
    bad = npl < 1e-12
    if np.any(bad):  # k || g degeneracy
        n_plane[bad] = _cross3(k_hat, np.array([0.0, 1.0, 0.0]))
        npl[bad] = np.linalg.norm(n_plane[bad], axis=1)
    e_sigma = n_plane / npl[:, None]
    e_pi = np.cross(e_sigma, k_hat_rows)
    e_pi /= np.linalg.norm(e_pi, axis=1)[:, None]

    v0_vec = beta * np.array([0.0, 0.0, 1.0])
    detuning = kg2 - omega**2
    eUg_over_m = eUg / M_E_EV
    g_dot_v0 = beta * g_vec[:, 2]
    kg_dot_v0 = np.einsum("ij,ij->i", kg_vec, np.broadcast_to(v0_vec, (n_lines, 3)))
    k_dot_g = np.einsum("ij,ij->i", k_vec, g_vec)

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    k_dot_v0 = np.einsum("ij,ij->i", k_vec, np.broadcast_to(v0_vec, (n_lines, 3)))

    A2 = np.zeros(n_lines)
    A2_pxr = np.zeros(n_lines)
    A2_cbs = np.zeros(n_lines)
    for e in (e_sigma, e_pi):
        v0_dot_e = e @ v0_vec
        g_dot_e = np.einsum("ij,ij->i", g_vec, e)
        A_PXR = chi / detuning * (kg_dot_v0 * g_dot_e - omega**2 * v0_dot_e)
        # relativistic A_CBS (Zhai SI Eq. 6): {a;b} = a.b - (a.v)(b.v),
        # 1/gamma prefactor; reduces to Feranchuk Eq. (14) at beta -> 0
        braced_ge = g_dot_e - g_dot_v0 * v0_dot_e
        braced_kg = k_dot_g - k_dot_v0 * g_dot_v0
        A_CBS = (
            -eUg_over_m
            / (gamma * g_dot_v0)
            * (braced_ge + v0_dot_e * braced_kg / g_dot_v0)
        )
        A2 += np.abs(A_PXR + A_CBS) ** 2
        A2_pxr += np.abs(A_PXR) ** 2
        A2_cbs += np.abs(A_CBS) ** 2

    order = np.argsort(E)
    return {
        "E_eV": E[order],
        "omega": omega[order],
        "A2": A2[order],
        "A2_PXR": A2_pxr[order],
        "A2_CBS": A2_cbs[order],
        "hkl": hkl[order].astype(int),
        "g_mag": g_mag[order],
    }


def photons_per_electron(
    crystal,
    hkl,
    photon_E_eV,
    theta_B_normal,
    beta,
    L_z_ang,
    L_abs_ang,
    dOmega_sr,
    polarization="pi",
    B_ang2=0.0,
    use_henke=False,
    geometry="symmetric",
    theta_obs=None,
):
    """
    Photons per electron into dOmega: Feranchuk-Spence Eq. (12) with the full
    Eq. (13)/(14) amplitudes, the Eq. (9) absorption-limited length, AND the
    exact delta-function Jacobian that the paper's "~" drops:

        dN = (alpha/2pi) * omega * (L_eff/beta) * |A_PXR+A_CBS|^2
             * dOmega / (1 - beta n.v)

    The 1/(1 - beta cos(theta_obs)) factor comes from integrating
    delta[omega(1 - v.n) - v.g] over omega (Eq. 9); it makes this function
    agree exactly with the integrated finite-segment lineshape used in
    cxr_montecarlo.mc_spectrum (validated to <1% in zhai_fig1c_check.py).
    Up to ~1.7x at relativistic-ish beta and forward angles.

    omega is in [1/Angstrom], L_eff in [Angstrom] (omega*L dimensionless,
    hbar = c = 1). polarization="both" sums sigma and pi.
    """
    amps, omega, _ = amplitudes_PXR_CBS_both(
        crystal,
        hkl,
        photon_E_eV,
        beta,
        theta_B_normal,
        B_ang2,
        use_henke,
        geometry,
        theta_obs,
    )
    pols = ("sigma", "pi") if polarization == "both" else (polarization,)
    A2 = sum(abs(amps[pol][0] + amps[pol][1]) ** 2 for pol in pols)

    # observation-direction z-component per geometry (v0 is along +z)
    if geometry == "symmetric":
        # detector at Omega = 2*theta_Bragg = pi - 2*theta_B_normal from beam
        n_z = -np.cos(2.0 * theta_B_normal)
    elif geometry == "fixed":
        n_z = np.cos(theta_obs)
    else:  # "lif": theta_B_normal IS theta_obs
        n_z = np.cos(theta_B_normal)

    L_eff = L_abs_ang * (1.0 - np.exp(-L_z_ang / L_abs_ang))
    return (
        ALPHA_FS
        / (2.0 * np.pi)
        * omega
        * (L_eff / beta)
        * A2
        * dOmega_sr
        / (1.0 - beta * n_z)
    )


def flux_per_second(
    crystal,
    hkl,
    photon_E_eV,
    theta_B_normal,
    beta,
    L_z_ang,
    L_abs_ang,
    dOmega_sr,
    current_A,
    polarization="pi",
    B_ang2=0.0,
    use_henke=False,
    geometry="symmetric",
    theta_obs=None,
):
    """photons_per_electron scaled by the electron rate current_A/e."""
    e_charge = 1.602176634e-19
    n_e_per_s = current_A / e_charge
    return n_e_per_s * photons_per_electron(
        crystal,
        hkl,
        photon_E_eV,
        theta_B_normal,
        beta,
        L_z_ang,
        L_abs_ang,
        dOmega_sr,
        polarization,
        B_ang2,
        use_henke,
        geometry,
        theta_obs,
    )


def dominant_reflections(
    crystal,
    n_families=4,
    E_ref_eV=1000.0,
    B_ang2=0.0,
    use_henke=False,
    g_max_invang=8.0,
):
    """
    Automatically select the strongest reflection FAMILIES of a crystal,
    Zhai-style (their Table 5 keeps the four planes of largest |chi_g| per
    crystal; everything weaker contributes < ~30%).

    Enumerates all reciprocal vectors with |g| <= g_max_invang, ranks by
        metric = |S(g)| e^{-W} / g^2
    which is proportional to |chi_g| evaluated at each reflection's OWN line
    energy (omega_res scales with g, and chi ~ S/omega^2). Symmetry-
    equivalent members are grouped by identical (|g|, metric) -- no explicit
    space-group code needed -- and ALL members of the top n_families are
    returned as a sorted list of (h, k, l) tuples (including Friedel mates).

    NOTE: this ranks by the crystal STRUCTURE only. Texture constraints are
    yours to impose -- e.g. HOPG must be restricted to (00l) by hand, since
    its in-plane reflections are incoherent across fiber-textured grains.
    """
    info = CRYSTALS[crystal]
    B = _reciprocal_basis(info["lattice"])
    a_vecs = _direct_lattice_vectors(info["lattice"])

    # exact per-axis index bounds: |h_i| <= g_max |a_i| / 2 pi
    nmax = [
        int(np.floor(g_max_invang * np.linalg.norm(a) / (2.0 * np.pi))) for a in a_vecs
    ]
    grids = np.meshgrid(*(np.arange(-n, n + 1) for n in nmax), indexing="ij")
    hkl = np.column_stack([G.ravel() for G in grids]).astype(float)
    g_vec = hkl @ B
    g_mag = np.linalg.norm(g_vec, axis=1)
    keep = (g_mag > 1e-9) & (g_mag <= g_max_invang)
    hkl, g_mag = hkl[keep], g_mag[keep]

    # |S(g)| with the per-element form-factor policy, vectorized over hkl
    dwf = debye_waller(g_mag, B_ang2)
    F_el = {}
    for el in {el for el, _ in info["basis"]}:
        F_el[el] = _atom_F(el, g_mag, E_ref_eV, use_henke)
    S = np.zeros(g_mag.shape, dtype=complex)
    for el, R_frac in info["basis"]:
        S += F_el[el] * np.exp(2j * np.pi * (hkl @ R_frac)) * dwf
    metric = np.abs(S) / g_mag**2

    # group symmetry mates: identical (|g|, metric) to rounding
    fams = {}
    for i in range(hkl.shape[0]):
        key = (round(float(g_mag[i]), 6), round(float(metric[i]), 9))
        fams.setdefault(key, []).append(tuple(int(x) for x in hkl[i]))
    ranked = sorted(fams.items(), key=lambda kv: -kv[0][1])

    out = []
    for (_, m), members in ranked[:n_families]:
        if m < 1e-9 * ranked[0][0][1]:
            break  # forbidden/negligible families
        out.extend(sorted(members))
    return out


def bremsstrahlung_background(photon_E_eV, Z, number_density_per_ang3, L_z_ang, dE_eV):
    """
    Incoherent bremsstrahlung background, Feranchuk-Spence Eq. (17) (their
    Akhiezer-Berestetskii estimate): photons per electron per steradian
    within a spectral bin dE_eV around photon_E_eV,
        dN/dn = (4 e^2 / 3 pi) Z^2 (e^2/m)^2 rho L_z ln(137/Z^{1/3}) dE/E
    with e^2 = alpha (hbar = c = 1) and rho = atoms per Angstrom^3.
    Isotropic estimate, angular and velocity dependence neglected (as in the
    paper); single-element approximation for compounds.
    """
    m_inv_ang = M_E_EV / HBARC_EV_ANG
    return (
        4.0
        * ALPHA_FS
        / (3.0 * np.pi)
        * Z**2
        * (ALPHA_FS / m_inv_ang) ** 2
        * number_density_per_ang3
        * L_z_ang
        * np.log(137.0 / Z ** (1.0 / 3.0))
        * dE_eV
        / photon_E_eV
    )


# ---- CXR / bremsstrahlung-background ratio (Eq. 18) ------------------------
def cxr_to_bremsstrahlung(photon_E_eV, number_density_per_ang3, beta, Z_avg, dE_over_E):
    """
    eta = [dN/dn]_CXR / [dN/dn]_BS, Feranchuk-Spence Eq. (18):
        eta ~ (rho / omega_n^3) * (6 pi^2 v0 / ln(137/Z^{1/3})) * (omega_n/dE)
    rho = number density [1/Angstrom^3], omega_n in [1/Angstrom] (= omega/c).
    The coherency factor xi_n = rho / omega_n^3 (Eq. 19) is the key scaling.
    Higher detector resolution (smaller dE/E) -> larger eta.
    """
    omega_per_ang = photon_E_eV / HC_EV_ANG * 2.0 * np.pi  # omega/c [1/Angstrom]
    xi_n = number_density_per_ang3 / omega_per_ang**3  # Eq.(19)
    eta = (
        xi_n
        * (6.0 * np.pi**2 * beta / np.log(137.0 / Z_avg ** (1.0 / 3.0)))
        / dE_over_E
    )
    return eta, xi_n


if __name__ == "__main__":
    beta = beta_from_Ee(30e3)
    # diamond (400) example
    E_line = omega_n(d_ang=0.8917, beta=beta, theta_B_normal=np.deg2rad(45), n=1)
    print("line energy [eV]:", E_line)
    dg = delta_g("diamond", [4, 0, 0], E_line)
    print("structure factor:", structure_factor("diamond", [4, 0, 0], 4478))
    print(f"delta_g = A_PXR/A_CBS: {np.real(dg):0.3f}")
