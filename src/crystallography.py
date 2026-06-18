"""
crystallography.py

General-purpose X-ray crystallography / diffraction primitives, shared by the
Monte-Carlo pipeline (montecarlo, sweep, the detector forward models) and the
Feranchuk-Spence analytic checks (checks/feranchuk_spence.py). Nothing here is
specific to the Feranchuk amplitude framework -- it is the reusable layer above
atomic_form_factors.py:

  * physical constants (hc, hbar c, alpha, m_e, r_e),
  * lattice geometry: direct/reciprocal vectors, |g| for any crystal system,
  * the crystal database (crystal_structures.toml -> CRYSTALS),
  * Debye-Waller, structure factor S(g), and the polarizability / crystal-
    potential Fourier components chi_g (PXR) and U_g (CBS),
  * photoabsorption length from Henke f2,
  * dominant_reflections (rank reflection families by |S| e^{-W} / g^2),
  * _rotation_between (minimal rotation matrix, used to orient crystals).

Units: energies eV, lengths Angstrom, angles radians.

Crystal structures are loaded from crystal_structures.toml (data/) into the
CRYSTALS dict; see that file for the format. Depends on atomic_form_factors.py
(cromer_mann_f0, atomic_form_factor, henke_dispersion, Z_TABLE).
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


# ---- absorption length ------------------------------------------------------
def absorption_length_ang(element, photon_E_eV, number_density_per_ang3):
    """
    L_abs [Angstrom] from Henke f2:  mu = 2 (omega/c) * (n_atoms r_e lambda^2 f2 / ... )
    Practical form: 1/L_abs = 2 k beta_index, beta_index = (r_e lambda^2 / 2pi) n f2.
    Returns L_abs in Angstrom. (For compounds, sum n_i f2_i.)

    The wide brem grid starts at 0 eV (config E_grid_brem = np.arange(0.0, ...)),
    so this is called with E=0; there lam->inf, mu->0, L_abs->inf, which the brem
    path already swallows via nan_to_num. The errstate guard just suppresses the
    benign "divide by zero" / "invalid value" RuntimeWarnings from that E=0 bin --
    the returned values (and so all numerics) are unchanged.
    """
    _, f2 = henke_dispersion(element, photon_E_eV)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = HC_EV_ANG / photon_E_eV
        beta_idx = R_E_ANG * lam**2 / (2.0 * np.pi) * number_density_per_ang3 * f2
        k = 2.0 * np.pi / lam
        mu = 2.0 * k * beta_idx  # 1/Angstrom
        return 1.0 / mu


# ---- crystal orientation ----------------------------------------------------
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


# ---- reflection ranking -----------------------------------------------------
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
