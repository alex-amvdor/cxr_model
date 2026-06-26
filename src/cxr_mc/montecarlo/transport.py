"""
montecarlo.transport

Single-scattering electron transport (Zhai SI S2): element data, elastic
scattering models (Browning free paths + NIST-Mott-calibrated screened-
Rutherford angles, analytic SR fallback), Joy-Luo stopping power, and the
vectorized event-driven trajectory simulator. Pure NumPy -- never touches the
GPU; the spectrum phase consumes the segment arrays it returns.
"""

import os
from functools import cache

import numpy as np

from .. import DATA_DIR
from .materials import _normalize_composition

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
