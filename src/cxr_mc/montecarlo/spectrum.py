"""
montecarlo.spectrum

Radiation from the transported segments (Zhai SI S1): the finite-interaction-
time CXR (PXR + CBS) line spectrum mc_spectrum, its solid-angle-integrated
wrapper, the Bethe-Heitler bremsstrahlung background, and the external-brem
loader. The array-heavy inner loops run on the GPU backend (``xp``) when
available.
"""

import numpy as np

from ..crystallography import (
    ALPHA_FS,
    CRYSTALS,
    HBARC_EV_ANG,
    M_E_EV,
    U_g,
    chi_g,
    reciprocal_g_vector,
)
from ._backend import REAL, _to_cpu, xp
from .geometry import _mosaic_quadrature, _orientation_R
from .materials import _layer_dz, _mu_total_inv_ang, _normalize_composition, _stack_tau
from .transport import TRANSPORT_ELEMENTS, beta_from_keV

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
    from ..atomic_form_factors import load_henke

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
        chi_re = xp.asarray(chi_tab.real, dtype=REAL)  # type: ignore[reportAttributeAccessIssue]
        chi_im = xp.asarray(chi_tab.imag, dtype=REAL)  # type: ignore[reportAttributeAccessIssue]
        u_re = xp.asarray(u_tab.real, dtype=REAL)  # type: ignore[reportAttributeAccessIssue]
        u_im = xp.asarray(u_tab.imag, dtype=REAL)  # type: ignore[reportAttributeAccessIssue]

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
