"""Multilayer ANALYTIC validation (docs/multilayer-materials.md, "Validation
plan"): two self-contained, quantitative checks that need NO measured data.

  A. CLOSED-FORM cross-stack absorption (slice 1). With identical emission
     segments, the only difference between the film-on-substrate spectrum and the
     film-only spectrum on a BACK exit is the substrate's optical depth. The
     escape factor T_abs is applied at each line's RESONANCE energy E_res (it
     multiplies that segment's lineshape), and the film escape path is shared, so
     the INTEGRATED flux of a film line is attenuated by exactly the Beer-Lambert
     factor at the line energy:

         integral(line) spec_stack / integral(line) spec_film
             == exp(-mu_sub(E_line) * t_sub / |n_z|)

     a closed-form check of the piecewise mu_i*dz_i path integral, end to end
     through mc_spectrum (not just the _stack_tau helper). [docs validation plan:
     "for a film line at energy E, the integrated flux ratio (with vs without
     substrate) must equal exp(-mu_sub(E) t_sub/|n_z|)".]

  B. DEPTH-DOSE vs the Kanaya-Okayama range (slice 2 transport). From
     simulate_trajectories the dose centroid d_bar is a sane fraction of the K-O
     range and the maximum penetration z_max ~ the range itself

         R_KO = 0.0276 * A * E^1.67 / (Z^0.889 * rho)   [um]

     with z_max following K-O's E^1.67 ENERGY scaling (same material, carbon) and
     its A/(Z^0.889 rho) MATERIAL scaling (carbon vs aluminium). [z_max -- not the
     centroid -- carries the scaling: K-O estimates the *range*, and the dose-
     profile shape (hence the centroid fraction) is Z-dependent.]

Run CPU-forced (the viz laptop has the cupy wheel but no CUDA toolkit):
    uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/multilayer_validation_check.py',run_name='__main__')"
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_model.montecarlo import (  # noqa: E402
    TRANSPORT_ELEMENTS,
    _dEds_compound,
    _mu_total_inv_ang,
    mc_spectrum,
    simulate_trajectories,
    tilted_geometry,
)
from cxr_model.sweep import crystal_params, substrate_composition  # noqa: E402

_N_A = 6.02214076e23  # rho[g/cm^3] = n[1/Ang^3] * A[g/mol] * 1e24 / N_A


def _rho_g_cm3(element, n_per_ang3):
    return n_per_ang3 * TRANSPORT_ELEMENTS[element]["A"] * 1e24 / _N_A


def _kanaya_okayama_ang(element, n_per_ang3, E0_keV):
    """Kanaya-Okayama electron range [Angstrom]."""
    p = TRANSPORT_ELEMENTS[element]
    rho = _rho_g_cm3(element, n_per_ang3)
    return 0.0276 * p["A"] * E0_keV**1.67 / (p["Z"] ** 0.889 * rho) * 1e4  # um -> Ang


# ---- A. closed-form cross-stack absorption (slice 1) -------------------------
def check_absorption_closed_form():
    cp = crystal_params("mose2")
    comp = cp["composition"]
    sub = substrate_composition("sio2")
    t_film, t_sub, E0 = 500.0, 1.5e4, 30.0  # 50 nm film, 1.5 um substrate (tau~0.5 at the line)
    # ONE strong, ISOLATED, relatively HARD line (MoSe2 (-1,0,3) ~ 1438 eV): no
    # cross-line blend, and mu_sub is flat enough across the line that the
    # integrated/peak ratio is the Beer-Lambert factor at the line energy. (The
    # soft lines ~150-550 eV sit where mu_sub ~ E^-3 is too steep for a clean
    # closed form.)
    hkl = [(-1, 0, 3)]
    E_grid = np.arange(900.0, 2100.0, 2.0)  # eV; brackets the 1438 eV line

    # BACK exit (positive tilt): the escape ray runs through the film then the
    # WHOLE substrate, so n_z > 0 and the substrate path length is the full t_sub.
    beam, n_hat = tilted_geometry(np.deg2rad(90.0), np.deg2rad(30.0))
    n_hat = n_hat / np.linalg.norm(n_hat)  # match mc_spectrum's normalization
    assert n_hat[2] > 0.0, "need a back exit (n_z>0) for this identity"
    segs = simulate_trajectories(
        E0, 600, t_film, composition=comp, E_cut_keV=5.0, seed=1, beam_dir=beam
    )

    def _spec(layers):
        return np.asarray(
            mc_spectrum(
                segs,
                E_grid,
                crystal="mose2",
                hkl_list=hkl,
                n_hat=n_hat,
                B_ang2=cp["B_ang2"],
                composition=comp,
                beam_uvw=cp["beam_uvw"],
                layers=layers,
            )
        )

    spec_film = _spec([(0.0, t_film, comp)])
    spec_stack = _spec([(0.0, t_film, comp), (t_film, t_film + t_sub, sub)])

    # At the dominant line's PEAK BIN the flux is dominated by segments whose
    # resonance E_res ~ E_peak, so the bin ratio is the Beer-Lambert factor at the
    # line energy. (A wide window blends in barely-attenuated tails of harder
    # lines -- fatal at soft energies where mu_sub is steep.)
    pk = int(np.argmax(spec_film))
    E_pk = float(E_grid[pk])
    mu_pk = float(np.asarray(_mu_total_inv_ang(sub, np.array([E_pk])))[0])  # 1/Ang
    expected = float(np.exp(-mu_pk * t_sub / abs(n_hat[2])))
    ratio_bin = float(spec_stack[pk] / spec_film[pk])
    # window diagnostic (expected to be looser than the peak-bin ratio)
    w = (E_grid > E_pk - 60.0) & (E_grid < E_pk + 60.0)
    ratio_win = float(
        np.trapezoid(spec_stack[w], E_grid[w]) / np.trapezoid(spec_film[w], E_grid[w])
    )
    rel = abs(ratio_bin - expected) / expected
    ok = rel < 0.02
    print("A. closed-form cross-stack absorption (back exit, slice 1):")
    print(
        f"   line E_pk={E_pk:.0f} eV  t_sub={t_sub:.0f}A  n_z={n_hat[2]:+.3f}  "
        f"mu_sub={mu_pk:.3e}/A  tau_sub={mu_pk * t_sub / abs(n_hat[2]):.3f}"
    )
    print(
        f"   peak-bin ratio stack/film={ratio_bin:.4f}  (window +-60eV={ratio_win:.4f})  vs  "
        f"exp(-mu_sub*t_sub/|n_z|)={expected:.4f}  (rel {rel:.2e})   "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


# ---- B. depth-dose vs the Kanaya-Okayama range (slice 2 transport) -----------
def _dose_depths(element, n_per_ang3, E0_keV, slab_ang, Ne=300, seed=0):
    """Depth-dose summary for a bulk single-element slab: the energy-weighted mean
    deposition depth d_bar (dose centroid) and the maximum penetration z_max (the
    deepest of Ne electrons ~ the practical range). Energy deposited per segment =
    |dE/ds|(E_seg) * L_seg (the model's own Joy-Luo stopping). Also the transmitted
    fraction, which must be ~0 so the slab captures the full range."""
    comp = [(element, n_per_ang3)]
    segs = simulate_trajectories(
        E0_keV,
        Ne,
        slab_ang,
        composition=comp,
        E_cut_keV=1.0,
        seed=seed,
        elastic_model="sr",  # analytic screened-Rutherford: no Mott table needed
    )
    z = segs["r_mid"][:, 2]
    dep = np.abs(_dEds_compound(comp, segs["E_keV"])) * segs["L_ang"]  # keV per segment
    d_bar = float(np.sum(dep * z) / np.sum(dep))
    return d_bar, float(z.max()), segs["n_transmitted"] / Ne


def check_depth_dose_kanaya_okayama():
    # carbon (diamond density) at two energies + aluminium at 30 keV
    runs = {}
    for tag, (el, n, E0, seed) in {
        "C20": ("C", 0.176, 20.0, 11),
        "C30": ("C", 0.176, 30.0, 12),
        "Al30": ("Al", 0.0603, 30.0, 13),
    }.items():
        R = _kanaya_okayama_ang(el, n, E0)
        d_bar, z_max, ft = _dose_depths(el, n, E0, slab_ang=3.0 * R, seed=seed)
        runs[tag] = dict(el=el, E0=E0, R=R, d_bar=d_bar, z_max=z_max, ft=ft)

    ok = True
    print("\nB. depth-dose vs Kanaya-Okayama range (slice 2 transport):")
    for r in runs.values():
        # the dose centroid is a sane fraction of R_KO; the max penetration is ~the
        # K-O range (the deepest electrons reach it); the slab caught the full range.
        fc, fr = r["d_bar"] / r["R"], r["z_max"] / r["R"]
        sane = (0.25 < fc < 0.55) and (0.78 < fr < 1.08) and (r["ft"] < 0.02)
        ok &= sane
        print(
            f"   {r['el']:>2} {r['E0']:4.0f} keV: R_KO={r['R'] / 1e4:5.2f} um  "
            f"d_bar={r['d_bar'] / 1e4:4.2f} um ({fc:.2f} R_KO)  "
            f"z_max={r['z_max'] / 1e4:4.2f} um ({fr:.2f} R_KO)  "
            f"transmit={r['ft']:.1%}  {'ok' if sane else 'BAD'}"
        )

    # K-O predicts the penetration RANGE, so scale z_max (not the dose centroid,
    # whose fraction of R_KO is Z-dependent because the profile shape changes with
    # Z -- higher-Z Al backscatters more, front-loading its dose).
    e_meas = runs["C30"]["z_max"] / runs["C20"]["z_max"]  # energy: same material
    e_pred = (30.0 / 20.0) ** 1.67
    e_rel = abs(e_meas - e_pred) / e_pred
    e_ok = e_rel < 0.10
    ok &= e_ok
    print(
        f"   energy   scaling z_max(C,30)/z_max(C,20)={e_meas:.3f}  vs  "
        f"(3/2)^1.67={e_pred:.3f}  (rel {e_rel:.2f})   {'PASS' if e_ok else 'FAIL'}"
    )
    m_meas = runs["Al30"]["z_max"] / runs["C30"]["z_max"]  # material: A/(Z^0.889 rho)
    m_pred = runs["Al30"]["R"] / runs["C30"]["R"]
    m_rel = abs(m_meas - m_pred) / m_pred
    m_ok = m_rel < 0.12
    ok &= m_ok
    print(
        f"   material scaling z_max(Al)/z_max(C)={m_meas:.3f}  vs  "
        f"R_KO(Al)/R_KO(C)={m_pred:.3f}  (rel {m_rel:.2f})   {'PASS' if m_ok else 'FAIL'}"
    )
    return ok


def main():
    ok = check_absorption_closed_form()
    ok &= check_depth_dose_kanaya_okayama()
    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
