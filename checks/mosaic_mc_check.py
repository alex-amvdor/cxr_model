"""Exact Monte-Carlo crystal-mosaicity validation (docs/crystal-mosaicity.md route 2).

mc_spectrum now averages the line spectrum over crystallite orientations via a 2-D
Gauss-Hermite product quadrature (mosaic_fwhm_rad / mosaic_nodes). This is the
design doc's validation plan, on real HOPG transport:

  1. PERFECT-CRYSTAL bit-for-bit: nodes<=1 or fwhm=None reproduces the no-mosaic
     spectrum exactly (the K=1 guard old checkpoints / default runs rely on).
  2. eta -> 0 converges to the perfect crystal.
  3. The line BROADENS and its peak drops monotonically with the mosaic grade
     (ZYA 0.4 < ZYB 0.8 < ZYH 3.5 deg).
  4. The orientation quadrature is CONVERGED at the default node count
     (nodes=5 vs 9 agree on the integrated line).
  5. Small-eta limit: the added width (in quadrature) matches the analytic
     E*|tan psi|*eta the shipped model uses -- the two routes agree where the
     analytic one is valid.
  6. The integrated YIELD changes (the exact route is NOT a pure area-preserving
     convolution, unlike the analytic term); the ratio is reported.

Run CPU-forced (the viz laptop has the cupy wheel but no CUDA toolkit):
    uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/mosaic_mc_check.py',run_name='__main__')"
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import itertools

from cxr_model.montecarlo import (
    mc_spectrum,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
    simulate_trajectories,
    tilted_geometry,
)
from cxr_model.sweep import crystal_params

E0_KEV = 30.0
THETA_OBS = np.deg2rad(90.0)
TILT_DEG = -45.0  # moderate: psi ~ 45 deg, tan(psi) ~ 1, away from grazing
THICKNESS = 290.0  # thin film -> narrow intrinsic line, cleanest broadening test
NE = 250
SEED = 11
E_GRID = np.arange(400.0, 1400.0, 0.5)  # brackets HOPG (002) ~860 eV; (004) excluded

cp = crystal_params("hopg")
COMP = cp["composition"]


def _line_fwhm_eV(y, E=E_GRID):
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    pk = y[i]
    if not np.isfinite(pk) or pk <= 0.0:
        return float(E[i]), np.nan

    def _cross(e0, e1, y0, y1, h):
        return e0 + (h - y0) * (e1 - e0) / (y1 - y0)

    h = 0.5 * pk
    L = i
    while L > 0 and y[L] > h:
        L -= 1
    eL = float(E[0]) if y[L] > h else _cross(E[L], E[L + 1], y[L], y[L + 1], h)
    R = i
    while len(y) - 1 > R and y[R] > h:
        R += 1
    eR = float(E[-1]) if y[R] > h else _cross(E[R - 1], E[R], y[R - 1], y[R], h)
    return float(E[i]), eR - eL


def main():
    beam, n_hat = tilted_geometry(THETA_OBS, np.deg2rad(TILT_DEG))
    segs = simulate_trajectories(
        E0_KEV, NE, THICKNESS, composition=COMP, E_cut_keV=5.0, seed=SEED, beam_dir=beam
    )

    def spec(fwhm_deg=None, nodes=1):
        fwhm_rad = None if fwhm_deg is None else np.deg2rad(fwhm_deg)
        return mc_spectrum(
            segs,
            E_GRID,
            crystal="hopg",
            hkl_list=cp["hkl_list"],
            n_hat=n_hat,
            B_ang2=cp["B_ang2"],
            composition=COMP,
            beam_uvw=cp["beam_uvw"],
            mosaic_fwhm_rad=fwhm_rad,
            mosaic_nodes=nodes,
        )

    perfect = spec()
    integral = lambda y: float(np.trapezoid(y, E_GRID))  # noqa: E731
    ok = True

    print(
        f"HOPG (002), {E0_KEV:.0f} keV, theta_obs 90 deg, tilt {TILT_DEG:g} deg, "
        f"{THICKNESS:.0f} A film, {segs['L_ang'].size} segments\n"
    )

    # 1. perfect-crystal bit-for-bit -------------------------------------------
    b1 = np.array_equal(perfect, spec(fwhm_deg=3.5, nodes=1))  # nodes<=1 -> identity
    b2 = np.array_equal(perfect, spec(fwhm_deg=None, nodes=5))  # no spread -> identity
    bit = b1 and b2
    ok &= bit
    print(f"1. perfect-crystal bit-for-bit (nodes=1, fwhm=None): {'PASS' if bit else 'FAIL'}")

    # 2. eta -> 0 ---------------------------------------------------------------
    tiny = spec(fwhm_deg=1e-4, nodes=5)
    rel = np.max(np.abs(tiny - perfect) / max(perfect.max(), 1e-300))
    conv0 = rel < 1e-5
    ok &= conv0
    print(
        f"2. eta -> 0 converges to perfect:                    "
        f"{'PASS' if conv0 else 'FAIL'}  (max rel {rel:.1e})"
    )

    # 3. broadening + peak drop. The half-max CORE width is the physical line width,
    #    but it is lumpy for a broad grade until the orientation nodes resolve it
    #    (node spacing < intrinsic core), so each grade uses a node count adequate
    #    for ITS spread. (A second-moment width is no good here -- it is dominated by
    #    the physical multiple-scattering Doppler skirt, not the mosaic core.) -----
    E_pk, fw_perfect = _line_fwhm_eV(perfect)
    grades = [("ZYA", 0.4, 9), ("ZYB", 0.8, 13), ("ZYH", 3.5, 41)]
    specs = {name: spec(fwhm_deg=d, nodes=n) for name, d, n in grades}
    cores = [fw_perfect] + [_line_fwhm_eV(specs[n])[1] for n, _, _ in grades]
    mono_w = all(a < b for a, b in itertools.pairwise(cores))
    peak_drop = all(perfect.max() > specs[n].max() for n, _, _ in grades)
    big = cores[-1] > 3.0 * cores[0]  # ZYH clearly broadens the core
    ok &= mono_w and peak_drop and big
    print(
        f"3. broadens (core width up) + peak drops with grade: "
        f"{'PASS' if (mono_w and peak_drop and big) else 'FAIL'}"
    )
    print(
        f"     core FWHM [eV]  perfect={cores[0]:6.1f} | "
        + " | ".join(f"{n} {c:6.1f}" for (n, _, _), c in zip(grades, cores[1:], strict=False))
    )
    # the half-max lineshape of a BROAD grade stays lumpy until nodes resolve it:
    hm = [_line_fwhm_eV(spec(fwhm_deg=3.5, nodes=n))[1] for n in (7, 15, 31)] + [cores[-1]]
    print(
        "     ZYH half-max FWHM at nodes 7/15/31/41: "
        + " ".join(f"{h:.0f}" for h in hm)
        + "  (lumpy until node spacing < core)"
    )

    # 4. node convergence at the default ---------------------------------------
    i5 = integral(spec(fwhm_deg=3.5, nodes=5))
    i9 = integral(spec(fwhm_deg=3.5, nodes=9))
    conv_n = abs(i5 - i9) / max(abs(i9), 1e-300) < 0.02
    ok &= conv_n
    print(
        f"4. quadrature converged (nodes 5 vs 9, ZYH):         "
        f"{'PASS' if conv_n else 'FAIL'}  (rel {abs(i5 - i9) / abs(i9):.1e})"
    )

    # 5. small-eta -> analytic E|tan psi|eta ------------------------------------
    # the added width (quadrature) should track the analytic slope where it is valid
    case = dict(
        crystal="hopg",
        theta_obs_rad=THETA_OBS,
        tilt_deg=TILT_DEG,
        tilt_azim_deg=0.0,
        E0_keV=E0_KEV,
        hkl_list=cp["hkl_list"],
        beam_uvw=cp["beam_uvw"],
        azimuth_rad=0.0,
    )
    psi = mosaic_psi_rad(case, E_pk)
    print(f"5. small-eta vs analytic E|tan psi|eta (psi={np.degrees(psi):.0f} deg):")
    band_ok = True
    for d in (0.2, 0.4):
        fw = _line_fwhm_eV(spec(fwhm_deg=d, nodes=7))[1]
        added = np.sqrt(max(fw**2 - fw_perfect**2, 0.0))  # quadrature-subtract intrinsic
        analytic = mosaic_fwhm_eV(E_pk, psi, np.deg2rad(d))
        ratio = added / analytic if analytic > 0 else np.nan
        within = 0.6 <= ratio <= 1.5
        band_ok &= within
        print(
            f"     eta={d:g} deg: added={added:5.1f}  analytic={analytic:5.1f}  "
            f"ratio={ratio:.2f}  {'ok' if within else 'OUT'}"
        )
    ok &= band_ok

    # 6. yield (reported): the exact average is not an area-preserving convolution,
    #    but for this fixed-detector, whole-line observable it stays NEAR-CONSERVED
    #    here -- the broadening, not a yield change, dominates (the literature's
    #    larger mosaic yield gains are for other geometries/observables). Sanity
    #    bound only -- a gross departure would flag an amplitude/weighting bug.
    y_ratio = integral(specs["ZYH"]) / max(integral(perfect), 1e-300)
    sane = 0.5 < y_ratio < 2.0
    ok &= sane
    print(
        f"6. integrated yield ZYH/perfect = {y_ratio:.3f} (near-conserved): "
        f"{'PASS' if sane else 'FAIL'}"
    )

    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
