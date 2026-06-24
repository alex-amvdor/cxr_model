"""Multilayer first-slice validation: mc_spectrum / mc_brem_spectrum with a
film-on-substrate absorber stack (docs/multilayer-materials.md).

Checks, on real MoSe2 transport segments:
  1. a single absorber layer [0, t_film] reproduces the single-slab path
     (layers=None) BIT-FOR-BIT -- the regression anchor;
  2. a substrate behind the film ATTENUATES the BACK-exit (positive-tilt) line
     and brem emission;
  3. the substrate is TRANSPARENT to the film's segments on a FRONT exit
     (negative tilt = high-flux geometry): the lines leave the entrance face,
     away from the substrate.

Run CPU-forced (the viz laptop has no CUDA toolkit):
    uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/multilayer_check.py',run_name='__main__')"
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_model.montecarlo import (  # noqa: E402
    simulate_trajectories,
    mc_spectrum,
    mc_brem_spectrum,
    tilted_geometry,
)
from cxr_model.sweep import crystal_params, film_on_substrate_layers  # noqa: E402

cp = crystal_params("mose2")
COMP = cp["composition"]
T_FILM = 500.0   # 50 nm film
T_SUB = 5e6      # 0.5 mm substrate
E0 = 30.0
THETA = np.deg2rad(90.0)
E_LINE = np.arange(350.0, 1750.0, 3.0)
E_BREM = np.arange(350.0, 30000.0, 50.0)


def _segments(tilt_deg):
    beam, n_hat = tilted_geometry(THETA, np.deg2rad(tilt_deg))
    segs = simulate_trajectories(E0, 400, T_FILM, composition=COMP, E_cut_keV=5.0, seed=1, beam_dir=beam)
    segs_b = simulate_trajectories(E0, 200, T_FILM, composition=COMP, E_cut_keV=1.0, seed=2, beam_dir=beam)
    return n_hat, segs, segs_b


def _line(segs, n_hat, layers):
    return mc_spectrum(
        segs, E_LINE, crystal="mose2", hkl_list=cp["hkl_list"], n_hat=n_hat,
        B_ang2=cp["B_ang2"], composition=COMP, beam_uvw=cp["beam_uvw"], layers=layers,
    )


def _brem(segs_b, n_hat, layers):
    return mc_brem_spectrum(segs_b, E_BREM, composition=COMP, n_hat=n_hat, layers=layers)


def main():
    stack = film_on_substrate_layers(COMP, T_FILM, "sio2", T_SUB)
    one = [(0.0, T_FILM, COMP)]
    ok = True
    for tilt, label in [(-30.0, "front exit (neg tilt, high-flux)"), (+30.0, "back exit (pos tilt)")]:
        n_hat, segs, segs_b = _segments(tilt)
        slab, one_l, st = _line(segs, n_hat, None), _line(segs, n_hat, one), _line(segs, n_hat, stack)
        b_slab, b_st = _brem(segs_b, n_hat, None), _brem(segs_b, n_hat, stack)

        rel = np.max(np.abs(slab - one_l) / np.maximum(np.abs(slab), 1e-300))
        bit = rel < 1e-9
        l_ratio = st.sum() / max(slab.sum(), 1e-300)
        b_ratio = b_st.sum() / max(b_slab.sum(), 1e-300)
        front = n_hat[2] < 0

        print(f"\n{label}:  n_z = {n_hat[2]:+.3f}")
        print(f"  bit-for-bit (layers=None == [0,t_film]): {'PASS' if bit else 'FAIL'}  (max rel {rel:.1e})")
        print(f"  line flux  slab={slab.sum():.4e}  stack={st.sum():.4e}  stack/slab={l_ratio:.4f}")
        print(f"  brem flux  slab={b_slab.sum():.4e}  stack={b_st.sum():.4e}  stack/slab={b_ratio:.4f}")

        ok &= bit
        if front:
            trans = abs(l_ratio - 1.0) < 1e-9 and abs(b_ratio - 1.0) < 1e-9
            print(f"  substrate transparent to film segments: {'PASS' if trans else 'FAIL'}")
            ok &= trans
        else:
            absb = (l_ratio < 0.9999) and (b_ratio < 0.9999)
            print(f"  substrate attenuates back-exit emission: {'PASS' if absb else 'FAIL'}")
            ok &= absb

    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
