"""Multilayer slice-3 validation: per-layer COHERENT radiation
(docs/multilayer-materials.md (2)).

mc_spectrum already radiates one crystal; slice 3 makes _spectrum_case radiate
EVERY crystalline layer of a film-on-substrate stack and sum them incoherently,
each self-absorbing through the whole stack. On real transport segments:

  1. A CRYSTALLINE substrate (silicon) gets its own electron segments and
     radiates NONZERO PXR/CBS lines of its own.
  2. The pipeline spectrum equals the incoherent sum of the per-layer
     contributions (film layer 0 + substrate layer 1) -- i.e. _spectrum_case
     really does sum the layers, with no cross-layer coherence.
  3. An AMORPHOUS substrate (sio2) radiates NO lines: its _spectrum_case
     spectrum is the film-layer contribution alone, BIT-FOR-BIT.
  4. Sanity: a crystalline substrate raises the total coherent line flux above
     the same film on an amorphous substrate.

Run CPU-forced (the viz laptop has the cupy wheel but no CUDA toolkit):
    uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/multilayer_slice3_check.py',run_name='__main__')"
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_model.montecarlo import (  # noqa: E402
    _segments_in_layer,
    _spectrum_case,
    _transport_case,
    mc_spectrum,
)
from cxr_model.sweep import Sweep, build_cases, substrate_radiator  # noqa: E402

E0 = 30.0
TILT = -30.0  # front exit (high flux); thin substrate so its lines escape
T_FILM = 300.0  # 30 nm MoSe2 film
T_SUB = 3000.0  # 0.3 um substrate -- thin enough that substrate lines aren't fully self-absorbed
E_LINE = np.arange(100.0, 3500.0, 3.0)  # wide: brackets BOTH the film and Si lines
E_BREM = np.arange(100.0, 30000.0, 100.0)


def _case(substrate):
    sw = Sweep(
        material="mose2",
        tilt_deg=TILT,
        energy_keV=E0,
        thickness_ang=T_FILM,
        substrate=substrate,
        substrate_thickness_ang=T_SUB,
        E_grid_line=E_LINE,
        E_grid_brem=E_BREM,
    )
    return build_cases(sw, n_electrons=600, n_electrons_brem=120)[0]


def _radiate_layer(segs, E_grid, n_hat, abs_layers, rad, L):
    """One layer's coherent spectrum, exactly as _spectrum_case computes it."""
    sL = _segments_in_layer(segs, L)
    if rad is None or sL["L_ang"].size == 0:
        return np.zeros(E_grid.shape)
    return mc_spectrum(
        sL,
        E_grid,
        crystal=rad["crystal"],
        hkl_list=rad["hkl_list"],
        n_hat=n_hat,
        B_ang2=rad["B_ang2"],
        composition=abs_layers[L][2],
        beam_uvw=rad["beam_uvw"],
        layers=abs_layers,
    )


def main():
    ok = True

    # --- crystalline substrate: MoSe2 film on a Si substrate --------------------
    case = _case("silicon")
    tp = _transport_case(case)
    out = _spectrum_case(case, tp)  # the slice-3 pipeline spectrum
    segs, n_hat, E_grid = tp["segs"], tp["n_hat"], tp["E_grid"]
    rads = case["layer_radiators"]
    n_sub = int(_segments_in_layer(segs, 1)["L_ang"].size)

    print(
        f"MoSe2 {T_FILM:.0f}A on silicon {T_SUB:.0f}A, {E0:.0f} keV, tilt {TILT:g} deg, "
        f"{segs['L_ang'].size} segments ({n_sub} in the substrate)\n"
    )

    spec_film = _radiate_layer(segs, E_grid, n_hat, case["abs_layers"], rads[0], 0)
    spec_sub = _radiate_layer(segs, E_grid, n_hat, case["abs_layers"], rads[1], 1)

    # 1. the crystalline substrate radiates its own lines -----------------------
    sub_int = float(np.trapezoid(spec_sub, E_grid))
    radiates = n_sub > 0 and sub_int > 0.0
    ok &= radiates
    print(
        f"1. crystalline substrate radiates its own lines:        "
        f"{'PASS' if radiates else 'FAIL'}  (Si line integral {sub_int:.3e})"
    )

    # 2. pipeline spectrum == incoherent sum of the per-layer parts -------------
    manual = spec_film + spec_sub
    denom = max(float(np.max(np.abs(manual))), 1e-300)
    rel = float(np.max(np.abs(out["spec"] - manual)) / denom)
    summed = rel < 1e-9
    ok &= summed
    print(
        f"2. pipeline == film(0) + substrate(1), incoherent sum:  "
        f"{'PASS' if summed else 'FAIL'}  (max rel {rel:.1e})"
    )

    # 3. amorphous substrate radiates NO lines (film-only, bit-for-bit) ---------
    assert substrate_radiator("sio2") is None
    case_a = _case("sio2")
    tp_a = _transport_case(case_a)
    out_a = _spectrum_case(case_a, tp_a)
    film_only = _radiate_layer(
        tp_a["segs"],
        tp_a["E_grid"],
        tp_a["n_hat"],
        case_a["abs_layers"],
        case_a["layer_radiators"][0],
        0,
    )
    bit = np.array_equal(out_a["spec"], film_only)
    ok &= bit
    print(f"3. amorphous substrate adds no lines (film-only b4b):    {'PASS' if bit else 'FAIL'}")

    # 4. sanity: crystalline substrate raises total coherent line flux ---------
    f_si = float(np.trapezoid(out["spec"], E_grid))
    f_sio2 = float(np.trapezoid(out_a["spec"], tp_a["E_grid"]))
    boosted = f_si > f_sio2
    ok &= boosted
    print(
        f"4. crystalline substrate boosts coherent flux:          "
        f"{'PASS' if boosted else 'FAIL'}  (Si {f_si:.3e} > SiO2 {f_sio2:.3e})"
    )

    print("\nALL CHECKS:", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
