"""
atomic_db_diff.py  -- how far do the numbers move if we swap the hard-coded atomic
data for xraydb? Compares, per element:
  * f0(g)            Cromer-Mann (project)  vs  Waasmaier-Kirfel (xraydb)
  * f' and f''(E)    Henke/CXRO  (project)  vs  Chantler/FFAST     (xraydb)
over the grids the pipeline actually uses (g <= 8 1/Ang; 0.1-30 keV line/abs band).

    uv run --with xraydb python checks/atomic_db_diff.py

EVALUATION ONLY -- nothing here is imported by the pipeline.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import atomic_form_factors as proj          # Henke + Cromer-Mann (project)
import atomic_form_factors_xraydb as xdb    # Chantler + Waasmaier (xraydb)

ELEMENTS = list(proj.CROMER_MANN)  # the structure-factor elements


def f0_table():
    g = np.linspace(0.0, 8.0, 81)  # |g| range of dominant_reflections
    print("f0(g): Cromer-Mann (ITC, project)  vs  Waasmaier-Kirfel (xraydb)")
    print(f"  {'el':>2}  {'f0(0) proj/xdb':>16}  {'max %':>7}  {'mean %':>7}")
    for el in ELEMENTS:
        a = proj.cromer_mann_f0(el, g)
        b = xdb.cromer_mann_f0(el, g)
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-9)
        print(f"  {el:>2}  {a[0]:7.2f}/{b[0]:<7.2f}  {100*rel.max():7.2f}  {100*rel.mean():7.2f}")


def disp_table():
    E = np.geomspace(100.0, 30000.0, 600)
    print("\nf'(E), f''(E): Henke/CXRO (project)  vs  Chantler/FFAST (xraydb), 0.1-30 keV")
    print(f"  {'el':>2}  {'|df`| max/mean [e]':>20}  {'f`` rel max/mean [%]':>22}")
    for el in ELEMENTS:
        fp_h, f2_h = proj.henke_dispersion(el, E)  # NaN outside the Henke range
        fp_c, f2_c = xdb.henke_dispersion(el, E)
        ok = np.isfinite(fp_h) & np.isfinite(f2_h) & (f2_h > 0)
        if not ok.any():
            print(f"  {el:>2}  (no overlapping Henke range)")
            continue
        dfp = np.abs(fp_h - fp_c)[ok]
        rel2 = (np.abs(f2_h - f2_c) / np.maximum(f2_h, 1e-3))[ok]
        print(
            f"  {el:>2}  {dfp.max():8.2f} / {dfp.mean():<8.2f}  "
            f"{100*rel2.max():9.1f} / {100*rel2.mean():<9.1f}"
        )


def headline():
    # Net effect on the PXR coupling: chi_g ~ S(g) built from (f0 + f'), so what
    # matters is the f0+f' real part at a representative reflection. Show it for a
    # few resonant elements at 1 keV (mid line-band), g = 2 1/Ang.
    print("\nReal part f0(g=2)+f' at 1 keV (the chi_g-relevant amplitude):")
    print(f"  {'el':>2}  {'proj':>8}  {'xdb':>8}  {'diff %':>7}")
    for el in ("C", "Se", "Mo", "Te", "W"):
        g = np.array([2.0])
        E = np.array([1000.0])
        re_p = float(proj.cromer_mann_f0(el, g)[0] + proj.henke_dispersion(el, E)[0][0])
        re_x = float(xdb.cromer_mann_f0(el, g)[0] + xdb.henke_dispersion(el, E)[0][0])
        d = 100 * abs(re_p - re_x) / max(abs(re_p), 1e-9)
        print(f"  {el:>2}  {re_p:8.3f}  {re_x:8.3f}  {d:7.2f}")


if __name__ == "__main__":
    print(f"xraydb {xdb.xraydb.__version__}; elements: {ELEMENTS}\n")
    f0_table()
    disp_table()
    headline()
