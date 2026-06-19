"""
atomic_form_factors.py

Atomic form factor F(g, E) = f0(g) + f'(E) + i f''(E)
combining Cromer-Mann (angular, energy-independent) with
Henke/CXRO (energy-dependent dispersion + absorption).

Conventions:
  g = |reciprocal lattice vector| = 2*pi / d_hkl   [1/Angstrom]
  E = photon energy                                 [eV]
  Henke files (./atomic_scattering_factors/<element>.csv): columns E[eV], f1, f2
      with f1 = Z + f' (forward scattering), f2 = f''.
  Cromer-Mann: f0(s) = sum_i a_i exp(-b_i s^2) + c,  s = sin(theta)/lambda = g/(4*pi).
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

# resolve relative to this file (src/) so imports work from any working
# directory; the data lives in the sibling data/ directory
SCATTERING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "atomic_scattering_factors",
)

# Atomic numbers
Z_TABLE = {
    "C": 6,
    "Li": 3,
    "F": 9,
    "Si": 14,
    "Ge": 32,
    "Mo": 42,
    "Se": 34,
    "W": 74,
    "S": 16,
    "Te": 52,
    "Zr": 40,
    "Hf": 72,
    "Pt": 78,  # 1T dichalcogenide metals
    # detector-window / light elements (Henke data only -- no Cromer-Mann
    # coefficients; used for absorption, not structure factors)
    "H": 1,
    "N": 7,
    "O": 8,
    "Al": 13,
}

# Cromer-Mann coefficients (a1,b1,a2,b2,a3,b3,a4,b4,c)
# Neutral atoms, from International Tables for Crystallography Vol. C, Table 6.1.1.4.
CROMER_MANN = {
    "C": (
        2.31000,
        20.8439,
        1.02000,
        10.2075,
        1.58860,
        0.56870,
        0.86500,
        51.6512,
        0.21560,
    ),
    "Si": (
        6.29150,
        2.43860,
        3.03530,
        32.3337,
        1.98910,
        0.67850,
        1.54100,
        81.6937,
        1.14070,
    ),
    "Ge": (
        16.0816,
        2.85090,
        6.37470,
        0.25160,
        3.70680,
        11.4468,
        3.68300,
        54.7625,
        2.13130,
    ),
    "Mo": (
        3.70250,
        0.27720,
        17.2356,
        1.09580,
        12.8876,
        11.0040,
        3.74290,
        61.6584,
        4.38750,
    ),
    "Se": (
        17.0006,
        2.40980,
        5.81960,
        0.27260,
        3.97310,
        15.2372,
        4.35430,
        43.8163,
        2.84090,
    ),
    "S": (
        6.90530,
        1.46790,
        5.20340,
        22.2151,
        1.43790,
        0.253600,
        1.58630,
        56.1720,
        0.866900,
    ),
    "Te": (
        19.9644,
        4.81742,
        19.0138,
        0.420885,
        6.14487,
        28.5284,
        2.52390,
        70.8403,
        4.35200,
    ),
    "W": (
        29.0818,
        1.72029,
        15.4300,
        9.22590,
        14.4327,
        0.321703,
        5.11982,
        57.0560,
        9.88750,
    ),
    "Zr": (
        17.8765,
        1.27618,
        10.9480,
        11.9160,
        5.41732,
        0.117622,
        3.65721,
        87.6627,
        2.06929,
    ),
    "Hf": (
        29.1440,
        1.83262,
        15.1726,
        9.59990,
        14.7586,
        0.275116,
        4.30013,
        72.0290,
        8.58154,
    ),
    "Pt": (
        27.0059,
        1.51293,
        17.7639,
        8.81174,
        15.7131,
        0.424593,
        5.78370,
        38.6103,
        11.6883,
    ),
    "Li": (1.1282, 3.9546, 0.7508, 1.0524, 0.6175, 85.3905, 0.4653, 168.261, 0.0377),
    "F": (3.5392, 10.2825, 2.6412, 4.2944, 1.517, 0.2615, 1.0243, 26.1476, 0.2776),
}


def cromer_mann_f0(element, g):
    """
    Energy-independent atomic form factor f0(g) from the Cromer-Mann fit.

    element : str, lowercase element name (e.g. 'C')
    g       : float or array, reciprocal lattice vector magnitude [1/Angstrom]
              (g = 2*pi/d_hkl; NOT 1/d).
    Returns f0 in electron units (f0(0) = Z).
    """
    if element not in CROMER_MANN:
        raise KeyError(
            f"No Cromer-Mann coefficients for '{element}'. Have: {list(CROMER_MANN)}"
        )
    coeffs = CROMER_MANN[element]
    a = coeffs[0:8:2]
    b = coeffs[1:8:2]
    c = coeffs[8]
    g = np.asarray(g, dtype=float)
    s = g / (4.0 * np.pi)  # s = sin(theta)/lambda
    s2 = s * s
    f0 = np.full_like(s, c)
    for ai, bi in zip(a, b):
        f0 = f0 + ai * np.exp(-bi * s2)
    return f0


@lru_cache(maxsize=None)
def load_henke(element, sentinel=-9999.0):
    """Load the Henke table for an element as (E [eV], f1, f2) arrays.
    Cached: each element's CSV is read from disk once per process."""
    path = os.path.join(SCATTERING_DIR, f"{element}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Henke file: {path}")
    # Data rows have a trailing tab -> phantom 4th column. Split on whitespace
    # and take only the first three columns. skiprows=1 drops the text header.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=1,
        header=None,
        usecols=[0, 1, 2],
        names=["E", "f1", "f2"],
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError(f"No rows parsed from {path}; check delimiter/columns.")
    E = df["E"].to_numpy()
    f1 = df["f1"].to_numpy()
    f2 = df["f2"].to_numpy()
    good = f1 > sentinel + 1.0
    if good.sum() == 0:
        raise ValueError(f"All f1 rows are sentinel in {path}.")
    return E[good], f1[good], f2[good]


def henke_dispersion(element, E_eV, on_out_of_range="nan"):
    """
    Energy-dependent dispersion corrections from Henke data, interpolated to E_eV.

    Returns (f_prime, f_double_prime) where f' = f1(E) - Z, f'' = f2(E),
    as arrays the same shape as E_eV.

    on_out_of_range :
        "nan"   -> points outside the tabulated range return NaN (default).
                   Keeps array length/alignment; downstream code must tolerate NaN.
        "raise" -> raise ValueError (the old strict behavior).
    """
    if element not in Z_TABLE:
        raise KeyError(f"No Z for '{element}'. Have: {list(Z_TABLE)}")
    Z = Z_TABLE[element]
    E_tab, f1_tab, f2_tab = load_henke(element)
    E_eV = np.asarray(E_eV, dtype=float)

    in_range = (E_eV >= E_tab.min()) & (E_eV <= E_tab.max())

    if on_out_of_range == "raise" and not np.all(in_range):
        bad = E_eV[~in_range]
        raise ValueError(
            f"E={bad} eV outside Henke range "
            f"[{E_tab.min():.1f}, {E_tab.max():.1f}] eV for {element}."
        )

    # np.interp clamps to endpoints outside the range; we overwrite those with NaN.
    f1 = np.interp(E_eV, E_tab, f1_tab)
    f2 = np.interp(E_eV, E_tab, f2_tab)
    fp = f1 - Z
    fp = np.where(in_range, fp, np.nan)
    f2 = np.where(in_range, f2, np.nan)
    return fp, f2


def atomic_form_factor(element, g, E_eV, on_out_of_range="nan"):
    """
    Full complex atomic form factor F(g, E) = f0(g) + f'(E) + i f''(E).

    g    : reciprocal lattice vector magnitude [1/Angstrom] (= 2*pi/d_hkl)
    E_eV : photon energy [eV]

    Out-of-Henke-range energies return NaN (same shape preserved) by default,
    so the result stays index-aligned with E_eV. Pass on_out_of_range="raise"
    for the old strict behavior.
    """
    f0 = cromer_mann_f0(element, g)  # real, energy-independent
    fp, fpp = henke_dispersion(element, E_eV, on_out_of_range)
    return (f0 + fp) + 1j * fpp  # NaN where E out of range
