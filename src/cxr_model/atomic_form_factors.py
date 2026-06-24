"""
atomic_form_factors.py

Atomic form factor F(g, E) = f0(g) + f'(E) + i f''(E), sourced from xraydb
(https://github.com/xraypy/XrayDB, MIT code / CC0 data) -- no hard-coded tables.

  f0(g)        Waasmaier & Kirfel (1995) parameterization  -> xraydb.f0
  f'(E),f''(E) Chantler / NIST FFAST anomalous corrections  -> xraydb.f1_chantler,
               xraydb.f2_chantler  (these return f' and f'' DIRECTLY, i.e. the
               anomalous parts only -- NOT the Henke-style f1 = Z + f')
  Z            atomic numbers                                -> xraydb.atomic_number

The public surface is unchanged from the previous Henke/CXRO + Cromer-Mann
implementation -- cromer_mann_f0, henke_dispersion, atomic_form_factor, Z_TABLE,
load_henke -- so callers (crystallography.py, montecarlo.py, the checks/) are
untouched. The names cromer_mann_f0 / henke_dispersion are kept for compatibility
even though the underlying data is now Waasmaier-Kirfel / Chantler.

Conventions:
  g = |reciprocal lattice vector| = 2*pi / d_hkl   [1/Angstrom]  (NOT 1/d)
  E = photon energy                                 [eV]
  f0(s), s = sin(theta)/lambda = g/(4*pi).
"""

from functools import cache

import numpy as np
import xraydb


# ---- atomic numbers ---------------------------------------------------------
@cache
def _z_of(element):
    """Atomic number for an element symbol, or None if xraydb doesn't know it."""
    try:
        return int(xraydb.atomic_number(element))
    except Exception:
        return None


class _ElementZTable:
    """Mapping element-symbol -> atomic number, backed by xraydb (replaces the
    old hard-coded Z_TABLE dict). Supports the two access patterns callers use:
        Z_TABLE[el]      -> int Z, or KeyError if the symbol is unknown
        el in Z_TABLE    -> bool (is it a valid element symbol?)
    """

    def __getitem__(self, element):
        z = _z_of(element)
        if z is None:
            raise KeyError(f"Unknown element symbol '{element}'.")
        return z

    def __contains__(self, element):
        return _z_of(element) is not None


Z_TABLE = _ElementZTable()


# ---- f0(g): non-resonant, energy-independent --------------------------------
def cromer_mann_f0(element, g):
    """
    Energy-independent atomic form factor f0(g) (Waasmaier-Kirfel via xraydb;
    name kept for API compatibility with the old Cromer-Mann implementation).

    element : str element symbol (e.g. 'C')
    g       : float or array, reciprocal lattice vector magnitude [1/Angstrom]
              (g = 2*pi/d_hkl; NOT 1/d).
    Returns f0 in electron units (f0(0) = Z), shaped like g.
    """
    g_arr = np.asarray(g, dtype=float)
    s = (g_arr / (4.0 * np.pi)).ravel()
    f0 = np.asarray(xraydb.f0(element, s), dtype=float)
    return f0.reshape(g_arr.shape)


# ---- f'(E), f''(E): resonant dispersion + absorption ------------------------
@cache
def _chantler_bounds(element):
    """(Emin, Emax) [eV] of the element's tabulated Chantler/FFAST grid."""
    E = np.asarray(xraydb.chantler_energies(element), dtype=float)
    return float(E.min()), float(E.max())


def henke_dispersion(element, E_eV, on_out_of_range="nan"):
    """
    Energy-dependent dispersion corrections (Chantler/FFAST via xraydb; name kept
    for API compatibility with the old Henke/CXRO implementation).

    Returns (f_prime, f_double_prime) as arrays shaped like E_eV, where f' and f''
    are the anomalous corrections directly (xraydb.f1_chantler / f2_chantler).

    on_out_of_range :
        "nan"   -> energies outside the tabulated range return NaN (default),
                   keeping array length/alignment; downstream code tolerates NaN.
        "raise" -> raise ValueError (the old strict behavior).
    """
    if element not in Z_TABLE:
        raise KeyError(f"Unknown element symbol '{element}'.")
    Emin, Emax = _chantler_bounds(element)
    E = np.asarray(E_eV, dtype=float)
    shape = E.shape
    Eflat = np.atleast_1d(E).ravel()

    # strict interior: f1_chantler can fail right at the table endpoints, and the
    # brem grid passes E=0 (<= Emin) which must read as out-of-range anyway.
    in_range = (Eflat > Emin) & (Eflat < Emax)

    if on_out_of_range == "raise" and not np.all(in_range):
        bad = Eflat[~in_range]
        raise ValueError(
            f"E={bad} eV outside Chantler range [{Emin:.1f}, {Emax:.1f}] eV for {element}."
        )

    fp = np.full(Eflat.shape, np.nan)
    fpp = np.full(Eflat.shape, np.nan)
    if in_range.any():
        Ein = Eflat[in_range]
        fp[in_range] = np.asarray(xraydb.f1_chantler(element, Ein), dtype=float)
        fpp[in_range] = np.asarray(xraydb.f2_chantler(element, Ein), dtype=float)
    return fp.reshape(shape), fpp.reshape(shape)


def atomic_form_factor(element, g, E_eV, on_out_of_range="nan"):
    """
    Full complex atomic form factor F(g, E) = f0(g) + f'(E) + i f''(E).

    g    : reciprocal lattice vector magnitude [1/Angstrom] (= 2*pi/d_hkl)
    E_eV : photon energy [eV]

    Out-of-range energies return NaN (shape preserved) by default, so the result
    stays index-aligned with E_eV. Pass on_out_of_range="raise" for strict mode.
    """
    f0 = cromer_mann_f0(element, g)  # real, energy-independent
    fp, fpp = henke_dispersion(element, E_eV, on_out_of_range)
    return (f0 + fp) + 1j * fpp  # NaN where E out of range


# ---- Henke-style (E, f1, f2) table -----------------------------------------
@cache
def load_henke(element):
    """Element's anomalous-scattering table as (E [eV], f1, f2) arrays on the
    native Chantler/FFAST energy grid (which densely samples absorption edges --
    montecarlo relies on this to resolve the edge jumps). f1 = Z + f' is the
    Henke-convention forward-scattering factor; f2 = f''. Cached per element.
    """
    E = np.asarray(xraydb.chantler_energies(element), dtype=float)
    E = E[E.min() < E]  # drop the single endpoint where f1_chantler is unstable
    Z = Z_TABLE[element]
    fp = np.asarray(xraydb.f1_chantler(element, E), dtype=float)
    f2 = np.asarray(xraydb.f2_chantler(element, E), dtype=float)
    return E, Z + fp, f2
