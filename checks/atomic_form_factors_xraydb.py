"""
atomic_form_factors_xraydb.py  -- PROTOTYPE / EVALUATION ONLY (not in the pipeline)

A drop-in xraydb-backed replacement for src/atomic_form_factors.py, mirroring its
public API so you can A/B the library against the project's hard-coded data. Run
the companion checks/atomic_db_diff.py to quantify how far the numbers move:

    uv run --with xraydb python checks/atomic_db_diff.py

PROVENANCE NOTE -- the data SOURCE differs, so numbers will shift:
  * f0(g):  xraydb -> Waasmaier & Kirfel (1995).   project -> Cromer-Mann (ITC Vol C).
            (Both are 9-ish-parameter fits to the same Hartree-Fock atoms; agree <~0.5%.)
  * f1/f2:  xraydb -> Chantler (NIST FFAST).        project -> Henke/CXRO.
            Chantler's f1 carries a relativistic term, so f' = f1 - Z has a small
            roughly-constant offset vs Henke, and edge POSITIONS differ slightly.

If adopted for real, this is the body that would slot behind the existing
cromer_mann_f0 / henke_dispersion / atomic_form_factor signatures in src/.
"""

import numpy as np
import xraydb


def Z_of(element):
    """Atomic number (replaces the hard-coded Z_TABLE)."""
    return xraydb.atomic_number(element)


def cromer_mann_f0(element, g):
    """f0(g) [electrons], g = |reciprocal vector| = 2*pi/d [1/Ang]. xraydb.f0 takes
    q = sin(theta)/lambda = g/(4 pi), and uses the Waasmaier-Kirfel parameterization."""
    g = np.asarray(g, dtype=float)
    s = g / (4.0 * np.pi)
    return np.asarray(xraydb.f0(element, s), dtype=float)


def henke_dispersion(element, E_eV):
    """(f', f'') at photon energy E_eV [eV], from Chantler (FFAST). NOTE xraydb's
    f1_chantler/f2_chantler already return the ANOMALOUS corrections f' and f''
    (NOT the Henke-style f1 = Z + f'), so they map directly onto the project's
    (f', f'') contract -- no Z subtraction. atleast_1d keeps scalars index-safe."""
    E = np.asarray(E_eV, dtype=float)
    fp = np.atleast_1d(np.asarray(xraydb.f1_chantler(element, E), dtype=float))   # f'
    fpp = np.atleast_1d(np.asarray(xraydb.f2_chantler(element, E), dtype=float))  # f''
    return fp, fpp


def atomic_form_factor(element, g, E_eV):
    """Full complex F(g, E) = f0(g) + f'(E) + i f''(E)."""
    f0 = cromer_mann_f0(element, g)
    fp, fpp = henke_dispersion(element, E_eV)
    return (f0 + fp) + 1j * fpp
