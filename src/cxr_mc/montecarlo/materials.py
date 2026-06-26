"""
montecarlo.materials

Composition handling and X-ray self-absorption shared across transport,
spectrum and detector: normalize a single-element / compound material spec,
the total linear attenuation summed over elements, and the layered
(film-on-substrate) Beer-Lambert optical depth.
"""

import numpy as np

from ..crystallography import absorption_length_ang
from ._backend import _GPU, REAL, _to_cpu, cp


def _normalize_composition(element, n_atoms_per_ang3, composition):
    """
    Accept either the single-element API (element=, n_atoms_per_ang3=) or a
    compound composition=[(element, number_density_1_per_Ang3), ...];
    return the latter form.
    """
    if composition is not None:
        return [(el, float(n)) for el, n in composition]
    if element is None or n_atoms_per_ang3 is None:
        raise ValueError(
            "specify the material: pass composition=[(element, n_per_Ang3), ...] "
            "(or both element= and n_atoms_per_ang3=). Refusing to fall back to a "
            "default element so a material can't be silently mis-loaded."
        )
    return [(element, float(n_atoms_per_ang3))]


def _mu_total_inv_ang(comp, E_eV):
    """Total linear attenuation 1/L_abs [1/Angstrom] summed over elements.

    absorption_length_ang (from crystallography) is CPU-only, so the sum
    is always computed on the CPU. The result is returned on the SAME device as
    E_eV: a GPU array if the caller passed one (mc_spectrum, mixing it with
    on-device factors), a numpy array otherwise (detector_efficiency, whose
    output is multiplied into the host-side spectra in the notebook). Keying off
    the input device -- not the global _GPU flag -- keeps the CPU post-processing
    path numpy even when a GPU is present."""
    E_cpu = _to_cpu(E_eV)
    mu = 0.0
    for el, n_i in comp:
        mu = mu + 1.0 / absorption_length_ang(el, E_cpu, n_i)
    if _GPU and isinstance(E_eV, cp.ndarray):
        return cp.asarray(mu, dtype=REAL)
    return mu


# ---- layered (film-on-substrate) self-absorption ----------------------------
def _layer_dz(z_mid, n_z, z_top, z_bot):
    """z-extent of the layer [z_top, z_bot] that a photon leaving depth z_mid
    along n_hat (z-component n_z) crosses on its way out: toward z=0 when n_z<0
    (the entrance face) or the back face when n_z>0. numpy ufuncs are used so the
    same code serves numpy or cupy z_mid. Returns an array shaped like z_mid."""
    if n_z < 0:  # escape ray spans depths [0, z_mid]
        return np.maximum(np.minimum(z_mid, z_bot) - z_top, 0.0)
    return np.maximum(z_bot - np.maximum(z_mid, z_top), 0.0)  # spans [z_mid, z_total]


def _stack_tau(layers, z_mid, n_z, E):
    """Beer-Lambert optical depth for a photon leaving each segment midpoint
    (depth z_mid) along n_hat through a LAYERED absorber stack:
        tau = (1/|n_z|) * sum_i mu_i(E) * dz_i
    layers = [(z_top, z_bot, composition), ...] top (entrance) first, contiguous,
    the deepest z_bot being the total stack thickness. z_mid and E are per-segment
    arrays (E the resonance energy); the result matches their device. A single
    layer over [0, total_thickness] reproduces the single-slab escape exactly,
    so passing layers=None elsewhere stays bit-for-bit identical."""
    inv = 1.0 / max(abs(float(n_z)), 1e-12)
    tau = 0.0
    for z_top, z_bot, comp in layers:
        dz = _layer_dz(z_mid, n_z, float(z_top), float(z_bot))
        tau = tau + _mu_total_inv_ang(comp, E) * dz * inv
    return tau
