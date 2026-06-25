"""cxr_mc -- coherent X-ray radiation (PXR + coherent bremsstrahlung) from
table-top electron beams in crystals.

The package is intentionally light at import time: ``import cxr_mc`` pulls in
no heavy dependencies (matplotlib / cupy / the MC pipeline). Import submodules
explicitly, e.g. ``from cxr_mc import crystallography`` or
``from cxr_mc.montecarlo import run_cases``.

See the README for the scientific overview and CLAUDE.md for working conventions.
"""

from pathlib import Path

__version__ = "0.1.0"

# Packaged data (crystal_structures.toml, mott_transport_cross_sections/,
# eaglexo_qe.csv, legacy atomic_scattering_factors/). Resolved relative to this
# file so it works installed (wheel) or from a source checkout.
DATA_DIR = Path(__file__).parent / "data"

__all__ = ["DATA_DIR", "__version__"]
