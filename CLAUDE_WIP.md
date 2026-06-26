# WIP snapshot — large-module refactor + marimo/altair

**Branch:** `refactor/large-modules` (off `main` @ `afe6660`). **Do NOT touch `main`.**
Decisions (from user, overnight autonomous run):
1. Refactor shape: **package + re-export** (preserve `from cxr_mc.montecarlo import X` exactly).
2. marimo: **port `feature/marimo-transfer` + ONE PoC notebook**.
3. altair: **add alongside** matplotlib (non-destructive new module).
4. Branches: **commit + push to origin** (NOT main, no PRs).

## Baseline
- `uv run pytest` trampoline is STALE (rename leftover) — use **`uv run python -m pytest`** instead. Baseline: **132 passed, exit 0**, green.
- `uv run ruff check .` / `ruff format .` for lint/format.
- CUDA-less box: CPU fallback; fast `tests/` are CPU-only.

## Task list (TaskCreate ids)
- #1 in_progress: montecarlo.py → package
- #2 pending: plots.py → package
- #3 pending: marimo port + PoC
- #4 pending: altair module

## TASK #1 — montecarlo.py (1725 lines) → `montecarlo/` package

### Planned split (all names re-exported from `__init__.py`):
- `_backend.py` — `cp`/`xp`/`_GPU`/`REAL`/`_to_cpu` (GPU probe + banner, runs once). **DONE (written).**
- `materials.py` — `_normalize_composition`, `_mu_total_inv_ang`, `_layer_dz`, `_stack_tau`. **DONE (written).**
- `transport.py` — `beta_from_keV`, `_sigma_browning_cm2`, `_alpha_sr_joy`, `_alpha_from_first_moment`,
  `_load_mott_transport`, `_mott_alpha_table`, `_NO_MOTT`, `_sample_cos_theta`, `_dEds_keV_per_ang`,
  `_dEds_compound`, `_rotate_directions`, `simulate_trajectories`, `TRANSPORT_ELEMENTS`,
  `MOTT_DIR`, `A0_SQ_CM2`. **TODO.**
- `geometry.py` — `tilted_geometry`, `detector_directions`, `_orientation_R`, `_small_tilt_R`,
  `_mosaic_quadrature`. **TODO.**
- `spectrum.py` — `_SEG_ARRAYS`, `_segments_in_layer`, `_polarization_pair`, `mc_spectrum`,
  `mc_spectrum_solid_angle`, `R_E_CM2`, `_brem_dsigma_dk`, `mc_brem_spectrum`, `load_external_brem`. **TODO.**
- `runner.py` — `run_case`, `_transport_case`, `_spectrum_case`, `_worker_init`, `run_cases`. **TODO.**
- `detector.py` — `detector_efficiency`, `eds_fwhm_eV`, `aperture_fwhm_eV`, `mosaic_fwhm_eV`,
  `mosaic_psi_rad`, `convolve_detector`. **TODO.**
- `__init__.py` — move the big physics module docstring here; explicit re-export of EVERY name
  below. **TODO.**

### Dependency DAG (acyclic, verified):
`_backend` → `materials` → `transport`/`geometry` → `spectrum` → `runner`; `detector` depends on
materials/geometry/transport/crystallography.

### Relative-import rule: submodules are one level deeper, so inside `montecarlo/<sub>.py`:
- `from . import DATA_DIR` → `from .. import DATA_DIR`
- `from .crystallography import ...` → `from ..crystallography import ...`
- `from .atomic_form_factors import load_henke` → `from ..atomic_form_factors import load_henke`
- intra-package: `from ._backend import ...`, `from .materials import ...`, etc.

### Per-submodule imports (planned):
- transport: `import os; import numpy as np; from functools import cache; from .. import DATA_DIR;
  from .materials import _normalize_composition`. (NOTE: transport is pure-numpy, no `_backend`.)
  `MOTT_DIR = str(DATA_DIR / "mott_transport_cross_sections")`. Keep `TRANSPORT_ELEMENTS` here.
- geometry: `import numpy as np; from ..crystallography import _direct_lattice_vectors, _rotation_between`.
- spectrum: `import numpy as np; from .._backend import xp, REAL, _to_cpu, _GPU, cp;
  from .materials import _normalize_composition, _mu_total_inv_ang, _layer_dz, _stack_tau;
  from .transport import beta_from_keV, TRANSPORT_ELEMENTS;
  from .geometry import _orientation_R, _mosaic_quadrature;
  from ..crystallography import CRYSTALS, chi_g, U_g, reciprocal_g_vector, ALPHA_FS, HBARC_EV_ANG, M_E_EV`.
  Keep `from ..atomic_form_factors import load_henke` LOCAL inside mc_spectrum (it is today).
- runner: `import os; import numpy as np; from .._backend import _GPU, cp;
  from .transport import simulate_trajectories; from .geometry import tilted_geometry;
  from .spectrum import mc_spectrum, mc_brem_spectrum, _segments_in_layer`.
  Keep `concurrent.futures` imports local inside functions.
- detector: `import numpy as np; from .materials import _mu_total_inv_ang;
  from .geometry import tilted_geometry, _orientation_R; from .transport import beta_from_keV;
  from ..crystallography import CRYSTALS, reciprocal_g_vector, HBARC_EV_ANG`.
  Keep `from scipy.ndimage import gaussian_filter1d` LOCAL inside convolve_detector.

### CRITICAL — names that EXTERNAL code imports from `cxr_mc.montecarlo` (must ALL be re-exported):
Public: `aperture_fwhm_eV, beta_from_keV, convolve_detector, detector_efficiency, eds_fwhm_eV,
load_external_brem, mosaic_fwhm_eV, mosaic_psi_rad, simulate_trajectories, tilted_geometry,
run_cases, run_case, mc_brem_spectrum, mc_spectrum, mc_spectrum_solid_angle, detector_directions,
TRANSPORT_ELEMENTS`.
Private (imported by tests/checks): `_normalize_composition, _layer_dz, _mu_total_inv_ang,
_stack_tau, _mosaic_quadrature, _small_tilt_R, _segments_in_layer, _spectrum_case,
_transport_case, _dEds_compound, _sigma_browning_cm2`.
Also re-export for safety: `xp, cp, _GPU, REAL, _to_cpu, _orientation_R, _polarization_pair,
_rotate_directions, _sample_cos_theta, _alpha_sr_joy, _alpha_from_first_moment,
_load_mott_transport, _mott_alpha_table, _dEds_keV_per_ang, _brem_dsigma_dk, _worker_init,
MOTT_DIR, A0_SQ_CM2, R_E_CM2, _NO_MOTT, _SEG_ARRAYS`.

### Pickling note (Windows spawn): `run_case/_transport_case/_spectrum_case/_worker_init` will live in
`cxr_mc.montecarlo.runner` (a real importable module) → pickle-by-reference still works. They stay
module-level. The GPU banner prints once per worker as before.

### Exact source line ranges in the ORIGINAL `src/cxr_mc/montecarlo.py` (now to be deleted):
- header/cupy/REAL/_to_cpu: 1-98 → split into `_backend` (done) + crystallography import block.
- crystallography import (lines 79-91): `ALPHA_FS, CRYSTALS, HBARC_EV_ANG, M_E_EV, U_g,
  _direct_lattice_vectors, _rotation_between, absorption_length_ang, chi_g, reciprocal_g_vector`.
- MOTT_DIR/A0_SQ_CM2/TRANSPORT_ELEMENTS: 101-121 (transport).
- beta_from_keV: 124-126 (transport).
- scattering models: 129-231 (transport).
- _dEds_keV_per_ang/_normalize_composition/_dEds_compound: 234-272 (transport; _normalize→materials).
- _mu_total_inv_ang: 275-291 (materials, DONE).
- _layer_dz/_stack_tau: 294-319 (materials, DONE).
- _rotate_directions: 322-335 (transport).
- tilted_geometry/detector_directions/_orientation_R/_small_tilt_R/_mosaic_quadrature: 338-489 (geometry).
- simulate_trajectories: 492-737 (transport).
- _SEG_ARRAYS/_segments_in_layer/_polarization_pair: 740-766 (spectrum).
- mc_spectrum: 769-1062 (spectrum).
- mc_spectrum_solid_angle: 1065-1108 (spectrum).
- R_E_CM2/_brem_dsigma_dk/mc_brem_spectrum: 1111-1245 (spectrum).
- run_case/_transport_case/_spectrum_case/_worker_init/run_cases: 1248-1567 (runner).
- load_external_brem: 1570-1599 (spectrum).
- detector model (detector_efficiency..convolve_detector): 1602-1725 (detector).

### Finish #1 by: write transport/geometry/spectrum/runner/detector + __init__; `git rm` the old
`montecarlo.py`; `uv run python -m pytest` must stay 132 passed; `ruff check .` clean; commit.

## TASK #2 — plots.py (2554 lines) → `plots/` package
Same pattern. Planned submodules: `_style` (COLORS, palettes, `_AXIS_SPECS`, `_HEATMAP_QUANTITIES`,
`_METRIC_LABELS`, `energy_color`, label helpers), `spectra`, `sweeps` (heatmaps/facet/metric/scan),
`detectors` (timepix/eaglexo), `trajectories`, `interactive` (browse/stream). Grep external importers
of `cxr_mc.plots` first (results.py, export.py, cli, notebooks) and re-export everything. plots.py
imports from `.montecarlo`, `.results`, `.timepix_response`, `.eaglexo_response`.

## TASK #3 — marimo
`feature/marimo-transfer` exists, is PRE-RENAME (`cxr_model`). Use the translate-patch port
(from session log 2026-06-26): `git diff <base> feature/marimo-transfer -- ':(exclude)TODO.md' |
sed 's/cxr_model/cxr_mc/g; s/cxr-model/cxr-mc/g'` then `git apply`. Then build ONE PoC marimo
notebook for analysis viz. Likely needs `marimo` + `altair` added to deps (pyproject `[dependency-groups]`).

## TASK #4 — altair
New non-destructive module (e.g. `src/cxr_mc/altair_plots.py` or `plots/altair_*`) for key
spectra/sweep figures; leave matplotlib `plots/` intact.

## Repo conventions
- New/edited physics needs derivation docstring + `Validation: <id>` + ledger row; verify physics in
  a FRESH context. (Refactor preserves physics verbatim — no new derivations, but keep all docstrings.)
- `docs/repo_map.md` MUST be updated after the refactor (regen via `docs:update-repo-map` skill, or hand-edit
  the montecarlo/plots sections to describe the new submodules).
- Notebooks commit output-free (nbstripout pre-commit). Pre-commit runs ruff/ruff-format/nbqa/nbstripout.
