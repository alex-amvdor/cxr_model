# CLAUDE.md

Agent / contributor operating guide for `cxr_model`. User-facing docs (physics,
install, materials, detectors, validation, provenance) live in
[`README.md`](README.md); design notes in [`docs/`](docs/); the backlog in
[`TODO.md`](TODO.md). This file holds only working conventions: build/test/run,
how to extend the code, and the non-obvious pitfalls.

## Behavioral guidelines

Bias toward caution over speed; for trivial tasks use judgment.

1. **Think before coding.** State assumptions explicitly; ask if uncertain. If
   multiple interpretations exist, surface them — don't pick silently. Push back
   when a simpler approach exists.
2. **Simplicity first.** Minimum code that solves the problem, nothing
   speculative. No abstractions for single-use code, no unrequested
   "flexibility," no error handling for impossible scenarios.
3. **Surgical changes.** Touch only what you must. Don't "improve" adjacent code
   or refactor what isn't broken; match existing style. Remove only the
   imports/variables your own changes orphaned — mention pre-existing dead code,
   don't delete it. Every changed line should trace to the request.
4. **Goal-driven execution.** Turn tasks into verifiable goals (a failing test
   to make pass, an anchor that must stay 1.00), then loop until verified.

## Build / test / run

- Dependencies live in the project venv managed by **uv**. Always run with
  `uv run python …`; a bare `python` on PATH has **no numpy** (a misleading
  "the code is broken" signal when it's just the wrong interpreter).
- **Tests:** `uv run python -m pytest -q` (CPU-only, fast; heavy MC physics
  validation lives in `checks/`, not here).
- **Analytic anchor:** `uv run python checks/feranchuk_check_script.py` (LiF, CPU).
- **CPU-force the MC anchors** on a machine with the cupy wheel but no CUDA by
  masking cupy, e.g.
  `uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/feranchuk_vs_zhai_check.py',run_name='__main__')"`
  (the 29 nm A/B ratio must stay **1.00**; same pattern for `checks/multilayer_check.py`).
- **Sweep (headless):** `cxr scan <material> [--quick] [--workers N]` (installed
  console script; or `uv run python scan.py <material> …` via the root shim)
  → writes `checkpoints/<material>.pkl`.
- **Remote compute, local viz:** `uv run python dev/remote.py {scan|start|pull|attach|status|logs|stop} …`
  — a *personal* author helper that runs `scan.py` on the `qlmc` GPU box over ssh and
  pulls the checkpoint back (host configurable via `CXR_REMOTE_*`; not shipped in the
  package). For a generic cluster, see README → *Running on a cluster*. Don't render
  PDFs on the box — pull and render locally.
- **Notebooks:** `scan.ipynb` (run a sweep) → `analysis.ipynb` (all viz). Both
  share the per-material grids in `src/cxr_model/config.py` — edit a grid there once and
  both pick it up. Notebooks are output-stripped on commit by `nbstripout`.
- **Docs site (optional):** `uv run --group docs sphinx-build -b html docs docs/_build/html`
  → open `docs/_build/html/index.html`. MyST renders `docs/*.md`; autosummary builds
  the API from docstrings (config in `docs/conf.py`). The `docs` dependency-group is
  not installed by default.
- **Docker (CPU):** `docker build -t cxr-model .` then
  `docker run --rm cxr-model pytest -q`. CPU-only (uv base + locked deps); the
  `Dockerfile` header documents the GPU base-swap.
- **Windows:** the Bash tool mangles `C:\…` paths and `&&` chains — use forward
  slashes (`C:/Users/…`) or prefer the dedicated Glob / Read / Grep / Edit tools.

## Code map

`src/cxr_model/` is the importable package (installable via `uv sync` / `pip
install -e .`; notebooks add `src/` to the path then `from cxr_model … import`);
the README has the per-module responsibility table. Packaged data resolves via
`cxr_model.DATA_DIR`, so imports work from any cwd. `*.pkl` / `*.png` are gitignored.

| Need to change… | Go to |
|---|---|
| crystal DB, structure factor, χ_g/U_g, reflections, constants | `src/cxr_model/crystallography.py` |
| electron transport + radiation + detector helpers + drivers | `src/cxr_model/montecarlo.py` |
| sweep definition (knobs → Cartesian product of cases) | `src/cxr_model/sweep.py` |
| per-material grids, default settings, sweep builders | `src/cxr_model/config.py` |
| checkpointed/resumable sweep driver, checkpoint loaders | `src/cxr_model/run.py` |
| result records, metrics, ranking/selection | `src/cxr_model/results.py` |
| all plotting | `src/cxr_model/plots.py` |
| Timepix3 / Eagle XO forward models | `src/cxr_model/timepix_response.py`, `src/cxr_model/eaglexo_response.py` |
| atomic form factors (xraydb-backed) | `src/cxr_model/atomic_form_factors.py` |

## Adding a material (or element)

These registries are NOT colocated; miss one and it fails *late* (a `KeyError`
in a worker), not at import. For a material reusing present elements, only the
(*) sites; a NEW ELEMENT needs all of them. Verify the whole chain with a single
`run_case` at tiny `Ne`.

1. (*) `src/cxr_model/data/crystal_structures.toml` — the `[material]` block (system, lattice,
   basis). 2H TMDs are isostructural with MoSe₂ (metal on 2c, chalcogen on 4f
   z=0.621 → δ=0.129 from the metal plane); sanity-check the M–X bond
   `sqrt((a/√3)² + (δc)²)` against the literature value.
2. (*) `src/cxr_model/config.py` — add to `_MATERIAL_GRIDS` (`MATERIALS` auto-derives).
3. (*) `src/cxr_model/sweep.py` — `MATERIAL_LABELS` **and** a `crystal_params()` branch
   (composition, `hkl_list`, `beam_uvw`, `B_ang2`).
4. `src/cxr_model/crystallography.py` — add to `_EDGE_PRONE` if any absorption edge lands
   in the ≤4.5 keV line grid (forces the complex resonant f0+f′+if″).
5. `src/cxr_model/montecarlo.py` — `TRANSPORT_ELEMENTS` (Z, A, `J_keV` = ICRU/NIST mean
   excitation energy in keV; e.g. Te = 0.485).
6. README — append the key to the materials table.

Atomic scattering data (Z, f0, f′, f″) comes from **xraydb** for any element —
no table to edit (see [`docs/atomic-data-sources.md`](docs/atomic-data-sources.md)).
NIST Mott transport tables (`src/cxr_model/data/mott_transport_cross_sections/`) are OPTIONAL:
a missing element falls back to analytic screened-Rutherford with a one-time warning.

## Physics conventions (read before touching geometry or amplitudes)

- **Frame:** incident beam along **+z**; detector at azimuth φ=0 in the x–z plane
  at polar angle θ_obs. At θ_obs=90° the detector is along +x.
- **Tilt sign (critical):** `tilt_deg` is the sample-normal polar tilt;
  `tilt_azim_deg` is its azimuth (0 = pitch in the beam–detector plane, 90 = yaw).
  **Negative tilt = entrance face toward the detector = HIGH flux** (radiation is
  born near the entrance and escapes a short path); positive tilt points the PXR
  lobe away → ~10× weaker. At θ_obs=90° the tilt must be nonzero (an untilted slab
  self-absorbs photons travelling along its faces; yaw alone gives identically zero).
- **Coherence:** the measured line is `|A_PXR + A_CBS|²` — never separable. CBS is
  weak only when F≈Z (low-g). Segments add incoherently; reflections within a
  segment add coherently.
- **HOPG fiber texture:** only (00l) reflections are coherent. Do **not** use
  `dominant_reflections` for `hopg` — pass (00l) only, beam along c-axis [001].
- **Units:** Ångström, eV (electron energies in keV where noted).
  `E2_EV_ANG = α·ħc = 14.3996 eV·Å`. χ_g = −r_e λ² S(g) e^(−W)/(πV). Resonance
  ω = v·g/(1−v·n̂). Segment lineshape |Q|² = t_L² sinc²(P·t_L).
- **Relativistic CBS** (matters ≳100 keV): braced {a;b} = a·b − (a·v)(b·v) with a
  1/γ prefactor; present at all amplitude sites.
- **Detector geometry is per-instrument** — never transfer numbers. Timepix3:
  θ_obs=90°, Δθ≈1.76°, Ω≈9.5e-4 sr. Zhai SEM/TEM values differ (see README).

## Gotchas

- **Worker re-import guard (spawn / forkserver):** `run_cases` uses a
  `ProcessPoolExecutor`; the start method is `spawn` (Windows) and — as of Python
  3.14 — `forkserver` (Linux), both of which re-import the entry module in each
  worker. Any script driving a sweep **must** have an `if __name__ == "__main__":`
  guard (`scan.py` does) or it relaunches itself recursively (surfaces as a
  `forkserver ConnectionResetError`). `python - <<EOF` fails the same way.
- **GPU ⇒ serial:** with CuPy present, `run_cases` runs serially (one CUDA
  context). Multiprocessing speedups apply only CPU-side. Workers run BELOW_NORMAL
  priority with BLAS pinned to 1 thread.
- **Mosaic MC nodes — moments converge, lineshape doesn't (yet):** `mosaic_route="mc"`
  averages `mc_spectrum` over `mosaic_nodes²` crystallite orientations (serial under
  CuPy, so a direct K× cost). Yield / second-moment width converge by ~5 nodes, but a
  *smooth broad lineshape* needs nodes scaling with mosaic/intrinsic width (HOPG ZYH ~35);
  too few → a lumpy multi-peak line, not a bug. Don't also apply the analytic term
  (`build_cases` makes the two routes mutually exclusive). See `docs/crystal-mosaicity.md`.
- **`apply_detector_qe` defaults False:** intrinsic spectra are shown unfiltered;
  the Timepix3 / Eagle XO views apply their own QE downstream.
- **Benign `divide by zero`:** the wide brem grid starts at 0 eV (λ→∞ at E=0), so
  `absorption_length_ang` is called at E=0; values are `nan_to_num`-clamped. Not a
  bug — don't "fix" it (the warning is suppressed in-function via `np.errstate`).
- **`dev/remote.py` line endings:** the tar-sync normalizes CRLF→LF for text files so
  the box stays `git status`-clean.
- **MoSe₂ structure:** Se z-fractional δ=0.129 (not 0.121 — a Wyckoff misread),
  verified by the 2.53 Å Mo–Se bond. A known ~×2 normalization source vs the paper.
- **Trajectory colouring (datashader):** aggregate tracks with
  `cvs.line(..., line_width=0)` then `tf.spread`; `line_width>0` paints a bogus
  radial gradient across the line. Sort segments by `(elec_id, t_ang)`.

## Validation

See README → *Validation* and the scripts in `checks/` (Feranchuk analytic
anchor, analytic-vs-MC agreement, kinematic audit, Zhai Fig 1c). The
Feranchuk–Spence analytic core (`checks/feranchuk_spence.py`) is a reference, not
part of the results pipeline.
