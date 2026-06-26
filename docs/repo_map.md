# Repository map

Navigation aid for `src/cxr_mc/` — the importable package. Read this before
exploring source. For *why* (physics, validation, provenance) see
[`README.md`](../README.md) and the design notes in [`docs/`](.); for the backlog
see [`TODO.md`](../TODO.md). Regenerate with the `docs:update-repo-map` command.

## Dependency layers (leaf → driver)

```
atomic_form_factors            (xraydb-backed atomic data; no sibling deps)
        │
crystallography                (crystal DB, structure factor, χ_g/U_g, μ)
        │
montecarlo                     (transport + radiation + detector helpers)
        │
sweep ─────────────┐
        │          │
results ◄── montecarlo, sweep
config  ◄── results, sweep
run     ◄── montecarlo, results
scan    ◄── config, run, sweep
plots   ◄── montecarlo, results, timepix_response, eaglexo_response
cli     ◄── scan, export          (the `cxr` console script)
```

`timepix_response` / `eaglexo_response` depend only on `crystallography`.
Packaged data resolves via `cxr_mc.DATA_DIR`, so imports work from any cwd.

## Entry points

- **`cxr` console script** → `cli:main` (`pyproject.toml [project.scripts]`),
  dispatching the `scan`, `export` and `slim` subcommands.
- **`cxr scan <material>`** → `scan:main` → `run.run_sweep` → writes
  `checkpoints/<material>.pkl`. Root shim: `scan.py`.
- **`cxr slim <checkpoint>`** → `slim:slim_checkpoint` → `results.slim_results`:
  shrink a checkpoint pickle for transfer (drop wide-brem / float32 / filter configs).
- **Notebooks**: `scan.ipynb` (sweep) → `analysis.ipynb` (viz); both read the
  per-material grids in `config.py`.
- **Sweep worker**: `montecarlo.run_case` (module-level so it pickles into the
  `run_cases` process pool).

---

## Core physics

### `crystallography.py`
Crystal database, structure factors, and X-ray optical constants — the physics
data layer under the Monte Carlo.
- Public: `load_crystals`, `reciprocal_g_vector`, `g_mag`, `debye_waller`,
  `structure_factor`, `chi_g`, `U_g`, `absorption_length_ang`,
  `dominant_reflections`, `beta_from_Ee`; the `CRYSTALS` registry.
- Deps: `atomic_form_factors`, `DATA_DIR`.

### `atomic_form_factors.py`
Atomic scattering factors (Z, f0, f′, f″) from **xraydb** for any element — no
hand-maintained table.
- Public: `cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`,
  `load_henke`.
- Deps: none (leaf; external xraydb).

### `montecarlo/` (package)
The simulation core: electron transport, the segment-sum PXR+CBS line spectrum,
bremsstrahlung, the parallel case runner, and detector-convolution helpers.
Split from a single module into submodules; **every public and internal name is
re-exported from the package**, so `from cxr_mc.montecarlo import X` is unchanged
(`tests/test_montecarlo_exports.py` freezes the export set).
- `_backend` — GPU/CPU array backend probe + banner: `xp`, `cp`, `REAL`,
  `_to_cpu`, `_GPU`.
- `materials` — `_normalize_composition`, `_mu_total_inv_ang`, `_layer_dz`,
  `_stack_tau` (composition + cross-stack self-absorption). Deps: `_backend`,
  `crystallography`.
- `transport` — `simulate_trajectories` (multilayer-stack aware via `layers=`),
  `beta_from_keV`, scattering/stopping helpers; the `TRANSPORT_ELEMENTS`
  registry. Pure NumPy. Deps: `materials`, `DATA_DIR`.
- `geometry` — `tilted_geometry`, `detector_directions`, `_orientation_R`,
  `_small_tilt_R`, `_mosaic_quadrature`. Deps: `crystallography`.
- `spectrum` — `mc_spectrum` (PXR+CBS, cross-stack self-absorption, exact mosaic
  average), `mc_spectrum_solid_angle`, `mc_brem_spectrum`, `load_external_brem`.
  Deps: `_backend`, `materials`, `transport`, `geometry`, `crystallography`.
- `detector` — `detector_efficiency`, `eds_fwhm_eV`, `aperture_fwhm_eV`,
  `mosaic_fwhm_eV`, `mosaic_psi_rad`, `convolve_detector`. Deps: `materials`,
  `geometry`, `transport`, `crystallography`.
- `runner` — `run_case`, `run_cases` (GPU-serial / CPU-pooled), `_transport_case`,
  `_spectrum_case`, `_worker_init`. Deps: `_backend`, `transport`, `geometry`,
  `spectrum`.
- Deps: `crystallography`, `DATA_DIR`.

## Sweep, config & drivers

### `sweep.py`
Turns a `Sweep` definition into the Cartesian product of `run_case` dicts.
- Public: `Sweep` (dataclass of all knobs), `build_cases`, `crystal_params`,
  `substrate_composition`, `film_on_substrate_layers`, `geometry_table`,
  `fmt_thickness`; the `MATERIAL_LABELS` registry.
- Deps: `crystallography`.

### `config.py`
Per-material grids and the default settings/sweep builders shared by the CLI and
both notebooks.
- Public: `default_settings`, `material_grid`, `material_sweep`,
  `trajectory_sweep`; the `_MATERIAL_GRIDS` / `MATERIALS` registries.
- Deps: `results` (`Settings`), `sweep` (`Sweep`).

### `run.py`
Checkpointed, resumable sweep driver and checkpoint loaders/repair.
- Public: `run_sweep`, `load_checkpoint`, `checkpoint_path_for`,
  `cases_from_results`, `repair_brem_wide`, `repair_checkpoint`.
- Deps: `montecarlo` (`run_cases`), `results` (`store_result`).

### `scan.py`
Headless sweep entry: parse args → build cases → `run_sweep` → checkpoint.
- Public: `main`, `run`, `add_subparser`.
- Deps: `config`, `run`, `sweep`.

## Results & plotting

### `results.py`
Result records, derived line metrics, and ranking/selection.
- Public: `Settings` (dataclass), `store_result`, `records`, `filter_results`,
  `select_results`, `slim_results`, `sweep_values`, `results_dataframe`,
  `line_metrics`, `line_index`,
  `line_quality`, `selection_score`, `top_geometries`, `summary_table`,
  `show_summary`, `show_top`, `best_azimuth`, `detected_background`.
- Deps: `montecarlo`, `sweep`.

### `plots/` (package)
All plotting — Matplotlib/Plotly. Split from a single module into submodules by
figure type; **every public and internal name is re-exported from the package**,
so `from cxr_mc.plots import X` is unchanged (`tests/test_plots_exports.py`
freezes the export set). Submodule DAG (leaf → driver):
`_style → _common → sweeps → {spectra, detectors, trajectories} → interactive`.
- `_style` — `COLORS`, `_ENERGY_PALETTE`, `energy_color` (per-energy colour map
  consistent across every figure). Leaf; no sibling deps.
- `_common` — shared figure plumbing: `_line_brem` (per-record line/brem split),
  `_per_tilt_figs` (one-figure-per-tilt loop), `_mode`, `_EFF_CACHE`. Deps:
  `montecarlo`, `results`.
- `spectra` — `plot_by_energy`, `plot_full_spectrum`, `plot_peak_vs_tilt`,
  `plot_mosaic_comparison`, `plot_best_spectra`, `plot_material_comparison`,
  `plot_tilt_panel`, the `_draw_*` spectral drawers. Deps: `_style`, `_common`,
  `montecarlo`, `results`.
- `sweeps` — `plot_heatmaps`, `facet_metric` (small-multiples over many knobs),
  `plot_metric_vs`, `plot_scan`; the `_HEATMAP_QUANTITIES` / `_METRIC_LABELS`
  tables + axis helpers. Deps: `_style`, `results`.
- `detectors` — `plot_timepix_efficiency` / `_detected` / `_poisson`,
  `plot_eaglexo_efficiency` / `_detected` / `_charge` / `_charge_map`. Deps:
  `_style`, `_common`, `sweeps`, `results`, `timepix_response`, `eaglexo_response`.
- `trajectories` — `plot_electron_trajectories`, `plot_trajectory_grid`,
  `plot_penetration_survival`. Deps: `_style`, `montecarlo`, `results`.
- `interactive` — `browse`, `browse_plotly`, `stream_chunk`, `plot_chunk`
  (the slider/streaming drivers that dispatch to the `spectra`/`detectors`
  drawers). Top of the DAG. Deps: `_style`, `_common`, `spectra`, `detectors`.
- `altair_spectra` — Altair/Vega-Lite renderer for the intrinsic spectra
  (`spectrum_chart`, `spectrum_frame`): a fast, interactive alternative to the
  matplotlib `spectra` figures, sharing `_common._line_brem` so the physics is
  identical. Intentionally **NOT** re-exported from the package (would break the
  frozen export guard) — import via `cxr_mc.plots.altair_spectra`. First slice of
  the matplotlib → altair migration (`tests/test_altair_plots.py`). Deps:
  `_common`, `results`, `altair`, `pandas`.
- Deps: `montecarlo`, `results`, `timepix_response`, `eaglexo_response`.

## Detector forward models

### `timepix_response.py`
Timepix3 charge-sensitive forward model (diffusion, absorption, energy
resolution, Poisson counts).
- Public: `TimepixResponse`, `build_response`, `get_response`,
  `absorption_efficiency`, `energy_fwhm_eV`, `sigma_diffusion_um`,
  `poisson_counts`.
- Deps: `crystallography`.

### `eaglexo_response.py`
Eagle XO detector forward model (geometry/solid angle, QE table, energy
resolution, Poisson counts).
- Public: `EagleResponse`, `geometry`, `sweep_geometry`, `get_response`, `qe`,
  `qe_absorption_model`, `load_qe_table`, `solid_angle_sr`, `energy_fwhm_eV`,
  `poisson_counts`.
- Deps: `crystallography`.

## CLI & packaging

### `cli.py`
The `cxr` console-script dispatcher.
- Public: `main`.
- Deps: `scan`, `export`, `slim`, `__version__`.

### `export.py`
`cxr export` subcommand — render figures / PDFs from a checkpoint.
- Public: `add_subparser`, `main`.

### `slim.py`
`cxr slim` subcommand — shrink a checkpoint pickle for transfer (drop the
full-range brem arrays, downcast spectra to float32, filter configs).
- Public: `slim_checkpoint`, `add_subparser`, `main`.
- Deps: `results` (`slim_results`).

### `__init__.py`
Package root: exposes `DATA_DIR` (packaged-data resolver) and `__version__`.

### `_compile_nb.py`
Internal notebook-compile helper; not part of the public API.
</content>
