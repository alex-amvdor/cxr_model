# cxr_model

**Coherent X-ray radiation (PXR + coherent bremsstrahlung) from table-top electron beams in crystals.**

A physics simulation that predicts the narrow, tunable X-ray lines a low-energy
electron beam (~30–60 keV) generates inside a crystal, and the flux a real
detector would measure. It replicates and extends Zhai et al., *Nat. Commun.*
**16**, 11218 (2025), *"Enhanced tunable X-rays from bulk crystals driven by
table-top free-electron energies,"* with the analytic core cross-checked against
Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

The driving question for the active work: **what line flux and enhancement
should a home-built 2×2 Timepix3 quad (or a Raptor Eagle XO CCD) see at
θ_obs = 90°**, for comparison against the paper's TEM/SEM measurements.

---

## The physics, in brief

A relativistic electron moving through a crystal carries a virtual photon field
that "sees" the periodic electron density and lattice potential. Two coherent
emission channels result, plus an incoherent background:

- **PXR (parametric X-ray radiation):** the electron's virtual photons Bragg-
  diffract off lattice planes into real, narrow X-ray lines. The line energy is
  set by the geometry, ω = **v·g** / (1 − **v·n̂**), so it is *tunable* with beam
  energy and observation/tilt angle rather than fixed like a fluorescence line.
- **CBS (coherent bremsstrahlung):** the periodic crystal potential puts coherent
  peaks on the bremsstrahlung continuum. CBS and PXR radiate into the same modes
  and **interfere** — the measured line is `|A_PXR + A_CBS|²`, never separable.
- **Incoherent bremsstrahlung:** the smooth background the lines sit on.

### How the pipeline computes it

```
   beam (30–60 keV)
        │
        ▼
  ┌──────────────────────┐   CASINO-style single-scattering Monte Carlo:
  │ electron transport   │   Joy–Luo slowing-down + Mott/screened-Rutherford
  │ (montecarlo.py)      │   elastic scattering → straight radiating segments
  └──────────┬───────────┘
             │ segments (position, direction, energy, length)
        ┌────┴───────────────────────────┐
        ▼                                 ▼
  ┌───────────────┐               ┌────────────────────┐
  │ coherent line │               │ bremsstrahlung      │
  │ spectrum      │               │ background          │
  │ |A_PXR+A_CBS|²│               │ Born + Elwert       │
  │ finite-t sinc²│               │                     │
  └───────┬───────┘               └─────────┬──────────┘
          └───────────────┬─────────────────┘
                          ▼
                ┌─────────────────────┐  Beer–Lambert self-absorption
                │ self-absorption     │  from each segment to the surface
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐  per-instrument forward models:
                │ detector response   │  Timepix3 (Si, ~1.9 keV threshold),
                └─────────────────────┘  Eagle XO (CCD, solid-angle × QE)
```

Each straight trajectory segment between elastic collisions radiates
independently (incoherent across segments, coherent across reciprocal vectors
within a segment) with the finite-interaction-time lineshape
`|Q|² = t_L² · sinc²(P·t_L)` — the physical replacement for the absorption-limited
delta-function of the closed-form theory.

---

## Repository layout

```
scan.ipynb         RUNNER:  pick MATERIAL → Sweep → run_sweep → checkpoints/<material>.pkl
analysis.ipynb     VIZ:     load that checkpoint → all figures (no sweeps here)
scan.py            root shim → cxr_model.scan (guarded; python scan.py, or cxr scan)
export_pdf.py      root shim → cxr_model.export (analysis.ipynb → PDF, or cxr export)
src/cxr_model/     importable package: physics modules + the cxr CLI entry point
src/cxr_model/data/  crystal_structures.toml, atomic_scattering_factors/, mott_transport_cross_sections/, *_qe.csv
checks/            validation scripts + notebooks (Feranchuk anchor, Zhai Fig 1c, kinematic audit)
dev/               author-only helpers (remote.py — run scan.py on a personal GPU box over ssh)
docs/              design notes & decision records (deferred features, library choices)
checkpoints/       per-material results pickles (gitignored)
results/           exported figures / PDFs (PNGs gitignored)
```

Packaged data resolves via `cxr_model.DATA_DIR`, so imports work from any working
directory and the data travels with an installed wheel. `*.pkl` checkpoints and
`*.png` images are gitignored; notebooks are output-stripped on commit by
`nbstripout` via `.gitattributes`.

### The `cxr_model` package modules

| Module | Responsibility |
|---|---|
| `crystallography.py` | Crystal DB loader (`CRYSTALS` from TOML), reciprocal vectors, structure factor / `chi_g` (PXR) / `U_g` (CBS), Debye–Waller, absorption length, `dominant_reflections()`. Physical constants. No GPU. |
| `montecarlo.py` | The transport + radiation pipeline: `simulate_trajectories` (MC electron transport), `mc_spectrum` (coherent lines), `mc_brem_spectrum` (Born+Elwert brem), detector helpers, `tilted_geometry`, and the case drivers `run_case`/`run_cases`. **Optional CuPy GPU** with automatic CPU fallback. |
| `sweep.py` | `Sweep` dataclass + `build_cases`. Every physical knob is a scalar (fixed) or a sequence (swept); cases = the Cartesian product. |
| `config.py` | Shared run config imported by **both** notebooks so they can't drift: `default_settings()`, the per-material sweep grids, and the builders `material_sweep(mat)` / `trajectory_sweep(mat)`. **This is where you tune a material's grid.** |
| `run.py` | `run_sweep(...)`: checkpointed, crash-safe, resumable driver around `run_cases`; `load_checkpoint`/`cases_from_results` for the viz side. |
| `results.py` | `Settings` dataclass, per-record metrics (`peak_flux`, `coherent_flux`, `line_flux`, `line_quality`, …), and ranking helpers (`selection_score`, `top_geometries`). |
| `plots.py` | All plotting: `browse`, `plot_heatmaps`, `plot_metric_vs`, `plot_best_spectra`, `plot_material_comparison`, and the electron-penetration figures. |
| `timepix_response.py` | Per-photon forward model of the Timepix3 (Si sensor): photoabsorption, e–h pairs, charge sharing, and the **~1.9 keV counting threshold** (the headline effect — it eats sub-2 keV line flux). |
| `eaglexo_response.py` | Raptor Eagle XO CCD: a clean `solid_angle × QE(E)` operator (windowless direct-detection CCD). |
| `atomic_form_factors.py` | Complex atomic form factor `F(g,E) = f0(g) + f'(E) + i·f''(E)` via **xraydb** (Waasmaier–Kirfel `f0` + Chantler/FFAST `f', f''`); no hard-coded tables. |

---

## Installation

The project is managed by [**uv**](https://docs.astral.sh/uv/) with a committed
lockfile (`uv.lock`) and requires **Python ≥ 3.14**.

```bash
git clone https://github.com/alex-amvdor/cxr_model.git
cd cxr_model
uv sync          # .venv + locked deps + an editable install of cxr_model (the cxr CLI)
```

Run anything with `uv run …` (or activate `.venv`). Note that a bare `python` on
your PATH will **not** have the dependencies — always use `uv run python …`.
`uv sync` installs the package, so `import cxr_model` works with no path hacks and
the `cxr` console script is on the venv PATH (`uv run cxr --help`).

**GPU is optional.** `cupy-cuda13x` (CUDA 13) is a dependency, but it imports
cleanly even with no usable GPU and the code **falls back to CPU automatically**
(you'll see `No GPU found … Falling back to CPU execution`). On a CUDA machine
you'll see `Using GPU`. Set `CXR_FP64=1` to force double precision for
reference/validation runs (the GPU path defaults to fp32).

Launch the notebooks with:

```bash
uv run jupyter lab
```

---

## Quickstart

The workflow is **two notebooks that share the grids in `config.py`** — edit a
material's thickness / energies / tilts / energy-grids there once and both
notebooks pick it up.

1. **`scan.ipynb`** (the runner): set `MATERIAL`, then
   `material_sweep(MATERIAL)` → `build_cases` → `run_sweep`, which writes
   `checkpoints/<material>.pkl` and streams the per-tilt statistics tables live.
2. **`analysis.ipynb`** (the viz): set the same `MATERIAL`, `load_checkpoint`,
   `cases_from_results`, then `browse` / heatmaps / Eagle XO / Timepix /
   penetration figures. No sweeps run here.

`COLLAPSE_AZIMUTH=True` (in `config.py`) keeps only the best azimuth per
(tilt, energy).

### Headless / non-interactive

```bash
cxr scan <material> [--quick] [--workers N]      # installed console script
uv run python scan.py <material> [--quick]       # identical, via the root shim
```

`--quick` runs a tiny smoke-test grid into an isolated `<material>_quick.pkl`.
(The entry point has the required `if __name__ == "__main__"` guard — see *Notes*.)

### Running on a cluster

`cxr scan` is headless and writes a single `checkpoints/<material>.pkl`, so it
drops into any batch scheduler — install once, then submit one job per material.
See [`docs/running-on-a-cluster.md`](docs/running-on-a-cluster.md) for a SLURM
`sbatch` template (including a job-array sweep over several materials). Pull the
checkpoints back and do all the matplotlib/PDF work locally.

> The author's own loop uses a small personal helper,
> [`dev/remote.py`](dev/remote.py), to push the working tree to one GPU box (ssh
> host `qlmc`, overridable via `CXR_REMOTE_{HOST,DIR,UV}`) and pull the checkpoint
> back. It is **not** part of the installed package and is specific to that setup;
> the cluster recipe above is the portable path.

---

## Materials

Crystals are defined in [`src/cxr_model/data/crystal_structures.toml`](src/cxr_model/data/crystal_structures.toml)
(lattice + basis). Current catalog (TOML keys):

| Key | Material | Structure |
|---|---|---|
| `diamond` | diamond | cubic |
| `silicon` | silicon | cubic |
| `lif` | LiF | cubic (rock salt) |
| `hopg` | highly-oriented pyrolytic graphite | hexagonal (fiber-textured) |
| `mose2`, `wse2`, `mote2` | 2H-MoSe₂, 2H-WSe₂, 2H-MoTe₂ | hexagonal (2H TMD) |
| `mos2`, `ws2` | 2H-MoS₂, 2H-WS₂ | hexagonal (2H TMD) |
| `ptse2`, `hfse2`, `zrse2` | PtSe₂, HfSe₂, ZrSe₂ | hexagonal (1T TMD) |

> **Note:** graphite is keyed `hopg` — there is no `graphite` key. HOPG is
> fiber-textured, so **only (00l) reflections are coherent** (random in-plane
> grain azimuths); it is treated specially (beam along the c-axis, (00l) only),
> not via `dominant_reflections`.

Adding a new material (or element) touches several non-colocated registries
(the TOML, `TRANSPORT_ELEMENTS`, an edge flag, the config grid, and the
`sweep.py` wiring). Atomic scattering data comes from **xraydb** for any element,
so there is no per-element table to edit. The full checklist lives in
[`CLAUDE.md`](CLAUDE.md) under *"Adding a material"*.

### Crystal mosaicity (optional, analytic — off by default)

Real crystals are mosaic: an incoherent ensemble of slightly misoriented crystallites with
a Gaussian *mosaic spread* η (rocking-curve FWHM; e.g. HOPG ZYA 0.4° / ZYB 0.8° / ZYH
3.5°). The simulation includes an **initial analytic model** of the resulting line
broadening, **off by default** and switchable per run:

```python
material_sweep("hopg", mosaic=True)                       # use the per-crystal value
material_sweep("hopg", mosaic=True, mosaic_fwhm_deg=3.5)  # override (e.g. ZYH grade)
```

**How it works.** A mosaic tilt rotates **g**, and only the numerator `v·g` of the
resonance `E_res = ħc·(v·g)/(1 − v·n̂)` depends on it, so the line picks up a Gaussian
broadening `FWHM_mosaic = E·|tan ψ|·η` (ψ = ∠(v, g)), added in quadrature with the EDS and
detector-aperture widths and applied through the same `convolve_detector` pass. The
**intrinsic** spectrum is unchanged — mosaicity enters only the detector-convolution FWHM,
so `plots.plot_mosaic_comparison` can overlay several grades from a single computed record.
The per-crystal spread is an **optional** `mosaic_fwhm_deg` in `crystal_structures.toml`
(only HOPG carries one today); crystals without it stay perfect, and `mosaic=False` is an
exact no-op.

**Limits.** It is *energy-shift only* (amplitudes held fixed across the mosaic cone),
diverges as ψ → 90° (capped at the peak energy), is usually sub-dominant to the
multiple-scattering Doppler width except in thin / near-perfect / high-mosaic cases, and
**is not yet validated** against measured line widths (see *Validation*). The exact
Monte-Carlo route (a per-orientation incoherent sum that broadens PXR+CBS, not just shifts
the energy) is designed but unimplemented — see
[`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

---

## Detectors

Detector geometry is **per-instrument — never transfer numbers between setups.**

| Setup | θ_obs | Δθ | Ω |
|---|---|---|---|
| Our **Timepix3** quad (28 mm at 0.4 m) | 90° | ≈1.76° | ≈9.5×10⁻⁴ sr |
| Zhai SEM (JEOL 7800) | 119° | 16.6° | 0.066 sr |
| Zhai TEM (JEOL 2010HR) | ≈112.5° | ≈12° | ≈0.034 sr |

The intrinsic spectra are detector-agnostic; the **Timepix3** and **Eagle XO**
forward models apply their own quantum efficiency / response downstream. The
Timepix3 ~1.9 keV counting threshold is the dominant instrument effect for these
soft lines.

---

## Physics conventions (read before touching geometry or amplitudes)

- **Frame:** incident beam along **+z**; detector at azimuth φ = 0 in the x–z
  plane at polar angle θ_obs. At θ_obs = 90° the detector is along +x.
- **Tilt sign (critical):** `tilt_deg` is the sample-normal polar tilt.
  **Negative tilt = entrance face toward the detector = HIGH flux** (radiation is
  born near the entrance and escapes a short path); positive tilt points the PXR
  lobe away and is ~10× weaker.
- **At θ_obs = 90°, the tilt must be nonzero** — an untilted slab self-absorbs
  photons travelling along its faces (yaw alone gives identically zero).
- **Coherence:** the measured line is `|A_PXR + A_CBS|²` — never separable.
  Segments add incoherently; reflections within a segment add coherently.
- **Units:** Ångström and eV (electron energies in keV where noted).
- **Relativistic CBS** corrections (the 1/γ braced products) matter ≳100 keV and
  are present at all amplitude sites.

---

## Validation (`checks/`)

- `feranchuk_spence.py` — the Feranchuk–Spence analytic core (PXR+CBS amplitudes,
  flux helpers). A reference, **not** part of the results pipeline.
- `feranchuk_check_script.py` — LiF analytic anchor.
- `feranchuk_vs_zhai_check.py` — analytic vs. MC pipeline agreement.
- `kinematic_validity_check.py` — DYN/recoil audit + van-der-Waals merit table.
- `zhai_fig1c_check.ipynb`, `cxr_analysis_feranchuk.ipynb` — figure reproductions.
- `src/cxr_model/_compile_nb.py` — compiles every notebook's code cells (syntax smoke test).

### What remains to be validated / approximated

These are known approximations or unvalidated additions — read before quoting absolute
numbers:

- **Crystal mosaicity** (the analytic model above) is **not** checked against measured HOPG
  rocking-curve / line widths; it is energy-shift only and unreliable near grazing, and the
  exact Monte-Carlo route is unimplemented — [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
- **Detector solid angle** is treated as a single observation direction `n̂` with a flat Ω
  flux scale and an analytic Gaussian polar-aperture broadening (`aperture_fwhm_eV`),
  exactly as the source papers do. A first-principles integral over the detector face is
  unimplemented; it matters for the wide SEM/TEM detectors (≈12–17°), not the small Timepix
  Ω — [`docs/detector-solid-angle.md`](docs/detector-solid-angle.md).
- **Atomic data** is sourced from **xraydb** (Waasmaier–Kirfel `f0` + Chantler/FFAST
  `f', f''`); the migration from the legacy Henke/CXRO + Cromer–Mann tables was adopted and
  re-validated against the Feranchuk/Zhai anchors —
  [`docs/atomic-data-sources.md`](docs/atomic-data-sources.md).
- **Timepix3 hardware** parameters (`SENSOR_THICKNESS_UM`, `BIAS_VOLTAGE_V`,
  `TEMPERATURE_K` in `timepix_response.py`) are **placeholders** pending the real quad
  values; the detected-spectrum figures inherit that uncertainty.

---

## Data provenance

- **Atomic scattering:** Waasmaier–Kirfel `f0` + Chantler/FFAST `f', f''`, supplied on
  demand by **xraydb** for any element (no per-element table to maintain). The legacy
  Henke/CXRO `.nff` CSVs in `src/cxr_model/data/atomic_scattering_factors/` are now unused (kept for
  provenance / A-B comparison) — [`docs/atomic-data-sources.md`](docs/atomic-data-sources.md).
- **Elastic transport:** NIST SRD 64 relativistic Mott *transport* cross sections
  (`src/cxr_model/data/mott_transport_cross_sections/`) calibrate the screened-Rutherford
  α(E) per element; free paths from the Browning fit. Elements without a NIST
  table fall back to the analytic screening with a one-time warning.
- **Crystal structures:** `src/cxr_model/data/crystal_structures.toml` (lattice + basis).
- **Detector QE:** `src/cxr_model/data/eaglexo_qe.csv`; Timepix Si response computed from Henke `f2`.

---

## Notes & gotchas

- **`__main__` guard required for sweep scripts.** `run_cases` uses a
  `ProcessPoolExecutor`; the workers re-import the entry module (`spawn` on
  Windows, `forkserver` on Linux as of Python 3.14). Any script that drives a
  sweep must be guarded with `if __name__ == "__main__":` or it relaunches itself
  recursively. Notebooks are guarded-equivalent; `scan.py` is guarded.
- **GPU ⇒ serial.** With CuPy active, `run_cases` runs serially (one CUDA
  context). Multiprocessing speedups apply only on the CPU path; cap
  `max_workers` near your physical core count.
- A benign `RuntimeWarning: divide by zero` can appear because the wide brem grid
  starts at 0 eV (λ→∞); the values are clamped downstream. It is not a bug.

---

## References

- W. Zhai et al., *"Enhanced tunable X-rays from bulk crystals driven by
  table-top free-electron energies,"* **Nat. Commun. 16, 11218 (2025).**
- I. D. Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

## Status & license

Academic research code under active development. No open-source license is
currently attached — contact the author regarding reuse.
Contributor/agent working conventions are documented in [`CLAUDE.md`](CLAUDE.md).
