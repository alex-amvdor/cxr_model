# CLAUDE.md

## Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merged with project-specific information.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Information

Physics simulation of **coherent X-ray radiation (PXR + coherent bremsstrahlung)**
from table-top electron beams in crystals. Replicates and extends Zhai et al.,
*Nat. Commun.* **16**, 11218 (2025) ("Enhanced tunable X-rays from bulk crystals
driven by table-top free-electron energies"), with the analytic core validated
against Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

Goal of the active work: predict measurable line flux/enhancement for a real
lab setup — a home-built **2×2 Timepix3 quad** (and an alternative **Raptor
Eagle XO** CCD) at θ_obs = 90°, ~30–60 keV beam, to be compared against the
paper's TEM/SEM measurements.

Repo: `https://github.com/alex-amvdor/cxr_model.git` (history was force-slimmed;
notebooks are stripped by `nbstripout` via `.gitattributes`).

### Layout

```
scan.ipynb         RUNNER: pick MATERIAL -> Sweep -> run_sweep -> checkpoints/<material>.pkl
analysis.ipynb     VIZ: load that checkpoint -> all figures (no sweep runs here)
scan.py                headless twin of scan.ipynb (guarded; run a sweep non-interactively)
remote.py              orchestrator: run scan.py on the GPU box over ssh, pull the checkpoint back
export_pdf.py          headless analysis.ipynb -> PDF export (run locally)
src/                   all importable modules (both notebooks do sys.path.insert(0,"src"))
data/                  crystal_structures.toml, atomic_scattering_factors/, mott_transport_cross_sections/, *_qe.csv
checks/                validation scripts + check notebooks (Feranchuk anchor, Fig 1c, kinematic audit)
checkpoints/           per-material results pickles (gitignored)
results/               exported figures/PDFs (PNGs gitignored)
```

Data paths resolve **relative to `src/`** (`Path(__file__).parent.parent/"data"`),
so imports work from any cwd. `*.pkl` checkpoints and `*.png` are gitignored.

### Module responsibilities (`src/`)

- **`crystallography.py`** — general X-ray/crystallography primitives shared by
  the pipeline (no Feranchuk specifics). Crystal DB loader (`CRYSTALS` from TOML),
  `reciprocal_g_vector`, `structure_factor`/`chi_g`/`U_g`, `debye_waller`,
  `absorption_length_ang`, `_rotation_between`, and
  `dominant_reflections(material, n_families, B_ang2)` (ranks reflections by
  |S|·e^(−W)/g², Zhai Table 5 methodology) — plus the physical constants. No GPU.
  (The Feranchuk-Spence analytic amplitudes — `amplitudes_PXR_CBS_both` with
  Eqs. 13/14, `cxr_lines_fixed`, `omega_n`, `delta_g`, flux helpers — now live in
  `checks/feranchuk_spence.py`, a validation reference, not the results pipeline;
  it imports its crystallography primitives from here.)
- **`montecarlo.py`** — the transport+radiation pipeline. CASINO-style
  single-scattering MC (`simulate_trajectories` — also returns per-segment age
  `t_ang` = Σ L/β and `elec_id`, used by the penetration plots), per-segment
  spectrum (`mc_spectrum`), Born+Elwert brem (`mc_brem_spectrum`), detector
  helpers (`eds_fwhm_eV`, `aperture_fwhm_eV`, `convolve_detector`,
  `detector_efficiency`), `tilted_geometry`, and the parallel drivers
  `run_case`/`run_cases`. **Optional CuPy GPU** (`xp = cp` if importable; prints
  "Using GPU"); spectra use fp32 on-device.
- **`sweep.py`** — `Sweep` dataclass + `build_cases`. Every physical knob is
  scalar (fixed) or sequence (swept); cases = Cartesian product. Cheap to import.
- **`config.py`** — shared run config imported by BOTH notebooks so they can't
  drift: `default_settings()`, the per-material sweep grids, and the builders
  `material_sweep(mat)` (full scan) / `trajectory_sweep(mat)` (penetration figures).
- **`run.py`** — `run_sweep(...)`: checkpointed, streaming driver around
  `run_cases`. Resumes per material from `checkpoints/<material>.pkl` (skips
  cached cases), re-pickles at config granularity (crash-safe), fires `on_chunk`
  once a whole group (same material/thickness/tilt = full azimuth sweep) finishes.
  `load_checkpoint(material)` + `cases_from_results(results)` let the viz notebook
  load results (and rebuild the case list) without re-running.
- **`results.py`** — `Settings` dataclass, `store_result` (case → `results`
  record), `best_azimuth` (collapse an azimuth sweep to the highest-**peak**
  spectrum), `show_summary` (full per-row table). Per-record scalar metrics for
  the maps/selection live in `line_metrics`: `peak_flux` (max coherent density,
  no peak-finding), `coherent_flux` (∫ of ALL lines), `line_flux` (∫ under the
  single dominant found line only), `line_eV`/`fwhm_eV`, `line_frac`, and
  `line_quality` ∈ [0,1] — a definition score that flags geometries with NO clean
  line (broad ramp, or many comparable peaks; dominance·contrast·narrowness).
  `selection_score(m, mode)` ranks records (`quality_peak` default = bright AND
  clean) for every "best geometry" path; `top_geometries`/`show_top` print a
  compact ranked top-N table (the readable alternative to the full dump).
- **`plots.py`** — all plotting. `browse(results, settings, kind=...)` pages
  one figure per polar tilt (inline backend, or `%matplotlib widget`/ipympl);
  kinds: `chunk`|`by_energy`|`full` (intrinsic spectra) and `eaglexo`|`timepix`
  (detector). `plot_heatmaps(..., x=, y=, panel=, select=)` is GENERAL — any two
  swept knobs as axes (default azimuth×tilt, panel per energy; pass
  `x="E0_keV", y="tilt_deg"` etc.), with best-per-cell reduction via
  `selection_score` and dominant-line maps gated by `min_line_quality`.
  `plot_metric_vs` (1-D scans of any metric vs any swept param), `plot_best_spectra`
  (top-N geometries across the whole sweep — the answer to "thousands of cases"),
  `plot_material_comparison` (cross-material best line energy vs flux),
  `plot_trajectory_grid`/`plot_penetration_profile` (datashader cascades + depth
  profiles). Beam energy → colour is consistent across every figure
  (`energy_color`); `_per_tilt_figs` is the shared wrapper body. Intrinsic spectra
  are single-axis — the EDS view was removed; Eagle XO is the detector view, and
  `apply_detector_qe` now defaults **False** so "intrinsic" means intrinsic.
- **`timepix_response.py`** — per-photon forward model of the Timepix3 (Si
  sensor): photoabsorption, e-h pairs (W=3.65 eV, Fano), charge sharing, and the
  **~1.9 keV counting threshold** (the headline effect — it eats sub-2 keV line
  flux, not just blurs it).
- **`eaglexo_response.py`** — Raptor Eagle XO CCD: a clean `solid_angle × QE(E)`
  operator (windowless direct-detection CCD, no threshold/charge-sharing).

`results` is always `{config_name: {E0_keV: record}}`. Functions take `results`
and `Settings` explicitly — no module globals.

### Running

Two notebooks sharing the per-material grids in `config.py` (edit a material's
thickness / energies / tilts / E-grids there; both notebooks pick it up):

1. **`scan.ipynb`** (runner): pick `MATERIAL` → `material_sweep(MATERIAL)` →
   `build_cases` → `run_sweep` → writes `checkpoints/<material>.pkl` (streams the
   per-tilt stats tables live).
2. **`analysis.ipynb`** (viz): same `MATERIAL` → `load_checkpoint` →
   `cases_from_results` → `browse` / heatmap / Eagle XO / penetration figures. No
   sweep runs here (only the cheap, CPU-only electron transport behind the
   penetration figures).

`COLLAPSE_AZIMUTH=True` keeps only the best azimuth per (tilt, energy).

**Remote compute, local viz** (the lab box has an RTX 5080; this laptop has the
matplotlib + PDF toolchain). The GPU sweep runs on the box, everything visual
stays local — no manual ssh, no hand-copying files:

```
python remote.py scan mose2          # sync code up -> run scan.py on the box -> pull checkpoint back
python remote.py scan mose2 --quick   # tiny grid smoke test (isolated <material>_quick.pkl)
python remote.py pull mose2           # just fetch an existing checkpoint
```

then open `analysis.ipynb` (same `MATERIAL`) or `export_pdf.py` locally. The
box is ssh host `qlmc` (`~/.ssh/config`, cloudflared proxy); override with
`CXR_REMOTE_{HOST,DIR,UV}`. `scan.py <material> [--quick] [--workers N]` is the
guarded headless runner `remote.py` invokes (also runnable directly on the box).
Don't run PDF export on the box — pull the checkpoint and render here.

Crystals (TOML keys): `diamond`, `silicon`, `lif`, `hopg`, `mose2`, `wse2`,
`mos2`, `ws2`, `ptse2`, `hfse2`, `zrse2`. (Note: the graphite entry is keyed
`hopg` — there is no `graphite` key, and nothing may pass one.)

### Physics conventions — read before touching geometry or amplitudes

- **Frame**: incident beam along **+z**; detector at azimuth φ=0 in the x–z
  plane at polar angle θ_obs. At θ_obs=90° the detector is along +x.
- **Tilt sign (critical)**: `tilt_deg` is the sample-normal polar tilt;
  `tilt_azim_deg` is its azimuth (0 = pitch in the beam–detector plane, 90 = yaw).
  **Negative tilt = entrance face toward the detector = HIGH flux** (radiation is
  born near the entrance and escapes a short path). Positive tilt points the PXR
  lobe away → ~10× weaker. Verified via `tilted_geometry`: negative tilt gives
  n̂·ẑ < 0 (detector seen through the entrance face). At θ_obs=90°, tilt must be
  nonzero (an untilted slab self-absorbs photons travelling along its faces; yaw
  alone gives identically zero).
- **Coherence**: the measured line is `|A_PXR + A_CBS|²` — never separable. CBS is
  weak only when F≈Z (low-g). Segments add incoherently; reflections within a
  segment add coherently.
- **HOPG fiber texture**: only (00l) reflections are coherent (random in-plane
  grain azimuths). Do **not** use `dominant_reflections` for `hopg` — pass (00l)
  only, beam along c-axis [001].
- **Units**: Ångström, eV (electron energies in keV where noted).
  `E2_EV_ANG = α·ħc = 14.3996 eV·Å`. χ_g = −r_e λ² S(g) e^(−W)/(πV).
  Resonance ω = v·g/(1−v·n̂). Segment lineshape |Q|² = t_L² sinc²(P·t_L).
- **Relativistic CBS** (matters ≳100 keV): braced {a;b} = a·b − (a·v)(b·v) with a
  1/γ prefactor; present at all amplitude sites.

### Gotchas

- **Worker re-import guard (spawn / forkserver)**: `run_cases` uses
  `ProcessPoolExecutor`; the worker (`run_case`/`_transport_case`) is module-level
  so it pickles. The start method is `spawn` on Windows and — **as of Python 3.14**
  — `forkserver` on Linux (was `fork`), and BOTH re-import the entry module in each
  worker. So any script that drives a sweep **must** have an
  `if __name__ == "__main__":` guard (`scan.py` does), or it relaunches itself
  recursively — surfacing as a `forkserver ConnectionResetError`. `python - <<EOF`
  fails the same way (workers import `<stdin>`). Unguarded scripts "worked" pre-3.14
  only because `fork` didn't re-import; notebooks are guarded-equivalent.
- **`apply_detector_qe` defaults False**: the intrinsic spectra (`spec`) are now
  shown un-filtered. The Timepix3 / Eagle XO views apply their OWN QE downstream;
  the legacy SDD polymer-window QE (`detector_efficiency`) is opt-in only.
- **`remote.py` line endings**: the tar-sync now normalizes CRLF→LF for text files
  (`_add_to_tar`), so the box receives LF-clean content and `git status` there
  stays clean — a later `git pull` is no longer blocked by cosmetic diffs.
- **GPU ⇒ serial**: with CuPy present, `run_cases` runs serially (one CUDA
  context), not in a process pool. Multiprocessing speedups only apply CPU-side.
- **Workers**: machine is an i7-13620H — 10 physical cores / 16 threads. Cap
  `max_workers` at ~10 (CPU path); the extra 6 are P-core hyperthreads (~15% more,
  but the laptop gets sluggish). Workers run BELOW_NORMAL priority with BLAS
  pinned to 1 thread.
- **Detector geometry is per-instrument** — do not transfer numbers. Zhai SEM
  (JEOL 7800): θ_obs=119°, Δθ=16.6°, Ω=0.066 sr. TEM (JEOL 2010HR, the MoSe₂/WSe₂
  data): θ_obs≈112.5°, Δθ≈12°, Ω≈0.0344 sr (Δθ from Huang et al. 2022; Ω = the
  same cone convention). Our Timepix3: θ_obs=90°, Δθ≈1.76°, Ω≈9.5e-4 sr (28 mm
  quad at 0.4 m). Applying SEM numbers to TEM data was the cause of "far broader
  peaks."
- **MoSe₂ structure**: Se z-fractional δ=0.129 (not 0.121 — a Wyckoff misread),
  verified by the 2.53 Å Mo–Se bond. This changed |S(002)|² by ×0.53 and is a
  known ~×2 source of disagreement vs the paper's normalization.
- **Brem grid**: `E_grid_line` is fine+narrow (lines cap at a few keV, expensive
  sinc²); `E_grid_brem` is coarse+wide (cheap, extend to the beam energy). It
  starts at 0 eV, so `absorption_length_ang` is called at E=0 (L_abs→∞, μ→0,
  swallowed by `nan_to_num`); the now-benign "divide by zero" RuntimeWarning is
  suppressed in-function via `np.errstate`, values unchanged.
- **Trajectory colouring (datashader)**: aggregate tracks with `cvs.line(...,
  line_width=0)` so each pixel takes the true electron energy, then `tf.spread` to
  thicken. `line_width>0` coverage-weights the aggregated value and paints a bogus
  radial gradient ACROSS the line (hot centre → cool edges) instead of along the
  path. Tracks are continuous per-electron polylines (sort segments by
  `(elec_id, t_ang)`), and every panel in a grid shares one frame so only the slab
  rotates.

### Validation (`checks/`)

- `feranchuk_spence.py` — the Feranchuk-Spence analytic core (PXR+CBS amplitudes,
  `cxr_lines_fixed`, flux helpers). A reference, NOT in the results pipeline; it
  imports the general primitives from `src/crystallography.py` (run checks with
  `src/` on `sys.path`).
- `feranchuk_check_script.py` — LiF analytic anchor.
- `feranchuk_vs_zhai_check.py` — analytic vs MC pipeline agreement (film flux
  ratio → 1.00 after the δ-function Jacobian fix).
- `kinematic_validity_check.py` — DYN/recoil/ξ_e audit + vdW merit table.
- `zhai_fig1c_check.ipynb`, `cxr_analysis_feranchuk.ipynb` — figure reproductions.
- `src/_compile_nb.py` — compiles every notebook's code cells (syntax smoke test).

### Data provenance

- Atomic scattering: Cromer-Mann f0 + Henke/CXRO f′,f″ (`data/atomic_scattering_factors/*.csv`, `.nff` format).
- Elastic transport: NIST SRD 64 relativistic Mott **transport** cross sections
  (`data/mott_transport_cross_sections/`), used to calibrate the screened-Rutherford
  α(E) per element; free paths from the Browning fit (valid ≤30 keV — extrapolated
  beyond, a caveat at 200 keV).
- Crystal structures: `data/crystal_structures.toml` (lattice + basis + B-factors).
- Detector QE: `data/eaglexo_qe.csv`; Timepix Si response computed from Henke f2.
