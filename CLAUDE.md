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

Repo: `https://github.com/alex-amvdor/cxr_model.git` (notebooks are stripped
by `nbstripout` via `.gitattributes`).

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
  `detector_efficiency`), the analytic crystal-mosaic broadening
  (`mosaic_fwhm_eV` + `mosaic_psi_rad`, with the shared `_orientation_R`),
  `tilted_geometry`, and the parallel drivers
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
  `plot_trajectory_grid`/`plot_penetration_survival` (datashader cascades +
  surviving-population-vs-depth curve). Beam energy → colour is consistent across every figure
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
python remote.py scan mose2          # FOREGROUND: sync up -> run scan.py -> pull checkpoint (holds the ssh session)
python remote.py scan mose2 --quick   # tiny grid smoke test (isolated <material>_quick.pkl)
python remote.py pull mose2 wse2       # just fetch one or more existing checkpoints
```

**Detached queue** (multiple materials, survives ssh disconnect — launch, walk
away, reconnect): `start` ships the code, writes a runner under
`<remote>/jobs/<jobid>/` (gitignored), and `nohup setsid`-launches it so it keeps
running after you disconnect; the runner calls `scan.py` once per material in
sequence (pid/meta/state/log per job). The job is detached on the box from the
start, so the ssh link is only ever a **viewer** — `--follow`/`attach` stream the
log live and exit when the job finishes, and **disconnecting is just Ctrl-C**
(or closing the terminal / a dropped link): it tears down the viewer only, the
job runs to completion. Reconnect with `attach`/`status`/`logs`, `pull` when the
state reads `done`.

```
python remote.py start mose2 wse2 mos2   # queue, run detached, return immediately
python remote.py start mose2 --follow     # launch, then track it live
python remote.py start mose2 --quick      # detached quick smoke test
python remote.py attach [JOBID]            # (re)connect + track live (default: latest)
python remote.py jobs                      # list jobs + their state
python remote.py status [JOBID]            # meta + state + alive? + log tail (default: latest)
python remote.py logs [JOBID] --follow     # live tail
python remote.py stop JOBID                # SIGTERM the job's process group
```

then open `analysis.ipynb` (same `MATERIAL`) or `export_pdf.py` locally. The
box is ssh host `qlmc` (`~/.ssh/config`, cloudflared proxy); override with
`CXR_REMOTE_{HOST,DIR,UV}`. `scan.py <material> [--quick] [--workers N]` is the
guarded headless runner `remote.py` invokes (also runnable directly on the box).
Don't run PDF export on the box — pull the checkpoint and render here. Material
keys passed to `start`/`scan` are validated against the crystal-key alphabet
(they're embedded in a remote shell command).

**Local tooling (read before running anything by hand):**

- The deps live in the project venv managed by **uv**. The bare `python` on PATH
  has **no numpy** — always run scripts/one-liners with `uv run python ...`, never
  plain `python` (a bare `python -c "import numpy"` fails and is a misleading
  "the code is broken" signal when it's just the wrong interpreter).
- On Windows the **Bash tool mangles `C:\...` paths and `&&` chains** (the
  backslashes get eaten, so `ls "C:\path" && echo x` fails). Use **forward
  slashes** (`C:/Users/alexa/...`) in Bash, or prefer the dedicated Glob / Read /
  Grep tools, which take native paths cleanly.
- A benign `RuntimeWarning: divide by zero` from `chi_g`/`absorption_length_ang`
  fires because the wide brem grid starts at 0 eV (λ→∞ at E=0); values are
  `nan_to_num`-clamped downstream. Not a bug — don't "fix" it.

Crystals (TOML keys): `diamond`, `silicon`, `lif`, `hopg`, `mose2`, `wse2`,
`mote2`, `mos2`, `ws2`, `ptse2`, `hfse2`, `zrse2`. (Note: the graphite entry is keyed
`hopg` — there is no `graphite` key, and nothing may pass one.)

**Adding a material (or a new element) — every registry that must be touched.**
These are NOT colocated; miss one and it fails *late* (a `KeyError` deep in a
worker), not at import. For a material that reuses already-present elements, only
the (*) sites; a NEW ELEMENT needs all of them:

1. (*) `data/crystal_structures.toml` — the `[material]` block (system, lattice,
   basis). 2H TMDs are isostructural with MoSe₂ (metal on 2c (1/3,2/3,1/4),
   chalcogen on 4f z=0.621 → δ=0.129 from the metal plane); sanity-check the M–X
   bond `sqrt((a/√3)² + (δc)²)` against the literature value.
2. (*) `src/config.py` — add to `_MATERIAL_GRIDS` (`MATERIALS` auto-derives).
3. (*) `src/sweep.py` — `MATERIAL_LABELS` **and** a `crystal_params()` branch
   (composition, `hkl_list`, `beam_uvw`, `B_ang2`).
4. `src/crystallography.py` — add to `_EDGE_PRONE` if any absorption edge lands
   in the ≤4.5 keV line grid (forces the complex resonant f0+f′+if″).
5. `src/montecarlo.py` — `TRANSPORT_ELEMENTS` (Z, A, `J_keV` = the ICRU/NIST mean
   excitation energy in keV — NOT a fudge factor; e.g. Te = 0.485).
6. this file — append the key to the "Crystals (TOML keys)" line above.

Atomic scattering data (Z, f0, f′, f″) now comes from **xraydb** for any element —
no table to edit (was: hand-typed `Z_TABLE` + `CROMER_MANN` coefficients + a CXRO
`.nff` download per element). See [docs/atomic-data-sources.md](docs/atomic-data-sources.md).

NIST Mott transport tables (`data/mott_transport_cross_sections/`) are OPTIONAL:
a missing element (W, S, Pt, Hf, Zr, Te…) falls back to analytic
screened-Rutherford screening with a one-time warning. Verify the whole chain
with a single `run_case` at tiny `Ne` — it exercises every registry above.

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
- **Workers**: Workers run BELOW_NORMAL priority with BLAS
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
- **Trajectory colouring (datashader)**: aggregate tracks with `cvs.line(..., line_width=0)` so each pixel takes the true electron energy, then `tf.spread` to
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

- Atomic scattering: Waasmaier–Kirfel f0 + Chantler/FFAST f′,f″, via **xraydb**
  (`src/atomic_form_factors.py`). The legacy Henke/CXRO `.nff` CSVs in
  `data/atomic_scattering_factors/` are now unused (kept for provenance / A-B).
- Elastic transport: NIST SRD 64 relativistic Mott **transport** cross sections
  (`data/mott_transport_cross_sections/`), used to calibrate the screened-Rutherford
  α(E) per element; free paths from the Browning fit (valid ≤30 keV — extrapolated
  beyond, a caveat at 200 keV).
- Crystal structures: `data/crystal_structures.toml` (lattice + basis + B-factors).
- Detector QE: `data/eaglexo_qe.csv`; Timepix Si response computed from Henke f2.

## TODOs

### General Project Refinement — Top Priority

The user's Principal Investigor wishes to publish this repository according to the following message received by the user:

> We are launching a new subtap on our website for **CODES**, meant to feature all the “public-ready” codes developed within the group.
>
> *Prep work:*
>
> * Itemize each self-standing code (i.e., it is “finished” and self-consistently functional,
>   this is not your code under development).
>   * Make sure it’s properly commented or documented (e.g., a read-me)
> * Per each code, make sure that it is uploaded to the QLMC Github account
> * Per each code, make sure that it is has a DOI assigned to it
>
> After you have the above ready, provide the following *Materials for Website:*
>
> * A title for your code + DOI code
> * A single, succinct, paragraph, describing what the code does and notably, what it doesn’t
> * A permalink of where the code resides on our Github

* **Don't do any of the external github/DOI assignment yet.** At this stage, we just want to refine the project/repo to be fully prepared for future publication,
* **Broadly evaluate the structure of this project**. Confirm whether or not this `CLAUDE.md` file is being used according to industry standard best practices, or if the project information/TODOs should be relocated elsewhere; if so, do so. Trim the fat as needed; remove any bloat that isn't critical.
* **Evaluate the python filestructure,** including the use of notebooks, source code, etc., and refactor/reorganize as necessary (if necessary). Analyze if `jupyter` kernel implementation, its interaction with `uv`, etc., is easily transferrable between end users who might clone the repo.
* **Evaluate the documentation as written thus far** (in `./docs/`, as well as in this file and in `README.md`), and confirm whether or not it is all written and organized according to industry standards. **Evaluate if this project is a good candidate for `sphinx`** or a similar python documentation rendering software; if it is, then implement the best-fit software.
* **Ensure codebase on main is fully functional and polished**. Move any in-progress or partially completed features to a `feature/`, `patch/`, `hotfix/`, or other branch
* **Ensure all documentation is fully up-to-date.** It should be optimized for readability and clarity (from both a physics and a codebase explanation perspective), with bloat minimized.
* **Evaluate package structure.** Analyze if project should be `uv`-installable so that commands are directly runnable from CLI; if so, plan the CLI syntax (i.e., `scan` structure, `remote` structure, etc), then implement.
* **Evaluate containerization with Docker**. Given the dependencies on `uv`, `jupyter`, and potentially `pyelpsa` (see TODOs below), etc., evaluate whether or not adding the optional capability for future endusers to simply build with `Docker` to skip the setup of the various dependencies is worthwhile. If so, implement it.

### Features, Patches, Bugfixes

* Feature: film-on-substrate multi-layer materials — a vdW film
  (MoSe₂/MoS₂/WS₂/MoTe₂) on its real substrate (SiO₂/Si or sapphire), with
  per-crystalline-layer radiation and full-stack self-absorption. The 2H
  groundwork (incl. 2H-MoTe₂) is on `main`; the multilayer engine itself is
  unstarted.

  * Full design + phasing in [docs/multilayer-materials.md](docs/multilayer-materials.md) (recommended first
    slice = cross-stack self-absorption). Only 1T′-MoTe₂ exact coordinates
    remain blocked on a reliable CIF.
* Patch: Improve checkpointing — current method leads to gigabyte-sized file
  transfers for material pickles that have had many cases run and contain much
  stale (or old but still useful) data. Some form of multiple pickles per material,
  or at least filtering of the remote pickled data for only the required information
  prior to plotting. Whichever is cleaner and more robust. (Partial mitigation landed
  plot-side: `select_results`/`sweep_values` slice a loaded checkpoint by value
  before plotting, but the on-disk pickle is still the full union, so the transfer-size
  problem itself is unsolved.)
* Feature: grazing-incidence soft x-ray diffraction grating (in combination with an
  EagleXO or an Alex detector), like those from Ultrafast Innovations.
* Feature: first-class support for custom large-parameter-space sweeps (beyond the
  current `plot_scan` heatmap/line auto-pick) — e.g. interactive slicing/faceting
  of many simultaneously-swept knobs.
* Feature: crystal mosaicity — the **exact Monte-Carlo route** (incoherent sum over
  Gaussian-spread crystallite orientations inside `mc_spectrum`, broadening PXR+CBS
  per orientation). The cheap **analytic** route already landed (see Done below); the
  MC route is the upgrade for the large-mosaic / broad-line regime where the analytic
  energy-shift-only model breaks (HOPG ZYH, ψ→90°). Shares machinery with the deferred
  **detector solid-angle integration** (both incoherently sum the spectrum over a
  distribution of a direction — g for mosaic, n̂ for the aperture).

  * Caveat: the electron multiple-scattering Doppler width often dominates the line,
    so mosaic is visible mainly in thin / near-perfect crystals.
    Full design + pros/cons: [docs/crystal-mosaicity.md](docs/crystal-mosaicity.md).
* Feature: detector solid-angle integration — replace the single-`n_hat` +
  flat-`domega_sr` + analytic `aperture_fwhm_eV` approximation with a first-principles
  integral of `mc_spectrum` over a grid of `n_hat` tiling the chip. Matters for the wide
  SEM/TEM detectors (≈12–17°), negligible for the tiny Timepix Ω. Main cost is the
  unit-convention refactor (drop the double-counted `domega_sr`/`aperture_fwhm_eV`) and
  the GPU-serial wall-clock multiplier, not the kernel. Full design + pros/cons:
  [docs/detector-solid-angle.md](docs/detector-solid-angle.md).
* Feature: Python unit management library — evaluate the use of Pint or a similar
  unit management Python library. Look at the most popular candidates (Pint, natu,
  Buckingham, Units, among others), evaluate/compare their value in making this
  project more robust, readable, or otherwise improved. If they are worthwhile,
  select the one that best fits this project, then implement it project-wide.
* Feature: NIST transport databases replacement with pyelsepa — evaluate the use
  of pyelsepa/ELSEPA in this project to replace the currently used NIST Mott transport tables.
  Currently, these tables are given by hardcoded data files at `data/mott_transport_cross_sections/`.
  User has begun the process of adding support for pyelsepa in the case it is found to be useful.

  * On user's home desktop, it is located at `C:\\dev\\pyelsepa\\`. The docker image has been built
    according to the github repo [github.com/eScatter/pyelsepa](https://github.com/eScatter/pyelsepa)
* Feature: Clean up and centralize physics anchors/checks — make Feranchuk/Zhai anchors
  in `./checks/` also produce plots to compare with those provided in the literature (for the user).
  Specifically, the figures in Zhai are critical, especially 1C. Preferably, a minimal pre-configured
  jupyter notebook for the data viz for the user interface, though backend source code files
  used by the notebook are fine.
