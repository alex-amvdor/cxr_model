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
4. `src/atomic_form_factors.py` — `Z_TABLE` **and** `CROMER_MANN` (9-coeff f0,
   ITC Vol C Table 6.1.1.4; check Σaᵢ+c ≈ Z).
5. `src/crystallography.py` — add to `_EDGE_PRONE` if any absorption edge lands
   in the ≤4.5 keV line grid (forces the complex Henke f).
6. `src/montecarlo.py` — `TRANSPORT_ELEMENTS` (Z, A, `J_keV` = the ICRU/NIST mean
   excitation energy in keV — NOT a fudge factor; e.g. Te = 0.485).
7. `data/atomic_scattering_factors/<El>.csv` — Henke f1/f2. Download the raw table
   from CXRO `https://henke.lbl.gov/optical_constants/sf/<el>.nff` (same format,
   10 eV–30 keV, `-9999.` sentinel below valid f1 — no conversion). Use `curl`,
   NOT WebFetch (WebFetch summarizes and will not reproduce the ~500 numeric rows).
8. this file — append the key to the "Crystals (TOML keys)" line above.

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

- Atomic scattering: Cromer-Mann f0 + Henke/CXRO f′,f″ (`data/atomic_scattering_factors/*.csv`, `.nff` format).
- Elastic transport: NIST SRD 64 relativistic Mott **transport** cross sections
  (`data/mott_transport_cross_sections/`), used to calibrate the screened-Rutherford
  α(E) per element; free paths from the Browning fit (valid ≤30 keV — extrapolated
  beyond, a caveat at 200 keV).
- Crystal structures: `data/crystal_structures.toml` (lattice + basis + B-factors).
- Detector QE: `data/eaglexo_qe.csv`; Timepix Si response computed from Henke f2.

## TODOs

* Feature: film-on-substrate multi-layer materials — a vdW film
  (MoSe₂/MoS₂/WS₂/MoTe₂) on its real substrate (SiO₂/Si or sapphire), with
  per-crystalline-layer radiation and full-stack self-absorption. Design + the 2H
  groundwork live on the `mote2-multilayer-materials` branch; blocked on a
  reliable 1T′-MoTe₂ CIF for exact coordinates.
* Patch: Improve checkpointing -- current method leads to gigabyte-sized file transfers for material pickles that have had many cases run and contain much stale (or old but still useful) data. Some form of multiple pickles per material, or at least filtering of the remote pickled data for only the required information prior to plotting. Whichever is cleaner and more robust. (Partial mitigation landed plot-side: `select_results`/`sweep_values` slice a loaded checkpoint by value before plotting — but the on-disk pickle is still the full union, so the transfer-size problem itself is unsolved.)
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
  distribution of a direction — g for mosaic, n̂ for the aperture). Caveat from the
  feasibility study: the electron multiple-scattering Doppler width often dominates the
  line, so mosaic is visible mainly in thin / near-perfect crystals. Full design + pros/cons:
  [docs/crystal-mosaicity.md](docs/crystal-mosaicity.md).
* Feature: detector solid-angle integration — replace the single-`n_hat` +
  flat-`domega_sr` + analytic `aperture_fwhm_eV` approximation with a first-principles
  integral of `mc_spectrum` over a grid of `n_hat` tiling the chip. Matters for the wide
  SEM/TEM detectors (≈12–17°), negligible for the tiny Timepix Ω. Main cost is the
  unit-convention refactor (drop the double-counted `domega_sr`/`aperture_fwhm_eV`) and
  the GPU-serial wall-clock multiplier, not the kernel. Full design + pros/cons:
  [docs/detector-solid-angle.md](docs/detector-solid-angle.md).

### Done (recent)

* ✅ Crystal mosaicity — INITIAL ANALYTIC model (switchable, per-crystal, optional).
  A mosaic tilt rotates g, and only the numerator `v·g` of `E_res` depends on it, so the
  line gets a Gaussian broadening `FWHM = E·|tan ψ|·η` (ψ = ∠(v,g), η = rocking-curve
  FWHM), added in quadrature with the EDS + aperture widths in `store_result` (capped at
  `E_pk`; the linearization diverges as ψ→90°). `montecarlo.mosaic_fwhm_eV` +
  `mosaic_psi_rad` (+ extracted `_orientation_R`, shared with `mc_spectrum`);
  `crystal_structures.toml` carries an OPTIONAL `mosaic_fwhm_deg` per crystal (HOPG = 0.8°
  ZYB; perfect crystals omit it); `load_crystals` surfaces it; `Sweep(mosaic=True[,
  mosaic_fwhm_deg=…])` is the on/off switch (`build_cases` → `case["mosaic_fwhm_rad"]`,
  `None` ⇒ perfect ⇒ exact no-op, so old checkpoints and `mosaic=False` are unchanged).
  `plots.plot_mosaic_comparison(r, settings, grades_deg=…)` overlays grades from ONE
  computed record (re-convolution only — intrinsic spec is fixed). Tests in
  `tests/test_mosaic.py`. NB analytic = energy-shift only (amplitudes held fixed across
  the cone); the exact per-orientation MC sum is the future upgrade above.

* ✅ Data selection when plotting — `results.select_results` (value-based slicing:
  scalar / list / predicate) + `results.sweep_values` (what's in a checkpoint).
  The notebook slices `res` before any plot, so a fat hopg.pkl no longer overplots
  every thickness. (The remote *transfer* size is still open — see above.)
* ✅ EagleXO detection scheme — a CCD integrates charge, so the figure of merit is
  recorded CHARGE, not a QE-shaped spectrum: `EagleResponse.charge_density` /
  `integrated_charge`, the per-tilt `browse(kind="eaglexo_charge")` view, and the
  `plot_eaglexo_charge_map` geometry map (detected charge rate, or well-fill
  fraction vs `FULL_WELL_E` with `exposure_s`). The misleading "measured spectrum"
  plot was removed (a bare CCD can't return a spectrum).
* ✅ Tiny electron-trajectory plots — `plot_trajectory_grid` panels are now SQUARE
  (shared frame squared) and capped at 3 columns; the figure grows in rows rather
  than crushing panels.
* ✅ Few-azimuth → lines, not banded heatmaps — `plot_scan` auto-picks lines vs
  heatmap from each axis's value count, and `plot_metric_vs` guards a single-valued
  x (auto-substitutes a genuinely-swept axis).
* ✅ CuPy "CUDA path could not be detected" warning on the GPU-less laptop —
  silenced at the cupy import in `montecarlo.py` (the CPU fallback is unchanged).
* ✅ Figure/UX polish pass — see the "Figure / plotting polish — follow-up review"
  section above (all 7 items resolved).

### Figure / plotting polish — follow-up review (2026-06-19)

Context: a prior session addressed several of the patches above (CuPy warning
suppressed; `select_results`/`sweep_values` value-slicing added; unified
`plot_scan` auto-picks heatmap vs lines; `_draw_chunk` collapse bug fixed;
EagleXO charge view + per-tilt `browse(kind="eaglexo_charge")` added; trajectory
panel sizing bumped). The items below were the **remaining** figure/UX problems
found while actually using the wired-in `analysis.ipynb` on the hopg checkpoint
(40 thicknesses × 15 polar tilts × 1 energy).

**Status: all 7 RESOLVED** (2026-06-19 session). Summary of what landed (the
detailed items are kept below for reference): (1) `plot_best_spectra` default
`ncols=3` + `constrained_layout`; (2) `plot_metric_vs` now guards a single-valued
`x` — warns and auto-substitutes a genuinely-swept axis — and the notebook plots
`line_flux`/`peak_flux` vs `tilt_deg`; (3) the thickness×tilt `coherent_flux` /
`coherent_brem_ratio` cells are now `plot_scan` heatmaps, with
`coherent_brem_ratio` given a real label+cmap via a new `_EXTRA_QUANTITIES`
registry (kept out of the default `_HEATMAP_QUANTITIES` so plain scans don't grow
a panel); (4) `plot_eaglexo_measured` removed (a bare CCD can't return a
spectrum; the charge view is the "what it measures" path), Eagle/Timepix kept
plots moved to `constrained_layout` so suptitles aren't clipped; (5)
`plot_timepix_poisson` widened (~4.6"/panel, ≥6.8" min) + `constrained_layout`;
(6) `plot_penetration_profile` replaced by `plot_penetration_survival`
(surviving-population % of N₀ vs depth, per-electron max depth via the new
`elec_id` key in `_trajectory_data`); (7) `plot_trajectory_grid` panels are now
square (shared frame squared via `_square_frame`) and capped at 3 columns, the
figure growing in rows. Verified: full pytest suite + notebook-compile + an
Agg-backend smoke pass over every touched figure on the hopg checkpoint.
(Implementation pointers below were guidance, not a spec.)

1. **`plot_best_spectra` ("Top N geometries by …") is too wide — drop to 3
   columns.** The figure runs off-screen to the right unless the window is
   full-screened. In `src/plots.py:plot_best_spectra`, the default is `ncols=4`
   with `figsize=(3.3*ncols, 2.6*nrows)`; change the default to **`ncols=3`** (≈10"
   wide) so it fits without full-screening. The trajectory grid (#7) should match
   this per-panel width.

2. **`line_flux` ("integrated flux under the dominant line") draws as stacked
   points at a single x, not a line.** The notebook cell
   `plot_metric_vs(res, settings, x="E0_keV", metric="line_flux", hue="tilt_deg")`
   puts beam energy on x, but hopg has only ONE energy (30 keV) → every polar tilt
   becomes a separate point stacked vertically at x=30 keV. Want: **line_flux vs
   polar tilt** (`x="tilt_deg"`), a real line. Fix in the notebook (use a swept
   axis), and/or make `plot_metric_vs` / `plot_scan` guard a single-valued `x` —
   warn and auto-substitute a genuinely-swept parameter (or refuse to connect
   meaningless points) instead of silently stacking them.

3. **`coherent_flux` and `coherent_brem_ratio` vs thickness are 15-line spaghetti
   — make them heatmaps.** The notebook cells
   `plot_metric_vs(x="thickness_ang", metric="coherent_flux"/"coherent_brem_ratio",
   hue="tilt_deg")` draw one line per polar tilt (15 lines). Per the unified-scan
   decision, thickness(40)×tilt(15) is dense on both axes and should be a
   **heatmap**: replace with
   `plot_scan(res, settings, x="thickness_ang", y="tilt_deg", quantities=["coherent_flux"])`
   and the same with `quantities=["coherent_brem_ratio"]`. Both keys already work
   as heatmap quantities (`coherent_brem_ratio` is ungated); confirm a sensible
   colormap + label for the ratio (it's not in `_HEATMAP_QUANTITIES`, so it gets
   the default label/cmap via `plot_scan`'s string-quantity path — give it a real
   one in `_HEATMAP_QUANTITIES`/`_METRIC_LABELS` if it's to be a first-class map).

4. **`plot_eaglexo_measured` is conceptually wrong (a CCD can't return a spectrum)
   and its title is clipped.** A bare Eagle XO integrates charge; it yields a
   *spectrum* only in the special low-occupancy single-photon-counting mode, so the
   default "Poisson 'measured' spectra" plot is misleading. **Remove**
   `plot_eaglexo_measured` (and demote `eaglexo_response.poisson_counts` /
   `resolve_energy` to an explicitly-labeled "photon-counting mode" extra), and make
   the "what it actually measures" view the integrated **charge / well-fill**
   (`plot_eaglexo_charge_map`, already added — possibly also a 2-D detected-charge
   image). Also: the `suptitle` is cut off (`constrained_layout`/suptitle spacing)
   — fix that wherever a kept plot still uses a suptitle.

5. **`plot_timepix_poisson` title is clipped and the figure is too narrow.** In
   `src/plots.py:plot_timepix_poisson`, `figsize=(min(3.7*len(energies), 11.5), 4.4)`
   → for a single energy that's only 3.7" wide (far too narrow) and the suptitle
   "…Poisson 'measured' spectra…" is cut off. Fix: per-panel width ≈4.5" with a
   sensible **minimum total width**, and use `constrained_layout` / proper suptitle
   spacing so the title isn't clipped. (The Timepix DOES count photons, so its
   measured spectrum is legitimate — unlike the Eagle XO in #4.)

6. **Replace the penetration energy/age profiles with an electron-population-vs-
   depth curve.** **Remove `src/plots.py:plot_penetration_profile` entirely** (both
   the mean-electron-energy-vs-depth *and* the mean-age-vs-depth panels and their
   calculations). Replace with a **surviving-population vs depth** plot: the
   fraction of the initial electrons still "alive" (transporting above the energy
   cutoff — not yet stopped or backscattered out) as a **% of N₀**, vs depth z below
   the entrance surface. Suggested definition: per electron take the deepest point
   it reaches, then `survival(z) = (#electrons reaching depth ≥ z) / N₀` — a
   monotonically decreasing penetration/survival curve, one per beam energy.
   `simulate_trajectories` already returns `elec_id`, segment midpoints/`r_mid`, and
   `n_backscattered`/`n_transmitted`, so the per-electron max depth is recoverable.
   Update the notebook's penetration cell to call the new function.

7. **Trajectory grid panels are still far too small — max 3 SQUARE panels per row,
   sized like #1.** `src/plots.py:plot_trajectory_grid` wraps the swept tilts into a
   √n grid whose panels inherit the (wide, non-square) shared data frame, so they
   read as skinny vertically-stacked strips. Want: **cap at 3 columns** and make
   each axis **square** (square subplot box, ≈3.3" wide to match
   `plot_best_spectra`), keeping `set_aspect("equal")` so the slab/tracks stay
   physically correct — let the data letterbox within the square box, or re-crop the
   shared frame toward square. For >3 tilts the figure grows in ROWS (taller /
   scrollable), never by shrinking panels. (This supersedes the partial sizing bump
   already applied; the column cap + square aspect is the missing piece.)
