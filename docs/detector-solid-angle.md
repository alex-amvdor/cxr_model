# Detector solid-angle integration

The line energy `E_res = ħc·(v·g)/(1 − v·n̂)` depends on the observation direction `n̂`.
A real detector subtends a finite solid angle, so different parts of its face see slightly
different line energies (and intensities). How that finite acceptance is handled is the
subject of this note.

---

## Current treatment — single angle (IMPLEMENTED)

The spectrum is computed at **one** `n̂` (from `montecarlo.tilted_geometry`), and the
finite detector is folded in by two separate approximations:

1. **Flux:** the per-steradian intensity is multiplied by a flat solid angle —
   `results.store_result` sets `scale = domega_sr · PER_NA`. This assumes
   `d²N/dE dΩ` is constant across the face.
2. **Line width:** an analytic, **polar-only**, symmetric Gaussian
   (`montecarlo.aperture_fwhm_eV`, from the detector polar span Δθ) added in quadrature
   with the EDS resolution and applied via `convolve_detector`.

This is exactly what the source papers do (Zhai et al. 2025; the Sci. Adv. 2023 / Adv.
Sci. 2022 predecessors): evaluate at a single θ_obs and treat Δθ_obs as a Gaussian line
broadening. It is an excellent approximation for the **small** Timepix acceptance
(Δθ ≈ 1.76°, Ω ≈ 9.5×10⁻⁴ sr).

---

## Improvement — integrate over the face (IMPLEMENTED as an opt-in tool)

### Status (TODO P2 #4)

`montecarlo.detector_directions()` + `montecarlo.mc_spectrum_solid_angle()` implement the
face integral as an **opt-in tool**, validated in `checks/detector_solid_angle_check.py`
and `tests/test_detector_solid_angle.py`:

- `detector_directions(theta_obs, tilt, …, n_side, chip_mm, dist_mm, domega_sr)` lays the
  `n_side × n_side` grid on the flat chip facing the source and returns sample-frame `n̂_i`
  plus weights `dΩ_i = dA_i cos ψ_i / r_i²`, rescaled so `Σ dΩ_i = domega_sr`.
- `mc_spectrum_solid_angle(…, n_hats, weights)` accumulates `Σ_i w_i · mc_spectrum(n̂=n̂_i)`,
  reusing the **validated single-angle** `mc_spectrum` per direction (no GPU-kernel surgery).
- **Regression:** `n_side = 1` reproduces `spec · Ω` to machine precision (max rel ≈ 5e-15).
- **Wide detector (Δθ ≈ 37°):** the integrated line shifts (≈ −50 eV) and broadens
  (13 → 123 eV) into the true asymmetric shape, and the integrated width is *narrower* than
  the symmetric `aperture_fwhm_eV` (196 eV), as anticipated below. **Timepix (Δθ ≈ 2°):**
  centroid shift ≈ 0.02 eV (negligible).

**Deliberately deferred** — the genuinely expensive, risky part (see Cons): the
unit-convention refactor that bakes Ω into the checkpoint pipeline (`results.store_result`
`scale`, dropping the `aperture_fwhm_eV` term, and the `integrated` flag every consumer must
branch on). The tool above returns the Ω-integrated spectrum directly and does **not** mutate
that single-`n̂` convention, so it is safe for the wide-detector study without destabilising
the sweep/plot pipeline.

### Functionality

Replace the single `n̂` with a grid of directions `{n̂_i}` tiling the detector face, compute
the spectrum for each, and accumulate a solid-angle-weighted sum:

- Add a `detector_directions(θ_obs, tilt, …, n_side, chip/dist)` helper alongside
  `tilted_geometry` that lays an `n_side × n_side` grid on the **flat rectangular chip** at
  distance d **facing the source**, in the **lab frame**, and returns per-cell directions
  plus solid-angle weights `dΩ_i = dA_i cos ψ_i / r_i²` (inverse-square + obliquity).
  Each direction is mapped into the sample frame through the same `R.T` as the single `n̂`.
- Wrap the per-reflection body of `mc_spectrum` in an outer loop over `n̂`, accumulating
  `Σ_i w_i · spec_i`. The result is the **Ω-integrated** line spectrum (already × Ω).
- `n_side = 1` returns the single central direction with weight Ω — i.e. **exactly today**.

The per-direction quantities that must be recomputed inside the loop: the polarization
pair `e_s/e_p`, `denom = 1 − v·n̂` / `E_res`, the sinc width `a_width`, the photon
kinematics `k = ω n̂` / `k+g` / detuning, and the `T_abs` escape branch (the sign of
`n̂_z` picks the exit face).

### Pros

- **First-principles.** One calculation replaces **both** approximations and yields the
  true **asymmetric** lineshape, the `n̂`-resolved line shift, **and** the intensity
  gradient across the face.
- **Self-validating.** At `n_side = 1` the integrated flux equals the old `spec · Ω`
  exactly (the regression anchor). In the small-Δθ limit the integrated width *supersedes*
  `aperture_fwhm_eV` (it converges to the true, generally narrower, 2-D width that the
  symmetric box-Gaussian only crudely approximates — do **not** expect exact agreement).
- **Small kernel surface:** `n̂` is already a single argument threaded through
  `mc_spectrum`.

### Cons

- **GPU wall-clock multiplier.** Cost ≈ `N_dir ×` the per-reflection sinc² matmul — the
  device hot loop — and the GPU path runs **serially** (one CUDA context), so there is no
  parallelism to hide it. A 5×5 grid is ~25× per line case; wide detectors want tens of
  directions. Loop directions (flat memory) rather than broadcasting an `n̂` axis (N_dir×
  memory).
- **The real cost is a unit-convention refactor, not the kernel.** Once `spec` bakes in Ω
  you must remove the now double-counted `domega_sr` from `results.store_result` `scale`
  **and** the `aperture_fwhm_eV` term from `fwhm` (keep the EDS-resolution term!). That
  single `r["scale"]` / `r["fwhm"]` convention is consumed by `detected_background`,
  `summary_table`, `line_metrics`, `best_azimuth`/`selection_score`, and the
  Timepix/EagleXO forward models — and the **brem** (kept single-`n̂`, flat-Ω) shares the
  scale field. A checkpoint mixing integrated and single-angle records becomes
  unit-inconsistent, so records need an `integrated` flag and every consumer must branch on
  it (or mixing must be forbidden).
- **Frame trap.** The chip tiling lives in the lab frame; every per-direction physics
  quantity is evaluated in the sample frame. Getting the frame (or the chip's in-plane
  roll relative to the scattering plane) wrong silently biases the asymmetry.
- **Brem caveat.** Brem emission is isotropic (1/4π) and only `n̂`-dependent through
  `T_abs`; keeping it single-`n̂` is fine for Timepix but is an extra approximation in the
  **wide + grazing** regime, where `L_esc ∝ 1/n̂_z` varies across the face.

### Effort

≈ **7–11 engineer-days**, dominated by the unit-convention split and its regression suite,
plus adding the wide-detector geometry (the regime where the feature actually matters).

### When it is worth it

- **Negligible** for the Timepix (Δθ ≈ 1.76°): sub-percent shift/asymmetry; the current
  flat-Ω + Gaussian is already excellent.
- **Worthwhile** for the wide SEM/TEM detectors (Δθ ≈ 12–17°), where the lineshape is
  visibly asymmetric, the intensity gradient across the face is real, and the
  single-angle assumptions break down.

Recommend keeping `n_side = 1` (exact, fast) as the default and reserving the integral for
the wide-detector study.
