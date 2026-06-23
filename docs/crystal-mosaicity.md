# Crystal mosaicity

Real crystals are **mosaic**: an incoherent ensemble of small, slightly misoriented
perfect crystallites whose orientations follow a Gaussian distribution about the mean.
The width is the **mosaic spread** η, quoted as a rocking-curve FWHM (e.g. HOPG grades
ZYA 0.4° / ZYB 0.8° / ZYH 3.5°; good bulk TMDs ~0.2–0.5°; CVD/exfoliated flakes up to a
few degrees). Because the PXR/Bragg coherence angular width is far smaller than η, the
crystallites radiate **incoherently**, and the observable is the single-crystal
spectral-angular distribution **averaged over the orientation distribution** (i.e. over
the spread of the reciprocal vector **g**). This is the standard kinematical mosaic-crystal
treatment (Darwin mosaic-block model; in PXR specifically: Nasonov; Feranchuk–Ivashin;
and the 2026 *Rad. Phys. Chem.* paper "The Effect of Crystalline Mosaicity on the
Spectral-Angular Distribution of Parametric X-ray Radiation").

Two implementations are possible. **(1)** is shipped; **(2)** is designed here.

---

## (1) Analytic broadening — IMPLEMENTED (the "initial" model)

**What it does.** A mosaic tilt rotates **g**, and only the numerator `v·g` of the
resonance `E_res = ħc·(v·g)/(1 − v·n̂)` depends on g. To first order a tilt δ shifts the
line by `dE/E = −tan(ψ)·δ`, ψ = ∠(v, g). A Gaussian mosaic of rocking-curve FWHM η
therefore broadens the line by a Gaussian of energy width

```
FWHM_mosaic = E · |tan ψ| · η
```

added **in quadrature** with the EDS-resolution and detector-aperture widths and applied
through the same Gaussian convolution path. The **intrinsic** spectrum is untouched —
mosaicity enters only the detector-convolution FWHM, so a record computed with
`mosaic=False` can be re-broadened at plot time.

**Where it lives.**
- `montecarlo.mosaic_fwhm_eV(E, ψ, η_rad)` — the formula above.
- `montecarlo.mosaic_psi_rad(case, E_pk)` — ψ for the reflection whose nominal
  (unscattered-beam) resonance is nearest the peak; uses the shared
  `montecarlo._orientation_R` (extracted from `mc_spectrum`).
- `results.store_result` — adds the term in quadrature, gated on
  `case["mosaic_fwhm_rad"]`, capped at `E_pk`.
- `crystal_structures.toml` — optional per-crystal `mosaic_fwhm_deg`; `crystallography.load_crystals` surfaces it.
- `sweep.Sweep(mosaic=…, mosaic_fwhm_deg=…)` → `build_cases` → `case["mosaic_fwhm_rad"]`.
- `plots.plot_mosaic_comparison` — overlay grades from one record.
- Tests: `tests/test_mosaic.py`.

**Limits (this is why it is the "initial" model):**
- **Energy-shift only.** It holds the amplitudes (polarization projection `g·e`, the PXR
  detuning `|k+g|²−ω²`, the CBS braced products) fixed across the mosaic cone — these all
  depend on g *direction*. Good while the line is narrow; approximate once it is broad.
- **Diverges at grazing.** `tan ψ → ∞` as ψ → 90° (g grazing v), so the term is capped at
  `E_pk` and is unreliable at steep tilts near grazing.
- **Often sub-dominant.** The electron multiple-scattering Doppler spread already broadens
  the line in thick/bulk crystals, so mosaic broadening is mostly visible in thin /
  near-perfect samples and for the high-mosaic ZYH grade.
- **Single representative ψ.** It uses the nominal beam direction (like `aperture_fwhm_eV`
  uses nominal θ_obs), not the per-segment scattered velocities.
- **Not yet validated** against measured HOPG rocking-curve / line widths.

---

## (2) Monte-Carlo mosaic average — DESIGNED, NOT IMPLEMENTED (the exact route)

### Functionality

Replace the post-hoc Gaussian with a true incoherent average **inside** `mc_spectrum`:

1. Draw `K` crystallite orientations from the mosaic distribution: a 2-D Gaussian tilt of
   the construction-frame normal with per-axis σ = η_FWHM / 2.3548 (the rocking curve is
   the 1-D projection). HOPG is the simplest case — a single c-axis tilt DOF for the (00l)
   reflections (in-plane azimuth is a symmetry no-op for g ∥ c).
2. For each orientation, apply the extra rotation to **g** at the existing `R_orient` hook
   (`montecarlo._orientation_R`), then recompute the full per-reflection block: the
   polarization pair `e_s/e_p`, `E_res`, the amplitudes `A_PXR/A_CBS`, and the sinc
   lineshape. `T_abs` (self-absorption) is **mosaic-independent** — it depends on `n̂` and
   depth, not on g — so it can be hoisted out of the orientation loop.
3. Sum the per-orientation spectra incoherently, weighted by the mosaic PDF (or
   equal-weighted if the K samples are drawn from the distribution). `K = 1` (no tilt)
   must reproduce today's result bit-for-bit.

### Pros

- **Exact** within the kinematical mosaic-block model: broadens **both** PXR and CBS,
  captures the amplitude / polarization variation across the cone, and produces the
  correct (generally asymmetric) lineshape for any η and geometry.
- **No grazing divergence** — there is no `tan ψ` linearization.
- Reproduces the **yield increase** the literature reports (mosaicity opens the
  diffracted-bremsstrahlung channel; a ~4× increase has been reported), which the
  energy-only convolution cannot.
- **Shares machinery** with the detector solid-angle integral
  ([detector-solid-angle.md](detector-solid-angle.md)) — both are "incoherently sum the
  spectrum over a distribution of a direction": g for mosaic, n̂ for the aperture. One
  "sum over (rotated-g, n̂) samples" accumulator serves both.

### Cons

- **Cost.** It multiplies the GPU hot loop (the per-reflection sinc² matmul, see the perf
  note in [../CLAUDE.md](../CLAUDE.md)) by `K`. With CuPy present the spectrum runs
  **serially** on one CUDA context, so `K` is a direct wall-clock multiplier with no
  parallelism to hide it. Prefer looping orientations (flat device memory) over a stacked
  g-axis (K× memory). If ever combined with the solid-angle integral the cost is
  **multiplicative** (`K × N_dir`).
- **RNG bookkeeping.** Needs a deterministic mosaic sub-stream distinct from the per-case
  transport `seed` (`sweep.build_cases`), or mosaic sampling couples to transport noise.
- **A true `K = 1` fast path** must be preserved so default runs and old checkpoints are
  unchanged.

### Effort

≈ **5–7 engineer-days**: the orientation loop + per-orientation recompute (keeping the
chunked-sinc and `sinc_cutoff` paths working), the RNG sub-stream, the `K = 1`
byte-identical guard, and a convergence/regression suite.

### When it is worth it

Only where the analytic model breaks: **broad lines / large η** (HOPG ZYH, steep tilt
near grazing) where the amplitude variation and asymmetry matter. For narrow lines the
analytic Gaussian already captures the width, and the Doppler skirt frequently dominates
both. Run a thin-vs-bulk linewidth comparison before committing.

### Validation plan

- `K → 1` and η → 0 reproduce the current spectrum bit-for-bit.
- Integrated line FWHM scales with η and converges to the analytic `E·|tan ψ|·η` in the
  small-η limit (away from grazing).
- Cross-check against the 2026 PXR-mosaicity paper: η 0.4° → 3.5° broadens the HOPG (002)
  rocking curve and lowers the peak by the reported factors.
