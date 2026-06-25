# Grazing-incidence soft X-ray grating spectrometer

An exploratory new measurement modality (TODO P3 #8): instead of reading the
PXR+CBS spectrum as energy-vs-counts at a fixed take-off angle (EDS/Timepix),
**disperse** it with a reflection grating at grazing incidence and read the
resulting *spatial* image on a position-sensitive detector (Raptor Eagle XO CCD,
or an "Alex"-type detector from Ultrafast Innovations). The grating's dispersion
gives spectral resolution set by geometry and pixel size rather than by the
detector's intrinsic energy resolution — potentially far better than the ~130 eV
EDS line width that dominates the soft-X-ray band today.

---

## Status

`src/cxr_model/grating.py` implements the **dispersion geometry** (the concrete,
testable physics) as a standalone forward model — nothing in the sweep/plot
pipeline imports it yet. Cross-checked in `tests/test_grating.py`.

| provided | meaning |
| --- | --- |
| `wavelength_angstrom(E)` / `groove_spacing_angstrom(ρ)` | unit conversions |
| `Grating(groove_density_per_mm, alpha_rad, order)` | a grating in a fixed mount |
| `Grating.diffraction_angle_rad(E)` | the grating equation `sinβ = mλ/d − sinα` (NaN if no propagating order) |
| `Grating.angular_dispersion_rad_per_angstrom(E)` | `dβ/dλ = m/(d cosβ)` |
| `disperse_spectrum(E, spec, grating, distance_mm)` | **flux-conserving** map of a spectrum to detector position (`∫I dx = ∫spec dE`) |
| `resolving_power(E, grating, distance_mm, pixel_mm)` | pixel-limited `λ/Δλ` |

## Physics and conventions

Reflection grating equation, angles from the grating **normal**:

```
d (sin α + sin β) = m λ          →     sin β = m λ / d − sin α
```

- `α` = incidence angle, `β` = diffraction angle (from normal); `m = 0` is
  specular (`β = −α`).
- **Grazing** incidence (grazing angle `θ_g = 90° − α` of a few degrees) is a
  choice of `α`, not a different equation — it is what buys usable reflectivity
  for soft X-rays. In this regime first orders never run out of propagating
  solutions across 100 eV–2 keV (the NaN branch is only hit by pathological
  rulings; see the test).
- `λ = hc / E` with `hc = 12398.42 eV·Å` (`crystallography.HC_EV_ANG`).

## What is NOT modelled yet (and why)

This is *dispersion geometry only*. Deliberately out of scope for the scaffold:

- **Grating reflectivity / groove efficiency** vs energy and angle — needs the
  coating optical constants and a groove-profile efficiency model (e.g. a scalar
  or rigorous-coupled-wave treatment). This sets the absolute throughput and the
  usable band, so it is the first thing to add before any flux comparison.
- **Aberrations / focusing** (spherical or VLS gratings, Rowland circle) — the
  flat-detector `x = L tan(β − β_ref)` map ignores defocus and coma.
- **Source size / beam divergence** — a real line image is the convolution of the
  dispersion with the source spot and slit; the pixel-limited `resolving_power`
  is an upper bound.

## Phased plan to a real modality

1. **(done)** Dispersion geometry + resolving power, tested.
2. Reflectivity/efficiency `R(E, α)` from coating optical constants (xraydb can
   supply `f', f''` → δ, β → Fresnel reflectivity at grazing angle) × a groove
   efficiency factor; multiply into `disperse_spectrum`.
3. A forward-model entry that takes the model's `mc_spectrum` output + a
   `Grating` + an `eaglexo_response`-style detector and returns the **dispersed,
   detected image** (counts vs pixel), so it slots in beside the Timepix / Eagle
   XO models.
4. Optionally expose grating parameters as `Sweep` knobs and add a `plots.py`
   panel; validate against a measured grating-spectrometer dataset when available
   (data-dependent, like P1 #1/#3).

Reflectivity (step 2) is the next high-value piece — until then the dispersed
profile is a *relative* spectrum, correct in position but not in throughput.
