# TODO — feature/grazing-grating

This branch carries one backlog item; the full triaged backlog lives on `main`.

## Grazing-incidence soft X-ray diffraction grating (P3 — exploratory)

*Dispersion scaffold IMPLEMENTED.*

With an Eagle XO or an Alex detector, like those from Ultrafast Innovations — a new
experimental modality: disperse the soft-X-ray spectrum with a grazing-incidence reflection
grating and read the spatial image, for spectral resolution beyond the ~130 eV EDS width.

`src/cxr_mc/grating.py` implements the dispersion geometry (grating equation
`sinβ = mλ/d − sinα`, angular dispersion, flux-conserving `disperse_spectrum`, pixel-limited
resolving power), cross-checked in `tests/test_grating.py`.

**Remaining (next high-value step):** grating reflectivity / groove efficiency `R(E,α)`
(xraydb → Fresnel at grazing angle), then a detected-image forward model beside the
Timepix / Eagle XO ones, and validation vs measured data. Design + phased plan:
[`docs/grazing-grating.md`](docs/grazing-grating.md).
