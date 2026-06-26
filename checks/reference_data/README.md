# Reference data for the validation figures

Drop digitized literature curves here and `checks/anchor_figures.py` overlays
them on the model figures automatically — turning the theory-anchored plots
into true *model-vs-measured* comparisons with no code change.

## `zhai_fig1c.csv` — Zhai et al. Fig. 1c

`checks/anchor_figures.py:reference_curve()` looks for `zhai_fig1c.csv` in this
directory. If it is absent, the figure falls back to theory-only (the
dispersion-relation line markers). If present, each series is overlaid on the
EDS-convolved panel, scaled to the model peak for that beam energy (shape
comparison; absolute calibration of a digitized figure is rarely meaningful).

### Schema

A comma-separated file with a header row and these columns:

| column      | meaning                                                        |
| ----------- | -------------------------------------------------------------- |
| `series`    | beam-energy label, e.g. `17.5keV`, `20keV`, `22.5keV`, `25keV` |
| `energy_eV` | photon energy of the digitized point, in eV                   |
| `intensity` | intensity at that point (arbitrary units are fine)            |

- `#`-prefixed lines are treated as comments and ignored.
- The series label is parsed tolerantly: `25`, `25keV`, `25.0 keV` all match the
  25 keV model curve (within 0.25 keV).
- See `zhai_fig1c.example.csv` for the exact format. That example contains
  **synthetic placeholder points only** — it is *not* loaded (the loader reads
  `zhai_fig1c.csv`, not the `.example.csv`). Replace it with real digitized data
  (e.g. from WebPlotDigitizer on the published figure) and rename to
  `zhai_fig1c.csv` to activate the overlay.

### Provenance note

When you add real data, record where it came from (paper figure + panel,
digitizer tool, date) at the top of the CSV as `#` comments, so the overlay
stays citable.
