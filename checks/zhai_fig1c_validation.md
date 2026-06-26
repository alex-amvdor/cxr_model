---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python (sci)
  language: python
  name: sci
---

# Zhai Fig. 1c -- model vs theory anchors

A minimal, pre-configured viz wrapper over [`anchor_figures.py`](anchor_figures.py).
It renders the Monte-Carlo model's PXR+CBS spectra against the first-principles
**theory anchors** (TODO P1 #2):

1. **dispersion-relation line energies** (Feranchuk-Spence Eq. 10) -- the MC peaks
   must land on these vertical markers;
2. the **Feranchuk Eq. (12) closed form** for the absolute line flux -- a single
   straight segment through `mc_spectrum` reproduces it to <1%;
3. the **bulk vs 29 nm film enhancement**, against the analytic no-transport ceiling.

If a digitized Fig 1c curve is dropped in `reference_data/zhai_fig1c.csv`
(schema in `reference_data/README.md`), it is overlaid automatically and this
becomes a true model-vs-measured comparison. See the module docstring for the
physics and references.

```{code-cell} ipython3
# Run from the repo root. On a machine WITHOUT a usable CUDA device (the cupy
# wheel is installed but there is no GPU), uncomment the next line to force the
# CPU path; on the GPU workstation leave it commented.
# import sys; sys.modules["cupy"] = None

import sys

sys.path.insert(0, "checks")  # anchor_figures, feranchuk_spence
sys.path.insert(0, "src")  # cxr_mc

%matplotlib inline
import anchor_figures as af
```

```{code-cell} ipython3
# Experimental conditions (Zhai SI S3). Override any field to explore other
# crystals / energies / detector geometries with the same machinery.
anchor = af.ZhaiAnchor()
af.theory_line_energies(anchor)  # {beam energy [keV]: Eq.(10) line energy [eV]}
```

```{code-cell} ipython3
# The Monte-Carlo spectra. SLOW on CPU; raise ne toward ~500 for publication
# quality (anchor_figures.main() defaults to 500).
model = af.model_spectra(anchor, ne=200, ne_brem=100)
reference = af.reference_curve(anchor)
print("reference data:", "LOADED" if reference else "none (theory-only overlay)")
```

```{code-cell} ipython3
from tabulate import tabulate

print(
    tabulate(
        af.validation_table(anchor, model),
        headers=[
            "E0\n[keV]",
            "MC peak\n[eV]",
            "Eq.10\n[eV]",
            "diff\n[eV]",
            "MC/closed\n(1 seg)",
            "line flux\n[ph/e/0.066sr]",
            "backscatter",
        ],
        tablefmt="github",
    )
)
```

```{code-cell} ipython3
# Fig 1c analog vs theory: intrinsic + EDS-convolved spectra, dotted Eq.(10)
# line energies, 29 nm film overlay, and the measured curve if present.
af.figure_spectra(anchor, model, reference);
```

```{code-cell} ipython3
# Absolute-flux anchor: single-segment MC/closed-form ratio (~1) and the
# MC-vs-Feranchuk line flux per beam energy.
af.figure_flux_anchor(anchor, model);
```

```{code-cell} ipython3
# Bulk vs 29 nm film enhancement at the top beam energy.
af.figure_enhancement(anchor, model);
```
