---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Bulk-crystal CXR — scan runner

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material checkpoint (`checkpoints/<material>.pkl`). The companion
**`analysis.ipynb`** loads that checkpoint and draws every figure — keeping
the long scan and the (re-runnable) plotting in separate kernels.

Set `MATERIAL` below, then run top to bottom. Every material's sweep grid lives
in `src/config.py`, shared with the analysis notebook so the two never drift.

```{code-cell} ipython3
import sys

sys.path.insert(0, "src")

from cxr_mc.sweep import build_cases, geometry_table
from cxr_mc.config import default_settings, material_sweep, COLLAPSE_AZIMUTH
from cxr_mc.plots import stream_chunk
from cxr_mc.run import run_sweep
from IPython.display import display
```

```{code-cell} ipython3
# Material: "hopg" | "diamond" | "silicon" | "mose2" | "wse2" | "ptse2"
#         | "hfse2" | "zrse2" | "ws2" | "mos2"
MATERIAL = "hopg"

settings = default_settings()
sweep = material_sweep(MATERIAL)  # full parametric grid (src/config.py)

cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
print(f"{len(cases)} cases across {len({c['name'] for c in cases})} configs")
display(geometry_table(cases))
```

```{code-cell} ipython3
# Run (resumes from the checkpoint, skipping cached cases). The per-tilt
# photon-counting tables stream live; all the figures are in analysis.ipynb.
results = {}
try:
    run_sweep(
        cases,
        results,
        on_chunk=lambda batch: stream_chunk(
            results, batch, settings, collapse_azimuth=COLLAPSE_AZIMUTH
        ),
    )
except EOFError:
    print("EOF Error -- the script has already processed all data")

print(f"\nDone -> checkpoints/{MATERIAL}.pkl")
print("Open analysis.ipynb with the same MATERIAL to visualize.")
```
