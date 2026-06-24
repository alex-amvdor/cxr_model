# TODO / Backlog

Durable backlog for `cxr_model`. The publication-readiness refinement epic
(Phases A–E) is **complete** — see [`REFINEMENT_PLAN.md`](REFINEMENT_PLAN.md) (an
internal record; delete it when the repo is published).

Items are kept on branches, not `main`, until finished — the PI publishes
*finished* code. Roughly ordered by value to the project goal.

## Features

- **Physics-anchor comparison plots.** Make the `checks/` Feranchuk/Zhai anchors
  emit figures versus the literature (especially **Zhai Fig 1C**), ideally a
  minimal pre-configured viz notebook (backend source modules feeding the notebook
  are fine). High value for the publication's validation story.

- **Multilayer film-on-substrate materials.** A vdW film (MoSe₂/MoS₂/WS₂/MoTe₂) on
  its real substrate (SiO₂/Si or sapphire), with per-crystalline-layer radiation
  and full-stack self-absorption. Status: **slice 1 (cross-stack self-absorption)
  is on `main`**; **slice 2 (multilayer electron transport)** is on branch
  `feature/multilayer-materials`. Remaining: **slice 3** (coherent lines from a
  *crystalline* substrate) + quantitative validation vs a measured dataset. Full
  design: [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

- **pyelsepa / ELSEPA transport.** Evaluate replacing the hardcoded NIST Mott
  transport tables in `src/cxr_model/data/mott_transport_cross_sections/` with
  on-demand ELSEPA.
  A docker image is built locally at `C:\dev\pyelsepa\`
  ([github.com/eScatter/pyelsepa](https://github.com/eScatter/pyelsepa)). NB: this
  is *electron*-scattering data — separate from the xraydb (photon) migration;
  xraydb cannot supply it. Robustness / removes a hardcoded dataset.

- **Pint (or similar) units.** Evaluate Pint / natu / Buckingham / Units for
  project-wide unit safety; implement the best fit if worthwhile. Scope carefully —
  a units refactor touches every amplitude site.

- **Crystal mosaicity — exact Monte-Carlo route.** An incoherent sum over
  Gaussian-spread crystallite orientations inside `mc_spectrum`, broadening PXR+CBS
  per orientation. The cheap analytic route already landed; the MC route is the
  upgrade for the large-mosaic / broad-line regime (HOPG ZYH, ψ→90°). Shares
  machinery with the detector solid-angle integration. Caveat: the electron
  multiple-scattering Doppler width often dominates, so mosaic is visible mainly in
  thin / near-perfect crystals. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

- **Detector solid-angle integration.** Replace the single-`n̂` + flat-`domega_sr`
  + analytic `aperture_fwhm_eV` approximation with a first-principles integral of
  `mc_spectrum` over a grid of `n̂` tiling the chip. Matters for the wide SEM/TEM
  detectors (≈12–17°), negligible for the tiny Timepix Ω. Design:
  [`docs/detector-solid-angle.md`](docs/detector-solid-angle.md).

- **Grazing-incidence soft X-ray diffraction grating** (with an Eagle XO or an
  Alex detector), like those from Ultrafast Innovations.

- **First-class custom large-parameter-space sweeps** — interactive
  slicing/faceting of many simultaneously-swept knobs, beyond the current
  heatmap/line auto-pick.

## Patches

- **Checkpointing GB-transfer.** The current single-pickle-per-material method
  produces gigabyte transfers full of stale data. Move to multiple pickles per
  material, or filter the remote pickle to only the required information before
  transfer. (Partial mitigation landed plot-side: `select_results`/`sweep_values`
  slice a loaded checkpoint in-memory before plotting, but the on-disk pickle is
  still the full union, so the transfer-size problem itself is unsolved.)
