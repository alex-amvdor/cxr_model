# TODO / Backlog

Durable, triaged backlog for `cxr_model`. Items live on `feature/…` / `patch/…`
branches, not `main`, until finished — the PI publishes *finished* code.
Priorities weigh value-to-goal (the line-flux / enhancement predictions and the
publication's validation story) against effort and risk.

> The publication-readiness epic (Phases A–E) is complete.

## P1 — high value (physics accuracy + publication validation)

1. **Crystal mosaicity — exact Monte-Carlo route.** *(feature — IMPLEMENTED; measured-data
   validation remains)* The incoherent per-orientation average inside `mc_spectrum`
   (`Sweep(mosaic_route="mc")`, 2-D Gauss-Hermite quadrature over crystallite tilt) is
   done: it broadens PXR+CBS, captures the asymmetric lineshape, has no `tan ψ` grazing
   divergence, and keeps the perfect-crystal path bit-for-bit. Cross-checked in
   `checks/mosaic_mc_check.py` (bit-for-bit, η→0, small-η→analytic, broadening, yield) and
   `tests/test_mosaic_mc.py`. The scoping check (`checks/mosaic_scoping_check.py`) showed
   the line is genuinely mosaic-broad for HOPG's real grades — ZYB/ZYH ≳ the
   multiple-scattering Doppler width even in bulk — which **corrects the earlier "Doppler
   dominates" caveat**. **Remaining:** validate the broadened line widths against a measured
   HOPG rocking-curve / EDS dataset (the headline reason it exists; data-dependent). Still
   shares the orientation-average pattern with the detector solid-angle integration (P2 #4).
   Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

2. **Physics-anchor comparison plots.** *(feature)* Make the `checks/`
   Feranchuk/Zhai anchors emit figures versus the literature (especially **Zhai
   Fig 1C**), ideally a minimal pre-configured viz notebook (backend source
   modules feeding the notebook are fine). High value for the publication's
   validation story, and relatively self-contained.

3. **Multilayer film-on-substrate — measured-data validation.** *(feature — slices 1–3
   IMPLEMENTED; measured-data validation remains)* A vdW film (MoSe₂/MoS₂/WS₂/MoTe₂) on its
   real substrate (SiO₂/Si or sapphire), with per-crystalline-layer radiation and full-stack
   self-absorption. Slice 1 (cross-stack self-absorption), slice 2 (multilayer electron
   transport — substrate backscatter + material-aware brem), and slice 3 (per-layer coherent
   radiation — a *crystalline* substrate emits its own lines via `Sweep(substrate="silicon")`)
   are done: bit-for-bit single-layer, cross-checked in `checks/multilayer_check.py` (slices
   1–2) and `checks/multilayer_slice3_check.py` (slice 3). **Remaining:** quantitative
   validation against a measured film-on-substrate dataset (data-dependent). Design:
   [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 — medium (experiment match + usability)

4. **Detector solid-angle integration.** *(feature)* Replace the single-`n̂` +
   flat-`domega_sr` + analytic `aperture_fwhm_eV` approximation with a
   first-principles integral of `mc_spectrum` over a grid of `n̂` tiling the chip.
   Matters for the wide SEM/TEM detectors in the source paper (≈12–17°),
   negligible for the tiny Timepix Ω. Shares machinery with the mosaic MC route
   (P1 #1). Design: [`docs/detector-solid-angle.md`](docs/detector-solid-angle.md).

5. **Checkpointing GB-transfer.** *(patch)* The single-pickle-per-material method
   produces gigabyte transfers full of stale data. Move to multiple pickles per
   material, or filter the remote pickle to only the required information before
   transfer. (Partial mitigation landed plot-side — `select_results` /
   `sweep_values` slice a loaded checkpoint in-memory — but the on-disk pickle is
   still the full union, so the transfer-size problem itself is unsolved.)

6. **pyelsepa / ELSEPA transport.** *(feature)* Evaluate replacing the hardcoded
   NIST Mott transport tables in `src/cxr_model/data/mott_transport_cross_sections/`
   with on-demand ELSEPA (a docker image is built locally at `C:\dev\pyelsepa\`,
   [github.com/eScatter/pyelsepa](https://github.com/eScatter/pyelsepa)). NB: this
   is *electron*-scattering data — separate from the xraydb (photon) migration;
   xraydb cannot supply it. The analytic screened-Rutherford fallback already
   works, so this is robustness / removing a hardcoded dataset, not a blocker.

## P3 — lower / exploratory

7. **Pint (or similar) units.** *(feature)* Evaluate Pint / natu / Buckingham for
   project-wide unit safety; implement the best fit if worthwhile. High churn — a
   units refactor touches every amplitude site — so scope carefully; low priority
   relative to the physics work above.

8. **Grazing-incidence soft X-ray diffraction grating.** *(feature)* With an Eagle
   XO or an Alex detector, like those from Ultrafast Innovations. A new
   experimental modality; exploratory.

9. **First-class custom large-parameter-space sweeps.** *(feature)* Interactive
   slicing/faceting of many simultaneously-swept knobs, beyond the current
   heatmap/line auto-pick. Viz/UX polish; the current auto-pick works.
