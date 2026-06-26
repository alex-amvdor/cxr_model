# TODO / Backlog

Items live on `feature/…` / `patch/…` branches, not `main`, until finished.
Priorities weigh value-to-goal (the line-flux / enhancement predictions and the
publication's validation story) against effort and risk.

## P1 — high value (physics accuracy + publication validation)

1. **Crystal mosaicity — exact Monte-Carlo route.** *IMPLEMENTED; measured-data
   validation remains.* The incoherent per-orientation average inside `mc_spectrum`
   (`Sweep(mosaic_route="mc")`, 2-D Gauss-Hermite quadrature over crystallite tilt) is
   done: it broadens PXR+CBS, captures the asymmetric lineshape, has no `tan ψ` grazing
   divergence, and keeps the perfect-crystal path bit-for-bit. Cross-checked in
   `checks/mosaic_mc_check.py` (bit-for-bit, η→0, small-η→analytic, broadening, yield) and
   `tests/test_mosaic_mc.py`. The scoping check (`checks/mosaic_scoping_check.py`) showed
   the line is genuinely mosaic-broad for HOPG's real grades — ZYB/ZYH ≳ the
   multiple-scattering Doppler width even in bulk. **Remaining:** validate the broadened line
   widths against a measured HOPG rocking-curve / EDS dataset (the headline reason it exists;
   data-dependent). Still shares the orientation-average pattern with the detector solid-angle
   integration (P2 #4). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Physics-anchor comparison plots.** *in progress: see feature/anchor-figures.* Make the `checks/`
   Feranchuk/Zhai anchors emit figures versus the literature (especially **Zhai
   Fig 1C**), ideally a minimal pre-configured viz notebook (backend source
   modules feeding the notebook are fine). High value for the publication's
   validation story, and relatively self-contained.
3. **Multilayer film-on-substrate — measured-data validation.** *IMPLEMENTED,
   only measured-data validation remains.* Models the vdW film (MoSe₂/MoS₂/WS₂/MoTe₂)
   on their real substrate (SiO₂/Si or sapphire). **Remaining:** quantitative validation
   against a *measured film-on-substrate dataset (data-dependent —
   no in-repo dataset yet).* Design: [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 — medium (experiment match + usability)

4. **Detector solid-angle integration.** *in progress: see feature/detector-solid-angle*
5. **Checkpointing GB-transfer.** *in progress: see feature/checkpoint-transfer*
6. **pyelsepa / ELSEPA transport.** *in progress: see feature/elsepa-transport*

## P3 — lower / exploratory

7. **jupyter to marimo.** *in progress: see feature/marimo-transfer* Explore moving
   from jupyter notebooks to marimo, primarily to improve data visualization.
8. **Pint units.** *see feature/units-evaluation*
9. **Grazing-incidence soft X-ray diffraction grating.** *in progress: see feature/grazing-grating*
10. **First-class custom large-parameter-space sweeps.** *in progress: see feature/sweep-faceting*

## PA — meta / agent-related / cleanup

10. **Repo Ownership and Name Change.** *cxr_model -> cxr-mc — DONE*
    1. **NOTE:** User still awaiting GitHub admin permissions on the repo;
       no code action needed — kept as a standing reminder.
11. **Agent Skill and Command Review.** Review project-specific skills and commands
    for usefulness, structure. Revise any which are likely useful but not perfected,
    remove any which are extraneous/irrelevant/not useful.
12. **Large Module Refactoring.** Refactor `montecarlo.py` and `plot.py` into submodules,
    they are extremely bloated. `montecarlo.py` especially has multiple discrete
    logical blocks which can be easily discretized.
13. **Notebook/Repo Reorg.** User does not like having analysis/scan/export notebooks
    and scripts cluttering the root. Move them either somewhere into /src/cxr_mc
    (post-rename), or create new subdirs. Consider a restructuring of /src/*;
    if that is decided to be worthwhile, triage into main priorities in TODO.md
