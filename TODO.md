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
   integration (now on `main`). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate — measured-data validation.** *IMPLEMENTED,
   only measured-data validation remains.* Models the vdW film (MoSe₂/MoS₂/WS₂/MoTe₂)
   on their real substrate (SiO₂/Si or sapphire). **Remaining:** quantitative validation
   against a *measured film-on-substrate dataset (data-dependent —
   no in-repo dataset yet).* Design: [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 — medium (experiment match + usability)

3. **pyelsepa / ELSEPA transport.** *in progress: see feature/elsepa-transport.*
   Paywalled ELSEPA source now added under the sibling `pyelsepa/elsepa/`
   (+ `adus_v1_0.tar.gz` in `pyelsepa/docker/`); no longer blocked on the data —
   next is building the docker image / wiring the adapter.

## P3 — lower / exploratory

4. **jupyter to marimo.** *in progress: see feature/marimo-transfer*
   Explore moving from jupyter notebooks to marimo, primarily to improve data visualization.
   In concert, look at move from matplotlib to altair
5. **Grazing-incidence soft X-ray diffraction grating.** *in progress: see feature/grazing-grating*

## PA — meta / agent-related / cleanup

6. **Repo Ownership and Name Change.** *cxr_model -> cxr-mc — DONE*
    1. **NOTE:** User still awaiting GitHub admin permissions on the repo;
       no code action needed — kept as a standing reminder.
7. **Agent Skill and Command Review.** Review project-specific skills and commands
    for usefulness, structure. Revise any which are likely useful but not perfected,
    remove any which are extraneous/irrelevant/not useful.
8. **Large Module Refactoring.** Refactor `montecarlo.py` and `plot.py` into submodules,
    they are extremely bloated. `montecarlo.py` especially has multiple discrete
    logical blocks which can be easily discretized.
9. **Notebook/Repo Reorg.** User does not like having analysis/scan/export notebooks
    and scripts cluttering the root. Move them either somewhere into /src/cxr_mc
    (post-rename), or create new subdirs. Consider a restructuring of /src/*;
    if that is decided to be worthwhile, triage into main priorities in TODO.md
10. **Polars investigation.** Investigate use of Polars dataframes to do
    packaging of large parameter sweep metadata.
11. **TODO Cleanup**. No reason to have in depth descriptions of exactly all TODOs here in user's opinion.
    Unless agent is strongly opposed, in order to save tokens, in-depth TODO details should be moved
    onto their relevant branches once created, with only minimal summaries described here.
    On those feature branches, TODO should be edited to reflect only the task(s) to be handled on that branch.
    If necessary, turn this into a skill/command.
