# TODO / Backlog

Items live on `feature/…` / `patch/…` branches, not `main`, until finished.
Full detail for an in-progress item lives on its branch (or its design doc);
`main` keeps only a one-line summary + pointer. Priorities weigh value-to-goal
(line-flux / enhancement predictions + the publication's validation story)
against effort and risk.

## P1 — high value (physics accuracy + publication validation)

1. **Crystal mosaicity — measured-data validation.** MC route implemented; validate
   broadened line widths vs. a measured HOPG rocking-curve / EDS dataset
   (data-dependent). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate — measured-data validation.** Model implemented;
   validate vs. a measured film-on-substrate dataset (data-dependent). Design:
   [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 — medium (experiment match + usability)

3. **pyelsepa / ELSEPA transport.** → `feature/elsepa-transport`. Adapter landed;
   gated on building the `elsepa` docker image (paywalled Fortran source acquired).

## P3 — lower / exploratory

4. **jupyter → marimo (+ matplotlib → altair).** → `feature/marimo-transfer`.
   Improve data viz/interactivity; Altair spectrum renderer is the first slice.
5. **Grazing-incidence soft X-ray diffraction grating.** → `feature/grazing-grating`.
   Dispersion scaffold implemented; next is grating reflectivity + detected-image model.

## PA — meta / cleanup

6. **Repo ownership & name change.** `cxr_model` → `cxr-mc` DONE; awaiting GitHub
   admin permissions (standing reminder, no code action).
7. **Agent skill & command review.** Triage project-specific skills/commands; fix the
   useful-but-rough ones, remove the extraneous.
8. **Notebook/repo reorg.** Move root analysis/scan/export notebooks + scripts into
   `src/cxr_mc/` or new subdirs; consider a `src/` restructure and re-triage if so.
9. **Polars investigation.** Evaluate Polars for packaging large parameter-sweep metadata.
10. **TODO formatting convention.** `main` carries minimal summaries; full detail lives
    on the item's branch, slimmed to only that branch's task. (This file follows it.)
    Enforced by the `/docs:todo-sync` command.
