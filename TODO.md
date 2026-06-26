# TODO / Backlog

Items live on `feature/…` / `bugfix/…` / `docs/…` branches, not `main`, until finished.
Full detail for an in-progress item lives on its branch (or its design doc);
`main` keeps only a one-line summary + pointer, enforced by /docs:todo-slim.
Priorities weigh value-to-goal (line-flux / enhancement predictions + the publication's validation story)
against effort and risk.

Item generation:
----------------

1. Create and move to branch of relevant type
2. Overwrite branch TODO.md with concise, 2-3 sentence problem summary + implementation path, scoped only to the relevant item, then publish to `origin`
3. Move to `main`, create 1 sentence summary of new item, then triage into existing TODO.md items and push tightly scoped `docs(todo)` commit to main

**NOTE:** If user has directly written a detailed item summary into `TODO.md` on main, , removing detailed item info from `main` once branch has been created

## P1 — high value (physics accuracy + publication validation)

1. **Crystal mosaicity — measured-data validation.** MC route implemented; validate
   broadened line widths vs. a measured HOPG rocking-curve / EDS dataset
   (data-dependent). Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
2. **Multilayer film-on-substrate — measured-data validation.** Model implemented;
   validate vs. a measured film-on-substrate dataset (data-dependent). Design:
   [`docs/multilayer-materials.md`](docs/multilayer-materials.md).

## P2 — medium (experiment match + usability)

1. **USER EDITED: pyelsepa / ELSEPA transport.** → `feature/elsepa-transport`.
   Gated on installing eScatter/cslib/pint for validation.
2. **USER ADDED: eScatter/cstool/Nebula investigation.** Evaluate merits of robustness,
   accuracy improvement, repo simplification, and any other critical items for a potential
   move to use of one or more of the listed libraries/tools. In particular, evaluate if the
   added physics of the Penn dielectric fxn, tracking secondary electrons, etc., is worth
   the added compute, or if the Mott/partial dirac/CSDA is enough. This item is tied
   intimately related to TODO #P2-1, above.
3. **Polars investigation.** Evaluate Polars for packaging large parameter-sweep metadata.

## P3 — lower / exploratory

1. **jupyter → marimo (+ matplotlib → altair).** → `feature/marimo-transfer`.
   Improve data viz/interactivity; Altair spectrum renderer is the first slice.
2. **Grazing-incidence soft X-ray diffraction grating.** → `feature/grazing-grating`.
   Dispersion scaffold implemented; next is grating reflectivity + detected-image model.

## PA — meta / cleanup

1. **Repo ownership & name change.** `cxr_model` → `cxr-mc` DONE; awaiting GitHub
   admin permissions (standing reminder, no code action).
2. **Agent skill & command review.** Triage project-specific skills/commands; fix the
   useful-but-rough ones, remove the extraneous.
3. **Notebook/repo reorg.** Move root analysis/scan/export notebooks + scripts into
   `src/cxr_mc/` or new subdirs; consider a `src/` restructure and re-triage if so.
