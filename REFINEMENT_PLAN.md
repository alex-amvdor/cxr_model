# Publication-Readiness Refinement — status

> Internal record of the publication-readiness epic (the PI's "public-ready
> codes" request). The phased work below is **complete**; the durable backlog
> lives in [`TODO.md`](TODO.md). Delete this file when the repo is published.

## Phases (A–E, all complete)

- **A — Docs single-sourcing + de-stale.** [`README.md`](README.md) is the
  user/scientific doc, [`CLAUDE.md`](CLAUDE.md) the contributor guide, `TODO.md`
  the backlog; stale post-xraydb / post-`mote2` content removed.
- **B — Installable package + CLI.** `src/` → `src/cxr_model/` (real package,
  hatchling build, package data via `cxr_model.DATA_DIR`) with the `cxr` console
  script (`cxr scan` / `cxr export`); root shims kept for the notebooks and the
  GPU box.
- **C — Branch hygiene.** `main` carries only finished code; multilayer **slice 1**
  is the documented opt-in capability, **slices 2–3** stay on
  `feature/multilayer-materials`.
- **D — Sphinx docs site.** `docs/` builds clean (MyST design notes + autosummary
  API): `uv run --group docs sphinx-build -b html docs docs/_build/html`.
- **E — Docker (CPU).** `Dockerfile` + `.dockerignore`; CPU-only, structured so a
  GPU image is a base-image swap.

## Deferred

- **Rebase `feature/multilayer-materials` onto `main`** before resuming multilayer
  slice 3 (it forks from a pre-restructure commit).
- Priority-2 feature / patch backlog → [`TODO.md`](TODO.md).
