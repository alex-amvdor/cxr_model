# Publication-Readiness Refinement — Plan & Triage

> **Internal working document** for the refinement agent. Delete (or move to an issue
> tracker) before the repo is published. Authored 2026-06-23 as a handoff.

## 0. Context & ground rules

The PI wants to feature "public-ready" group codes on a website subtab (title + DOI +
one-paragraph blurb + GitHub permalink). **This task is to get `cxr_model` into
publication-ready shape only** — per CLAUDE.md §"General Project Refinement":

- **Do NOT** create the external GitHub repo, assign a DOI, or write the website blurb yet.
- **Do NOT break the validated physics.** After any structural change, run the safety net:
  - `uv run python -m pytest -q` (82 tests)
  - `uv run python checks/feranchuk_check_script.py` (analytic LiF anchor, CPU)
  - MC anchors on a GPU-less machine must be CPU-forced by masking cupy, e.g.
    `uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/feranchuk_vs_zhai_check.py',run_name='__main__')"`
    (29 nm A/B must stay **1.00**); same pattern for `checks/multilayer_check.py`.
- Environment: Python ≥3.14, **uv**-managed, optional CuPy GPU (this laptop has the wheel
  but **no CUDA toolkit** → CPU-force or run on the `qlmc` box). Keep both paths working.

## 1. Repo snapshot (as of this handoff)

- **Layout:** `src/` (11 flat modules, imported via `sys.path.insert(0,"src")` — NOT a
  package), `checks/` (validation), `docs/` (4 design notes + README), `tests/` (82 pass),
  notebooks `scan.ipynb`/`analysis.ipynb`, scripts `scan.py`/`remote.py`/`export_pdf.py`,
  `pyproject.toml`+`uv.lock`, `README.md`, `CLAUDE.md`.
- **Packaging:** `pyproject.toml` has `[project]`+deps and `[tool.pytest]` but **no
  `[build-system]` and no `[project.scripts]`** → the project is not installable and has no
  CLI entry points. Scripts run as `uv run python scan.py …`.
- **main** is green and validated. It carries multilayer **slice 1** (an opt-in,
  bit-for-bit-safe capability); the in-progress **slice 2** lives on
  `feature/multilayer-materials`, off main (see §2).

## 2. CRITICAL handoff notes — read before touching anything

1. **Multilayer feature is mid-flight; the CLAUDE.md TODO entry is STALE** (it says the
   engine is "unstarted" — it is **not**). Placement is now **RESOLVED**:
   - **Slice 1** (cross-stack self-absorption) — on **`main`** (`b8c9422`, `8e955be`).
   - **Slice 2** (multilayer electron transport → substrate backscatter + material-aware
     bremsstrahlung) — on branch **`feature/multilayer-materials`** (`b67d5be`), off main.
   - Both are **opt-in**: `substrate=None` is bit-for-bit the old single-material path
     (regression-anchored). Remaining: slice 3 (coherent lines from a *crystalline*
     substrate) + quantitative validation vs a measured film-on-substrate dataset.
   - Status/design: [`docs/multilayer-materials.md`](docs/multilayer-materials.md).
   - **For the agent:**
     - Fix the stale CLAUDE.md "Features" entry (slices 1–2 done; only slice 3 + validation
       remain) — fold into the Phase A doc pass.
     - `main`'s `docs/multilayer-materials.md` still reads *"designed, not implemented"* even
       though slice 1 IS on main (its implementation-status block lives on the feature
       branch). Reconcile main's copy to reflect slice 1.
     - `feature/multilayer-materials` branches off the **pre-amend** TODO commit (`3543192`),
       not the current main tip (the commit that added this plan). Before resuming slice 3,
       `git rebase main` the branch — it's clean (only `REFINEMENT_PLAN.md` differs).
2. **README.md and CLAUDE.md duplicate** large sections (layout, module table, physics
   conventions, validation, data provenance, gotchas). Pick **one** source of truth
   (README for users; CLAUDE.md for agent conventions) and delete the duplication.
3. **README.md is STALE** post-xraydb-migration + mote2 addition. Fix at minimum:
   - Module table `atomic_form_factors.py` row says "Cromer–Mann + Henke/CXRO, `Z_TABLE`" →
     now **Waasmaier–Kirfel f0 + Chantler f',f'' via xraydb**, no `Z_TABLE`/`CROMER_MANN`.
   - "Adding a new material" prose lists `Z_TABLE` + Cromer–Mann coeffs + a Henke `.csv` →
     those are gone (xraydb supplies any element); only TOML + config + sweep + edge flag.
   - Materials table omits **`mote2`** (2H-MoTe₂ is in the code/TOML).
   - "Data provenance" + "What remains to be validated" still say the library was
     "evaluated but not adopted" / atomic swap "would require re-validation" → it **was
     adopted and re-validated** (see [`docs/atomic-data-sources.md`](docs/atomic-data-sources.md)).
   - (CLAUDE.md was already updated for these; README lagged.)

## 3. Refinement epic — phased plan (Priority 1, publication-blocking)

Ordered to minimize risk and unblock later phases. Each phase ends with the §0 safety net.

### Phase A — Documentation single-sourcing + de-stale  *(low risk, high value, do first)*
- De-stale README (§2.3).
- Single-source README ↔ CLAUDE.md (§2.2). Target end state:
  - **README.md** = the user/scientific doc (what it does, physics, install, quickstart,
    materials, detectors, validation, provenance, refs).
  - **CLAUDE.md** = a *concise* agent/contributor operating guide (build/test/run commands,
    the "Adding a material" checklist, the key gotchas, pointers to README/docs) — **strip
    the duplicated project narrative and move the TODOs out** (→ `TODO.md` or GitHub Issues;
    TODOs do not belong in CLAUDE.md). Aim < ~100 lines.
  - Consider a short **CONTRIBUTING.md** for dev workflow if useful.
- Acceptance: no factual staleness; no README/CLAUDE duplication; TODOs relocated.

### Phase B — Make it an installable package + CLI  *(medium effort, high value)*
- Restructure `src/` → `src/cxr_model/` with `__init__.py`; replace `sys.path.insert`
  hacks with real imports (`from cxr_model import …`) in notebooks, scripts, checks, tests.
  Keep `data/` resolvable (package-data or an importlib.resources accessor).
- Add `[build-system]` (hatchling) so `uv sync`/`uv pip install -e .` installs the package.
- Add a CLI via `[project.scripts]`. **Recommended syntax:** a single `cxr` entry with
  subcommands (typer or argparse):
  - `cxr scan <material> [--quick] [--workers N]`  (wraps `scan.py`)
  - `cxr remote scan <material> [--quick]` / `cxr remote pull <material>` (wraps `remote.py`)
  - `cxr export <material>` (wraps `export_pdf.py`)
  Keep the `if __name__=="__main__"` guards (spawn/forkserver re-import — see gotchas).
- Acceptance: fresh `git clone` + `uv sync` → `cxr scan diamond --quick` runs; `import
  cxr_model` works with no path hacks; notebooks run post-clone; tests still green.

### Phase C — Branch hygiene + main polish  *(low effort)*
- Multilayer placement is already resolved (slice 1 on main, slice 2 on
  `feature/multilayer-materials` — §2.1); just reconcile main's
  `docs/multilayer-materials.md` status and the CLAUDE.md TODO entry, and rebase the
  feature branch onto the current main tip per §2.1.
- Audit for any other half-finished code on main; move to `feature/…`/`patch/…` branches.
- Optional: if you'd rather main carry **zero** in-progress code, also move slice 1 to the
  feature branch (clean `git revert` on main — no force-push needed); otherwise keep it as
  the documented opt-in capability it is.
- Acceptance: main contains only finished, functional, documented code.

### Phase D — Sphinx docs site  *(optional; evaluate, then implement if pursued)*
- **Recommendation:** the code is richly docstringed and already has 4 design notes, so it
  is a *reasonable* Sphinx candidate — but the PI only needs README + blurb + DOI, so treat
  a hosted site as polish, not a blocker. If pursued, keep it **lightweight**: `sphinx` +
  `myst-parser` (render the existing `docs/*.md`) + `autodoc`/`autosummary` over
  `cxr_model`. Do it **after** Phase B (autodoc needs the importable package).
- Acceptance (if pursued): `sphinx-build docs _build/html` clean; API + design notes render.

### Phase E — Docker  *(optional; evaluate, then implement if worthwhile)*
- **Recommendation:** a **CPU-only** `Dockerfile` (uv base image → `uv sync`) is low-effort
  and lets cloners skip env setup — worthwhile as an optional convenience. **GPU** Docker
  (nvidia runtime + matching CUDA/cupy) is materially more complex — defer/optional. Note
  `pyelsepa` ships its own image (see backlog), so don't try to bundle it here.
- Acceptance (if pursued): `docker build` → container where `uv run pytest` passes (CPU).

## 4. Feature / patch backlog (Priority 2 — after refinement, on branches)

The PI publishes *finished* code, so these are **not** publication-blocking; keep them on
branches. Roughly ordered by value to the project's goal + publication polish:

1. **Physics-anchor comparison plots** (CLAUDE.md): make `checks/` Feranchuk/Zhai anchors
   emit figures vs the literature (esp. **Zhai Fig 1C**), ideally a minimal viz notebook.
   *High value for the publication's validation story.*
2. **Checkpointing GB-transfer patch**: multi-pickle per material or filtered remote pull
   (plot-side `select_results`/`sweep_values` already slice in-memory; on-disk size unsolved).
   *Usability.*
3. **pyelsepa/ELSEPA transport** (CLAUDE.md): evaluate replacing the hardcoded NIST Mott
   tables in `data/mott_transport_cross_sections/` with on-demand ELSEPA. User has a built
   docker image at `C:\dev\pyelsepa\` ([github.com/eScatter/pyelsepa](https://github.com/eScatter/pyelsepa)).
   NB: this is electron-scattering data — **separate** from the xraydb (photon) migration;
   xraydb cannot supply it. *Robustness / removes a hardcoded dataset.*
4. **Pint (or similar) units** (CLAUDE.md): evaluate Pint/natu/… for project-wide unit
   safety; implement the best fit if worthwhile. *Robustness/readability — scope carefully;
   a units refactor touches every amplitude site.*
5. **Multilayer slice 3** (crystalline-substrate coherent lines) + validation —
   `feature/multilayer-materials`.
6. Deferred (designed, lower priority): mosaicity exact-MC, detector solid-angle integral,
   grazing-incidence grating, interactive large-sweep slicing. See `docs/` design notes.

## 5. Suggested order

Phase A → B → C (publication-ready core), then D/E if a hosted site / container is wanted,
then dip into the backlog on branches. Gate every phase on the §0 safety net.
