# TODO Sync

Reconcile `TODO.md` across `main` and the feature/patch branches to the repo's
TODO-formatting convention (main backlog item PA — "TODO formatting convention").

The convention:

- `main` carries the full backlog, but **one-line summaries only** — a terse
  description + a pointer to the item's branch (`→ feature/…`) or design doc.
- Each `feature/…` / `patch/…` branch's `TODO.md` carries **only that branch's
  task**, with the full detail. The full backlog does NOT live on a branch.

Steps:

1. Start on `main` with a clean tree (`git status`). Read `main:TODO.md` — it is
   the authoritative backlog.
2. Drop finished items: cross-check each against `git log` / merged branches. An
   item whose work is merged to `main` is done — remove it, don't summarize it.
3. Slim every surviving `main` item to one line: summary + `→ branch` or
   `Design: docs/…` pointer. Renumber cleanly. Do not delete the convention item.
4. For each active `feature/…` / `patch/…` branch, rewrite its `TODO.md` to hold
   only its own task. Pull the freshest status text from whichever side
   (branch or main) is more current; keep the richer one.
5. Fix stale references while rewriting (e.g. renamed paths, `cxr_model → cxr_mc`).
6. Commit **doc-only** on each branch (one commit per branch). Do **NOT** push and
   do **NOT** edit anything but `TODO.md` — leave pushing to the user. Return to
   `main` and report each branch's commit + ahead-of-origin count.

Switching branches needs a clean tree; commit each branch before moving on.
