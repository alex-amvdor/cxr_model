# Physics validation system

How we get from *"the spectra match the literature"* to *"every load-bearing equation has been personally certified"* before publication. The driving rationale, the full phased plan, and the task backlog live in the author's notes; this file is the in-repo spec so any contributor (human or agent) can operate the system.

## Why output-matching isn't enough

Agreement with Zhai/Feranchuk is necessary but not sufficient — it can hide **compensating errors** (two mistakes that cancel in the one quantity plotted), **narrow-regime agreement** (right at the one point checked, wrong elsewhere in the sweep), and **tuned coincidences**. Publication trust needs **provenance** (every formula traces to a derivation), **independent verification** (re-derived by someone other than the implementer), and **coverage** (across the parameter space, not one point).

## The pieces

- **The ledger** — [`docs/physics-validation-ledger.md`](../physics-validation-ledger.md) is the single source of truth: one row per atomic physics claim, keyed by a stable `id`, anchored on `file::symbol`. The unit of trust is the **equation, not the module**.
- **In-code back-reference** — every annotated physics function carries a one-line `Validation: <id>` marker in its docstring, tying code↔ledger both ways. A physics `def` with no marker is an unledgered claim — find them with:
  ```bash
  # physics symbols missing a Validation: back-reference
  grep -L "Validation:" src/cxr_mc/{montecarlo,crystallography,atomic_form_factors,eaglexo_response,timepix_response}.py
  ```
- **Re-derivation write-ups** — `docs/validation/<id>.md` holds each independent derivation, its diff against the implementation, and the adjudication. This is the audit trail and the seed of the paper's validation appendix.
- **Anchors** — regression tests (mostly under `checks/`) that pin a claim to a reference value with a tolerance.

## Status lifecycle

```
unverified → filtered → rederived → anchored → signed-off
                  ↓          ↓          ↓
                       discrepancy  (any failed check — tracked loudly)
```

| status | meaning |
|--------|---------|
| `unverified` | ledgered, nothing checked yet |
| `filtered` | units + limiting cases + sign/convention checks pass |
| `rederived` | an independent fresh-context derivation matches the implementation |
| `anchored` | a regression test pins it to a reference value, green in CI |
| `signed-off` | **a human** read the source and the diff and certified it — the only state that gates publication |
| `discrepancy` | a check failed; under investigation |

## Workflow

**Cheap filters first** (before any expensive re-derivation): dimensional consistency, limiting cases (η→0, t→∞, non-relativistic, single-segment→closed-form), sign/symmetry/convention. Survivors advance; failures go straight to `discrepancy`.

**Adversarial re-derivation** (the core of independent verification):
1. Pick an `unverified`/`filtered` id.
2. A **fresh context — ideally a different model — that has NOT seen the implementation** gets only `{the cited source, what the function should compute, its signature}` and writes the independent expression to `docs/validation/<id>.md`.
3. Diff the independent expression against the code (symbolic/dimensional; numeric where possible).
4. The author adjudicates → `signed-off` or `discrepancy`.

**Pin it or lose it:** every `signed-off`/`anchored` claim has a regression test so a future edit can't silently break certified physics.

## Rules for contributors (human or agent)

- Every physics function carries a derivation docstring: **source (paper + eq #), assumptions, ≥1 limiting case, and a `Validation: <id>` marker.** No "trust me" formulas.
- New physics lands **with** a ledger row + a limiting-case test, or it doesn't land.
- Verification is done by a **different context/model** than the one that wrote the code.
- Only a **human** moves a claim to `signed-off`.

## Design note

The ledger is plain markdown (not an in-code decorator DSL + generator) on purpose: a physicist edits a table, not a parser; it renders on GitHub; it's git-diffable; and an agent can update it and grep for gaps with no tooling. If the ledger and code ever drift in practice, add a CI check that cross-references `Validation:` markers against ledger `id`s — but not before drift is actually observed.
