# cxr-mc Claude notes

Keep this file short.

Read `docs/repo_map.md` before exploring source files.
`README.md`has the science-facing overview
`docs/`has the design notes
`docs/physics-validation-ledger.md` tracks which physics is verified; `docs/validation/README.md` is the method
`TODO.md` contains the task backlog

- `TODO.md` on `main` contains full triaged list of tasks + top-level summaries
- `TODO.md` on branches contains details scoped to their specific task; see `TODO.md` for conventions.

## Canonical commands

Run tests:

```bash
uv run pytest
```

Run a single test:

```bash
uv run pytest path/to/test.py
```

Lint:

```bash
uv run ruff check .
```

Format:

```bash
uv run ruff format .
```

Type check:

```bash
uv run pyright
```

Notebook cleanup:

```bash
uv run nbstripout
```

Claude should always prefer these commands.

## Working rules

- Prefer edits in `src/cxr_mc/` over notebook logic.
- Notebook changes should stay output-free on commit.
- Do not duplicate README or TODO content here.
- When asked to locate something, use `scripts/dev.py repo-map` first.
- Keep changes surgical and verify with the smallest useful command.
- New/edited physics needs a derivation docstring (source eq, assumptions, a limiting case) + a `Validation: <id>` marker + a row in the validation ledger. Verify physics with a fresh context, never the one that wrote it; only a human marks a claim `signed-off`.
