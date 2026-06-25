---
name: notebook-workflow
description: Edit, validate, and sanitize the repository notebooks.
---

# Use this skill when

Use this skill when changing `scan.ipynb`, `analysis.ipynb`, or any notebook under `checks/`.

# Rules

- Do not put reusable physics logic in a notebook cell if it belongs in `src/cxr_mc/`.
- Keep committed notebooks output-stripped.
- Prefer `scripts/dev.py nbqa` before editing notebook code, and `scripts/dev.py nbstrip` after.
- If a notebook starts growing a real workflow, move the reusable parts into source code and leave the notebook as a thin driver.

# Canonical notebook commands

- `uv run python scripts/dev.py nbqa`
- `uv run python scripts/dev.py nbstrip`
- `uv run python scripts/dev.py verify`
