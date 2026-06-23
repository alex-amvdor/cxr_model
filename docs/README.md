# Design notes & decision records

In-depth notes on **deferred features** and **technology choices** for `cxr_model`.
These complement, not duplicate, the two top-level docs:

- [`../README.md`](../README.md) — user-facing overview, how to run, current physics, and
  **what remains to be validated**.
- [`../CLAUDE.md`](../CLAUDE.md) — contributor conventions, module responsibilities, the
  "adding a material" checklist, and the TODO / Done log.

This folder is for the longer "why / how / pros / cons / effort" write-ups that would
bloat those files.

| Note | Topic | Status |
|---|---|---|
| [crystal-mosaicity.md](crystal-mosaicity.md) | The analytic mosaic broadening (shipped) and the exact Monte-Carlo route (designed, not implemented) | analytic ✅ · MC ⏳ |
| [detector-solid-angle.md](detector-solid-angle.md) | Single-angle approximation (shipped) vs. a first-principles solid-angle integral (designed, not implemented) | ⏳ |
| [atomic-data-sources.md](atomic-data-sources.md) | Hard-coded Henke + Cromer-Mann data vs. an external library (xraydb / xraylib / …); evaluation + recommendation | evaluated, not adopted |

Legend: ✅ implemented · ⏳ designed, not implemented.
