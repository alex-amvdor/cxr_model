# Design notes & decision records

In-depth "why / how / trade-offs" write-ups that would bloat the top-level docs.
They complement:

- [`../README.md`](../README.md) — user-facing overview, physics, install, validation.
- [`../CLAUDE.md`](../CLAUDE.md) — contributor conventions, the module map, and the
  "adding a material" checklist.
- [`../TODO.md`](../TODO.md) — the feature / patch backlog.

| Note | Topic | Status |
|---|---|---|
| [running-on-a-cluster.md](running-on-a-cluster.md) | Headless `cxr scan` under SLURM (`sbatch` + job-array templates) | guide |
| [crystal-mosaicity.md](crystal-mosaicity.md) | Analytic mosaic line-broadening (shipped) vs. the exact Monte-Carlo route | analytic ✅ · MC ⏳ |
| [detector-solid-angle.md](detector-solid-angle.md) | Single-direction approximation (shipped) vs. a first-principles solid-angle integral | ⏳ |
| [multilayer-materials.md](multilayer-materials.md) | Film-on-substrate stacks: cross-stack self-absorption, per-layer radiation, multilayer transport | slice 1 ✅ (main) · 2 ✅ (branch) · 3 ⏳ |
| [atomic-data-sources.md](atomic-data-sources.md) | Hard-coded Henke/Cromer–Mann tables vs. an external library (xraydb) | adopted ✅ |

Legend: ✅ implemented · ⏳ designed, not implemented.
