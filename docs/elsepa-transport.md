# ELSEPA electron-transport cross sections

How the Monte-Carlo electron transport gets its elastic-scattering angular
distribution, and an evaluation of replacing the hardcoded NIST dataset with
on-demand ELSEPA (TODO P2 #6).

---

## Current treatment (IMPLEMENTED)

`montecarlo` samples each elastic deflection from a screened-Rutherford angular
distribution whose screening parameter `α(E)` is **calibrated to reproduce the
NIST SRD-64 relativistic Mott transport cross section** σ_tr(E):

- `_load_mott_transport(element)` reads
  `src/cxr_mc/data/mott_transport_cross_sections/DisplayCalcTCSTableFor<El>.csv`
  (50 eV – 300 keV, 401 points, σ_tr in a0² units) → `(E_eV, σ_tr_cm²)`.
- `_mott_alpha_table` solves `<1−cosθ>(α) = σ_tr / σ_el` (σ_el from the Browning
  fit) for `α(E)`, so the cheap screened-Rutherford sampler matches the Mott
  transport moment.
- Elements **without** a NIST table (e.g. W) fall back to the analytic Joy
  screened-Rutherford `α` (`elastic_model='sr'`), warned once.

So the hardcoded CSVs are a **drop-in dataset**: anything that produces σ_tr(E)
in that file layout feeds the model unchanged. This is `elastic_model='mott'`;
`'sr'` ignores the tables entirely and already works.

---

## ELSEPA option (ADAPTER LANDED, gated; NOT yet run here)

[ELSEPA](https://github.com/eScatter/pyelsepa) (Salvat, Jablonski & Powell 2005)
is a relativistic Dirac partial-wave elastic code — the same family of physics
NIST SRD-64 itself is built on, but runnable on demand for **any** Z and energy
grid, including elements with no committed NIST table. `pyelsepa` wraps the
Fortran in a Docker container; `elscata(Settings(IZ=Z, EV=energies))['tcstable']`
returns the transport cross sections (column order: E, total elastic, **1st
transport**, 2nd transport).

### The adapter — `dev/elsepa_tables.py`

| function | Docker? | role |
| --- | --- | --- |
| `write_mott_transport_csv(path, Z, E, σ_cm²)` | no | write σ_tr in the exact NIST CSV layout (drop-in) |
| `read_mott_transport_csv(path)` | no | parse it back, byte-for-byte like the model loader |
| `compare_tables(E1,s1,E2,s2)` | no | max/median rel. diff of two tables (log-log interp) |
| `elsepa_transport_cross_section(Z, E)` | **yes** | run ELSEPA → `(E, σ_tr1_cm²)` |
| `regenerate(el, Z)` / `compare_to_nist(el, Z)` | **yes** | write a CSV / validate vs NIST |

The pure pieces are unit-tested (`tests/test_elsepa_tables.py`): a table written
by `write_mott_transport_csv` loads back through the real
`montecarlo._load_mott_transport`, and `read_mott_transport_csv` reproduces that
loader exactly on the committed carbon table — so the interop contract is proven
without Docker.

### Blocker / status

The actual run needs three things; the ELSEPA image now builds, and the
remaining gap is a working `pyelsepa` Python environment:

1. **`elsepa` Docker image — BUILT.** ✅ The paywalled `adus_v1_0.tar.gz` source is
   now in `C:/dev/pyelsepa/docker`. The upstream `Dockerfile` (`FROM debian:8`)
   fails — jessie's apt repos are EOL (404). A modernized build works and produces
   the same `/opt/elsepa/elscata` binary pyelsepa invokes:
   `docker build -f Dockerfile.modern -t elsepa C:/dev/pyelsepa/docker`
   (`Dockerfile.modern`: `debian:12` + `gfortran -O3 -std=legacy` for the F77).
2. **Docker** — present (29.x). ✅
3. **A `pyelsepa`-importable Python env — NOT yet here.** pyelsepa imports
   `cslib` (eScatter's "component library": `cslib.units/.settings/.predicates`
   + `DataFrame`). ⚠️ The PyPI package named `cslib` is an *unrelated* computer-
   vision library (pulls `skimage`/`visdom`); the one pyelsepa needs lives at
   `github.com/eScatter/cslib` and is not on PyPI. It must be installed into a
   **separate** venv — pyelsepa pins `numpy==1.13.0`/`pint==0.8.1`, which would
   wreck the cxr_mc scientific stack, so never `pip install pyelsepa` into the
   project venv. Install path-wise with `--no-deps` + modern `numpy/pint/docker`
   alongside the eScatter `cslib`.

So `elsepa_transport_cross_section` still raises a clear `RuntimeError` until that
env exists, and **nothing in the model depends on it at runtime** — the committed
NIST tables and the SR fallback are untouched. The validation run below (regenerate
C/Si, `--compare` vs NIST) is the only step left once `cslib` is in place.

### Caveats to verify on a real build

- **Column index.** `_extract_sigma_tr1_cm2` assumes tcstable column 2 is the 1st
  transport cross section (ELSEPA's documented order). Confirm against your build
  and pass `col=` if it differs.
- **Units.** pyelsepa carries Pint units via `cslib`; the adapter converts to
  cm² when present and otherwise assumes a0². (Note: **pyelsepa depends on Pint**
  — a concrete data point for the units evaluation, TODO P3 #7.)
- **Settings.** `MNUCL/MELEC/MEXCH/MCPOL/...` are taken from pyelsepa's example
  (free-atom Dirac–Fock, default exchange/polarization). Match these to whatever
  model NIST SRD-64 used before trusting absolute agreement.

### Validation gate before adoption

Regenerate C and Si and run `--compare`: ELSEPA σ_tr should agree with NIST to a
few percent across 50 eV – 300 keV. Only then regenerate the rest and/or add
tables for the SR-fallback elements (W).

---

## Recommendation

Keep the **NIST tables + SR fallback as the default** (no runtime Docker
dependency, already validated). Treat ELSEPA as an **optional offline
regenerator** for robustness and for **extending coverage to elements with no
NIST table** (today W silently falls back to analytic SR). The drop-in CSV format
means adoption is just dropping regenerated files into the data dir — no model
code changes. Worth doing once the `elsepa` image is buildable; not a blocker.
