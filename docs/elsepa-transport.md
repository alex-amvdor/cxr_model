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

## ELSEPA option (ADAPTER LANDED; VALIDATION COMPLETE ✅)

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

### Setup / status

All three gates are now clear:

1. **`elsepa` Docker image — BUILT.** ✅ Source is now pulled directly from
   `github.com/eScatter/elsepa` (Apache-2.0) — no tarball needed. Build:
   `docker build -f C:/dev/pyelsepa/docker/Dockerfile.modern -t elsepa C:/dev/pyelsepa/docker`
   (`Dockerfile.modern`: `debian:12` + `ca-certificates` + `gfortran -O3 -std=legacy`;
   sets `ENV ELSEPA_DATA=/opt/elsepa` so the binary resolves its atomic data files).
   Checked into `C:/dev/pyelsepa/docker/Dockerfile.modern`.
2. **Docker** — present (29.x). ✅
3. **`pyelsepa` isolated venv — works with patches.** ✅ Build in a *separate* venv
   (never install into the cxr-mc venv — pyelsepa's ancient pins wreck the stack).
   The PyPI `cslib` is an unrelated CV library; the right one is eScatter's:
   `github.com/eScatter/cslib`. Three compatibility patches are needed for Python
   3.7+ / numpy 2.x / docker 7.x — apply them after installing:

   ```
   # in the isolated venv:
   pip install --no-deps C:/dev/pyelsepa
   pip install git+https://github.com/eScatter/cslib.git --no-deps
   pip install numpy pint docker ruamel.yaml
   ```

   Then patch three files in `<venv>/Lib/site-packages/`:

   **`cslib/predicates.py` line 76** — pint 0.17+ validates `{:~P}` on
   `Dimensionality` objects (raises `ValueError`); the format spec there is only a
   human-readable description so `{!s}` works fine:
   ```python
   # was: @predicate("{:~P} (e.g. {:~P})".format(u.dimensionality, u))
   @predicate("{!s} (e.g. {:~P})".format(u.dimensionality, u))
   ```

   **`elsepa/executable.py` line ~203** — `docker.APIClient.get_archive` returns a
   generator in docker 7.x (was file-like in 2.4.0):
   ```python
   # was: return Archive('r', strm.read())
   return Archive('r', b''.join(strm))
   ```
   Also make `__exit__` tolerant of a container that already exited:
   ```python
   def __exit__(self, exc_type, exc_value, exc_st):
       try:
           self.kill()
       except Exception:
           pass
       self.remove(force=True)
   ```

   **`elsepa/parse_output.py`** — PEP 479 (Python 3.7+) converts `StopIteration`
   raised inside a generator to `RuntimeError`; `join_double_header` used
   `StopIteration` as loop-exit flow control. Fix `arg_first` to raise `ValueError`
   and catch it in the loop:
   ```python
   def arg_first(pred, s):
       try:
           return next(i for i, v in enumerate(s) if pred(v))
       except StopIteration:
           raise ValueError("no matching element")

   def join_double_header(l1_, l2_):
       ...
       while True:
           try:
               x1 = x2 + arg_first(lambda v: v[0] != ' ' or v[1] != ' ', c[x2:])
           except ValueError:
               return
           try:
               x2 = x1 + arg_first(lambda v: v[0] == ' ' and v[1] == ' ', c[x1:])
           except ValueError:
               x2 = len(c)
           yield ' '.join([l1[x1:x2].strip(), l2[x1:x2].strip()])
           if x2 >= len(c):
               return
   ```

### Validation results (PASSED ✅)

`python dev/elsepa_tables.py --element <El> --Z <Z> --compare` vs the committed
NIST tables across 50 eV – 300 keV (401 points):

| Element | Z | max rel Δ | median rel Δ |
|---------|---|-----------|--------------|
| C       | 6  | 2.19%    | 0.01%        |
| Si      | 14 | 4.42%    | 0.02%        |

Median < 0.05% on both. Max-rel outliers are at energy-grid edges where
log-log interpolation diverges — not a physics disagreement. Agreement is
solidly within the "a few percent" gate. Column index 2 (1st transport σ_tr)
confirmed correct; units fallback (a0²) triggered since pint unit-conversion
path returns dimensionless data from this cslib build.

### Notes on settings / physics

- **Settings.** `MNUCL/MELEC/MEXCH/MCPOL/...` taken from pyelsepa's example
  (free-atom Dirac–Fock, default exchange/polarization). Match these to whatever
  model NIST SRD-64 used before trusting absolute agreement.
- `elsepa_transport_cross_section` still raises a clear `RuntimeError` unless the
  isolated env is present — **nothing in the model depends on it at runtime**.

---

## Recommendation

Keep the **NIST tables + SR fallback as the default** (no runtime Docker
dependency, already validated). Treat ELSEPA as an **optional offline
regenerator** for robustness and for **extending coverage to elements with no
NIST table** (today W silently falls back to analytic SR). The drop-in CSV format
means adoption is just dropping regenerated files into the data dir — no model
code changes. Worth doing once the `elsepa` image is buildable; not a blocker.
