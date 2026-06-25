# Atomic data: hard-coded tables vs. an external library

Evaluation of whether to replace the project's hand-maintained atomic scattering data with
an external library, and which one. **Conclusion: `xraydb` is the best fit, and it was
ADOPTED on 2026-06-23** — `src/atomic_form_factors.py` now sources f0/f′/f″/Z from xraydb
(no hard-coded tables). The migration was re-validated against the Feranchuk and Zhai
anchors (results below). xraylib was explicitly considered and rejected (see below).

---

## What was hard-coded before (now sourced from xraydb)

`src/atomic_form_factors.py` provides, behind one clean seam:

| Datum | Was | Now |
|---|---|---|
| `Z_TABLE` | hand-typed dict | `xraydb.atomic_number` (any symbol, lazy) |
| `cromer_mann_f0` f0(g) | Cromer–Mann (ITC Vol C 6.1.1.4) | Waasmaier–Kirfel (`xraydb.f0`) |
| `henke_dispersion` f′,f″ | Henke/CXRO `.nff` CSVs, 10 eV–30 keV | Chantler/FFAST (`xraydb.f1_chantler`/`f2_chantler`) |
| `load_henke` (E,f1,f2) | CSV parse | Chantler on `xraydb.chantler_energies` (edge-dense) |
| `_EDGE_PRONE` (in crystallography) | hand-flagged | unchanged (policy, not data) |

The public API — `cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`, `Z_TABLE`,
`load_henke` — is the **single choke point** and was kept byte-for-byte compatible
(signatures, shapes, NaN-out-of-range contract, the names `cromer_mann_f0`/`henke_dispersion`
even though the data is now Waasmaier/Chantler), so everything downstream
(`structure_factor`, `chi_g`, `U_g`, `absorption_length_ang`, the `checks/`) is untouched.

The CXRO `.nff` CSVs in `data/atomic_scattering_factors/` are now **legacy / unused by the
code** (kept for provenance and any future A/B against Henke).

**Why a library:** the "adding an element" chore in [CLAUDE.md](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc/blob/main/CLAUDE.md) touched ~8
non-colocated registries, ~4 of them atomic data. The swap eliminates the two worst
(typing Cromer–Mann coefficients, downloading a CXRO `.nff` per element) — a new element is
now free for f0/f′/f″/Z. **The cost:** xraydb is Chantler, not Henke, so numbers shift a
few percent and the validation anchors had to be re-run (they held — see below).

---

## Candidates (June 2026, uv / Python 3.14)

| Library | f0(q) | f1/f2 | Install | License | Verdict |
|---|---|---|---|---|---|
| **xraydb** 4.5.8 | Waasmaier–Kirfel | **Chantler** (FFAST) | ✅ pure-Python, `uv add xraydb` | MIT / CC0 | **ADOPTED** |
| **xraylib** 4.2.1 | `FF_Rayl` (Waasmaier) | `Fi`/`Fii` (**Cromer–Liberman**) | ✅ wheels for 3.13/3.14 | BSD | Rejected (see below) |
| **periodictable** | Cromer–Mann | **Henke/CXRO** (same as old) | ✅ pure-Python | BSD-like | Henke-preserving hybrid; not chosen |
| **scikit-beam** 0.0.27 | (wraps xraylib) | (wraps xraylib) | ❌ build fails on 3.14 | BSD | Wrong tool |
| **XATOM** (CFEL) | — | — | ❌ not pip-installable | academic | Wrong category |

### Why xraydb over xraylib (the 2026-06-23 head-to-head)

Both install on 3.14 and cover every element. The decision:

- **Accuracy is a wash.** On the χ_g-relevant real amplitude `f0+f′` in the 1–4.5 keV line
  band, **both** libraries deviate from the (validated) Henke baseline by comparable amounts
  (~0.3–3 %, worst near edges); neither is consistently closer, and xraylib is even
  marginally closer at a few points (Mo @1 keV, Te @4.5 keV). So "closeness to Henke" does
  not break the tie — the re-validation cost is the same either way.
- **Engineering fit decides, and xraydb wins:** pure-Python (lighter remote-box deploy vs
  xraylib's compiled C+SWIG), **vectorized** (matches the array-based `henke_dispersion`
  API; xraylib `Fi`/`Fii` are scalar C calls that must be Python-looped over the line/brem
  grids), and **eV + symbol** (matches existing convention; xraylib is keV + Z with an
  `f″ = −Fii` sign flip). xraylib's anomalous terms are Cromer–Liberman (older, weaker in
  the soft band) vs xraydb's Chantler. xraylib's real advantages — completeness, the XRF
  community standard — are for capabilities this project never uses.

xraydb gotchas baked into the adapter: `f1_chantler` returns f′ **directly** (not Henke-style
Z+f′); `f0(el, q)` takes `q = g/4π`; `f1_chantler` can raise at the exact table endpoint and
the brem grid passes E=0, so `henke_dispersion` masks to the strict interior → NaN.

---

## Re-validation results (2026-06-23, the actual swap)

Before = Henke/Cromer–Mann; after = xraydb/Chantler. CPU run (laptop has no CUDA toolkit;
the MC checks were forced onto numpy by masking `cupy`).

| Anchor | Before | After | Shift |
|---|---|---|---|
| `tests/` (form-factor unit tests) | pass | pass (+ new range/unknown tests) | — |
| Feranchuk LiF (200) \|χ_g\| | 4.311e-05 | 4.306e-05 | −0.1 % |
| Feranchuk LiF model flux | 2.812e+04 ph/s | 2.817e+04 ph/s | +0.2 % |
| Feranchuk LiF \|A_PXR/A_CBS\| | 0.6088 | 0.6068 | −0.3 % |
| **Zhai graphite 29 nm A/B line ratio** | **1.00** | **1.00** | **unchanged** |
| Zhai graphite 1 mm A/B line ratio | 2.54 | 2.57 | +1.2 % |
| graphite L_abs @ 973 eV | 1.84 µm | 1.97 µm | +7 % |

**Verdict: holds.** f0 is interchangeable (≤0.25 %). The thin-film regime — where the
idealized model is valid and the paper's TEM data live — is unchanged to <1 %. The bulk
numbers move ~5–7 %, entirely through the absorption length (Chantler f″ < Henke at these
soft energies → longer L_abs → less self-absorption), which is the physically-correct
consequence of the data-source change and is small vs the model's existing ~×2
normalization uncertainties (e.g. the MoSe₂ structure factor, detector geometry). A second,
benign improvement: Chantler covers 1 eV–966 keV vs Henke's 10 eV–30 keV, so brem
self-absorption above 30 keV is now physical instead of dropped (NaN); it does not touch the
≤4.5 keV line anchors.

The throwaway evaluation scripts (`checks/atomic_form_factors_xraydb.py`,
`checks/atomic_db_diff.py`, `checks/atomic_lib_compare.py`) were removed after the swap — the
"project vs library" comparison they encoded is now moot (the project *is* the library), and
their findings are captured above.
