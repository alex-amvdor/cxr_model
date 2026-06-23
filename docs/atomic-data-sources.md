# Atomic data: hard-coded tables vs. an external library

Evaluation of whether to replace the project's hand-maintained atomic scattering data with
an external library, and which one. **Conclusion: worthwhile only if more elements/
materials will be added; if adopted, `xraydb` is the best fit. Not adopted yet** — the
prototype and numeric diff live in `../checks/`.

---

## What is hard-coded today

`src/atomic_form_factors.py` + `data/` provide, behind one clean seam:

| Datum | Source | Used for |
|---|---|---|
| `Z_TABLE` | hand-typed | structure factor, CBS `U_g` |
| `CROMER_MANN` f0(g) | International Tables Vol C, Table 6.1.1.4 | angular form factor → `S(g)`, `chi_g` |
| Henke f1/f2 CSVs (`data/atomic_scattering_factors/*.csv`) | CXRO / Henke, 10 eV–30 keV | dispersion f′,f″ and absorption (μ from f2) |
| `_EDGE_PRONE` | hand-flagged | force the complex Henke f near edges |

The public API — `cromer_mann_f0`, `henke_dispersion`, `atomic_form_factor`, `Z_TABLE`,
`load_henke` — is the **single choke point**; everything downstream (`structure_factor`,
`chi_g`, `U_g`, `absorption_length_ang`) goes through it. That makes a backend swap a
"replace the body, keep the signature" job — see `../checks/atomic_form_factors_xraydb.py`.

**Why consider a library at all:** the "adding an element" chore in
[../CLAUDE.md](../CLAUDE.md) touches ~8 non-colocated registries, ~4 of them atomic data
(`Z_TABLE`, `CROMER_MANN`, the CXRO `.nff` download, `_EDGE_PRONE`). A library makes new
elements nearly free and gives edges/lines/μ for free. **The catch:** every library uses
different source tables, so numbers shift and the Feranchuk/Zhai validation anchors must be
re-run.

---

## Candidates

Install + API verified live under `uv` on this Python 3.14.2 box (June 2026).

| Library | f0(q) | f1/f2 | Install (uv / Py 3.14) | License | Verdict |
|---|---|---|---|---|---|
| **xraydb** 4.5.8 | Waasmaier–Kirfel | **Chantler** (FFAST) | ✅ pure-Python, `uv add xraydb` (verified) | MIT (code) / CC0 (data) | **Best fit** |
| **xraylib** 4.2.1 | Waasmaier/Hubbell (`FF_Rayl`) | tabulated (`Fi`/`Fii`) | ✅ binary wheels for 3.13/3.14 now on PyPI (verified) | BSD | Most authoritative; compiled C/SWIG (heavier) |
| **periodictable** | Cromer–Mann | **Henke/CXRO** (same as us) | ✅ pure-Python | BSD-like | Henke-preserving hybrid option |
| **scikit-beam** 0.0.27 | (wraps xraylib) | (wraps xraylib) | ❌ build fails on 3.14; huge stack (silx/pyfai/h5py) | BSD | Wrong tool here |
| **XATOM** (CFEL, Son/Santra) | — | — | ❌ not freely pip-installable | academic, request | Wrong category |

Notes:
- **xraydb** gotchas: `f1_chantler` returns f′ **directly** (the anomalous correction, not
  Henke-style Z+f′); `f0(el, q)` takes `q = g/4π = sinθ/λ`.
- **xraylib** gotcha: `Fii` (f″) is returned **negative** by its sign convention.
- **scikit-beam**'s atomic layer (`XrayLibWrap`) is literally an xraylib wrapper, and its
  own X-ray DB repo is **archived**, redirecting to xraypy/XrayDB (= xraydb). Its useful
  parts (XPCS correlation, powder integration) are for analysing *experimental* scattering
  data — not relevant to a from-scratch PXR simulation.
- **XATOM** is an X-ray-induced **atomic ionization-dynamics** toolkit (feeds XMDYN), not a
  crystallographic form-factor database; it does not expose f0/f1/f2 lookups.

---

## How far would the numbers move?

`../checks/atomic_db_diff.py` (run `uv run --with xraydb python checks/atomic_db_diff.py`)
compares the project (Henke + Cromer–Mann) against xraydb (Chantler + Waasmaier) over the
grids the pipeline uses:

- **f0(g): ≤ 0.25 % max, < 0.1 % mean** across all 13 structure-factor elements →
  Cromer–Mann and Waasmaier–Kirfel are interchangeable. **Not a concern.**
- **Dispersion (Henke → Chantler):** the χ_g-relevant real amplitude `f0 + f′` shifts
  **~2–3 %** for the resonant heavies (Se 2.7 %, Mo 2.1 %, Te 3.4 %, W 2.9 %) away from
  edges, with **large localized disagreement right at absorption edges** (where the
  pipeline samples densely and where Henke vs Chantler place edges slightly differently).

So the **table source for f1/f2 — not the f0 fit — is what moves the physics**, and a swap
would perturb the line intensities and the validation anchors at the few-percent level
(more at edges).

---

## Recommendation & effort

- **Best fit: `xraydb`.** Pure-Python, MIT/CC0, installs cleanly on the uv/3.14 stack,
  full coverage (f0, f1/f2, Z, edges, μ), most ergonomic. Accept that it is Chantler, not
  Henke, and re-validate.
- **Most authoritative: `xraylib`** — now also cleanly installable (3.14 wheels, BSD); pick
  it if you want the canonical XRF-community tables and accept a compiled dependency.
- **Minimal-drift hybrid: `periodictable`** for f1/f2 (it bundles the *same* CXRO Henke
  tables, so absorption/dispersion numbers barely change) + keep Cromer–Mann for f0.
- **Avoid** scikit-beam and XATOM for this purpose.

**Effort ≈ 3–5 engineer-days, dominated by re-validation, not plumbing:** reimplement the
four `atomic_form_factors.py` functions against the library (~1–2 days), confirm edge
sampling is preserved, then re-run the `../checks/` anchors and judge the shifts (~1–2
days). Gate the new backend behind the existing seam so you can A/B it against the
hard-coded path during validation.
