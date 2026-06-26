# Units library evaluation (Pint / natu / Buckingham)

Should `cxr_mc` adopt a units library for project-wide dimensional safety
(TODO P3 #7)? **Recommendation: no project-wide refactor.** Keep the current
named-convention + `ħ=c=1`-at-boundaries approach; at most adopt a *lightweight,
boundary-only* units check for I/O config. The evidence below — measured with
`pint 0.25.3` on this repo's Python — drives that call.

Reproduce it with `uv run --with pint python dev/units_poc.py` (Pint is **not** a
project dependency; the PoC pulls it ephemerally).

---

## What the codebase does today

- Quantities are unit-named at every site (`HBARC_EV_ANG`, `E_grid_eV`,
  `thickness_ang`, `domega_sr`, `L_abs_ang`, …) — energies in eV, lengths in
  Ångström, angles in radians, cross sections in cm²/a0².
- The amplitude core works in **natural units `ħ=c=1`** and restores units only
  at the boundaries (`feranchuk_spence` / `montecarlo` docstrings say so
  explicitly). There, energy, momentum and inverse length are the *same*
  dimension.
- The spectrum hot path runs on the **GPU array module** (`xp` = CuPy) with fp32.

So units are already documented and converted deliberately; the open question is
whether a library would catch enough real bugs to justify the churn.

## Measured evidence (pint 0.25.3)

| probe | result | implication |
| --- | --- | --- |
| vectorized `q*2+1` on 2M elements | **~1.0×** vs bare numpy | array ops are fine — Pint delegates to numpy ufuncs |
| **scalar** `q*1.0000001 + 0.5 eV` | **~536×** (59 ns → 31 µs) | **dealbreaker for the scalar-complex amplitude sites** |
| complex magnitude (`abs((1+2j)·u)`) | works (`complex` magnitude) | not a blocker |
| `1 eV + 1/Å` | **`DimensionalityError`** | the `ħ=c=1` core *fights* Pint's dimensional model |
| churn: unit-bearing constant uses across `src/` | **~140 sites** (montecarlo 50, crystallography 27, detectors 44, …) | a project-wide wrap is large and risky |

Two of these are decisive:

1. **Scalar overhead ×536.** The Feranchuk–Spence amplitudes (`A_PXR`, `A_CBS`,
   the polarization sums) are scalar/tiny-array complex arithmetic evaluated per
   reflection per segment. Wrapping those in Pint quantities would dominate
   runtime. Pint is only "free" on big vectorized ops, which is *not* where this
   code spends its scalar amplitude math.
2. **`ħ=c=1` conflict.** In the core, `E = ħc·β·g/(1−β·n̂)` mixes eV and 1/Å on
   purpose. Pint rejects `eV + 1/Å`. Using Pint there means either defining a
   custom natural-unit registry (so the dimensions collapse) — which throws away
   the very checking you adopted Pint for — or converting in/out of natural units
   at every boundary, adding friction and more conversion code than it removes.

## GPU caveat

The hot path is CuPy. Pint wraps array-likes via the numpy protocols, and CuPy
implements them, but mixing Pint + CuPy is unproven here (host/device transfers
on `.to()`, fp32 magnitude handling, `asnumpy` round-trips). Even if it worked,
probe #2 already rules out the scalar amplitude sites. (Aside: `pyelsepa`, the
ELSEPA wrapper from P2 #6, *does* depend on Pint — confirming Pint is a fine
choice for a thin boundary wrapper, which is all it needs.)

## Options weighed

- **Pint** — mature, great for I/O boundaries, but ×536 scalar cost and the
  `ħ=c=1` conflict make it wrong for the core.
- **natu** — designed for natural units (would handle `ħ=c=1` gracefully), but
  niche and effectively unmaintained; adopting an abandoned dependency in a
  publication codebase is a liability.
- **Buckingham / `astropy.units`** — Buckingham is for dimensional-analysis
  reduction, not runtime quantity safety; astropy.units has the same scalar-
  overhead and natural-unit issues as Pint and pulls in a heavy dependency.

## Recommendation

1. **Do not** wrap the amplitude/transport core. Keep `ħ=c=1`-at-boundaries with
   the existing unit-named constants and conventions; that is where the speed and
   the natural-unit clarity live.
2. **Optional, low-risk:** a units check at the *config I/O boundary only* — beam
   energy, detector geometry, thicknesses entered by the user — where ops are few
   and not hot. A tiny hand-rolled checked-constructor (assert-on-construction)
   buys ~80% of the safety with none of the churn or dependency; Pint would also
   work there. `dev/units_poc.py` sketches both.
3. Revisit only if a units bug actually bites in practice — the current
   convention discipline has held through the publication-readiness epic.
