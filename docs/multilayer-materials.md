# Multilayer film-on-substrate materials

Real lab samples are not free-standing single crystals: they are a thin vdW **film**
(MoSe₂ / MoS₂ / WS₂ / MoTe₂, tens of nm) grown or transferred onto a **substrate**
(amorphous SiO₂, crystalline Si, or sapphire Al₂O₃, hundreds of µm). To predict the
**measurable** line flux for the Timepix3 / Eagle XO setup — the project's active goal —
the model must represent that stack: each crystalline layer radiates its own lines, and
every photon is attenuated by the **whole stack** on its way to the detector.

Today the pipeline models exactly **one** single-crystal slab. This note specifies the
upgrade to an ordered stack of layers.

> **Implementation status.** **Slice 1 — cross-stack self-absorption (§1 below) — is
> implemented and on `main`** (opt-in: `substrate=None` is bit-for-bit the old single-material
> path, regression-anchored). **Slice 2** — multilayer electron transport (substrate
> backscatter + material-aware bremsstrahlung) — lives on branch
> `feature/multilayer-materials`. **Remaining:** slice 3 (coherent lines from a *crystalline*
> substrate) and quantitative validation against a measured film-on-substrate dataset.

---

## What a stack changes (three coupled effects)

1. **Cross-stack self-absorption** — *first-order, do this first.* A line born in the film
   travels out through the rest of the film **and the substrate** (or the substrate then
   the film, depending on exit face). For soft lines (≤4.5 keV) the substrate is optically
   thick, so its attenuation is often the dominant correction to the measured film flux.
2. **Per-layer radiation** — each **crystalline** layer is its own PXR/CBS source with its
   own structure factor, reflections, and B-factor; the spectra add **incoherently** (they
   are physically separate crystals). An **amorphous** layer (fused-silica SiO₂) radiates
   no coherent lines — it only absorbs and produces bremsstrahlung.
3. **Multilayer electron transport** — free path, stopping power, and the scattering element
   all change at a layer boundary. Matters when the film is not thin vs the electron range
   (or for the substrate's own emission); a refinement, not the first slice.

---

## What's already in place (the seams to generalize)

The codebase is closer than it looks — the single-material path is a clean special case:

| Concern | Today (single slab) | Code site |
|---|---|---|
| Sample def | `crystal` + `composition` + `hkl_list` + `B_ang2` + `thickness_ang` in the case dict | `sweep.crystal_params`, `sweep.build_cases` |
| Transport | one `composition`, slab `0≤z≤thickness`, faces only at `z=0`/`z=thickness` | `montecarlo.simulate_trajectories` (boundary truncation L467–480) |
| Line spectrum | one `crystal`/`hkl_list`/`B_ang2` | `montecarlo.mc_spectrum` |
| **Self-absorption** | **single** straight path to one face × **one** material's µ | `mc_spectrum` `T_abs` (L753–760) |
| Brem | one `composition`, same single-µ escape | `mc_brem_spectrum` |
| Geometry | whole slab shares one normal/tilt | `montecarlo.tilted_geometry` |
| Compound µ | `composition=[(el,n),…]` already supported | `_normalize_composition`, `_mu_total_inv_ang` |
| Crystalline Si | `silicon` is already a crystal | `data/crystal_structures.toml`, `crystal_params` |

Two things the recent **xraydb migration** already unblocked: substrate elements (O, Al for
SiO₂ / sapphire) need **no** hand-added atomic data — `henke_dispersion`/`load_henke` resolve
any element — and `composition`-based compound absorption already works for amorphous layers.

---

## Data model

A **Stack** is an ordered list of **Layers**, top (beam-entrance) first:

```python
# conceptual; the case dict carries this instead of scalar crystal/thickness/…
layers = [
    Layer(thickness_ang=500.,  crystal="mose2",  composition=[("Mo",n),("Se",n)],
          hkl_list=[...], B_ang2=0.6, beam_uvw=(0,0,2)),     # crystalline film
    Layer(thickness_ang=5e6,   crystal=None,     composition=[("Si",n),("O",2n)]), # amorphous SiO2 substrate
]
```

- `crystal=None` (or `hkl_list=[]`) ⇒ **amorphous**: absorbs + brems, no lines.
- A **single-layer** stack must reproduce today's result **bit-for-bit** (the regression
  anchor) — so the scalar `crystal`/`thickness_ang`/… path stays valid and is internally
  promoted to a one-layer stack.

**Flow through the pipeline** (each bullet is the generalization of an existing function):

- `sweep.crystal_params(material)` → also accept a **stack key** that returns
  `layers=[…]`. New `stack_params(name)` registry (e.g. `"mose2_on_si"`,
  `"mose2_on_sio2"`); single materials keep returning a one-layer stack.
- `sweep.build_cases` → put `layers` (a list of plain dicts) in the case instead of the
  scalar `crystal`/`composition`/`hkl_list`/`B_ang2`; `thickness_ang` becomes the **sum**
  (kept for labels/back-compat) with per-layer thicknesses in `layers`.
- `montecarlo._transport_case` → pass the layer stack to a stack-aware
  `simulate_trajectories` (see transport options below).
- `montecarlo._spectrum_case` → loop crystalline layers, accumulate `mc_spectrum` per
  layer (cross-stack `T_abs`), sum; `mc_brem_spectrum` summed per layer likewise.
- `results.store_result` / `plots` → label by stack name; the per-record metrics are
  unchanged (they consume `spec`/`brem`, which stay one array per case).
- **Checkpoints**: the case dict gains `layers`; old single-material checkpoints still load
  (absence of `layers` ⇒ one-layer stack). No format break.

---

## Geometry & conventions

Beam enters at `z=0` along `+z` (unchanged). Layer boundaries
`0 = z₀ < z₁ < … < z_N = Σ tᵢ`; layer *i* occupies `[z_{i-1}, z_i]`, layer 0 is the
entrance film, the substrate is the deepest. A point at depth `z` is in the layer whose
interval contains it. The **whole stack shares one normal and tilt** (`tilted_geometry`
is unchanged) — vdW films are conformal/parallel to the substrate.

---

## (1) Cross-stack self-absorption — the key generalization

Replace the single-material escape (`mc_spectrum` L753–760)

```python
L_esc = z_mid/(-n_hat[2])  if n_hat[2]<0  else (thickness - z_mid)/n_hat[2]
T_abs = exp(-L_esc * mu(E))
```

with a **piecewise optical depth** along the same straight ray `r(s)=r_mid + s·n̂`. The ray
runs in `z` from `z_mid` to the exit face (`z=0` if `n̂_z<0`, else `z=z_N`); within each
crossed layer *i* it travels `ℓ_i = Δz_i / |n̂_z|`, where `Δz_i` is the overlap of
`[z_mid → z_exit]` with `[z_{i-1}, z_i]`. Then

```
T_abs(E) = exp( − (1/|n̂_z|) · Σ_i  μ_i(E) · Δz_i )
```

`μ_i(E)` is `_mu_total_inv_ang(layer_i.composition, E)` — already vectorized over `E_res`.
`N≤~3` layers, so the per-segment cost is a tiny fixed loop over layers (cheap vs the sinc²
matmul). `N=1` collapses to today's formula exactly. **This single change** lets the film
spectrum see the substrate, and is the highest-value, lowest-risk slice.

---

## (2) Per-layer radiation

Wrap the per-reflection body of `mc_spectrum` in an outer loop over **crystalline** layers:

- Assign each segment to a layer by its midpoint depth `z_mid` (consistent with the existing
  midpoint approximation; segments are short vs layer thickness, so boundary-straddling is
  negligible — note it as a caveat).
- For layer *L*: take its segments, use **L's** `crystal`/`hkl_list`/`B_ang2`/`beam_uvw`,
  and apply the **cross-stack** `T_abs` (through all layers on the escape path, not just L).
- Sum the per-layer spectra incoherently (separate crystals ⇒ no cross-layer coherence).
- Amorphous layers are skipped for lines; their segments still feed brem and they still
  absorb. Brem (`mc_brem_spectrum`) is summed per layer with the same cross-stack `T_abs`.

---

## (3) Multilayer electron transport (phased)

`simulate_trajectories` currently uses one `composition` and truncates flights only at
`z=0`/`z=thickness`. Three escalating options:

- **(C) Single-material transport (first slice).** Transport in the **film** material
  throughout; only the *radiation/absorption* is stack-aware. Exact when the substrate
  barely perturbs the electrons in the film region, or when only the film's (substrate-
  attenuated) emission is wanted. Zero transport-code change.
- **(B) Thin-film approximation.** Switch the transport `composition` to the substrate's
  once an electron passes `z = t_film`. Good when the film ≪ electron range (the usual vdW
  case): the film is a thin entrance skin, the bulk of the cascade is in the substrate.
- **(A) Full per-layer transport.** At each step pick the layer from the electron's `z`,
  use that layer's rates/stopping/scattering element, and **truncate flights at internal
  boundaries** so material is constant within a flight (CASINO-style multilayer). Exact;
  the most code. Needed for thick films and for the substrate's own line emission to be
  quantitatively right.

**Recommendation:** ship (C), then (A). (B) is a stopgap only if a specific thick-film case
needs it before (A) lands.

---

## Registry / "adding a stack" checklist

1. `data/crystal_structures.toml` — add any crystalline substrate not present (Si exists;
   add sapphire `Al2O3` if its weak lines are wanted; fused-silica SiO₂ is amorphous → no
   entry, just a `composition`). **No atomic-data edits** — xraydb covers O/Al.
2. `src/sweep.py` — `stack_params(name)` returning `layers=[…]`; add to `MATERIAL_LABELS`.
3. `src/config.py` — a grid entry for the stack (thicknesses become per-layer).
4. `src/montecarlo.py` — the stack-aware `T_abs`, the per-layer radiation loop, and
   (phase A) the per-layer transport.
5. `src/results.py` / `src/plots.py` — labels only (metrics unchanged).
6. `CLAUDE.md` — document the stack key.

---

## Effort

- **(1) cross-stack absorption + (C) transport + one validated `mose2_on_si` case:**
  ≈ **2–3 engineer-days** (the `T_abs` path integral, the layer data model + back-compat
  promotion, regression that single-layer == today bit-for-bit).
- **(2) per-layer radiation (multi-crystalline / substrate emission):** ≈ **2–3 days**.
- **(3A) full per-layer transport:** ≈ **3–4 days** (internal-boundary truncation + the
  per-layer rate/stopping/scatter switch + a CASINO-style depth-dose cross-check).

Total ≈ **7–10 engineer-days** for the full feature; the **first slice is ~2–3** and
delivers the dominant substrate-attenuation physics on its own.

---

## Validation plan

- **Regression:** a one-layer stack reproduces the current `spec`/`brem` **bit-for-bit**
  (the hard anchor); `mosaic`/`tilt`/checkpoint paths unchanged.
- **Film-on-vacuum == film-only:** a substrate of zero thickness (or vacuum) must equal the
  present single-film result.
- **Cross-stack absorption vs analytic:** for a film line at energy E, the integrated flux
  ratio (with vs without substrate) must equal `exp(−μ_sub(E)·t_sub/|n̂_z|)` for a thin
  film (all emission at ~one depth) — a closed-form check of the path integral.
- **Depth-dose (phase A):** the per-layer energy deposition vs depth matches a CASINO /
  Kanaya-Okayama range estimate across the film/substrate interface.
- **Geometry sign:** negative tilt (entrance face toward detector) still gives the high-flux
  branch; the substrate attenuation must *increase* as the exit path through it lengthens.

---

## The 1T′-MoTe₂ block (scope note)

The CLAUDE.md TODO's "blocked on a reliable 1T′-MoTe₂ CIF" is **partial**: it only affects
the **1T′-MoTe₂** layer's exact coordinates. The general machinery and every **2H** film
(MoSe₂ / MoS₂ / WS₂, and 2H-MoTe₂ which is already in `crystal_structures.toml`) on a
Si / SiO₂ / sapphire substrate need no MoTe₂ CIF, so the feature is fully actionable now;
1T′-MoTe₂ slots in as one more layer definition once the CIF is in hand.
