# eScatter cstool / Nebula evaluation (Penn dielectric + secondary electrons)

Should `cxr_mc` move its electron transport onto the eScatter stack — **Nebula**
(GPU electron–matter Monte Carlo) fed by **cstool** (cross-section compiler) — to
gain a full inelastic model (optical-data + **full Penn** dielectric function),
**secondary-electron** generation, and acoustic-phonon scattering (TODO P2 #2)?

**Recommendation: no.** Keep the in-house segment transport. The physics Nebula
adds over what we have lives almost entirely *below* the 5 keV cutoff this model
deliberately discards, the one piece that does touch the primary beam
(energy-loss straggling) is quantitatively negligible for our line widths, and
the elastic-accuracy win that *is* worth having is already in hand via the ELSEPA
adapter (P2 #1) — to which cstool is largely redundant. Adopting Nebula would
*increase* dependencies and compute, not simplify the repo, and its outputs
(SE/BSE images, SE yields) are not the per-segment radiating trajectories the
PXR+CBS spectrum layer consumes. The order-of-magnitude evidence below drives the
call; reproduce the straggling bound with the snippet in *Energy-loss straggling*.

---

## What the transport does today, and why

`montecarlo/transport.py` is a CASINO-style **single-scattering** MC whose *only*
job is to emit the straight radiating segments `(r_mid, v_hat, L, E, t)` that the
PXR+CBS amplitude code integrates over. Three design choices matter here:

- **Elastic:** Browning free paths + screened-Rutherford angles calibrated
  per-element to the **NIST SRD-64 relativistic Mott transport** cross sections
  (`elastic_model="mott"`), analytic SR fallback otherwise. This is the dominant
  trajectory-shaping physics and it is already Mott-grade.
- **Stopping:** Joy–Luo modified-Bethe **CSDA** — energy drains deterministically
  along each flight, *no straggling* (transport.py:384). It captures the mean
  energy-vs-depth, which is what sets the depth-dependent line energy.
- **Primaries only, `E_cut_keV = 5.0`:** an electron is dropped below 5 keV
  because "segments below the cutoff don't radiate in the spectral window of
  interest anyway." There is no inelastic event list and **no secondary-electron
  cascade** — by construction.

The radiated line is built from the *primary* relativistic electron's path; PXR/CBS
amplitude scales steeply with the emitter's energy/velocity (the virtual-photon
field), so the keV–tens-of-keV primary is the entire source. (README "How the
pipeline computes it".)

## What cstool / Nebula are, and what they add

Nebula (van Kessel & Hagen, *SoftwareX* **12**, 100605, 2020;
[nebula-simulator.github.io](https://nebula-simulator.github.io/)) is an
open-source **GPU** simulator of electron–matter interaction built for
**SE/BSE SEM image simulation, secondary-electron yields, and e-beam
lithography**. cstool compiles its per-material cross-section files. Its physics
models, relative to ours:

| Channel | Nebula / cstool | cxr_mc today | Net new physics for *us* |
| --- | --- | --- | --- |
| Elastic | Mott (ELSEPA) + acoustic-phonon | NIST-Mott-calibrated SR | ~none (we are already Mott; phonon matters at <100 eV) |
| Inelastic | optical-data + **full Penn** dielectric ⇒ IMFP + energy-loss + **straggling** | CSDA mean loss, no straggling | straggling on the primary (bounded below) |
| Secondaries | full **SE cascade** (direct excitation, plasmon decay, inner-shell ionization) | none; `E_cut = 5 keV` | irrelevant to keV-line emission (see below) |
| Output | SE/BSE images, SE yields, reflected-e spectra | per-segment radiating trajectories | wrong observable — no X-ray emission, no segment list |

The added physics is real and well-validated **for what Nebula targets** — the
low-energy cascade that governs SE yield and SEM contrast.

## Evaluation against the four axes named in the backlog item

**1. Accuracy improvement — small and mostly out-of-band.**
- *Elastic:* no gain. We already calibrate to relativistic Mott; cstool's elastic
  is the same ELSEPA partial-wave source the P2 #1 adapter pulls directly.
- *Secondary electrons:* SEs are born at ≲50 eV (the SE peak is ~2–5 eV; even fast
  δ-rays are sub-keV here). PXR/CBS flux from a sub-keV electron at our 2–8 keV
  lines is negligible, and every such electron is below `E_cut = 5 keV` anyway.
  Tracking the cascade adds particles that *cannot* contribute to the observable.
- *Inelastic straggling on the primary:* the one in-band effect. Bounded next
  section → **≲10 eV** of line broadening at the shallow, high-flux depths, versus
  ~130 eV detector resolution. Sub-dominant; CSDA's mean-loss treatment is enough.

**2. Robustness — high coupling cost, wrong observable.** Nebula is mature and
GPU-fast, but it emits SE/BSE images and yields, *not* the `(position, direction,
energy, length)` radiating-segment list the PXR+CBS layer needs. We would have to
fork/instrument its kernels to log primary segments, then still keep our entire
amplitude/spectrum/detector stack on top. That is a larger, more fragile surface
than the ~250-line NumPy transport we maintain now.

**3. Repo simplification — it is the opposite.** Adopting Nebula adds a GPU C++/
CUDA engine, the cstool cross-section toolchain, and a material-data pipeline as
dependencies, on top of (not in place of) the spectrum code. cstool also overlaps
P2 #1: both wrap ELSEPA for elastic data, so we would be carrying two routes to
the same cross sections. Net dependency and build complexity goes **up**.

**4. Compute — pays for particles we throw away.** SE cascades multiply tracked
particles by 1–3 orders of magnitude per primary; Nebula is GPU-built precisely
because that is expensive. For us every cascade particle is below the radiating
window — compute spent to produce nothing in the spectrum.

## Energy-loss straggling — the one in-band gap, bounded

CSDA omits the *fluctuation* in energy loss. In the thin-absorber (Landau/Vavilov)
regime that applies to the shallow radiating depths, the loss-fluctuation scale is
the Landau width `ξ = (K/2)(Z/A) ρ x / β²` (K = 0.307 keV·cm²/mol), with
FWHM ≈ 4ξ. Mapping an energy spread to a PXR line shift via `ω ∝ β` (so
`dω/ω ≈ (1/βγ³)(δE/m_ec²)/β`), for Si at 40 keV:

| radiating depth x | mean CSDA loss | straggle FWHM ≈ 4ξ | line contribution @ 4 keV |
| --- | --- | --- | --- |
| 0.1 µm | 0.13 keV | ~51 eV | ~2 eV |
| 0.3 µm | 0.38 keV | ~154 eV | ~7 eV |
| 1.0 µm | 1.28 keV | ~512 eV | ~23 eV |

The high-flux radiation is born shallow (negative tilt, entrance face toward the
detector — README "Tilt sign"), so the relevant rows are the top two:
**straggling adds ≲10 eV to the line, well under the ~130 eV detector resolution
and the mosaic broadening.** This is an order-of-magnitude bound (single element,
Landau approximation, `ω ∝ β`), **not** a validated number — but it is ~20×
below the resolution floor, so the conclusion is robust to its own factor-of-few
uncertainty. If a future *measured* line width ever demands it, a cheap per-step
Gaussian straggling term (Bohr/Landau ξ) can be bolted onto the existing CSDA
drain in `transport.py` with no new dependency — far cheaper than importing
Nebula's machinery to get it.

## Recommendation

1. **Do not** adopt cstool/Nebula or move transport onto the eScatter stack. The
   added physics (Penn inelastic + SE cascade) serves SE-yield/SEM imaging, an
   observable orthogonal to coherent X-ray line flux; it is mostly below our
   5 keV cutoff and the in-band remainder (straggling) is negligible for our line
   widths.
2. **Keep** the in-house segment transport (Mott-calibrated elastic + CSDA). The
   elastic-accuracy improvement worth having comes from **ELSEPA via P2 #1**, which
   already delivers it without Nebula and without cstool's redundant elastic route.
3. **If** a measured PXR line width later shows unexplained broadening, add a
   lightweight Bohr/Landau straggling term to the CSDA step (a few lines, no
   dependency) before reconsidering a full inelastic engine.
4. **Revisit only** if the project scope changes to need SE-induced background,
   electron-imaging observables, or much higher beam energy / much thicker
   crystals (where the straggling rows above grow and δ-ray radiation could enter
   the window).

## References

- A. J. van Kessel & C. W. Hagen, "Nebula: Monte Carlo simulator of electron–
  matter interaction," *SoftwareX* **12**, 100605 (2020).
- Nebula docs / cstool: https://nebula-simulator.github.io/ ·
  https://github.com/Nebula-simulator/Nebula
- Penn, "Electron mean-free-path calculations using a model dielectric function,"
  *Phys. Rev. B* **35**, 482 (1987).
- P2 #1 ELSEPA adapter + validation: `docs/elsepa-transport.md` (on
  `feature/elsepa-port`).
