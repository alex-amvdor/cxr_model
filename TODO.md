# TODO — feature/cstool-nebula-eval

This branch carries one backlog item; the full triaged backlog lives on `main`.

## eScatter cstool / Nebula investigation (P2)

*EVALUATED — recommendation: do not adopt. Decision record in
[`docs/cstool-nebula-evaluation.md`](docs/cstool-nebula-evaluation.md).*

Question: move electron transport onto Nebula (GPU MC) + cstool (cross-section
compiler) to gain the full Penn-dielectric inelastic model + secondary-electron
cascades, vs. keeping the current Mott-calibrated elastic + Joy–Luo CSDA?

**Finding:** not worth it for coherent X-ray line-flux prediction —
- Nebula/cstool's new physics (Penn inelastic, SE cascade, acoustic phonon) serves
  SE-yield / SEM imaging, an observable orthogonal to ours, and lives mostly below
  the `E_cut = 5 keV` this model already discards.
- The one in-band gap, energy-loss straggling, is bounded to ≲10 eV of line
  broadening at the shallow high-flux depths vs. ~130 eV detector resolution
  (order-of-magnitude estimate in the doc).
- The elastic-accuracy win actually worth having is already delivered by the
  ELSEPA adapter (P2 #1); cstool's elastic route is redundant with it.
- Adopting Nebula raises dependencies + compute and does not emit the per-segment
  radiating trajectories the PXR+CBS layer needs — a complexity increase, not a
  simplification.

**Cheap fallback if ever needed:** a Bohr/Landau straggling term on the existing
CSDA step in `transport.py` (a few lines, no dependency) — see the doc.

**Next:** branch is complete (doc only, no code/physics change). Fold the
recommendation into `main`'s P2 #2 one-liner; this branch can be deleted once the
doc lands on `main`, or kept as the decision record's home.
