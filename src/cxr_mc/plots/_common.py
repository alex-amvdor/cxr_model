"""_common

Shared figure plumbing: per-record line/brem split and the per-tilt figure loop.
"""

import matplotlib.pyplot as plt

from ..montecarlo import (
    convolve_detector,
    detector_efficiency,
)
from ..results import (
    detected_background,
)

# cache of the (expensive) Timepix efficiency-curve response, keyed by hardware +
# MC settings, so re-running the detector cell with unchanged settings doesn't
# rebuild it. (The per-grid detected-spectra response is already cached inside
# timepix_response.get_response.)
_EFF_CACHE = {}


def _mode(settings):
    return "EDS-convolved" if getattr(settings, "convolve_with_det", False) else "intrinsic"


def _line_brem(r, settings, convolve=None):
    """Detected line and brem densities (per eV, before the unit scale) for one
    record, honoring the QE / brem-source flags. ``convolve`` overrides
    settings.convolve_with_det when given (True/False), so a caller can draw the
    intrinsic (convolve=False) and detector-convolved (convolve=True) spectra
    side by side."""
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    qe = detector_efficiency(r["E_grid"]) if settings.apply_detector_qe else 1.0
    line_in = r["spec"] * qe
    line_det = convolve_detector(r["E_grid"], line_in, r["fwhm"]) if do_conv else line_in
    brem_det = detected_background(r, settings, convolve=do_conv) / r["scale"]
    return line_det, brem_det


# ---- per-tilt figure helper --------------------------------------------------
def _per_tilt_figs(recs, settings, draw, figsize, *, empty_msg="no results yet", **kw):
    """Shared body of every ``plot_*`` wrapper: one freshly-drawn figure PER POLAR
    TILT. ``draw(fig, tilt_recs, settings, **kw)`` renders a single tilt onto a
    cleared figure (the same ``_draw_*`` the interactive ``browse`` uses), so the
    wrappers and the slider stay in lockstep. Handles the empty-records check, the
    per-tilt grouping, and collecting the figure list."""
    if not recs:
        print(empty_msg)
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=figsize)
        draw(fig, [r for r in recs if r["case"]["tilt_deg"] == t], settings, **kw)
        figs.append(fig)
    return figs
