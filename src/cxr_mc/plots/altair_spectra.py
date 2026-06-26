"""altair_spectra

Altair / Vega-Lite renderer for the intrinsic CXR spectra -- a fast, interactive
alternative to the matplotlib :mod:`cxr_mc.plots.spectra` figures. The headline
motivation for the jupyter -> marimo migration is that the matplotlib spectra are
sluggish over the fine energy grids; Vega-Lite renders them in the browser and
pans/zooms interactively at a fraction of the redraw cost.

Non-destructive: this module reuses the exact per-record data prep
(:func:`cxr_mc.plots._common._line_brem`) that the matplotlib path uses, so the
physics and units are identical -- only the renderer differs. The matplotlib
``plots/`` package is left untouched, and these names are intentionally NOT
re-exported from ``cxr_mc.plots`` (that package has a frozen export-set guard);
import them from the submodule:

    from cxr_mc.plots.altair_spectra import spectrum_chart

Functions return :class:`altair.Chart` objects, which render directly in marimo
and Jupyter.

Note on size: Vega-Lite caps a spec at 5000 data rows by default. A dense spectrum
(several thousand grid points x several beam energies) can exceed that; enable a
larger transport once at the top of a notebook with
``altair.data_transformers.enable("vegafusion")`` (shipped with ``marimo[recommended]``)
or ``altair.data_transformers.disable_max_rows()``. This module does not mutate that
global state itself.
"""

import altair as alt
import numpy as np
import pandas as pd

from ..results import records
from ._common import _line_brem

_FRAME_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]


def _tilt_records(results, tilt_deg=None):
    """Records for ONE polar tilt: the one nearest ``tilt_deg`` (default: the
    first/lowest tilt present). Returns ``[]`` when ``results`` is empty."""
    recs = records(results)
    if not recs:
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    t = tilts[0] if tilt_deg is None else min(tilts, key=lambda x: abs(x - tilt_deg))
    return [r for r in recs if r["case"]["tilt_deg"] == t]


def spectrum_frame(recs, settings, *, include_brem=True, collapse_azimuth=True):
    """Tidy long-form spectrum table for ``recs`` (already restricted to one polar
    tilt): one row per (beam energy, energy-grid point, component). Mirrors the
    intrinsic per-energy view of :func:`cxr_mc.plots.spectra._draw_by_energy`
    -- ``component`` is ``"total"`` (line + brem) or ``"brem"`` (the dashed
    underlay), both already multiplied by the per-record ``scale``. Columns:
    ``energy_eV, intensity, E0_keV, azimuth_deg, component``."""
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    frames = []
    for E0 in energies:
        grp = [r for r in recs if r["case"]["E0_keV"] == E0]
        if collapse_azimuth and len(grp) > 1:
            grp = [max(grp, key=lambda r: float(np.max(r["spec"])))]
        for r in grp:
            E = np.asarray(r["E_grid"], dtype=float)
            line_det, brem_det = _line_brem(r, settings, convolve=False)
            line_det = np.asarray(line_det, dtype=float)
            brem_det = np.asarray(brem_det, dtype=float)
            total = (line_det + brem_det) if include_brem else line_det
            az = float(r["case"]["tilt_azim_deg"])
            frames.append(
                pd.DataFrame(
                    {
                        "energy_eV": E,
                        "intensity": total * r["scale"],
                        "E0_keV": float(E0),
                        "azimuth_deg": az,
                        "component": "total",
                    }
                )
            )
            if include_brem:
                frames.append(
                    pd.DataFrame(
                        {
                            "energy_eV": E,
                            "intensity": brem_det * r["scale"],
                            "E0_keV": float(E0),
                            "azimuth_deg": az,
                            "component": "brem",
                        }
                    )
                )
    if not frames:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def spectrum_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    include_brem=True,
    collapse_azimuth=True,
    width=720,
    height=360,
):
    """Interactive Altair line chart of the INTRINSIC spectra at ONE polar tilt,
    one line per beam energy (brem drawn as a faint dashed underlay). The Altair
    counterpart of :func:`cxr_mc.plots.plot_by_energy` / ``browse(kind="by_energy")``
    -- same data prep, Vega-Lite renderer with pan/zoom. Pass ``tilt_deg`` to pick a
    tilt (default: the lowest present). Returns an :class:`altair.Chart`, or
    ``None`` when there are no records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = spectrum_frame(
        recs, settings, include_brem=include_brem, collapse_azimuth=collapse_azimuth
    )
    if df.empty:
        return None

    case = recs[0]["case"]
    title = (
        f"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} um, "
        f"theta_tilt={case['tilt_deg']:.1f} deg -- intrinsic"
    )

    base = alt.Chart(df).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)"),
        y=alt.Y("intensity:Q", title="Intensity (Phs/eV/s/nA)"),
        color=alt.Color("E0_keV:N", title="beam energy (keV)"),
        tooltip=["E0_keV:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = [base.transform_filter(alt.datum.component == "total").mark_line(strokeWidth=1.4)]
    if include_brem:
        layers.append(
            base.transform_filter(alt.datum.component == "brem").mark_line(
                strokeWidth=0.7, strokeDash=[4, 3], opacity=0.7
            )
        )
    return alt.layer(*layers).properties(width=width, height=height, title=title).interactive()
