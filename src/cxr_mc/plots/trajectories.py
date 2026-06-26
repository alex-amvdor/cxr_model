"""trajectories

Electron-trajectory and penetration/survival figures.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..montecarlo import (
    simulate_trajectories,
    tilted_geometry,
)
from ..results import (
    records,
)
from ._style import (
    energy_color,
)

# ---- electron trajectory + penetration view ----------------------------------
# Datashader rasterizes the (tens of thousands of) trajectory line-segments into
# ONE image per panel -- fast, and tiny on disk vs a matplotlib LineCollection of
# every segment -- while matplotlib keeps the crisp slab / beam / detector overlay
# and the energy colorbar (so the nbconvert PDF export still works). The segment
# colour is the electron's kinetic energy along the track (turbo); ds.max keeps it
# crisp under the line-width antialiasing (ds.mean would blend track edges low).
C_ANG_PER_FS = 2997.924580  # speed of light [Ang/fs]: age sum(L/beta)[Ang] -> fs
_TRAJ_CMAP = "turbo"


def _case_of(rec_or_case):
    """Accept either a results record (carries 'case') or a raw case dict."""
    return rec_or_case.get("case", rec_or_case)


def _trajectory_cases(cases_or_results):
    """Flatten a build_cases list OR a results store into a list of case dicts."""
    if isinstance(cases_or_results, dict):
        return [r["case"] for r in records(cases_or_results)]
    return [_case_of(c) for c in cases_or_results]


def _turbo_hex(n=256):
    """The turbo colormap as a hex list (the form datashader.shade wants)."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps[_TRAJ_CMAP]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def _beam_detector_basis(beam, n_hat):
    """Orthonormal 2D basis of the BEAM-DETECTOR plane: e1 = beam (-> +x, into the
    slab); e2 = the in-plane part of the detector direction (-> +y, "up"). Working
    in this plane (rather than a fixed x-z slice) keeps the beam horizontal AND the
    detector arrow pointing the right way for ANY polar/azimuthal tilt."""
    e1 = np.asarray(beam, float)
    e1 = e1 / np.linalg.norm(e1)
    nh = np.asarray(n_hat, float)
    nh = nh / np.linalg.norm(nh)
    perp = nh - np.dot(nh, e1) * e1
    if np.linalg.norm(perp) < 1e-9:  # detector ~parallel to beam: any in-plane up
        for ref in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            perp = ref - np.dot(ref, e1) * e1
            if np.linalg.norm(perp) > 1e-9:
                break
    return e1, perp / np.linalg.norm(perp)


def _trajectory_data(case, Ne, seed):
    """Simulate one case and project the cascade into the beam-detector plane.
    Returns 2D segment endpoints (M,2,2) in display units, per-segment energy/age/
    depth, the slab + detector unit vectors in that plane, and the back/through
    fractions."""
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    segs = simulate_trajectories(
        case["E0_keV"],
        Ne,
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=seed,
        beam_dir=beam,
    )
    e1, e2 = _beam_detector_basis(beam, n_hat)
    L, v, r = segs["L_ang"], segs["v_hat"], segs["r_mid"]
    start = r - 0.5 * L[:, None] * v
    u, ulab = (1e4, r"$\mu$m") if case["thickness_ang"] >= 1e4 else (10.0, "nm")

    # Continuous per-electron tracks (not a loose segment cloud): order segments by
    # (electron, age) so each electron's segment START points form a polyline --
    # consecutive starts share an endpoint, so they trace the real zig-zag path --
    # then break with NaN between electrons. This is what makes the tracks read as
    # paths (the old per-segment LineCollection got faint at the cool, slow tail).
    order = np.lexsort((segs["t_ang"], segs["elec_id"]))
    sx = (start @ e1)[order] / u
    sy = (start @ e2)[order] / u
    sE = segs["E_keV"][order]
    brk = np.flatnonzero(np.diff(segs["elec_id"][order]) != 0) + 1
    px = np.insert(sx, brk, np.nan)
    py = np.insert(sy, brk, np.nan)
    pE = np.insert(sE, brk, np.nan)

    z = np.array([0.0, 0.0, 1.0])  # slab normal in the sample frame
    ndet = np.array([n_hat @ e1, n_hat @ e2])
    ndet = ndet / np.linalg.norm(ndet)
    nslab = np.array([z @ e1, z @ e2])
    nn = np.linalg.norm(nslab)
    nslab = nslab / nn if nn > 1e-9 else np.array([1.0, 0.0])
    return dict(
        px=px,
        py=py,
        pE=pE,
        pts=np.column_stack([px, py]),  # for the shared-frame extent
        E=segs["E_keV"],
        t_fs=segs["t_ang"] / C_ANG_PER_FS,
        z_u=r[:, 2] / u,  # penetration depth below the surface, display units
        elec_id=segs["elec_id"],  # emitting electron index, per segment
        L=segs["L_ang"],
        ndet=ndet,
        nslab=nslab,
        u=u,
        ulab=ulab,
        thick=case["thickness_ang"] / u,
        eta=100.0 * segs["n_backscattered"] / segs["Ne"],
        thru=100.0 * segs["n_transmitted"] / segs["Ne"],
        Ne=segs["Ne"],
    )


def _trajectory_frame(pts_list, pct=99.0, pad=0.12, beam_frac=0.16):
    """ONE shared (xlo, xhi, ylo, yhi) for a set of panels, from the robust
    (1st/99th-percentile) extent of all their track vertices, expanded to include
    the origin and padded. Sharing it across tilts is what makes only the slab
    rotate frame-to-frame (the old per-panel autoscale was the "scaling is
    inconsistent" complaint). Symmetric in y (beam axis centred); the left margin
    always clears the beam arrow + label."""
    pts = np.concatenate([np.asarray(s).reshape(-1, 2) for s in pts_list], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    xlo = min(0.0, float(np.percentile(pts[:, 0], 100 - pct)))
    xhi = max(0.0, float(np.percentile(pts[:, 0], pct)))
    ymax = float(np.percentile(np.abs(pts[:, 1]), pct))
    sx = max(xhi - xlo, 1e-6)
    xlo -= pad * sx
    xhi += pad * sx
    yhi = max(ymax * (1.0 + pad), 1e-6)
    aL = beam_frac * (xhi - xlo)
    xlo = min(xlo, -1.5 * aL)  # room for the beam arrow + label
    return (float(xlo), float(xhi), float(-yhi), float(yhi))


def _square_frame(frame):
    """Expand the shorter side of a (xlo, xhi, ylo, yhi) frame symmetrically so it
    is SQUARE -- no data is cropped, the extra room becomes centred margin. With
    set_aspect("equal") this lets square subplot boxes hold the tracks without the
    skinny-strip letterboxing the wide native frame produced (the trajectory-grid
    sizing fix)."""
    xlo, xhi, ylo, yhi = frame
    w, h = xhi - xlo, yhi - ylo
    if w > h:
        pad = 0.5 * (w - h)
        ylo, yhi = ylo - pad, yhi + pad
    elif h > w:
        pad = 0.5 * (h - w)
        xlo, xhi = xlo - pad, xhi + pad
    return (float(xlo), float(xhi), float(ylo), float(yhi))


def _draw_trajectory_panel(
    ax,
    data,
    frame,
    E0,
    *,
    E_cut=5.0,
    px=820,
    spread_px=1,
    cmap=None,
    label=True,
    label_fs=8.5,
):
    """Render ONE penetration cross-section into ``ax`` over the shared ``frame``:
    grey slab, datashader-rasterized energy-coloured tracks, red beam + green
    detector arrows.

    The tracks are aggregated with ``line_width=0`` so each pixel takes the true
    electron energy of the track through it -- antialiased (line_width>0) lines
    instead coverage-weight that value, which paints a bogus radial gradient
    ACROSS the line thickness (hot centre -> cool edges) rather than along the
    path. ``tf.spread`` then thickens the crisp 1-px lines back to visibility
    WITHOUT reintroducing that artifact (it copies each pixel's colour outward)."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    xlo, xhi, ylo, yhi = frame
    nslab, ndet, thick = data["nslab"], data["ndet"], data["thick"]

    # slab polygon: front face through the origin, extending `thick` into +nslab
    tang = np.array([-nslab[1], nslab[0]])
    W = 6.0 * max(xhi - xlo, yhi - ylo)
    slab = np.array([-W * tang, W * tang, W * tang + thick * nslab, -W * tang + thick * nslab])
    ax.fill(slab[:, 0], slab[:, 1], facecolor="0.80", edgecolor="0.55", lw=1.0, zorder=1)

    # continuous NaN-separated per-electron tracks -> datashader raster, colour =
    # electron energy (ds.max keeps it crisp under the line-width antialiasing)
    df = pd.DataFrame({"x": data["px"], "y": data["py"], "E": data["pE"]})
    asp = (yhi - ylo) / (xhi - xlo)
    cvs = ds.Canvas(
        plot_width=px,
        plot_height=max(int(px * asp), 60),
        x_range=(xlo, xhi),
        y_range=(ylo, yhi),
    )
    agg = cvs.line(df, "x", "y", agg=ds.max("E"), line_width=0)  # crisp: true E/pixel
    img = tf.shade(agg, cmap=cmap or _turbo_hex(), span=(E_cut, E0), how="linear")
    if spread_px:
        img = tf.spread(img, px=spread_px, shape="circle")  # thicken, colour kept
    ax.imshow(
        np.asarray(img.to_pil()),
        extent=(xlo, xhi, ylo, yhi),
        origin="upper",
        aspect="equal",
        interpolation="none",
        zorder=2,
    )

    # beam (red) + detector (green) arrows, anchored at the entry point
    aL = 0.16 * (xhi - xlo)
    ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(-aL, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(ndet[0] * aL, ndet[1] * aL),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#119911", lw=2.0),
        zorder=4,
    )
    if label:
        ax.text(
            -aL * 0.5,
            0.03 * (yhi - ylo),
            "beam",
            color="red",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            zorder=5,
        )
        tx = float(np.clip(ndet[0] * aL * 1.1, xlo + 0.06 * (xhi - xlo), xhi - 0.06 * (xhi - xlo)))
        ty = float(np.clip(ndet[1] * aL * 1.1, ylo + 0.06 * (yhi - ylo), yhi - 0.1 * (yhi - ylo)))
        ax.text(
            tx,
            ty,
            "detector",
            color="#0a6a0a",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            zorder=5,
        )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")


def _traj_colorbar(ax, E_cut, E0, label="electron energy (keV)"):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    sm = cm.ScalarMappable(norm=Normalize(E_cut, E0), cmap=_TRAJ_CMAP)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    return cb


def plot_electron_trajectories(
    rec_or_case,
    *,
    Ne=200,
    seed=0,
    frame=None,
    E_cut=5.0,
    colorbar=True,
    spread_px=1,
    label=True,
    ax=None,
):
    """One electron-penetration cross-section in the beam-detector plane: the beam
    enters horizontally at the origin (red), the crystal is the grey slab (which
    rotates with the tilt), the detector direction is the green arrow, and the
    cascade is datashader-rasterized, coloured by electron energy.

    ``frame`` (xlo, xhi, ylo, yhi) fixes the axes so repeated calls at the same
    (material, thickness, energy) share ONE frame and only the slab rotates; None
    auto-fits this case. ``ax`` draws into an existing axis (used by the grid)."""
    case = _case_of(rec_or_case)
    data = _trajectory_data(case, Ne, seed)
    if frame is None:
        frame = _trajectory_frame([data["pts"]])
    if ax is None:
        xlo, xhi, ylo, yhi = frame
        asp = (yhi - ylo) / (xhi - xlo)
        # size the FIGURE to the data aspect so the equal-aspect axes fills it
        # (no floating-title letterbox): reserve ~1.7" width for ylabel+colorbar
        # and ~1.1" height for title+xlabel, then constrained_layout packs it.
        axw = 5.4
        _, ax = plt.subplots(
            figsize=(axw + 1.7, float(np.clip(axw * asp + 1.1, 3.2, 8.4))),
            constrained_layout=True,
        )
    _draw_trajectory_panel(
        ax, data, frame, case["E0_keV"], E_cut=E_cut, spread_px=spread_px, label=label
    )
    ax.set_xlabel(f"distance along beam ({data['ulab']})")
    ax.set_ylabel(f"transverse distance ({data['ulab']})")
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['E0_keV']:g} keV, "
        rf"$\theta$={case.get('tilt_deg', 0.0):g}$\degree$, "
        rf"$\phi$={case.get('tilt_azim_deg', 0.0):g}$\degree$  —  {data['Ne']} e$^-$ "
        rf"({data['eta']:.0f}% back, {data['thru']:.0f}% through)",
        fontsize=10,
    )
    if colorbar:
        _traj_colorbar(ax, E_cut, case["E0_keV"])
    return ax


def plot_trajectory_grid(
    cases_or_results,
    energy=None,
    *,
    Ne=150,
    seed=0,
    E_cut=5.0,
    spread_px=1,
    max_panels=12,
    ncols=None,
    max_width_in=13.0,
    max_height_in=11.0,
    panel_in=3.3,
    min_panel_in=2.5,
):
    """Electron-penetration cross-sections at ONE beam energy, a panel per
    (polar, azimuthal) tilt -- the trajectory analogue of plot_heatmaps. Every
    panel shares ONE (squared) frame, so across the grid ONLY the slab rotates;
    the cascade is datashader-rasterized and energy-coloured with a single shared
    colorbar.

    ``cases_or_results`` is a build_cases list or a results store; ``energy`` picks
    the beam energy (default the lowest). If both polar and azimuthal tilt are
    swept it lays out a polar x azimuth grid (like the heatmaps); otherwise it
    wraps the swept tilt into at most 3 columns (override with ``ncols``). ``Ne``
    (electrons/panel) trades detail for speed -- the electron transport, not the
    drawing, is the cost.

    Panel SIZE: each panel is SQUARE (the shared frame is squared so set_aspect
    "equal" stays exact) and targets ``panel_in`` inches (~3.3", matching
    plot_best_spectra), never shrinking below ``min_panel_in``. For more tilts than
    fit in 3 columns the figure grows TALLER (scrollable) rather than crushing
    panels -- ``max_height_in`` no longer shrinks them. ``max_panels`` caps how
    many (polar, azimuth) combos are drawn (evenly subsampled past the cap); raise
    it (and/or ``panel_in``) for a denser grid, lower it for bigger panels."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    energies = sorted({c["E0_keV"] for c in cases})
    energy = energies[0] if energy is None else min(energies, key=lambda e: abs(e - energy))
    grp = [c for c in cases if c["E0_keV"] == energy]

    polars = sorted({c["tilt_deg"] for c in grp})
    azims = sorted({c["tilt_azim_deg"] for c in grp})
    grid2d = len(polars) > 1 and len(azims) > 1
    # one representative case per (polar, azimuth) combo, in a stable order
    bycombo = {}
    for c in sorted(grp, key=lambda c: (c["tilt_deg"], c["tilt_azim_deg"])):
        bycombo.setdefault((c["tilt_deg"], c["tilt_azim_deg"]), c)
    combos = list(bycombo)
    if len(combos) > max_panels:  # subsample evenly so the grid stays on-screen
        keep = np.unique(np.linspace(0, len(combos) - 1, max_panels).round().astype(int))
        combos = [combos[i] for i in keep]
        grid2d = False
    data = {cb: _trajectory_data(bycombo[cb], Ne, seed) for cb in combos}
    # square the shared frame so the panels read as squares (the native frame is
    # wide -> skinny strips); no data is cropped, set_aspect("equal") stays exact.
    frame = _square_frame(_trajectory_frame([d["pts"] for d in data.values()]))

    if grid2d:
        nrows, ncols = len(polars), len(azims)
        cell = [[(p, a) for a in azims] for p in polars]
    else:
        n = len(combos)
        ncols = ncols or min(3, n)  # cap at 3 columns; wrap the rest into ROWS
        nrows = int(np.ceil(n / ncols))
        cell = [
            [combos[r * ncols + col] if r * ncols + col < n else None for col in range(ncols)]
            for r in range(nrows)
        ]

    # SQUARE panels (the "trajectory plots are tiny / skinny" fix): the frame is
    # square so asp == 1 and pw == ph. Target ``panel_in`` (~3.3", matching
    # plot_best_spectra), clamp to the per-column width budget, never shrink below
    # min_panel_in -- for many tilts the FIGURE grows TALLER (scrollable) rather
    # than crushing every panel. ``max_height_in`` is no longer used to shrink.
    xlo, xhi, ylo, yhi = frame
    asp = (yhi - ylo) / (xhi - xlo)  # == 1 after _square_frame
    pw = min(panel_in, (max_width_in - 1.1) / ncols)
    pw = max(pw, min_panel_in)
    ph = pw * asp
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * pw + 1.1, nrows * ph + 0.7),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for r in range(nrows):
        for col in range(ncols):
            ax = axes[r][col]
            combo = cell[r][col]
            if combo is None or combo not in data:
                ax.axis("off")
                continue
            d = data[combo]
            _draw_trajectory_panel(
                ax, d, frame, energy, E_cut=E_cut, spread_px=spread_px, label=False
            )
            ax.set_title(
                rf"$\theta$={combo[0]:g}$\degree$, $\phi$={combo[1]:g}$\degree$",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
            if r == nrows - 1:
                ax.set_xlabel(f"along beam ({d['ulab']})", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"transverse ({d['ulab']})", fontsize=8)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    mappable = cm.ScalarMappable(norm=Normalize(E_cut, energy), cmap=_TRAJ_CMAP)
    cb = fig.colorbar(mappable, ax=axes, shrink=0.8, aspect=30, pad=0.01)
    cb.set_label("electron energy (keV)")
    case0 = bycombo[combos[0]]
    fig.suptitle(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"{energy:g} keV — electron penetration (red beam, green detector; "
        rf"only the slab rotates)",
        fontsize=11,
    )
    return fig


def plot_penetration_survival(
    cases_or_results,
    *,
    Ne=500,
    seed=0,
    n_bins=80,
    tilt=None,
    depth_frac=True,
):
    """Surviving electron population vs penetration depth -- the fraction of the
    incident electrons (% of N0) that reach AT LEAST a depth z below the entrance
    surface, one monotonically-decreasing curve per beam energy. This is where the
    beam stops: the curve falls from 100% at the surface to 0 at the deepest
    penetration, and a higher-energy beam reaches deeper.

    Per electron the deepest segment it reaches sets its penetration depth (the
    max over its segment midpoints' depth below the slab normal); then
    ``survival(z) = (# electrons reaching depth >= z) / N0``.
    ``tilt`` selects the polar tilt (nearest; default the one closest to normal
    incidence). ``depth_frac`` plots depth as a fraction of the slab thickness (so
    thin and thick slabs overlay); set False for absolute depth."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    tilts = sorted({c["tilt_deg"] for c in cases})
    want = 0.0 if tilt is None else tilt
    t = min(tilts, key=lambda x: abs(x - want))
    energies = sorted({c["E0_keV"] for c in cases})

    fig, ax = plt.subplots(figsize=(8, 5))
    xmax = 1.0
    for E0 in energies:
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is None:
            continue
        d = _trajectory_data(c, Ne, seed)
        # deepest point each electron reaches (max over its segment depths), then
        # clip the tiny negative excursions of backscattered electrons that exit
        # just above the entrance face.
        max_depth = np.full(d["Ne"], -np.inf)
        np.maximum.at(max_depth, d["elec_id"], d["z_u"])
        max_depth = np.clip(max_depth[np.isfinite(max_depth)], 0.0, None)
        thick = d["thick"]
        x = max_depth / thick if depth_frac else max_depth
        xmax = 1.0 if depth_frac else max(xmax, float(thick))
        zs = np.linspace(0.0, 1.0 if depth_frac else float(thick), n_bins)
        surv = 100.0 * np.array([float((x >= z).mean()) for z in zs])
        ax.plot(zs, surv, "-", color=energy_color(E0, energies), lw=1.9, label=f"{E0:g} keV")
    case0 = next(c for c in cases if c["tilt_deg"] == t)
    ulab = r"$\mu$m" if case0["thickness_ang"] >= 1e4 else "nm"
    xlab = "depth / thickness" if depth_frac else f"penetration depth ({ulab})"
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"surviving electrons (% of $N_0$)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy", fontsize=9)
    ax.set_title(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}$={t:g}$\degree$ — electron penetration / survival",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
