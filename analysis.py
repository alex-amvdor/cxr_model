import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import sys

    import marimo as mo

    sys.path.insert(0, "src")

    import matplotlib.pyplot as plt
    from IPython.display import display

    from cxr_mc.config import default_settings, trajectory_sweep
    from cxr_mc.plots import (
        browse,
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_material_comparison,
        plot_metric_vs,
        plot_penetration_survival,
        plot_scan,
        plot_timepix_efficiency,
        plot_timepix_poisson,
        plot_trajectory_grid,
    )
    from cxr_mc.results import (
        filter_results,
        records,
        select_results,
        show_top,
        sweep_values,
    )
    from cxr_mc.run import cases_from_results, load_checkpoint
    from cxr_mc.sweep import build_cases

    return (
        browse,
        build_cases,
        cases_from_results,
        default_settings,
        display,
        filter_results,
        load_checkpoint,
        mo,
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_material_comparison,
        plot_metric_vs,
        plot_penetration_survival,
        plot_scan,
        plot_timepix_efficiency,
        plot_timepix_poisson,
        plot_trajectory_grid,
        plt,
        records,
        select_results,
        show_top,
        sweep_values,
        trajectory_sweep,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bulk-crystal CXR — analysis & visualization

    Loads the checkpoint written by **`scan.ipynb`** and draws every figure — no
    sweep runs here (only the cheap, CPU-only electron transport behind the
    penetration figures). The heavy lifting lives in `src/`:

    - **`config.py`** — the shared `Settings` + per-material sweep grids.
    - **`results.py`** — post-processing, the `best_azimuth` reduction, stats table.
    - **`plots.py`** — all the figures (datashader-rasterized trajectories,
      intrinsic spectra, the Eagle XO detector view, parametric heatmaps).

    Set `MATERIAL` to match the scan you ran, then run top to bottom.
    """)
    return


@app.cell
def _(
    cases_from_results,
    default_settings,
    filter_results,
    load_checkpoint,
    show_top,
):
    # Material must match the scan you ran in scan.ipynb / scan.py.
    MATERIAL = "hopg"

    settings = default_settings()
    results = load_checkpoint(MATERIAL)  # {name: {E0: record}}
    cases = cases_from_results(results)  # rebuild the case list from the records
    res = filter_results(results, cases)  # all loaded cases for this material

    # Compact, ranked "best geometries" table -- the short, readable view (sorted by
    # bright + well-defined line). For the full per-row dump use show_summary(...).
    show_top(res, settings, top_n=15, select="quality_peak")
    return MATERIAL, cases, res, settings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Slicing a big checkpoint

    A material's checkpoint accumulates every case ever run, so a loaded `res` can be
    huge — and the per-tilt browsers will overlay every extra swept dimension (e.g.
    all 40 thicknesses). `sweep_values` shows what's in there; `select_results` slices
    it to the cases you want by **value** (exact, a list, or a predicate). Pass the
    sliced store to any plot below.
    """)
    return


@app.cell
def _(display, records, res, select_results, sweep_values):
    # What's actually in this checkpoint -- swept knobs and their values. A material's
    # pickle accumulates EVERY case ever run, so `res` can be large.
    display(sweep_values(res))

    # Slice it to just the cases you want, by VALUE (exact, a list, or a predicate) --
    # the fix for "I loaded hopg to look at a few tilts and got every thickness
    # overplotted." The result drops into any plot/browser below.
    res_sel = select_results(
        res,
        tilt_deg=lambda t: t > -40,  # a predicate for a range...
        # tilt_deg=[-23.9, -14.4],   # ...or an explicit list of swept values
        # thickness_ang=lambda x: x <= 5e4,
    )
    print(f"{len(records(res_sel))} of {len(records(res))} records after slicing")
    # e.g.  browse(res_sel, settings, kind="chunk")   /   plot_scan(res_sel, settings)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intrinsic spectra

    One figure per polar tilt, click through with the tilt slider. `chunk` is the
    best-azimuth total / CXR-only pair; `by_energy` overlays every beam energy;
    `full` is the full measured range (sharp lines on the wide brem, log-log).
    """)
    return


@app.cell
def _(browse, res, settings):
    browse(res, settings, kind="chunk")  # best-azimuth: total | CXR-only
    return


@app.cell
def _(browse, res, settings):
    browse(res, settings, kind="by_energy")  # every beam energy overlaid
    return


@app.cell
def _(browse, res, settings):
    browse(res, settings, kind="full")  # full measured range, log-log
    return


@app.cell
def _(cases, plot_scan, res, settings):
    # Unified parametric scan: plot_scan auto-picks a HEATMAP (when both axes sweep
    # many values) or LINE plots (when an axis is fixed / has only a few values), one
    # figure per quantity (peak/coherent/line flux, line energy, FWHM, line/total,
    # line-definition quality, total flux). Defaults: azimuth x polar tilt, panel per
    # energy. For hopg (a thickness x tilt sweep, single azimuth) it auto-switches to
    # lines vs polar tilt instead of a banded heatmap.
    _ = plot_scan(res, settings, cases=cases, line_metric="prominence")

    # Pick any two swept knobs as axes; plot_scan chooses the representation:
    #   plot_scan(res, settings, x="thickness_ang", y="tilt_deg")  # dense study -> heatmap
    #   plot_scan(res, settings, x="tilt_deg", y="E0_keV")         # few values  -> lines
    #   plot_scan(res, settings, force="heatmap")  # or force="lines" to override
    #   plot_heatmaps(...) / plot_metric_vs(...)   # the specialized drawers, if you
    #                                              # always want a map or always lines
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry selection & parameter scans

    Thousands of cases can't all be shown. Instead of paging every polar tilt, rank
    every geometry by a selection score and show only the best few. The default
    score is `peak flux x line-definition quality` (bright **and** a clean single
    line), so spurious tall spikes don't win — change it with `select=` (`"peak"`,
    `"line_flux"`, `"coherent_flux"`, `"quality_peak"`, `"quality_line"`).

    `plot_scan` (above) is the unified scan — it auto-picks a heatmap or line plots.
    `plot_metric_vs` is the explicit 1-D drawer it delegates to: any metric vs any
    swept parameter, one line per a second. Energy / tilt scans work with the default
    sweep; a thickness scan needs the sweep to vary `thickness_ang`.
    """)
    return


@app.cell
def _(cases, plot_best_spectra, plot_metric_vs, plot_scan, plt, res, settings):
    # The best dozen geometries across the whole sweep (bright + clean line).
    plot_best_spectra(res, settings, top_n=12, select="quality_peak")
    plt.show()

    # 1-D scans vs polar tilt (the swept axis here): line flux & peak flux, one line
    # per beam energy. (plot_metric_vs also guards a single-valued x -- e.g. a
    # single-energy sweep -- by auto-substituting a parameter that actually sweeps.)
    plot_metric_vs(res, settings, x="tilt_deg", metric="line_flux", hue="E0_keV")
    plt.show()
    plot_metric_vs(res, settings, x="tilt_deg", metric="peak_flux", hue="E0_keV")
    plt.show()

    # thickness x tilt is dense on BOTH axes -> heatmaps, not 15-line spaghetti.
    # coherent flux (all lines) and the CXR / incoherent-brem ratio across geometry:
    plot_scan(
        res,
        settings,
        cases=cases,
        x="thickness_ang",
        y="tilt_deg",
        quantities=["coherent_flux"],
    )
    plt.show()
    plot_scan(
        res,
        settings,
        cases=cases,
        x="thickness_ang",
        y="tilt_deg",
        quantities=["coherent_brem_ratio"],
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eagle XO detector response

    The Raptor **Eagle XO** direct-detection CCD modelled as `solid_angle x QE(E)`
    (`eaglexo_response.py`): soft PXR lines pass at ~90% QE while the hard brem is
    crushed by the thin back-thinned sensor. This is the detector view (it replaces
    the old EDS detector-convolved spectra).

    > Point the sweep at the real Eagle solid angle with
    > `material_sweep(..., **eaglexo_response.sweep_geometry("4240", distance_m=...))`,
    > and set the working distance in `eaglexo_response.py` before trusting absolute rates.
    """)
    return


@app.cell
def _(plot_eaglexo_efficiency):
    plot_eaglexo_efficiency(sensor="4240")  # QE + solid angle + resolution
    return


@app.cell
def _(browse, res, settings):
    browse(res, settings, kind="eaglexo")  # detected (solid) vs incident (dotted)
    return


@app.cell
def _():
    # NB: there is no "measured spectrum" plot for a bare Eagle XO -- it's an
    # INTEGRATING CCD (accumulates charge, can't resolve photons), so it returns
    # brightness, not a spectrum. The energy-resolving photon-counting mode
    # (eaglexo_response.poisson_counts / resolve_energy) is a special low-occupancy
    # extra. What the camera actually reports is the recorded CHARGE -> the next cell.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recorded charge (the CCD reports brightness, not counts)

    The Eagle XO is a CCD: it **integrates charge** and cannot resolve individual
    photons, so its figure of merit is collected charge, not photon flux. Each
    absorbed photon deposits `E / W_Si` electrons, so the recorded signal is the
    detected spectrum weighted by photon **energy** — `plot_eaglexo_charge` shows
    where that signal comes from, and `plot_eaglexo_charge_map` maps the integrated
    charge rate (or, with `exposure_s`, the well-fill fraction) across geometry.
    """)
    return


@app.cell
def _(browse, cases, plot_eaglexo_charge_map, plt, res, settings):
    # Where the recorded charge comes from (per polar tilt, energy-weighted): a CCD
    # integrates charge, not photons, so each photon is weighted by E/W_Si and the
    # hard brem carries far more charge per photon than its count suggests. Compare
    # this with the photon-density view above (browse kind="eaglexo").
    browse(res, settings, kind="eaglexo_charge")

    # Geometry map of the SIGNAL the CCD actually reports: integrated detected charge
    # rate (e-/s), best geometry per cell. Auto-switches to lines for a thin axis
    # (hopg: single azimuth -> charge vs polar tilt). Pass exposure_s=... for the
    # well-fill fraction (saturation vs FULL_WELL_E) instead.
    plot_eaglexo_charge_map(res, settings, cases=cases)
    plt.show()
    # plot_eaglexo_charge_map(res, settings, cases=cases, exposure_s=600.0)  # saturation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Timepix3 detector response (optional comparison)

    The 2×2 Timepix3 quad forward model (`timepix_response.py`): photoabsorption →
    charge sharing → per-pixel threshold counting → ToT noise → Poisson. Set the real
    quad thickness / bias — σ_diff ∝ 1/√bias is the most sensitive knob.
    """)
    return


@app.cell
def _(browse, plot_timepix_efficiency, plot_timepix_poisson, res, settings):
    TPX_THICKNESS_UM = 300.0  ### FILL IN -- Si sensor thickness [um]
    TPX_BIAS_V = 100.0  ### FILL IN -- applied bias [V]

    plot_timepix_efficiency(thickness_um=TPX_THICKNESS_UM, bias_v=TPX_BIAS_V)
    plot_timepix_poisson(
        res,
        settings,
        integration_s=600.0,
        thickness_um=TPX_THICKNESS_UM,
        bias_v=TPX_BIAS_V,
    )
    browse(
        res,
        settings,
        kind="timepix",
        thickness_um=TPX_THICKNESS_UM,
        bias_v=TPX_BIAS_V,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Electron penetration

    Cross-sections of the electron cascade in the beam-detector plane,
    datashader-rasterized and coloured by electron energy along each track. All
    panels at one beam energy share ONE (square) frame, so across tilts only the slab
    rotates (red = beam, green = detector). Then the surviving-population curve: the
    fraction of the incident electrons still reaching each depth below the entrance
    surface — where the beam stops, one curve per beam energy. These run the cheap,
    CPU-only electron transport directly — no checkpoint required.
    """)
    return


@app.cell
def _(
    MATERIAL,
    build_cases,
    plot_penetration_survival,
    plot_trajectory_grid,
    plt,
    settings,
    trajectory_sweep,
):
    traj_sweep = trajectory_sweep(MATERIAL, n_tilts=9, energies=(30, 60))
    traj_cases = build_cases(traj_sweep, settings.n_electrons, settings.n_electrons_brem)

    # one penetration grid (polar-tilt panels) per beam energy
    for E0 in sorted({c["E0_keV"] for c in traj_cases}):
        plot_trajectory_grid(traj_cases, energy=E0, Ne=120)

    # surviving electron population (% of N0) vs penetration depth, one curve per energy
    plot_penetration_survival(traj_cases, Ne=500)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-material comparison

    The headline figure: for every material whose checkpoint exists, take its single
    best geometry/energy and plot the dominant coherent line **energy vs flux**, one
    point per crystal, coloured by line-definition quality — "which crystal gives the
    brightest well-defined line, and where," for comparison against the paper's
    catalogue. (Needs the scans to have been run for several materials.)
    """)
    return


@app.cell
def _(load_checkpoint, plot_material_comparison, plt, settings):
    from cxr_mc.sweep import MATERIAL_LABELS

    # load every material checkpoint that exists; skip the empties
    by_material = {}
    for m in MATERIAL_LABELS:
        r = load_checkpoint(m)
        if r:
            by_material[MATERIAL_LABELS[m]] = r

    if len(by_material) >= 2:
        plot_material_comparison(by_material, settings, select="quality_peak")
        plt.show()
    else:
        print("run scan.ipynb for more materials to populate this comparison")
    return


if __name__ == "__main__":
    app.run()
