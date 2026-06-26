"""Re-export contract for the ``plots`` package.

plots was split from a single module into a package; the package must keep
re-exporting every public AND internal name that external code (notebooks,
tests) imports as ``from cxr_mc.plots import X``. This freezes the set so a
dropped name fails here loudly rather than at some consumer's import. Importing
the package pulls in matplotlib but draws nothing, so it stays in the fast suite.
"""

import cxr_mc.plots as p

# Every top-level name the old plots.py defined, now re-exported from the package.
# Adding a name is fine; REMOVING one (or failing to re-export it) breaks this test.
FROZEN_EXPORTS = frozenset(
    {
        # _style
        "COLORS",
        "_ENERGY_PALETTE",
        "energy_color",
        # _common
        "_EFF_CACHE",
        "_mode",
        "_line_brem",
        "_per_tilt_figs",
        # interactive
        "browse",
        "_tilt_browser",
        "browse_plotly",
        "_draw_chunk",
        "plot_chunk",
        "stream_chunk",
        # spectra
        "plot_tilt_panel",
        "_draw_by_energy",
        "plot_by_energy",
        "_draw_full_spectrum",
        "plot_full_spectrum",
        "plot_peak_vs_tilt",
        "plot_mosaic_comparison",
        "plot_best_spectra",
        "plot_material_comparison",
        # detectors
        "_thr_keV",
        "plot_timepix_efficiency",
        "_tpx_detected",
        "_draw_timepix_detected",
        "plot_timepix_detected",
        "plot_timepix_poisson",
        "SI_K_EDGE_EV",
        "_domega_of",
        "plot_eaglexo_efficiency",
        "_eag_detected",
        "_draw_eaglexo_detected",
        "plot_eaglexo_detected",
        "_eag_charge_rate",
        "_draw_eaglexo_charge",
        "plot_eaglexo_charge",
        "plot_eaglexo_charge_map",
        # trajectories
        "C_ANG_PER_FS",
        "_TRAJ_CMAP",
        "_case_of",
        "_trajectory_cases",
        "_turbo_hex",
        "_beam_detector_basis",
        "_trajectory_data",
        "_trajectory_frame",
        "_square_frame",
        "_draw_trajectory_panel",
        "_traj_colorbar",
        "plot_electron_trajectories",
        "plot_trajectory_grid",
        "plot_penetration_survival",
        # sweeps
        "_AXIS_SPECS",
        "_HEATMAP_QUANTITIES",
        "_EXTRA_QUANTITIES",
        "_METRIC_LABELS",
        "_resolve_quantity",
        "_FLUX_GATED",
        "_axis_label",
        "_axis_disp",
        "_value_label",
        "_cell_edges",
        "plot_heatmaps",
        "facet_metric",
        "_isnum",
        "plot_metric_vs",
        "plot_scan",
    }
)


def test_every_frozen_name_is_importable():
    missing = sorted(n for n in FROZEN_EXPORTS if not hasattr(p, n))
    assert not missing, f"plots package no longer exports: {missing}"


def test_all_matches_frozen_set():
    assert set(p.__all__) == FROZEN_EXPORTS


def test_public_names_resolve_to_subpackage():
    # the re-exported callables must come from the new submodules, not a leftover
    # top-level plots.py
    assert p.plot_by_energy.__module__ == "cxr_mc.plots.spectra"
    assert p.browse.__module__ == "cxr_mc.plots.interactive"
    assert p.plot_heatmaps.__module__ == "cxr_mc.plots.sweeps"
    assert p.plot_timepix_efficiency.__module__ == "cxr_mc.plots.detectors"
    assert p.plot_trajectory_grid.__module__ == "cxr_mc.plots.trajectories"
