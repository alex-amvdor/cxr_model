import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys

    import marimo as mo

    sys.path.insert(0, "src")

    from IPython.display import display

    from cxr_mc.config import COLLAPSE_AZIMUTH, default_settings, material_sweep
    from cxr_mc.plots import stream_chunk
    from cxr_mc.run import run_sweep
    from cxr_mc.sweep import build_cases, geometry_table

    return (
        COLLAPSE_AZIMUTH,
        build_cases,
        default_settings,
        display,
        geometry_table,
        material_sweep,
        mo,
        run_sweep,
        stream_chunk,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bulk-crystal CXR — scan runner

    Runs the Monte-Carlo CXR parameter sweep for one material and writes the
    per-material checkpoint (`checkpoints/<material>.pkl`). The companion
    **`analysis.ipynb`** loads that checkpoint and draws every figure — keeping
    the long scan and the (re-runnable) plotting in separate kernels.

    Set `MATERIAL` below, then run top to bottom. Every material's sweep grid lives
    in `src/config.py`, shared with the analysis notebook so the two never drift.
    """)
    return


@app.cell
def _(build_cases, default_settings, display, geometry_table, material_sweep):
    # Material: "hopg" | "diamond" | "silicon" | "mose2" | "wse2" | "ptse2"
    #         | "hfse2" | "zrse2" | "ws2" | "mos2"
    MATERIAL = "hopg"

    settings = default_settings()
    sweep = material_sweep(MATERIAL)  # full parametric grid (src/config.py)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    print(f"{len(cases)} cases across {len({c['name'] for c in cases})} configs")
    display(geometry_table(cases))
    return MATERIAL, cases, settings


@app.cell
def _(COLLAPSE_AZIMUTH, MATERIAL, cases, run_sweep, settings, stream_chunk):
    # Run (resumes from the checkpoint, skipping cached cases). The per-tilt
    # photon-counting tables stream live; all the figures are in analysis.ipynb.
    results = {}
    try:
        run_sweep(
            cases,
            results,
            on_chunk=lambda batch: stream_chunk(
                results, batch, settings, collapse_azimuth=COLLAPSE_AZIMUTH
            ),
        )
    except EOFError:
        print("EOF Error -- the script has already processed all data")

    print(f"\nDone -> checkpoints/{MATERIAL}.pkl")
    print("Open analysis.ipynb with the same MATERIAL to visualize.")
    return


if __name__ == "__main__":
    app.run()
