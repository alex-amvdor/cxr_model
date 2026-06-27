"""
montecarlo.runner

Parallel case driver: the per-case transport + spectrum + brem worker
(run_case and its CPU/GPU phase split) and run_cases, which pipelines the
CPU transport across a worker pool behind the single-CUDA-context GPU phase.
The phase functions stay module-level so they pickle into Windows spawn
workers.
"""

import os
from typing import Any

import numpy as np

from ._backend import _GPU, cp
from .geometry import tilted_geometry
from .spectrum import _segments_in_layer, mc_brem_spectrum, mc_spectrum
from .transport import simulate_trajectories


def run_case(case):
    """
    Worker for one (crystal, beam energy) Monte Carlo case: transport + line
    spectrum + bremsstrahlung. Module-level so it can be pickled into worker
    processes on Windows (notebook-defined functions cannot).

    case: a plain dict --
        required: crystal, composition, hkl_list, B_ang2, E0_keV, thickness_ang,
                theta_obs_rad, Ne, Ne_brem, seed, and EITHER a single
                E_grid = (start_eV, stop_eV, step_eV) OR the decoupled pair
                E_grid_line / E_grid_brem (each a (start, stop, step) tuple):
                the lines are evaluated on the fine NARROW E_grid_line, the
                smooth bremsstrahlung on the coarse WIDE E_grid_brem (extend the
                latter to the beam energy for the full measured spectrum, without
                paying the line cost up there -- the lines top out at a few keV).
        optional: tilt_deg (0), tilt_azim_deg (0), beam_uvw (None),
                azimuth_rad (0), E_cut_lines_keV (5), E_cut_brem_keV (1),
                spec_chunk (40000) / brem_chunk (20000): segments per GPU matmul
                -- lower these to cap peak GPU memory on a busy/shared device,
                sinc_cutoff (None = exact lineshapes; windowing buys nothing
                for bulk targets, where scattering Doppler-spreads the lines
                across the whole grid),
                mosaic_mc_fwhm_rad (None) / mosaic_mc_nodes (1): the exact
                Monte-Carlo crystal-mosaicity average (mc_spectrum); None/1 ->
                perfect crystal,
                brem_step_eV (10; legacy single-E_grid fallback only)

    Returns dict(E_grid, spec, brem [on E_grid], E_grid_brem, brem_wide [the
                full-range background], eta, n_segments) plus crystal/E0.
    """
    return _spectrum_case(case, _transport_case(case))


def _transport_case(case):
    """CPU-only phase of run_case: the line + brem trajectory transport (pure
    numpy, never touches the GPU). Returns the segments + geometry + grids the
    spectrum phase consumes. run_cases farms this out to a worker pool so the
    transport of upcoming cases overlaps the GPU work on the current one."""
    if "E_grid_line" in case:
        E_grid = np.arange(*case["E_grid_line"])
        E_brem = np.arange(*case["E_grid_brem"])
    else:
        E_grid = np.arange(*case["E_grid"])
        step_b = case.get("brem_step_eV", 10.0)
        E_brem = np.arange(E_grid[0], E_grid[-1] + step_b, step_b)
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    # film-on-substrate stack drives multilayer transport too (substrate
    # backscatter / substrate brem); None -> single-material slab (unchanged).
    layers = case.get("abs_layers")
    segs = simulate_trajectories(
        case["E0_keV"],
        case["Ne"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=case["seed"],
        beam_dir=beam,
        layers=layers,
    )
    segs_b = simulate_trajectories(
        case["E0_keV"],
        case["Ne_brem"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        seed=case["seed"] + 1,
        beam_dir=beam,
        layers=layers,
    )
    return dict(E_grid=E_grid, E_brem=E_brem, n_hat=n_hat, segs=segs, segs_b=segs_b)


def _spectrum_case(case, tp):
    """GPU phase of run_case: line spectrum + brem from the already-transported
    segments ``tp`` (from _transport_case). Runs in the main process, so only one
    CUDA context ever touches the device."""
    E_grid, E_brem, n_hat = tp["E_grid"], tp["E_brem"], tp["n_hat"]
    segs, segs_b = tp["segs"], tp["segs_b"]
    # optional film-on-substrate stack (None -> single slab, unchanged)
    abs_layers = case.get("abs_layers")
    n_lay = int(segs.get("n_layers", 1))

    # LINES: each CRYSTALLINE layer radiates its own PXR/CBS lines, summed
    # INCOHERENTLY (separate crystals -> no cross-layer coherence); every line
    # self-absorbs through the WHOLE stack (layers=abs_layers). `layer_radiators`
    # is a per-layer list aligned with the stack -- a dict of crystal params for a
    # crystalline layer (film or crystalline substrate), None for an amorphous one
    # (no coherent lines). layer_radiators absent -> single slab: the film radiates
    # from ALL its segments via the case's scalar crystal keys (bit-for-bit the
    # pre-multilayer path). See docs/multilayer-materials.md (per-layer radiation).
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),  # None -> perfect crystal
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = case.get("spec_chunk") or 40000
    if radiators is None:
        spec = mc_spectrum(
            segs,
            E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case.get("beam_uvw"),
            azimuth_rad=case.get("azimuth_rad", 0.0),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            **mosaic_kw,
        )
    else:
        spec = np.zeros(E_grid.shape, dtype=float)
        for L, rad in enumerate(radiators):
            if rad is None:  # amorphous layer -> no coherent lines
                continue
            sL = _segments_in_layer(segs, L)
            if sL["L_ang"].size == 0:
                continue
            spec = spec + mc_spectrum(
                sL,
                E_grid,
                crystal=rad["crystal"],
                hkl_list=rad["hkl_list"],
                n_hat=n_hat,
                B_ang2=rad["B_ang2"],
                composition=abs_layers[L][2],
                beam_uvw=rad.get("beam_uvw"),
                azimuth_rad=case.get("azimuth_rad", 0.0),
                sinc_cutoff=case.get("sinc_cutoff"),
                chunk=spec_chunk,
                layers=abs_layers,
                **mosaic_kw,
            )

    # BREM: EVERY layer radiates with its OWN composition (each Z^2 cross
    # section); each layer's brem self-absorbs through the whole stack. Summed
    # over layers; a single layer is exactly the old single-material brem.
    brem_chunk = case.get("brem_chunk") or 20000
    if n_lay == 1:
        brem_wide = mc_brem_spectrum(
            segs_b,
            E_brem,
            composition=case["composition"],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
        )
    else:
        brem_wide = np.zeros(E_brem.shape, dtype=float)
        for L in range(n_lay):
            sL = _segments_in_layer(segs_b, L)
            if sL["L_ang"].size == 0:
                continue
            brem_wide = brem_wide + mc_brem_spectrum(
                sL,
                E_brem,
                composition=abs_layers[L][2],
                n_hat=n_hat,
                chunk=brem_chunk,
                layers=abs_layers,
            )
    brem = np.interp(E_grid, E_brem, brem_wide)  # brem under the lines (line grid)
    # Hand this case's GPU scratch back to the OS so the CuPy memory pool can't
    # accumulate (and fragment) across a long sweep until it fills the card.
    if _GPU:
        cp.get_default_memory_pool().free_all_blocks()
    return dict(
        E_grid=E_grid,
        spec=spec,
        brem=brem,
        E_grid_brem=E_brem,
        brem_wide=brem_wide,
        eta=segs["n_backscattered"] / segs["Ne"],
        n_segments=int(segs["L_ang"].size),
        crystal=case["crystal"],
        E0_keV=case["E0_keV"],
    )


def _worker_init():
    """
    Runs once in each worker process: drop to BELOW_NORMAL priority so the
    desktop stays responsive. Workers still use idle CPU at full speed; the
    OS just schedules interactive applications first.
    """
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[reportAttributeAccessIssue]
        # typed signatures matter: the untyped pseudo-handle (-1) gets
        # truncated on 64-bit and the call silently fails
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        try:
            if hasattr(os, "nice"):
                os.nice(10)  # type: ignore[reportAttributeAccessIssue]  # POSIX fallback
        except Exception:
            pass


def run_cases(cases, max_workers=None, progress=True, callback=None):
    """
    Run a list of case dicts through run_case, results in input order.

    GPU present (the usual path): the CPU transport is PIPELINED across a worker
    pool while THIS process drives the spectrum/brem serially on the single CUDA
    context -- the ~40% transport idle overlaps the GPU work, with no device
    contention (multiple CUDA contexts are what crawled the old max_workers>1).
    Workers run ONLY transport (pure CPU/numpy), never the GPU. Callbacks fire in
    input order as each case's GPU phase finishes.

    No GPU: cases run through a worker pool (or serially), completion order.

    max_workers: None -> sized automatically (a few transport workers when a GPU
        is present; ~3/4 of the CPUs otherwise). An integer pins the count; 0
        runs everything serially in this process (debugging / safe fallback).
    progress: tqdm bar over completed cases.
    callback: callable(i, case, out) invoked in THIS process as each case
        finishes; stream/checkpoint/plot without waiting for the batch.
        Exceptions propagate and abort the run.

    Crawl protections: workers run BELOW_NORMAL priority (_worker_init) and get
    single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, inherited) -- N workers
    x M BLAS threads is the classic oversubscription freeze.
    """

    def _maybe_bar(iterable):
        if not progress:
            return iterable
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, total=len(cases), desc="cases")
        except ImportError:
            # tqdm.auto picks the widget bar inside Jupyter, and that bar
            # raises ImportError AT CONSTRUCTION if ipywidgets is missing --
            # fall back to the plain-text console bar before giving up.
            try:
                from tqdm import tqdm

                return tqdm(iterable, total=len(cases), desc="cases")
            except ImportError:
                return iterable

    n = len(cases)
    results: list[Any] = [None] * n
    if n == 0:
        return results

    def _serial():
        for i in _maybe_bar(range(n)):
            out = run_case(cases[i])
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
        return results

    def _single_thread_blas():
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = "1"

    # ---- GPU: pipeline CPU transport (worker pool) behind the serial GPU ------
    if _GPU:
        if max_workers == 0:
            return _serial()
        if max_workers is None:
            ncpu = os.process_cpu_count() or os.cpu_count() or 8
            nw = max(2, min(n, ncpu // 2))  # ~physical cores; transport is the tail
        else:
            nw = min(max_workers, n)
        if nw < 2:
            return _serial()
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor

        prefetch = nw + 2  # keep the transport pool ahead
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as ex:
            inflight = {i: ex.submit(_transport_case, cases[i]) for i in range(min(prefetch, n))}
            for i in _maybe_bar(range(n)):
                tp = inflight.pop(i).result()  # transport (already overlapped)
                j = i + prefetch
                if j < n:
                    inflight[j] = ex.submit(_transport_case, cases[j])
                out = _spectrum_case(cases[i], tp)  # GPU, THIS process only
                results[i] = out
                if callback is not None:
                    callback(i, cases[i], out)
        return results

    # ---- no GPU: serial in-process, or a full-case worker pool ---------------
    if max_workers is None:
        ncpu = os.process_cpu_count() or os.cpu_count() or 8
        max_workers = max(1, min(n, ncpu * 3 // 4))
    if max_workers == 0:
        return _serial()
    _single_thread_blas()
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
        futures = {ex.submit(run_case, c): i for i, c in enumerate(cases)}
        for fut in _maybe_bar(as_completed(futures)):
            i = futures[fut]
            out = fut.result()
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
    return results
