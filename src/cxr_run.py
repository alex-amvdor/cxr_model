"""
cxr_run.py
==========

Drive a sweep through ``cxr_montecarlo.run_cases`` with on-disk checkpointing
and per-group streaming feedback.

:func:`run_sweep` resumes from a per-material checkpoint
(``checkpoints/<material>.pkl``, skipping cached cases), stores each finished
case into ``results``, re-pickles that material's checkpoint at config
granularity (crash-safe), and invokes ``on_chunk(group_config_names)`` once a
whole **group**
has finished. By default a group is everything that shares
(material, thickness, polar tilt) -- i.e. the full azimuth sweep at one tilt --
so the streamed plot/table waits until every azimuth is in and can collapse to
the best azimuth per energy (see cxr_plots.stream_chunk).

run_cases shows a tqdm bar over completed cases; it renders once at the top of
the cell, so as streamed plots/tables pile up below it scrolls out of view --
hence the explicit "tilt N/M" progress line printed with each chunk here, which
stays next to the latest output. With a GPU present the run is serial (one CUDA
context); see cxr_montecarlo.run_cases.
"""

import os
import pickle
import time
from collections import defaultdict

from cxr_montecarlo import run_cases
from cxr_results import store_result


def _default_group_key(case):
    """Everything but the azimuth (and energy): the azimuth sweep at one tilt."""
    return (case["crystal"], case["thickness_ang"], case["tilt_deg"])


def run_sweep(
    cases,
    results,
    *,
    checkpoint_dir="checkpoints",
    checkpoint_path=None,
    resume=True,
    max_workers=None,
    progress=True,
    group_key=None,
    on_chunk=None,
):
    """Run ``cases`` into ``results`` (mutated in place).

    cases : list of case dicts from cxr_sweep.build_cases.
    results : the dict to fill ({name: {E0: record}}).
    checkpoint_dir : directory for the per-material checkpoints. A sweep is
        single-material, so each writes ``<checkpoint_dir>/<material>.pkl``
        holding ONLY that material's configs -- different materials never share a
        pickle, and ``results`` can hold several materials in one kernel without
        them clobbering each other on disk. Created if missing; seeded once from
        the legacy combined ``cxr_run_checkpoint.pkl`` if that's still around.
    checkpoint_path : explicit override for the pickle path; None (default)
        derives the per-material path above. A rerun with resume=True skips every
        config already in the pickle.
    max_workers : forwarded to run_cases (None -> serial on GPU, ~3/4 cores on
        CPU); >1 on a GPU is coerced to serial there.
    progress : forwarded to run_cases (the per-case tqdm bar).
    group_key : case -> hashable. Configs sharing a key form one group;
        on_chunk fires once the WHOLE group has finished. Default groups by
        (material, thickness, polar tilt), so the azimuth sweep at a tilt is one
        group -- on_chunk gets every azimuth at once.
    on_chunk : optional callback(list_of_config_names) fired once per completed
        group, with all of that group's config names (cached + freshly run).
    """
    if group_key is None:
        group_key = _default_group_key

    # per-material checkpoint: a sweep is single-material, so name the pickle for
    # the crystal and keep them together in their own subdir.
    material = cases[0]["crystal"] if cases else "mixed"
    if checkpoint_path is None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{material}.pkl")

    def _crystal_of(rec_map):
        return next(iter(rec_map.values()))["case"]["crystal"]

    def _save():
        """Pickle just THIS material's configs -- ``results`` may also hold other
        materials run earlier in the same kernel, which belong in their own pkl."""
        subset = {n: results[n] for n in results if _crystal_of(results[n]) == material}
        with open(checkpoint_path, "wb") as f:
            pickle.dump(subset, f)

    # one-time migration: seed a missing per-material pickle from the old single
    # combined checkpoint (subset to this material), so an in-progress run's cache
    # survives the switch to per-material files.
    legacy = "cxr_run_checkpoint.pkl"
    if resume and not os.path.exists(checkpoint_path) and os.path.exists(legacy):
        try:
            with open(legacy, "rb") as f:
                old = pickle.load(f)
            sub = {n: v for n, v in old.items() if _crystal_of(v) == material}
            if sub:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(sub, f)
                print(
                    f"migrated {len(sub)} {material} configs from {legacy} -> {checkpoint_path}"
                )
        except Exception as e:  # corrupt/locked legacy pickle -> just skip it
            print(f"(legacy checkpoint migration skipped: {e})")

    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            loaded = pickle.load(f)
        results.update(loaded)
        print(
            f"resumed {sum(len(v) for v in loaded.values())} {material} cases from {checkpoint_path}"
        )

    todo = [
        c
        for c in cases
        if not (c["name"] in results and c["E0_keV"] in results[c["name"]])
    ]
    print(f"{len(todo)} of {len(cases)} cases to run ({len(cases) - len(todo)} cached)")

    # all config names in each group (cached + to-run), in first-seen order
    group_names = defaultdict(list)
    seen = set()
    for c in cases:
        if c["name"] not in seen:
            seen.add(c["name"])
            group_names[group_key(c)].append(c["name"])

    # per group: how many of its configs still have unfinished energies;
    # per config: how many beam energies it still owes
    group_remaining = defaultdict(int)
    energies_remaining = {}
    for c in todo:
        if c["name"] not in energies_remaining:
            group_remaining[group_key(c)] += 1
        energies_remaining[c["name"]] = energies_remaining.get(c["name"], 0) + 1
    n_groups = len(group_remaining)
    progress_state = {"done": 0}

    def _cb(i, case, out):
        store_result(results, case, out)
        name = case["name"]
        energies_remaining[name] -= 1
        if energies_remaining[name] == 0:  # this config (all energies) is done
            _save()  # crash-safe at config granularity (this material's subset)
            g = group_key(case)
            group_remaining[g] -= 1
            if group_remaining[g] == 0 and on_chunk is not None:
                # the whole azimuth sweep at this tilt is in -> stream it, with a
                # progress line that stays next to the streamed plot/table (the
                # tqdm bar is up top and scrolls away)
                progress_state["done"] += 1
                print(
                    f"\n=== {case['thickness_ang'] / 1e4:g} um, tilt "
                    f"{case['tilt_deg']:g} deg done "
                    f"-- {progress_state['done']}/{n_groups} tilt-groups ==="
                )
                on_chunk(group_names[g])

    # Fully-cached tilt-groups have no cases left to run, so the callback above
    # never fires for them. On resume, replay them through on_chunk first (in
    # tilt order) so their best-azimuth plots/tables are redrawn too.
    if on_chunk is not None:
        for g in sorted(g for g in group_names if group_remaining.get(g, 0) == 0):
            rep = next(iter(results[group_names[g][0]].values()))["case"]
            print(
                f"\n=== cached: {rep['thickness_ang'] / 1e4:g} um, "
                f"tilt {rep['tilt_deg']:g} deg (already computed) ==="
            )
            on_chunk(group_names[g])

    t0 = time.perf_counter()
    run_cases(todo, max_workers=max_workers, progress=progress, callback=_cb)
    print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")


# ---- brem-only repair --------------------------------------------------------
def repair_brem_wide(
    results, only_nonfinite=True, progress=True, save_every=0, save_cb=None
):
    """Regenerate ``brem_wide`` (and the line-grid ``brem``) for cached records
    using the CURRENT ``mc_brem_spectrum`` -- WITHOUT re-running the expensive
    line spectrum.

    Records computed before the brem-absorption fix have NaN ``brem_wide`` above
    the Henke ceiling (~30 keV) and at E=0, which the log plots drop (the
    full-range curve clips at 30 keV). This re-runs only the cheap bremsstrahlung
    transport (``Ne_brem`` electrons, same seed as the original case) per record
    and rewrites ``r["brem_wide"]`` / ``r["brem"]`` in place; the line ``spec`` is
    left untouched. Everything needed is already stored in ``r["case"]``.

    only_nonfinite : skip records whose ``brem_wide`` is already all-finite
        (default), so only the stale ones are touched. Set False to redo all.
    save_every / save_cb : if both set, call ``save_cb(results)`` every
        ``save_every`` repaired records (and once at the end) -- used by
        :func:`repair_checkpoint` to checkpoint progress so a crash/OOM doesn't
        lose everything; a re-run then resumes (only_nonfinite skips the saved ones).
    Returns the number of records repaired. Mutates ``results`` in place; re-pickle
    the checkpoint afterwards (or use :func:`repair_checkpoint`).
    """
    import numpy as np
    from cxr_montecarlo import (
        simulate_trajectories,
        mc_brem_spectrum,
        tilted_geometry,
    )

    todo = []
    for name in results:
        for r in results[name].values():
            bw = r.get("brem_wide")
            if only_nonfinite and bw is not None and np.isfinite(np.asarray(bw)).all():
                continue
            todo.append(r)
    if not todo:
        print("nothing to repair (all brem_wide already finite)")
        return 0
    print(f"repairing brem for {len(todo)} record(s)...")
    t0 = time.perf_counter()
    for k, r in enumerate(todo, 1):
        c = r["case"]
        if "E_grid_brem" in c:  # (start, stop, step) tuple
            E_brem = np.arange(*c["E_grid_brem"])
        elif r.get("E_grid_brem") is not None and np.asarray(r["E_grid_brem"]).size:
            E_brem = np.asarray(r["E_grid_brem"], float)
        else:  # legacy single-grid record
            step_b = c.get("brem_step_eV", 10.0)
            eg = np.asarray(r["E_grid"], float)
            E_brem = np.arange(eg[0], eg[-1] + step_b, step_b)
        beam, n_hat = tilted_geometry(
            c["theta_obs_rad"],
            np.deg2rad(c.get("tilt_deg", 0.0)),
            np.deg2rad(c.get("tilt_azim_deg", 0.0)),
        )
        segs_b = simulate_trajectories(
            c["E0_keV"],
            c["Ne_brem"],
            c["thickness_ang"],
            composition=c["composition"],
            E_cut_keV=c.get("E_cut_brem_keV", 1.0),
            seed=c["seed"] + 1,
            beam_dir=beam,
        )
        brem_wide = mc_brem_spectrum(
            segs_b, E_brem, composition=c["composition"], n_hat=n_hat
        )
        r["brem_wide"] = brem_wide
        r["E_grid_brem"] = E_brem
        r["brem"] = np.interp(np.asarray(r["E_grid"], float), E_brem, brem_wide)
        if save_cb is not None and save_every and (k % save_every == 0):
            save_cb(results)
            if progress:
                print(f"    ...saved progress ({k}/{len(todo)})")
        if progress and (k % 50 == 0 or k == len(todo)):
            print(f"  {k}/{len(todo)}  ({time.perf_counter() - t0:.0f} s)")
    if save_cb is not None:
        save_cb(results)  # final save
    return len(todo)


def repair_checkpoint(checkpoint_path, save_every=100, **kw):
    """Load a per-material checkpoint, repair its ``brem_wide`` (see
    :func:`repair_brem_wide`), and re-pickle it in place -- saving progress every
    ``save_every`` records (atomic temp+replace) so a crash/OOM is RESUMABLE: just
    call again and only_nonfinite picks up where it left off. Returns the repaired
    ``results`` dict (also usable directly in the notebook)."""
    if not os.path.exists(checkpoint_path):
        print(f"no such checkpoint: {checkpoint_path}")
        return {}
    with open(checkpoint_path, "rb") as f:
        results = pickle.load(f)

    def _save(res):
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(res, f)
        os.replace(tmp, checkpoint_path)  # atomic: never leaves a half-written pkl

    n = repair_brem_wide(results, save_every=save_every, save_cb=_save, **kw)
    print(f"re-saved {checkpoint_path}" if n else "checkpoint unchanged")
    return results
