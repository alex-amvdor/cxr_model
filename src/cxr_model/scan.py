"""Headless CXR scan runner (the library twin of scan.ipynb).

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material checkpoint (checkpoints/<material>.pkl). Use this to run sweeps
non-interactively -- in particular over SSH on the GPU box; see cxr_model.remote,
which drives this and pulls the checkpoint back so the (matplotlib / PDF)
data-vis can stay on the laptop.

    cxr scan mose2                # the full per-material grid (config)
    cxr scan mose2 --quick        # tiny grid: smoke test / pipeline check
    cxr scan mose2 --workers 0    # serial (no transport worker pool)

(equivalently ``python scan.py mose2`` via the root shim).

The ``if __name__ == "__main__"`` guard on the entry point is REQUIRED, not
stylistic: run_cases farms the electron transport out to a process pool, and the
default start method is 'spawn' on Windows and -- as of Python 3.14 --
'forkserver' on Linux. BOTH re-import the entry module in every worker, so without
the guard the sweep relaunches itself recursively (a process-spawn cascade that
surfaces as a forkserver ConnectionResetError). Inside the guard the re-import is
a harmless no-op.
"""

import argparse
import os

import numpy as np

from .config import default_settings, material_sweep
from .sweep import build_cases
from .run import run_sweep


def _build_parser(ap):
    ap.add_argument("material", help="crystal key, e.g. mose2 / hopg / silicon / ptse2")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="run_cases max_workers (default auto; 0 = serial, no transport pool)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="tiny grid (5 polar tilts x 2 azimuths x 2 energies) for a smoke test",
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.set_defaults(func=run)
    return ap


def add_subparser(sub):
    """Register the ``scan`` subcommand on an argparse subparsers object."""
    return _build_parser(sub.add_parser("scan", help="run one material's MC sweep -> checkpoint"))


def run(args):
    settings = default_settings()
    if args.quick:
        sweep = material_sweep(
            args.material,
            tilt_deg=np.linspace(-45.0, 45.0, 5),
            tilt_azim_deg=np.array([-30.0, -10.0]),
            energy_keV=[30, 60],
        )
    else:
        sweep = material_sweep(args.material)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    print(
        f"{args.material}: {len(cases)} cases across "
        f"{len({c['name'] for c in cases})} configs"
        + (" (quick grid)" if args.quick else "")
    )

    # A --quick smoke test writes to its OWN checkpoint (<material>_quick.pkl), so
    # its coarse off-grid points never contaminate the real per-material sweep.
    ckpt = (
        os.path.join(args.checkpoint_dir, f"{args.material}_quick.pkl")
        if args.quick
        else None  # None -> run_sweep derives <material>.pkl
    )
    results = {}
    run_sweep(
        cases,
        results,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=ckpt,
        max_workers=args.workers,
    )
    n = sum(len(v) for v in results.values())
    stem = f"{args.material}_quick" if args.quick else args.material
    print(f"done -> {args.checkpoint_dir}/{stem}.pkl ({n} records)")


def main(argv=None):
    ap = _build_parser(
        argparse.ArgumentParser(prog="scan.py", description="headless CXR scan runner")
    )
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
