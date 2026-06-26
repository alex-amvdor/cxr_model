"""``cxr slim`` -- shrink a per-material checkpoint for transfer (TODO P2 #5).

The GPU box writes one pickle per material holding the full union of every swept
config at full resolution (``results.store_result``); pulling it to the viz
laptop is gigabyte-scale and mostly stale for any single plot. This command
writes a smaller pickle -- dropping the full-range bremsstrahlung arrays and/or
downcasting the spectra to float32 -- that still loads and plots exactly like the
full one via ``run.load_checkpoint``.

    cxr slim checkpoints/hopg.pkl --drop-wide-brem --downcast
    cxr slim checkpoints/hopg.pkl -o hopg.slim.pkl --downcast

Value-based config filtering (keep only some tilts/energies) is available
programmatically via ``results.slim_results(..., tilt_deg=..., E0_keV=...)``.
"""

import argparse
import os
import pickle

from .results import slim_results


def slim_checkpoint(in_path, out_path=None, *, drop_wide_brem=False, downcast=False, **constraints):
    """Load a checkpoint, slim it (:func:`results.slim_results`), write a smaller
    pickle (atomic temp+replace), and report the size saved. ``out_path`` defaults
    to ``<stem>.slim<ext>``. Extra keyword args are case-field constraints passed
    straight to ``slim_results``. Returns the slim results dict."""
    with open(in_path, "rb") as f:
        results = pickle.load(f)
    slim = slim_results(results, drop_wide_brem=drop_wide_brem, downcast=downcast, **constraints)
    if out_path is None:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}.slim{ext or '.pkl'}"
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(slim, f)
    os.replace(tmp, out_path)  # atomic: never leave a half-written pickle
    before, after = os.path.getsize(in_path), os.path.getsize(out_path)
    n_in = sum(len(v) for v in results.values())
    n_out = sum(len(v) for v in slim.values())
    pct = 100.0 * (1.0 - after / before) if before else 0.0
    print(
        f"slimmed {in_path} ({before / 1e6:.1f} MB, {n_in} records) -> "
        f"{out_path} ({after / 1e6:.1f} MB, {n_out} records); {pct:.0f}% smaller"
    )
    return slim


def _cli(args):
    """CLI handler -- runs slim_checkpoint and returns None (the dict it returns
    must not reach sys.exit via the console-script wrapper)."""
    slim_checkpoint(
        args.checkpoint,
        args.out,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
    )


def add_subparser(sub):
    """Register the ``slim`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser("slim", help="shrink a checkpoint pickle for transfer")
    ap.add_argument("checkpoint", help="path to checkpoints/<material>.pkl")
    ap.add_argument("-o", "--out", default=None, help="output path (default: <stem>.slim.pkl)")
    ap.add_argument(
        "--drop-wide-brem",
        action="store_true",
        help="drop the full-range brem arrays (brem_wide, E_grid_brem) -- the largest fields",
    )
    ap.add_argument(
        "--downcast",
        action="store_true",
        help="store spectral arrays as float32 (halves their bytes)",
    )
    ap.set_defaults(func=_cli)
    return ap


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cxr-slim", description="shrink a checkpoint pickle for transfer"
    )
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
