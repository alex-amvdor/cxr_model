"""``cxr`` command-line entry point.

A single console script with subcommands, wired in pyproject.toml as
``cxr = "cxr_model.cli:main"``:

    cxr scan <material> [--quick] [--workers N]   # run a sweep -> checkpoint
    cxr export [stem]                             # analysis.ipynb -> results/<stem>.pdf
"""

import argparse

from . import __version__, scan, export


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cxr",
        description="Coherent X-ray radiation (PXR + coherent bremsstrahlung) toolkit.",
    )
    ap.add_argument("--version", action="version", version=f"cxr_model {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    scan.add_subparser(sub)
    export.add_subparser(sub)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
