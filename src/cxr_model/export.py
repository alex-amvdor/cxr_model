"""Export analysis.ipynb to results/<material>_cxr_<date>.pdf via webpdf.

The output is named after the ACTIVE Sweep's material in the notebook's
parameters cell (commented-out example sweeps are skipped) plus today's date, so
successive exports are self-describing instead of all landing on analysis.pdf.
Pass an explicit stem to override:  ``cxr export my_custom_name``.

Run from the repo root (it reads ``analysis.ipynb`` and writes into ``results/``).
"""

import asyncio
import datetime
import json
import re
import sys

NOTEBOOK = "analysis.ipynb"


def _material(nb_path):
    """Swept material from the notebook's ACTIVE ``Sweep(...)`` in the parameters
    cell (commented example sweeps are skipped). 'cxr' if it can't be found."""
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
    except OSError:
        return "cxr"
    pat = re.compile(r"material\s*=\s*['\"]([A-Za-z0-9_]+)['\"]", re.IGNORECASE)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            if line.lstrip().startswith("#"):  # skip commented-out example sweeps
                continue
            m = pat.search(line)
            if m:
                return m.group(1).lower()
    return "cxr"


def _default_stem():
    return f"{_material(NOTEBOOK)}_cxr_{datetime.date.today():%Y-%m-%d}"


def _export(stem=None):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.set_event_loop_policy = lambda *a, **k: None  # block jupyter reverting it

    stem = stem or _default_stem()
    print(f"exporting {NOTEBOOK} -> results/{stem}.pdf")

    from nbconvert.nbconvertapp import main as nbmain

    sys.argv = [
        "jupyter-nbconvert",
        "--to",
        "webpdf",
        "--output-dir",
        "results",
        "--output",
        stem,
        NOTEBOOK,
    ]
    nbmain()


def add_subparser(sub):
    """Register the ``export`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser("export", help="render analysis.ipynb -> results/<stem>.pdf")
    ap.add_argument(
        "stem",
        nargs="?",
        default=None,
        help="output filename stem (default: <material>_cxr_<date>)",
    )
    ap.set_defaults(func=lambda args: _export(args.stem))
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    _export(argv[0] if argv else None)


if __name__ == "__main__":
    main()
