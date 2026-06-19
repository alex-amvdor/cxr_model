"""Export analysis.ipynb to results/<material>_cxr_<date>.pdf via webpdf.

The output is named after the ACTIVE Sweep's material (commented-out example
sweeps in the parameters cell are skipped) plus today's date, so successive
exports are self-describing instead of all landing on analysis.pdf. Pass an
explicit stem to override:  python export_pdf.py my_custom_name
"""

import sys
import asyncio
import json
import re
import datetime

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


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.set_event_loop_policy = lambda *a, **k: (
            None
        )  # block jupyter reverting it

    stem = sys.argv[1] if len(sys.argv) > 1 else _default_stem()
    print(f"exporting {NOTEBOOK} -> results/{stem}.pdf")

    from nbconvert.nbconvertapp import main

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
    main()
