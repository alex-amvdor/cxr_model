"""Compile every code cell of the project notebooks to catch syntax errors.

Run from the repo root: ``python src/_compile_nb.py``."""

import json
import os
import sys

nbs = [
    "scan.ipynb",
    "analysis.ipynb",
    os.path.join("checks", "cxr_analysis_feranchuk.ipynb"),
    os.path.join("checks", "zhai_fig1c_check.ipynb"),
]
bad = 0
for path in nbs:
    nb = json.load(open(path, encoding="utf-8"))
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        n_code += 1
        # nbformat allows source as a list of lines OR a single string; normalize
        # to text, then strip IPython magics / shell escapes line-by-line the way
        # the headless runner does (a per-character pass would mangle a "%magic").
        raw = c["source"]
        text = "".join(raw) if isinstance(raw, list) else raw
        src = "".join(
            ln for ln in text.splitlines(keepends=True) if not ln.lstrip().startswith(("%", "!"))
        )
        cid = c.get("id", str(i))
        try:
            compile(src, f"{path}:cell[{i}]({cid})", "exec")
        except SyntaxError as e:
            bad += 1
            print(f"  SYNTAX ERROR in {path} cell[{i}] ({cid}): {e}")
    print(f"{path}: {n_code} code cells compiled" + ("" if not bad else "  <-- errors above"))

print("\n" + ("ALL CELLS COMPILE" if bad == 0 else f"{bad} CELLS FAILED"))
sys.exit(1 if bad else 0)
