"""Headless execution of cxr_analysis_zhai.ipynb (temporary test harness).
The __main__ guard is required: the notebook's run cell uses process-based
parallelism, and spawned workers re-import this module."""
import json
import os

import matplotlib
matplotlib.use("Agg")

if __name__ == "__main__":
    # the notebook lives in the repo root (parent of checks/) and its cells use
    # CWD-relative paths -- src/ on sys.path, the run checkpoint -- so exec it
    # from there rather than from checks/
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    nb = json.load(open("cxr_analysis_zhai.ipynb", encoding="utf-8"))
    ns = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(l for l in c["source"] if not l.lstrip().startswith("%"))
        exec(compile(src, f"cell{i}", "exec"), ns)
    print("--- all cells OK ---")
