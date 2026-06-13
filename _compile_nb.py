"""Compile every code cell of the edited notebooks to catch syntax errors."""
import json
import sys

nbs = ["cxr_analysis_zhai.ipynb", "cxr_analysis_feranchuk.ipynb",
       "zhai_fig1c_check.ipynb"]
bad = 0
for path in nbs:
    nb = json.load(open(path, encoding="utf-8"))
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        n_code += 1
        # strip IPython magics / shell escapes the way the headless runner does
        src = "".join(l for l in c["source"]
                      if not l.lstrip().startswith(("%", "!")))
        cid = c.get("id", str(i))
        try:
            compile(src, f"{path}:cell[{i}]({cid})", "exec")
        except SyntaxError as e:
            bad += 1
            print(f"  SYNTAX ERROR in {path} cell[{i}] ({cid}): {e}")
    print(f"{path}: {n_code} code cells compiled"
          + ("" if not bad else "  <-- errors above"))

print("\n" + ("ALL CELLS COMPILE" if bad == 0 else f"{bad} CELLS FAILED"))
sys.exit(1 if bad else 0)
