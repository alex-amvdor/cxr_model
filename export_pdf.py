"""analysis.ipynb -> results/<material>_cxr_<date>.pdf -- thin shim to cxr_mc.export.

Kept at the repo root for muscle-memory ``python export_pdf.py [stem]``; prefer
the installed CLI ``cxr export [stem]``. The real logic lives in cxr_mc/export.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cxr_mc.export import main

if __name__ == "__main__":
    main()
