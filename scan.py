"""Headless CXR scan runner -- thin shim to cxr_mc.scan.

Kept at the repo root so cxr_mc.remote (which runs ``python scan.py <material>``
on the GPU box) and muscle-memory ``python scan.py`` keep working from a checkout
with no install. The real logic -- and the rationale for the __main__ guard
(spawn/forkserver re-import the entry module per worker) -- lives in
cxr_mc/scan.py. Prefer the installed CLI: ``cxr scan <material>``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cxr_mc.scan import main

if __name__ == "__main__":
    main()
