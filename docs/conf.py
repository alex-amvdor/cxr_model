"""Sphinx configuration for the cxr_model documentation site.

Lightweight by design: MyST renders the existing Markdown design notes and
autosummary/autodoc pull the API straight from the package docstrings. Build
with::

    uv run --group docs sphinx-build -b html docs docs/_build/html

then open ``docs/_build/html/index.html``.
"""

import os
import sys

# Make the package importable for autodoc even from a non-installed checkout
# (an editable ``uv sync`` also puts it on the path).
sys.path.insert(0, os.path.abspath("../src"))

# -- Project -----------------------------------------------------------------
project = "cxr_model"
author = "Alex Amador"
copyright = "2026, Alex Amador"
release = "0.1.0"
version = "0.1.0"

# -- Extensions --------------------------------------------------------------
extensions = [
    "myst_parser",            # render the docs/*.md design notes
    "sphinx.ext.autodoc",     # API docs from docstrings
    "sphinx.ext.autosummary", # per-module summary tables + stub pages
    "sphinx.ext.napoleon",    # Google/NumPy docstring styles
    "sphinx.ext.viewcode",    # [source] links
    "sphinx.ext.intersphinx", # cross-link to numpy/scipy/python
    "sphinx.ext.mathjax",     # the docstrings carry LaTeX
]

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# cupy is the optional GPU dependency. Mock it so the docs build on any machine
# (no CUDA wheel / no GPU required) and importing the MC modules never touches a
# device; the package already falls back to CPU at runtime.
autodoc_mock_imports = ["cupy"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# The package docstrings are plain text (inline math like |g|, indented parameter
# blocks) rather than reStructuredText, so docutils emits cosmetic parse warnings
# when autodoc renders them. Suppress that category — the pages still render fine
# — instead of churning validated physics modules to satisfy an RST parser.
suppress_warnings = ["docutils"]

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- General -----------------------------------------------------------------
root_doc = "index"
# docs/README.md is the GitHub folder index (a pointer table); the Sphinx
# landing page is index.md, so leave README.md out of the build.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "cxr_model"
