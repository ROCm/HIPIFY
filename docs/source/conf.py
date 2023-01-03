# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from format_toc import format_toc


# -- Project information -----------------------------------------------------

project = "HIPIFY Documentation"
copyright = "2022, Advanced Micro Devices Ltd. "
author = "Advanced Micro Devices <a href=\"https://\">Disclaimer and Licensing Info</a> "


# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
        "sphinx.ext.duration",
        "sphinx.ext.doctest",
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.intersphinx",
        "sphinx_external_toc",
        "sphinx_design",
        "sphinx_copybutton",
        "myst_nb",
    ]

# MyST Configuration
myst_enable_extensions = ["colon_fence", "linkify"]
myst_heading_anchors = 3

format_toc()
external_toc_path = "_toc.yml"
external_toc_exclude_missing = False


intersphinx_mapping = {
        "rtd": ("https://docs.readthedocs.io/en/stable/", None),
        "python": ("https://docs.python.org/3/", None),
        "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    }
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["code_word_breaks.js"]
html_theme_options = {
    "home_page_in_toc": True,
    "use_edit_page_button": True,
    "repository_url": "https://github.com/RadeonOpenCompute/ROCm_Documentation_Overview/",
#TODO: import branch based on current git checkout
    "repository_branch": "main",
    "path_to_docs": "docs",

    "show_navbar_depth" : "2",
    "body_max_width " : "none",
    "show_toc_level": "0"
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ["_static"]
