# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# for PDF output on Read the Docs
project = "HIPIFY Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."

extensions = ["rocm_docs"]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "hipify"

# Theme-related settings
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "repository_url": "https://github.com/ROCm/HIPIFY",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

# Generate llms.txt
rocm_docs_generate_llms = True
