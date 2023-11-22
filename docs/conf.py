# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from rocm_docs import ROCmDocs

# for PDF output on Read the Docs
project = "HIPIFY Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved."

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs("HIPIFY Documentation")
docs_core.setup()

external_projects_current_project = "hipify"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
