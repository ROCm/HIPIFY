# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from rocm_docs import setup_rocm_docs

(
    copyright,
    author,
    project,
    extensions,
    myst_enable_extensions,
    myst_heading_anchors,
    external_toc_path,
    external_toc_exclude_missing,
    intersphinx_mapping,
    intersphinx_disabled_domains,
    templates_path,
    epub_show_urls,
    exclude_patterns,
    html_theme,
    html_title,
    html_static_path,
    html_css_files,
    html_js_files,
    html_extra_path,
    html_theme_options,
    html_show_sphinx,
    html_favicon,
) = setup_rocm_docs("HIPIFY Documentation")
