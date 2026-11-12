# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sie_mom'
copyright = '2025, German Farinas Perez'
author = 'German Farinas Perez'

version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'breathe',
    'sphinx.ext.imgmath',
    'sphinxcontrib.tikz',
    'sphinxcontrib.bibtex',
]

templates_path = ['_templates']
exclude_patterns = []


math_number_all = False
math_eqref_format = "({number})"

# BibTeX configuration
bibtex_bibfiles = ['references_cem.bib', 'references_metasurfaces.bib', 'references_general_theory.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'SIE-MoM'
html_theme_options = {
    'home_page_in_toc': True,
}
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Options for imgmath -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.imgmath

imgmath_image_format = 'svg'
imgmath_latex_preamble = r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
'''

# -- Options for sphinxcontrib.tikz ------------------------------------------
# https://pypi.org/project/sphinxcontrib-tikz/

tikz_latex_preamble = r'''
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
'''
# -- Breathe configuration ---------------------------------------------------
breathe_projects = {
    "sie_mom": "../build/xml/",
}
breathe_default_project = "sie_mom"

# CUDA pointer qualifiers in Doxygen-generated function declarations.
cpp_id_attributes = ['__restrict__']

# Reduce hyperref PDF bookmark warnings when section titles include inline math.
latex_elements = {
        'preamble': r'''
\pdfstringdefDisableCommands{%
    \def\mathcal#1{#1}%
    \def\mathbf#1{#1}%
    \def\boldsymbol#1{#1}%
    \def\hat#1{#1}%
    \def\times{x}%
    \def\phi{phi}%
}
''',
}
