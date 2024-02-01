# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Multi-hadron Interpolators'
copyright = '2024, William Detmold, William I. Jay, Gurtej Kanwar, Phiala E. Shanahan, and Michael L. Wagman'
author = 'William Detmold, William I. Jay, Gurtej Kanwar, Phiala E. Shanahan, and Michael L. Wagman'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

# html_theme = 'alabaster'
html_title = project
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'navigation_with_keys': False
}
html_sidebars = {
    '**': []
}
html_static_path = ['_static']
