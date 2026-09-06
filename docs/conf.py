project = 'dasio'
copyright = '2026, Jiaxuan Li'
author = 'Jiaxuan Li'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'myst_parser',
]
html_theme = 'furo'
html_theme_options = {
    'source_repository': 'https://github.com/jxli2a/dasio',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'light_css_variables': {'content-width': '60em'},
}
default_role = 'code'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'


sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'within_subsection_order': 'FileNameSortKey',
    'capture_repr': (),
    'download_all_examples': False,
    'plot_gallery': 'True',
}
exclude_patterns = ['_build']
