import os
import sys

from sphinx_gallery.sorting import FileNameSortKey

# Make main 'docs' conf.py importable
sys.path.insert(0, os.path.abspath("../../docs/source"))

# Import everything from the main docs configuration
from conf import *  # noqa: F403

# Now update the main docs configuration for recipes-specific needs
# TODO
# CF_DOCS_MODE = os.environ.get("CF_DOCS_MODE", "none")
# if CF_DOCS_MODE in ["dev-recipes", "dev-recipes-scrub", "latest", "archive"]:

# We need to use and configure sphinx-gallery for the recipes
extensions.append("sphinx_gallery.gen_gallery")  # noqa: F405

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "recipes",  # path to recipe files
    "gallery_dirs": "recipes",  # path to save gallery generated output
    "run_stale_examples": False,
    # Below setting can be buggy: see:
    # https://github.com/sphinx-gallery/sphinx-gallery/issues/967
    # "reference_url": {"cf": None},
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("cf",),
    "inspect_global_variables": True,
    "within_subsection_order": FileNameSortKey,
    "default_thumb_file": "_static/cf-recipe-placeholder-squarecrop.png",
    "image_scrapers": (
        "matplotlib",
    ),  # Ensures Matplotlib images are captured
    "plot_gallery": True,  # Enables plot rendering
    "reset_modules": ("matplotlib",),  # Helps with memory management
    "capture_repr": (),
}
