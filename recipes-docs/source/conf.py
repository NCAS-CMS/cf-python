import os
import sys

# Make main 'docs' conf.py importable
sys.path.insert(0, os.path.abspath("../../docs/source"))

# Import everything from the main docs configuration
from conf import *  # noqa

# Now update the main docs configuration for recipes-specific needs
# TODO
