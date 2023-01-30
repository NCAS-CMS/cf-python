import datetime
import faulthandler
import os
import unittest

import pycodestyle

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class styleTest(unittest.TestCase):
    """Test PEP8 compliance on all Python '.py' files in the 'cf'
    directory."""

    def setUp(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Note these must be specified relative to the roor dir of the repo:
        non_cf_python_files = [
            "scripts/cfa",
            "docs/source/conf.py",
            "setup.py",
        ]
        root_dir_relative_to_pwd = os.path.abspath(
            os.path.join(os.pardir, os.pardir)
        )
        self.non_cf_dir_python_paths = [
            os.path.join(root_dir_relative_to_pwd, *(path.split("/")))
            for path in non_cf_python_files
        ]

    def test_pep8_compliance(self):
        pep8_check = pycodestyle.StyleGuide()

        # Directories to skip in the recursive walk of the directory:
        skip_dirs = ("__pycache__", "c-lib")
        # These are pycodestyle errors and warnings to explicitly ignore. For
        # descriptions for each code see:
        # https://pep8.readthedocs.io/en/latest/intro.html#error-codes
        pep8_check.options.ignore += (  # ignored because...
            "W605",  # ... false positives on regex and LaTeX expressions
            "E272",  # ... >1 spaces to align keywords in long import listings
            "E402",  # ... justified lower module imports in units and __init__
            "E501",  # ... docstring examples include output lines >79 chars
            "E712",  # ... query module requires a (specific) '== False' check
            "E731",  # ... one sensible (default sort key) lambda expression
            # Black auto-formatting doesn't abide by this code (e.g. see
            # https://github.com/psf/black/issues/315)
            "E203",
        )

        # First add Python files which lie outside of the cf directory:
        python_files = self.non_cf_dir_python_paths
        # Then find all Python source code ('.py') files in the 'cf' directory,
        # including all unskipped sub-directories within e.g. test directory:
        for root_dir, dirs, filelist in os.walk(".."):
            if os.path.basename(root_dir) in skip_dirs:
                continue
            python_files += [
                os.path.join(root_dir, fname)
                for fname in filelist
                if fname.endswith(".py")
            ]

        # Ignore non-existent files which lie outside of the cf-python
        # directory
        python_files = [
            python_file
            for python_file in python_files
            if os.path.isfile(python_file)
        ]

        pep8_issues = pep8_check.check_files(python_files).total_errors
        self.assertEqual(
            pep8_issues,
            0,
            f"Detected {pep8_issues!s} PEP8 errors or warnings:",
        )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
