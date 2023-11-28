import datetime
import doctest
import faulthandler
import importlib
import os
import pkgutil
import unittest
from argparse import ArgumentParser
from random import choice, shuffle

faulthandler.enable()  # to debug seg faults and timeouts

import cf


def randomise_test_order(*_args):
    """Return a random choice from 1 or -1.

    When set as the test loader method for standard (merge)sort
    comparison to order all methods in a test case (see
    'sortTestMethodsUsing'), ensures they run in a random order, meaning
    implicit reliance on setup or state, i.e. test dependencies, become
    evident over repeated runs.

    """
    return choice([1, -1])


def add_doctests(test_suite):
    """Set up doctest tests and add them to a given test suite."""
    # Tell doctest comparisons to treat any sequence of whitespace including
    # newlines as equal and to take '...' in output to mean any text
    doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    # TODO: incrementally add-in doctesting to eventually cover whole codebase
    # but for now setup each module gradually by adding specific sub-module
    # names to 'if' statement below to include all under those in the check.
    for importer, name, ispkg in pkgutil.walk_packages(
        cf.__path__, cf.__name__ + "."
    ):
        test_suite.addTests(
            doctest.DocTestSuite(
                name,
                optionflags=doctest_flags,
                globs={
                    "cf": importlib.import_module("cf"),
                    "numpy": importlib.import_module("numpy"),
                },
            )
        )


# Set the tests to run from the cf/test/ directory even if this script is run
# from another directory (even one outside the repo root). This makes it easier
# to set up the individual tests without errors due to e.g. bad relative dirs:
test_dir = os.path.dirname(os.path.realpath(__file__))
# Build the test suite from the tests found in the test files:
test_loader = unittest.TestLoader
# Randomise the order to run the test methods within each test case (module),
# i.e. within each test_<TestCase>, e.g. for all in test_AuxiliaryCoordinate:
test_loader.sortTestMethodsUsing = randomise_test_order

testsuite_setup_0 = unittest.TestSuite()
testsuite_setup_0.addTests(
    test_loader().discover(test_dir, pattern="create_test_files.py")
)

testsuite_setup_0b = unittest.TestSuite()
testsuite_setup_0b.addTests(
    test_loader().discover(test_dir, pattern="create_test_files_2.py")
)
# Build the test suite from the tests found in the test files.
testsuite_setup_1 = unittest.TestSuite()
testsuite_setup_1.addTests(
    test_loader().discover(test_dir, pattern="setup_create_field.py")
)

testsuite = unittest.TestSuite()
all_test_cases = test_loader().discover(test_dir, pattern="test_*.py")
# Randomise the order to run the test cases (modules, i.e. test_<TestCase>)
# TODO: change to a in-built unittest way to specify the above (can't find one
# after much searching, but want to avoid mutating weakly-private attribute).
shuffle(all_test_cases._tests)
testsuite.addTests(all_test_cases)


# Run the test suite's first set-up stage.
def run_test_suite_setup_0(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite_setup_0)
    runner.run(testsuite_setup_0b)


def run_doctests_only(verbosity=2):
    """Run only doctest tests to test docstring code examples."""
    testsuite_doctests = unittest.TestSuite()  # use a new dedicated test suite
    add_doctests(testsuite_doctests)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    outcome = runner.run(testsuite_doctests)
    if not outcome.wasSuccessful():  # set exit code
        exit(1)  # else is zero for sucess as standard


# Run the test suite's second set-up stage.
def run_test_suite_setup_1(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite_setup_1)


# Run the test suite.
def run_test_suite(verbosity=2, include_doctests=False):
    if include_doctests:  # add doctests to the test suite to run
        add_doctests(testsuite)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    outcome = runner.run(testsuite)
    # Note unittest.TextTestRunner().run() does not set an exit code, so (esp.
    # for CI / GH Actions workflows) we need $? = 1 set if any sub-test fails:
    if not outcome.wasSuccessful():
        exit(1)  # else is zero for sucess as standard


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--doctest",
        dest="doctest",
        action="store_true",
        help="run only the doctest tests",
    )
    args = parser.parse_args()

    original_chunksize = cf.chunksize()
    print("--------------------")
    print("CF-PYTHON TEST SUITE")
    print("--------------------")
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    print("Running tests from", os.path.abspath(os.curdir))

    if args.doctest:
        print("Note: running only doctest tests\n")
        run_doctests_only()
    else:
        print("")
        cf.chunksize(original_chunksize)

        run_test_suite_setup_0()
        run_test_suite_setup_1()
        # TODO: when doctesting is ready such that all modules have the right
        # prep in their code docstring examples, set include_doctests=True.
        run_test_suite()

        cf.chunksize(original_chunksize)
