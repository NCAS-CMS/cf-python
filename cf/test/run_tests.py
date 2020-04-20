import datetime
import os
from random import choice, shuffle
import unittest

import cf


def randomise_test_order(*_args):
    '''Return a random choice from 1 or -1.

    When set as the test loader method for standard (merge)sort comparison
    to order all methods in a test case (see 'sortTestMethodsUsing'), ensures
    they run in a random order, meaning implicit reliance on setup or state,
    i.e. test dependencies, become evident over repeated runs.
    '''
    return choice([1, -1])


# Build the test suite from the tests found in the test files.
test_loader = unittest.TestLoader
# Randomise the order to run the test methods within each test case (module),
# i.e. within each test_<TestCase>, e.g. for all in test_AuxiliaryCoordinate:
test_loader.sortTestMethodsUsing = randomise_test_order

testsuite_setup_0 = unittest.TestSuite()
testsuite_setup_0.addTests(
    test_loader().discover('.', pattern='create_test_files.py')
)

# Build the test suite from the tests found in the test files.
testsuite_setup_1 = unittest.TestSuite()
testsuite_setup_1.addTests(
    test_loader().discover('.', pattern='setup_create_field.py')
)

testsuite = unittest.TestSuite()
all_test_cases = test_loader().discover('.', pattern='test_*.py')
# Randomise the order to run the test cases (modules, i.e. test_<TestCase>)
# TODO: change to a in-built unittest way to specify the above (can't find one
# after much searching, but want to avoid mutating weakly-private attribute).
shuffle(all_test_cases._tests)
testsuite.addTests(all_test_cases)


# Run the test suite's first set-up stage.
def run_test_suite_setup_0(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite_setup_0)


# Run the test suite's second set-up stage.
def run_test_suite_setup_1(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite_setup_1)


# Run the test suite.
def run_test_suite(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    outcome = runner.run(testsuite)
    # Note unittest.TextTestRunner().run() does not set an exit code, so (esp.
    # for CI / GH Actions workflows) we need $? = 1 set if any sub-test fails:
    if not outcome.wasSuccessful():
        exit(1)  # else is zero for sucess as standard


if __name__ == '__main__':
    original_chunksize = cf.CHUNKSIZE()
    print('--------------------')
    print('CF-PYTHON TEST SUITE')
    print('--------------------')
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print('')
    print('Running tests from', os.path.abspath(os.curdir))

    cf.CHUNKSIZE(original_chunksize)

    run_test_suite_setup_0()
    run_test_suite_setup_1()
    run_test_suite()

    cf.CHUNKSIZE(original_chunksize)
