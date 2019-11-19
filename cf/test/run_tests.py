import datetime
import os
import unittest

import cf

# Build the test suite from the tests found in the test files.
testsuite_setup_0 = unittest.TestSuite()
testsuite_setup_0.addTests(unittest.TestLoader().discover('.', pattern='create_test_files.py'))

# Build the test suite from the tests found in the test files.
testsuite_setup_1 = unittest.TestSuite()
testsuite_setup_1.addTests(unittest.TestLoader().discover('.', pattern='setup_create_field.py'))

testsuite = unittest.TestSuite()
testsuite.addTests(unittest.TestLoader().discover('.', pattern='test_*.py'))

# Run the test suite.
def run_test_suite_setup_0(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite_setup_0)

    
# Run the test suite.
def run_test_suite_setup_1(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite_setup_1)

    
# Run the test suite.
def run_test_suite(verbosity=2):
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(testsuite)

    
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
