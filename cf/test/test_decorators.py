import copy
import datetime
import unittest

import cf

import logging

# Note: it is important we test on the cf logging config rather than the
# generic Python module logging (i.e. 'cf.logging' not just 'logging').
# Also, mimic the use in the codebase by using a module-specific logger:
log_name = __name__
logger = cf.logging.getLogger(log_name)


class dummyClass:
    '''Dummy class acting as container to test methods as proper instance
       methods, mirroring their context in the codebase.
    '''
    def __init__(self, verbose=None):
        self.verbose = verbose

        self.debug_message = "A major clue to solving the evasive bug"
        self.detail_message = "In practice this will be very detailed."
        self.info_message = "This should be short and sweet"
        self.warning_message = "Best pay attention to this!"

    def func_2(self, good_kwarg=True, traceback=False, bad_kwarg=False):
        '''Dummy function, otherwise trivial, where a True boolean passed as
           a traceback keyword argument will ultimately raise an error.
        '''
        if traceback:
            cf.functions._DEPRECATION_ERROR_KWARGS(
                self, 'another_func', traceback=True)
        return good_kwarg

    @cf.decorators._deprecated_kwarg_check('traceback')
    def decorated_func_2(
            self, good_kwarg=True, traceback=False):
        '''Dummy function equivalent to 'func_2', but a decorator manages the
           logic to raise the error on use of a deprecated keyword argument.
        '''
        return good_kwarg

    # Not testing 'bad_kwarg' here other than to the extent that it does not
    # stop 'traceback from causing the expected deprecation-related error.
    @cf.decorators._deprecated_kwarg_check('traceback', 'bad_kwarg')
    def multikwarg_decorated_func_2(
            self, good_kwarg=True, traceback=False, bad_kwarg=False):
        '''Dummy function equivalent to 'func_2', but a decorator manages the
           logic to raise the error on use of a deprecated keyword argument.
        '''
        return good_kwarg

    @cf.decorators._manage_log_level_via_verbose_attr
    def decorated_logging_func(self):
        '''Dummy method to test _manage_log_level_via_verbose_attr.

        In particular, to test it interfaces with self.verbose correctly.
        '''
        logger.debug(self.debug_message)
        logger.detail(self.detail_message)
        logger.info(self.info_message)
        logger.warning(self.warning_message)


# --- End: class


class DecoratorsTest(unittest.TestCase):
    '''Test decorators module.

    These are unit tests on the self-contained decorators applied to an
    artificial, trivial & not cf-python specific class, so for the cases where
    decorators are imported directly from cf, there is no need to duplicate
    such tests which are already in the cf test suite.
    '''
    def setUp(self):
        self.test_only = []

    def test_deprecated_kwarg_check(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        test_class = dummyClass()

        # Test without (or with default) deprecated keyword argument
        res_1 = test_class.func_2(good_kwarg="good")
        res_2 = test_class.decorated_func_2(good_kwarg="good")
        res_3 = test_class.func_2(
            good_kwarg="good", traceback=False)
        res_4 = test_class.decorated_func_2(
            good_kwarg="good", traceback=False)
        res_5 = test_class.multikwarg_decorated_func_2(
            good_kwarg="good", traceback=False)
        self.assertEqual(res_1, res_2)
        self.assertEqual(res_2, "good")
        self.assertEqual(res_3, res_4)
        self.assertEqual(res_4, "good")

        # Test with deprecated keyword argument
        with self.assertRaises(cf.functions.DeprecationError):
            test_class.func_2(good_kwarg="good", traceback=True)
        with self.assertRaises(cf.functions.DeprecationError):
            test_class.decorated_func_2(
                good_kwarg="good", traceback=True)
        with self.assertRaises(cf.functions.DeprecationError):
            test_class.func_2(traceback=True, bad_kwarg="bad")
        with self.assertRaises(cf.functions.DeprecationError):
            test_class.multikwarg_decorated_func_2(
                traceback=True, bad_kwarg="bad")

    def test_manage_log_level_via_verbose_attr(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Order of decreasing severity/verbosity is crucial to one test below
        levels = ['WARNING', 'INFO', 'DETAIL', 'DEBUG']

        # Note we test assertions on the root logger object, which is the
        # one output overall at runtime, but the specific module logger name
        # should be registered within the log message:
        example_class = dummyClass()
        log_message = [
            'WARNING:{}:{}'.format(log_name, example_class.warning_message),
            'INFO:{}:{}'.format(log_name, example_class.info_message),
            'DETAIL:{}:{}'.format(log_name, example_class.detail_message),
            'DEBUG:{}:{}'.format(log_name, example_class.debug_message)
        ]

        for level in levels:
            # Important! Need to initialise class inside this loop not
            # outside it or, it retains the verbosity attribute value set
            # for the previous loop (0, i.e disable, so nothing emerges!)
            test_class = dummyClass()
            cf.LOG_LEVEL(level)  # reset to level

            # Default verbose(=None) cases: LOG_LEVEL should determine output
            with self.assertLogs(level=cf.LOG_LEVEL()) as catch:
                test_class.decorated_logging_func()

                for msg in log_message:
                    # LOG_LEVEL should prevent messages less severe appearing:
                    if levels.index(level) >= log_message.index(msg):
                        self.assertIn(msg, catch.output)
                    else:  # less severe, should be effectively filtered out
                        self.assertNotIn(msg, catch.output)

            # Cases where verbose is set; value should override LOG_LEVEL...

            # Highest verbosity case (note -1 == 'DEBUG', highest verbosity):
            # all messages should appear, regardless of global LOG_LEVEL:
            test_class.verbose = -1
            with self.assertLogs(level=cf.LOG_LEVEL()) as catch:
                test_class.decorated_logging_func()
                for msg in log_message:
                    self.assertIn(msg, catch.output)

            # Lowest verbosity case ('WARNING' / 1) excluding special case of
            # 'DISABLE' (see note above): only warning messages should appear,
            # regardless of global LOG_LEVEL value set:
            test_class.verbose = 1
            with self.assertLogs(level=cf.LOG_LEVEL()) as catch:
                test_class.decorated_logging_func()
                for msg in log_message:
                    if msg.split(":")[0] == 'WARNING':
                        self.assertIn(msg, catch.output)
                    else:
                        self.assertNotIn(msg, catch.output)

            # Boolean cases for testing backwards compatibility...

            # ... verbose=2 should be equivalent to verbose=3 now:
            test_class.verbose = True
            with self.assertLogs(level=cf.LOG_LEVEL()) as catch:
                test_class.decorated_logging_func()
                for msg in log_message:
                    if msg.split(":")[0] == 'DEBUG':
                        self.assertNotIn(msg, catch.output)
                    else:
                        self.assertIn(msg, catch.output)

            # ... verbose=0 should be equivalent to verbose=0 now, so
            # test along with 'DISABLE' special case below...

            # Special 'DISABLE' (0) case: note this needs to be last as we
            # reset the LOG_LEVEL to it but need to use 'NOTSET' for the
            # assertLogs level, which sends all log messages through:
            test_class.verbose = 0
            with self.assertLogs(level='NOTSET') as catch:
                # Note get 'AssertionError' if don't log anything at all, so
                # avoid this and allow check for disabled logging, first log
                # something then disable and check nothing else emerges:
                logger.info("Purely to keep 'assertLog' happy: see comment!")
                cf.LOG_LEVEL('DISABLE')
                test_class.decorated_logging_func()
                for msg in log_message:  # nothing else should be logged
                    self.assertNotIn(msg, catch.output)

            # verbose=0 should be equivalent in behaviour to verbose=0
            test_class.verbose = False
            with self.assertLogs(level='NOTSET') as catch:
                logger.info("Purely to keep 'assertLog' happy: see previous!")
                test_class.decorated_logging_func()
                for msg in log_message:  # nothing else should be logged
                    self.assertNotIn(msg, catch.output)


# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
