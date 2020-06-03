import copy
import datetime
import unittest

import cf


class dummyClass:
    '''Dummy class acting as container to test methods as proper instance
       methods, mirroring their context in the codebase.
    '''

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

# --- End: class


class DecoratorsTest(unittest.TestCase):
    '''Test decorators module.

    These are unit tests on the self-contained decorators applied to an
    artificial, trivial & not cf-python specific class, so for the cases where
    decorators are imported directly from cfdm, there is no need to duplicate
    such tests which are already in the cfdm test suite.
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


# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
