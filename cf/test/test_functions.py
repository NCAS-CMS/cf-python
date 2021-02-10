import datetime
import faulthandler
import os
import platform
import sys
import unittest
import inspect

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class functionTest(unittest.TestCase):
    def setUp(self):
        self.test_only = ()

    def test_example_field(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for n in range(8):
            f = cf.example_field(n)
            _ = f.array
            _ = f.dump(display=False)

        with self.assertRaises(Exception):
            _ = cf.example_field(-999)

    def test_keyword_deprecation(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Use as test case 'i' kwarg, the deprecated old name for
        # 'inplace':
        a = cf.Data([list(range(100))])
        a.squeeze(inplace=True)  # new way to specify operation tested below

        b = cf.Data([list(range(100))])
        with self.assertRaises(cf.functions.DeprecationError):
            b.squeeze(i=True)

    def test_aliases(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        self.assertEqual(cf.log_level(), cf.LOG_LEVEL())
        self.assertEqual(cf.free_memory(), cf.FREE_MEMORY())
        self.assertEqual(cf.free_memory_factor(), cf.FREE_MEMORY_FACTOR())
        self.assertEqual(cf.fm_threshold(), cf.FM_THRESHOLD())
        self.assertEqual(cf.total_memory(), cf.TOTAL_MEMORY())
        self.assertEqual(cf.regrid_logging(), cf.REGRID_LOGGING())
        self.assertEqual(cf.relaxed_identities(), cf.RELAXED_IDENTITIES())
        self.assertEqual(cf.tempdir(), cf.TEMPDIR())
        self.assertEqual(cf.chunksize(), cf.CHUNKSIZE())
        self.assertEqual(cf.set_performance(), cf.SET_PERFORMANCE())
        self.assertEqual(cf.of_fraction(), cf.OF_FRACTION())
        self.assertEqual(
            cf.collapse_parallel_mode(), cf.COLLAPSE_PARALLEL_MODE()
        )

    def test_configuration(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # This test assumes 'total_memory' remains constant throughout
        # the test run, which should be true generally in any
        # reasonable context.

        # Test getting of all config. and store original values to
        # test on:
        org = cf.configuration()
        self.assertIsInstance(org, dict)

        # Check all keys that should be there are, with correct value type:
        self.assertEqual(len(org), 11)  # update expected len if add new key(s)

        # Floats expected as values for most keys. Store these for
        # later as floats need assertAlmostEqual rather than
        # assertEqual tests:
        keys_with_float_values = [
            "atol",
            "rtol",
            "of_fraction",
            "free_memory_factor",
            "chunksize",
        ]
        for key in keys_with_float_values:
            self.assertIsInstance(org[key], float)

        # Other types expected:
        self.assertIsInstance(org["collapse_parallel_mode"], int)
        self.assertIsInstance(org["relaxed_identities"], bool)
        self.assertIsInstance(org["bounds_combination_mode"], str)
        self.assertIsInstance(org["regrid_logging"], bool)
        # Log level may be input as an int but always given as
        # equiv. string
        self.assertIsInstance(org["log_level"], str)
        self.assertIsInstance(org["tempdir"], str)

        # Store some sensible values to reset items to for testing, ensuring:
        # 1) they are kept different to the defaults (i.e. org values); and
        # 2) floats differ sufficiently that they will be picked up as
        #    qdifferent by the assertAlmostEqual decimal places (8, see
        #    below)
        reset_values = {
            "rtol": 5e-7,
            "atol": 2e-7,
            "tempdir": "/my-custom-tmpdir",
            "of_fraction": 0.1,
            "free_memory_factor": 0.25,
            "regrid_logging": True,
            "collapse_parallel_mode": 2,
            "relaxed_identities": True,
            "bounds_combination_mode": "XOR",
            "log_level": "INFO",
            "chunksize": 8e9,
        }

        # Test the setting of each lone item.
        expected_post_set = dict(org)  # copy for safety with mutable dict
        for setting, value in reset_values.items():
            cf.configuration(**{setting: value})
            post_set = cf.configuration()

            # Expect a dict that is identical to the original to start
            # with but as we set values incrementally they should be
            # reflected:
            expected_post_set[setting] = value

            # Can't trivially do a direct test that the actual and
            # expected return dicts are the same as there are float
            # values which have limited float precision so need
            # assertAlmostEqual testing:
            for name, val in expected_post_set.items():
                if isinstance(val, float):
                    self.assertAlmostEqual(
                        post_set[name], val, places=8, msg=setting
                    )
                else:
                    self.assertEqual(post_set[name], val)
        # --- End: for

        # Test the setting of more than one, but not all, items
        # simultaneously:
        new_values = {
            "regrid_logging": True,
            "tempdir": "/bin/bag",
            "of_fraction": 0.33,
        }
        cf.configuration(**new_values)
        post_set = cf.configuration()
        for name, val in new_values.items():  # test values that should change
            self.assertEqual(post_set[name], val)
        # ...and some values that should not:
        self.assertEqual(post_set["log_level"], "INFO")
        self.assertAlmostEqual(post_set["rtol"], 5e-7)

        # Test setting all possible items simultaneously (back to originals):
        cf.configuration(**org)
        post_set = cf.configuration()
        for name, val in org.items():
            if isinstance(val, float):
                self.assertAlmostEqual(post_set[name], val, places=8)
            else:
                self.assertEqual(post_set[name], val)
        # --- End: for

        # Test edge cases & invalid inputs...
        # ... 1. Falsy value inputs on some representative items:
        pre_set_config = cf.configuration()
        with self.assertRaises(ValueError):
            cf.configuration(of_fraction=0.0)
        with self.assertRaises(ValueError):
            cf.configuration(free_memory_factor=0.0)
        new_values = {
            "tempdir": "",
            "atol": 0.0,
            "regrid_logging": False,
        }
        cf.configuration(**new_values)
        post_set = cf.configuration()
        for name, val in new_values.items():  # test values that should change
            self.assertEqual(post_set[name], val)
        # ...and some values that should not:
        self.assertEqual(post_set["log_level"], pre_set_config["log_level"])
        self.assertAlmostEqual(post_set["rtol"], pre_set_config["rtol"])

        # 2. None as an input kwarg rather than as a default:
        pre_set_config = cf.configuration()
        set_of = 0.45
        cf.configuration(of_fraction=set_of, rtol=None, log_level=None)
        post_set = cf.configuration()
        # test values that should change
        self.assertEqual(post_set["of_fraction"], set_of)
        # ...and values that should not:
        self.assertEqual(post_set["rtol"], pre_set_config["rtol"])
        self.assertAlmostEqual(
            post_set["log_level"], pre_set_config["log_level"]
        )

        # 3. Gracefully error with invalid inputs:
        with self.assertRaises(ValueError):
            cf.configuration(of_fraction="bad")

        with self.assertRaises(ValueError):
            cf.configuration(log_level=7)

        # 4. Check invalid kwarg given logic processes **kwargs:
        with self.assertRaises(TypeError):
            cf.configuration(bad_kwarg=1e-15)

        old = cf.configuration()
        try:
            cf.configuration(atol=888, rtol=999, log_level="BAD")
        except ValueError:
            self.assertEqual(cf.configuration(), old)
        else:
            raise RuntimeError(
                "A ValueError should have been raised, but wasn't"
            )

        # Reset so later test fixtures don't spam with output
        # messages:
        cf.log_level("DISABLE")

    def test_context_managers(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # rtol, atol, chunksize
        for func in (
            cf.atol,
            cf.rtol,
            cf.chunksize,
            cf.free_memory_factor,
            cf.of_fraction,
        ):
            old = func()
            new = old * 1.001
            with func(new):
                self.assertEqual(func(), new)
                self.assertEqual(func(new * 1.001), new)
                self.assertEqual(func(), new * 1.001)

            self.assertEqual(func(), old)

        # collapse_parallel_mode
        func = cf.collapse_parallel_mode

        org = func(0)
        old = func()
        new = old + 1
        with func(new):
            self.assertEqual(func(), new)
            self.assertEqual(func(new + 1), new)
            self.assertEqual(func(), new + 1)

        self.assertEqual(func(), old)
        func(org)

        # log_level
        func = cf.log_level

        org = func("DETAIL")
        old = func()
        new = "DEBUG"
        with func(new):
            self.assertEqual(func(), new)

        self.assertEqual(func(), old)
        func(org)

        del org._func
        with self.assertRaises(AttributeError):
            with org:
                pass
        # --- End: with

        # bounds_combination_mode
        func = cf.bounds_combination_mode

        org = func("XOR")
        old = func()
        new = "AND"
        with func(new):
            self.assertEqual(func(), new)

        self.assertEqual(func(), old)
        func(org)

        del org._func
        with self.assertRaises(AttributeError):
            with org:
                pass
        # --- End: with

        # Full configuration
        func = cf.configuration

        org = func(rtol=cf.Constant(10), atol=20, log_level="WARNING")
        old = func()
        new = dict(rtol=cf.Constant(20), atol=40, log_level="DISABLE")

        with func(**new):
            self.assertEqual(cf.atol(), 40)

        self.assertEqual(func(), old)
        func(**org)

    def test_Constant(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        c = cf.atol()
        self.assertIs(c._func, cf.atol)

    def test_Configuration(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        c = cf.Configuration()
        self.assertIs(c._func, cf.configuration)

    def test_environment(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        e = cf.environment(display=False)
        ep = cf.environment(display=False, paths=False)
        self.assertIsInstance(e, str)
        self.assertIsInstance(ep, str)

        components = ["Platform: ", "udunits2 library: ", "numpy: ", "cfdm: "]
        for component in components:
            self.assertIn(component, e)
            self.assertIn(component, ep)
        for component in [
            "cf: {} {}".format(cf.__version__, os.path.abspath(cf.__file__)),
            "Python: {} {}".format(platform.python_version(), sys.executable),
        ]:
            self.assertIn(component, e)
            self.assertNotIn(component, ep)  # paths shouldn't be present here
        for component in [
            "cf: {}".format(cf.__version__),
            "Python: {}".format(platform.python_version()),
        ]:
            self.assertIn(component, ep)


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
