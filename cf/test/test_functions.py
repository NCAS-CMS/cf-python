import datetime
import faulthandler
import os
import platform
import sys
import unittest

import dask.array as da
import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class functionTest(unittest.TestCase):
    def setUp(self):
        self.test_only = ()

    def test_example_field(self):
        for f in cf.example_fields():
            f.dump(display=False)

        with self.assertRaises(ValueError):
            cf.example_field(-999)

    def test_keyword_deprecation(self):
        # Use as test case 'i' kwarg, the deprecated old name for
        # 'inplace':
        a = cf.Data([list(range(100))])
        a.squeeze(inplace=True)  # new way to specify operation tested below

        b = cf.Data([list(range(100))])
        with self.assertRaises(cf.functions.DeprecationError):
            b.squeeze(i=True)

    def test_aliases(self):
        self.assertEqual(cf.log_level(), cf.LOG_LEVEL())
        self.assertEqual(cf.free_memory(), cf.FREE_MEMORY())
        self.assertEqual(cf.total_memory(), cf.TOTAL_MEMORY())
        self.assertEqual(cf.regrid_logging(), cf.REGRID_LOGGING())
        self.assertEqual(cf.relaxed_identities(), cf.RELAXED_IDENTITIES())
        self.assertEqual(cf.tempdir(), cf.TEMPDIR())
        self.assertEqual(cf.chunksize(), cf.CHUNKSIZE())

    def test_configuration(self):
        # This test assumes 'total_memory' remains constant throughout
        # the test run, which should be true generally in any
        # reasonable context.

        # Test getting of all config. and store original values to
        # test on:
        org = cf.configuration()
        self.assertIsInstance(org, dict)

        # Check all keys that should be there are, with correct value type:
        self.assertEqual(len(org), 8)  # update expected len if add new key(s)

        # Types expected:
        self.assertIsInstance(org["atol"], float)
        self.assertIsInstance(org["rtol"], float)
        self.assertIsInstance(org["chunksize"], int)
        self.assertIsInstance(org["relaxed_identities"], bool)
        self.assertIsInstance(org["bounds_combination_mode"], str)
        self.assertIsInstance(org["regrid_logging"], bool)
        self.assertIsInstance(org["tempdir"], str)
        # Log level may be input as an int but always given as
        # equiv. string
        self.assertIsInstance(org["log_level"], str)

        # Store some sensible values to reset items to for testing, ensuring:
        # 1) they are kept different to the defaults (i.e. org values); and
        # 2) floats differ sufficiently that they will be picked up as
        #    qdifferent by the assertAlmostEqual decimal places (8, see
        #    below)
        reset_values = {
            "rtol": 5e-7,
            "atol": 2e-7,
            "tempdir": "/my-custom-tmpdir",
            #            "free_memory_factor": 0.25,
            "regrid_logging": True,
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

        # Test the setting of more than one, but not all, items
        # simultaneously:
        new_values = {
            "regrid_logging": True,
            "tempdir": "/bin/bag",
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

        # Test edge cases & invalid inputs...
        # ... 1. Falsy value inputs on some representative items:
        pre_set_config = cf.configuration()
        #        with self.assertRaises(ValueError):
        #            cf.configuration(free_memory_factor=0.0)
        new_values = {"tempdir": "", "atol": 0.0, "regrid_logging": False}
        cf.configuration(**new_values)
        post_set = cf.configuration()
        for name, val in new_values.items():  # test values that should change
            self.assertEqual(post_set[name], val)
        # ...and some values that should not:
        self.assertEqual(post_set["log_level"], pre_set_config["log_level"])
        self.assertAlmostEqual(post_set["rtol"], pre_set_config["rtol"])

        # 2. None as an input kwarg rather than as a default:
        pre_set_config = cf.configuration()
        set_atol = 0.45
        cf.configuration(atol=set_atol, rtol=None, log_level=None)
        post_set = cf.configuration()
        # test values that should change
        self.assertEqual(post_set["atol"], set_atol)
        # ...and values that should not:
        self.assertEqual(post_set["rtol"], pre_set_config["rtol"])
        self.assertAlmostEqual(
            post_set["log_level"], pre_set_config["log_level"]
        )

        # 3. Gracefully error with invalid inputs:
        with self.assertRaises(ValueError):
            cf.configuration(atol="bad")

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
        # rtol, atol
        for func in (cf.atol, cf.rtol):
            old = func()
            new = old * 1.001
            with func(new):
                self.assertEqual(func(), new)
                self.assertEqual(func(new * 1.001), new)
                self.assertEqual(func(), new * 1.001)

            self.assertEqual(func(), old)

        # chunksize
        func = cf.chunksize

        org = func(1000)
        old = func()
        new = 2000.123
        with func(new):
            self.assertEqual(func(), int(new))

        self.assertEqual(func(), old)
        func(org)

        del org._func
        with self.assertRaises(AttributeError):
            with org:
                pass

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
        c = cf.atol()
        self.assertIs(c._func, cf.atol)

    def test_Configuration(self):
        c = cf.Configuration()
        self.assertIs(c._func, cf.configuration)

    def test_environment(self):
        e = cf.environment(display=False)
        ep = cf.environment(display=False, paths=False)
        self.assertIsInstance(e, str)
        self.assertIsInstance(ep, str)

        components = ["Platform: ", "udunits2 library: ", "numpy: ", "cfdm: "]
        for component in components:
            self.assertIn(component, e)
            self.assertIn(component, ep)
        for component in [
            f"cf: {cf.__version__} {os.path.abspath(cf.__file__)}",
            f"Python: {platform.python_version()} {sys.executable}",
        ]:
            self.assertIn(component, e)
            self.assertNotIn(component, ep)  # paths shouldn't be present here
        for component in [
            f"cf: {cf.__version__}",
            f"Python: {platform.python_version()}",
        ]:
            self.assertIn(component, ep)

    def test_indices_shape(self):
        import dask.array as da

        shape = (10, 20)

        self.assertEqual(cf.indices_shape((slice(2, 5), 4), shape), [3, 1])
        self.assertEqual(
            cf.indices_shape(([2, 3, 4], np.arange(1, 6)), shape), [3, 5]
        )

        index0 = [False] * 5
        index0[2:5] = [True] * 3
        self.assertEqual(
            cf.indices_shape((index0, da.arange(1, 6)), shape), [3, 5]
        )

        index0 = da.full((5,), False, dtype=bool)
        index0[2:5] = True
        index1 = np.full((6,), False, dtype=bool)
        index1[1:6] = True
        self.assertEqual(cf.indices_shape((index0, index1), shape), [3, 5])

        index0 = da.arange(5)
        index0 = index0[index0 < 3]
        self.assertEqual(cf.indices_shape((index0, []), shape), [3, 0])

        self.assertEqual(
            cf.indices_shape((da.from_array(2), np.array(3)), shape), [1, 1]
        )
        self.assertEqual(
            cf.indices_shape((da.from_array([]), np.array(())), shape), [0, 0]
        )

        self.assertEqual(cf.indices_shape((slice(1, 5, 3), 3), shape), [2, 1])
        self.assertEqual(cf.indices_shape((slice(5, 1, -2), 3), shape), [2, 1])
        self.assertEqual(cf.indices_shape((slice(5, 1, 3), 3), shape), [0, 1])
        self.assertEqual(cf.indices_shape((slice(1, 5, -3), 3), shape), [0, 1])

        # keepdims=False
        self.assertEqual(
            cf.indices_shape((slice(2, 5), 4), shape, keepdims=False), [3]
        )
        self.assertEqual(
            cf.indices_shape(
                (da.from_array(2), np.array(3)), shape, keepdims=False
            ),
            [],
        )
        self.assertEqual(cf.indices_shape((2, 3), shape, keepdims=False), [])

    def test_size(self):
        self.assertEqual(cf.size(9), 1)
        self.assertEqual(cf.size("foobar"), 1)
        self.assertEqual(cf.size([9]), 1)
        self.assertEqual(cf.size((8, 9)), 2)

        x = np.arange(9)
        self.assertEqual(cf.size(x), x.size)

        x = da.arange(9)
        self.assertEqual(cf.size(x), x.size)

    def test_CFA(self):
        self.assertEqual(cf.CFA(), cf.__cfa_version__)

    def test_normalize_slice(self):
        self.assertEqual(cf.normalize_slice(slice(1, 4), 8), slice(1, 4, 1))
        self.assertEqual(cf.normalize_slice(slice(None), 8), slice(0, 8, 1))
        self.assertEqual(
            cf.normalize_slice(slice(6, None, -1), 8), slice(6, None, -1)
        )
        self.assertEqual(cf.normalize_slice(slice(-2, 4), 8), slice(6, 4, 1))

        # Cyclic slices
        self.assertEqual(
            cf.normalize_slice(slice(-2, 3), 8, cyclic=True), slice(-2, 3, 1)
        )
        self.assertEqual(
            cf.normalize_slice(slice(6, 3), 8, cyclic=True), slice(-2, 3, 1)
        )
        self.assertEqual(
            cf.normalize_slice(slice(6, 3, 2), 8, cyclic=True), slice(-2, 3, 2)
        )

        self.assertEqual(
            cf.normalize_slice(slice(2, -3, -1), 8, cyclic=True),
            slice(2, -3, -1),
        )
        self.assertEqual(
            cf.normalize_slice(slice(2, 5, -1), 8, cyclic=True),
            slice(2, -3, -1),
        )
        self.assertEqual(
            cf.normalize_slice(slice(2, 5, -2), 8, cyclic=True),
            slice(2, -3, -2),
        )

        with self.assertRaises(IndexError):
            cf.normalize_slice([1, 2], 8)

        for index in (
            slice(1, 6),
            slice(6, 1, -1),
            slice(None, 4, None),
            slice(1, 6, 0),
            [1, 2],
            5,
        ):
            with self.assertRaises(IndexError):
                cf.normalize_slice(index, 8, cyclic=True)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
