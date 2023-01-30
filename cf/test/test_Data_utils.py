import datetime
import faulthandler
import unittest

import cftime
import dask.array as da
import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DataUtilsTest(unittest.TestCase):
    def test_Data_Utils__da_ma_allclose(self):
        """TODO."""
        # Create a range of inputs to test against.
        # Note that 'a' and 'a2' should be treated as 'allclose' for this
        # method, the same result as np.ma.allclose would give because all
        # of the *unmasked* elements are 'allclose', whereas in our
        # Data.equals method that builds on this method, we go even further
        # and insist on the mask being identical as well as the data
        # (separately, i.e. unmasked) all being 'allclose', so inside our
        # cf.Data objects 'a' and 'a2' would instead *not* be considered equal.
        a_np = np.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
        a = da.from_array(a_np)
        a2 = da.from_array(np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0]))
        b_np = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
        b = da.from_array(b_np)
        c_np = np.ma.array([1.0, 2.0, 100.0], mask=[1, 0, 0])
        c = da.from_array(c_np)
        d = da.from_array(np.array([1.0, 2.0, 3.0]))
        e = a + 5e-04  # outside of tolerance to set, namely rtol=1e-05
        f = a + 5e-06  # within set tolerance to be specified, as above

        # Test the function with these inputs as both numpy and dask arrays...
        allclose = cf.data.dask_utils._da_ma_allclose

        self.assertTrue(allclose(a, a).compute())
        self.assertTrue(allclose(a2, a).compute())
        self.assertTrue(allclose(b, a).compute())

        # ...including testing the 'masked_equal' parameter
        self.assertFalse(allclose(b, a, masked_equal=False).compute())

        self.assertFalse(allclose(c, a).compute())
        self.assertTrue(allclose(d, a).compute())
        self.assertFalse(allclose(e, a).compute())

        self.assertTrue(allclose(f, a, rtol=1e-05).compute())

        # Test when array inputs have different chunk sizes
        a_chunked = da.from_array(a_np, chunks=(1, 2))
        self.assertTrue(
            allclose(da.from_array(b_np, chunks=(3,)), a_chunked).compute()
        )
        self.assertFalse(
            allclose(
                da.from_array(b_np, chunks=(3,)), a_chunked, masked_equal=False
            ).compute()
        )
        self.assertFalse(
            allclose(da.from_array(c_np, chunks=(3,)), a_chunked).compute()
        )

        # Test the 'rtol' and 'atol' parameters:
        self.assertFalse(allclose(e, a, rtol=1e-06).compute())
        b1 = e / 10000
        b2 = a / 10000
        self.assertTrue(allclose(b1, b2, atol=1e-05).compute())

    def test_Data_Utils_is_numeric_dtype(self):
        """TODO."""
        is_numeric_dtype = cf.data.utils.is_numeric_dtype
        for a in [
            np.array([0, 1, 2]),
            np.array([False, True, True]),
            np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0]),
            np.array(10),
        ]:
            self.assertTrue(is_numeric_dtype(a))

        for b in [
            np.array(["a", "b", "c"], dtype="S1"),
            np.empty(1, dtype=object),
        ]:
            self.assertFalse(is_numeric_dtype(b))

    def test_Data_Utils_convert_to_datetime(self):
        """TODO."""
        a = cftime.DatetimeGregorian(2000, 12, 3, 12)
        for x in (2.5, [2.5]):
            d = da.from_array(x)
            e = cf.data.utils.convert_to_datetime(
                d, cf.Units("days since 2000-12-01")
            )
            self.assertEqual(e.compute(), a)

        a = [
            cftime.DatetimeGregorian(2000, 12, 1),
            cftime.DatetimeGregorian(2000, 12, 2),
            cftime.DatetimeGregorian(2000, 12, 3),
        ]
        for x in ([0, 1, 2], [[0, 1, 2]]):
            d = da.from_array([0, 1, 2], chunks=2)
            e = cf.data.utils.convert_to_datetime(
                d, cf.Units("days since 2000-12-01")
            )
            self.assertTrue((e.compute() == a).all())

    def test_Data_Utils_convert_to_reftime(self):
        """TODO."""
        a = cftime.DatetimeGregorian(2000, 12, 3, 12)
        d = da.from_array(np.array(a, dtype=object))

        e, u = cf.data.utils.convert_to_reftime(d)
        self.assertEqual(e.compute(), 0.5)
        self.assertEqual(u, cf.Units("days since 2000-12-03", "standard"))

        units = cf.Units("days since 2000-12-01")
        e, u = cf.data.utils.convert_to_reftime(d, units=units)
        self.assertEqual(e.compute(), 2.5)
        self.assertEqual(u, units)

        a = "2000-12-03T12:00"
        d = da.from_array(np.array(a, dtype=str))

        e, u = cf.data.utils.convert_to_reftime(d)
        self.assertEqual(e.compute(), 0.5)
        self.assertEqual(u, cf.Units("days since 2000-12-03", "standard"))

        units = cf.Units("days since 2000-12-01")
        e, u = cf.data.utils.convert_to_reftime(d, units=units)
        self.assertEqual(e.compute(), 2.5)
        self.assertEqual(u, units)

        a = [
            [
                cftime.DatetimeGregorian(2000, 12, 1),
                cftime.DatetimeGregorian(2000, 12, 2),
                cftime.DatetimeGregorian(2000, 12, 3),
            ]
        ]
        d = da.from_array(np.ma.array(a, mask=[[1, 0, 0]]), chunks=2)

        e, u = cf.data.utils.convert_to_reftime(d)
        self.assertTrue((e.compute() == [-99, 0, 1]).all())
        self.assertEqual(u, cf.Units("days since 2000-12-02", "standard"))

        units = cf.Units("days since 2000-12-03")
        e, u = cf.data.utils.convert_to_reftime(d, units=units)
        self.assertTrue((e.compute() == [-99, -1, 0]).all())
        self.assertEqual(u, units)

        d = cf.Data(
            ["2004-02-29", "2004-02-30", "2004-03-01"], calendar="360_day"
        )
        self.assertEqual(d.Units, cf.Units("days since 2004-02-29", "360_day"))
        self.assertTrue((d.array == [0, 1, 2]).all())

        d = cf.Data(["2004-02-29", "2004-03-01"], dt=True)
        self.assertEqual(d.Units, cf.Units("days since 2004-02-29"))
        self.assertTrue((d.array == [0, 1]).all())

    def test_Data_Utils_unique_calendars(self):
        """TODO."""
        a = [
            [
                cftime.DatetimeGregorian(2000, 12, 1),
                cftime.DatetimeGregorian(2000, 12, 2),
                cftime.DatetimeGregorian(2000, 12, 3),
            ]
        ]
        d = da.from_array(np.ma.array(a, mask=[[1, 0, 0]]), chunks=2)
        c = cf.data.utils.unique_calendars(d)
        self.assertIsInstance(c, set)
        self.assertEqual(c, set(["standard"]))

        a = cftime.DatetimeGregorian(2000, 12, 1)
        d = da.from_array(np.array(a, dtype=object))
        c = cf.data.utils.unique_calendars(d)
        self.assertEqual(c, set(["standard"]))

        # ------------------------------------------------------------
        # TODO: re-instate when dask has fixed this:
        #       https://github.com/dask/dask/pull/9627
        # ------------------------------------------------------------
        # d[()] = np.ma.masked
        # c = cf.data.utils.unique_calendars(d)
        # self.assertEqual(c, set())

        a = [
            cftime.DatetimeGregorian(2000, 12, 1),
            cftime.DatetimeAllLeap(2000, 12, 2),
            cftime.DatetimeGregorian(2000, 12, 3),
        ]
        d = da.from_array(np.ma.array(a, mask=[1, 0, 0]), chunks=2)
        c = cf.data.utils.unique_calendars(d)
        self.assertEqual(c, set(["all_leap", "standard"]))

    def test_Data_Utils_first_non_missing_value(self):
        """TODO."""
        for method in ("index", "mask"):
            # Scalar data
            d = da.from_array(0)
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), 0
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

            d[()] = np.ma.masked
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), None
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

            # 1-d data
            d = da.arange(8)
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), 0
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

            d[0] = np.ma.masked
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), 1
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

            # 2-d data
            d = da.arange(8).reshape(2, 4)
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), 0
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

            d[0] = np.ma.masked
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), 4
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

            d[...] = np.ma.masked
            self.assertEqual(
                cf.data.utils.first_non_missing_value(d, method=method), None
            )
            self.assertEqual(
                cf.data.utils.first_non_missing_value(
                    d, cached=99, method=method
                ),
                99,
            )

        # Bad method
        with self.assertRaises(ValueError):
            cf.data.utils.first_non_missing_value(d, method="bad")

    def test_Data_Utils_conform_units(self):
        for x in (1, [1, 2], "foo", np.array([[1]])):
            self.assertEqual(cf.data.utils.conform_units(x, cf.Units("m")), x)

        d = cf.Data([1000, 2000], "m")
        e = cf.data.utils.conform_units(d, cf.Units("m"))
        self.assertIs(e, d)
        e = cf.data.utils.conform_units(d, cf.Units("km"))
        self.assertTrue(e.equals(cf.Data([1, 2], "km"), ignore_data_type=True))

        with self.assertRaises(ValueError):
            cf.data.utils.conform_units(d, cf.Units("s"))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
