import datetime
import faulthandler
import unittest

import cftime
import dask.array as da
import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DataUtilsTest(unittest.TestCase):
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
