import datetime
import unittest

import numpy
import cftime

import cf

from cf import Units


class DatetimeTest(unittest.TestCase):
    def test_Datetime(self):
        d = cf.dt(2003)
        d = cf.dt(2003, 2)
        d = cf.dt(2003, 2, 30, calendar="360_day")
        d = cf.dt(2003, 2, 30, 0, 0, calendar="360_day")
        d = cf.dt(2003, 2, 30, 0, 0, 0, calendar="360_day")
        d = cf.dt(2003, 4, 5, 12, 30, 15)
        d = cf.dt(2003, month=4, day=5, hour=12, minute=30, second=15)

    def test_Datetime_rt2dt(self):
        self.assertEqual(
            cf.cfdatetime.rt2dt(1, Units("days since 2004-2-28")),
            numpy.array(cf.dt(2004, 2, 29, calendar="standard"), dtype="O"),
        )
        self.assertTrue(
            (
                cf.cfdatetime.rt2dt([1, 3], Units("days since 2004-2-28"))
                == numpy.array(
                    [
                        datetime.datetime(2004, 2, 29),
                        datetime.datetime(2004, 3, 2),
                    ]
                )
            ).all()
        )
        a = numpy.array(
            [
                cf.dt(2004, 2, 29, calendar=None),
                cf.dt(2004, 3, 2, calendar="gregorian"),
            ],
            dtype="O",
        )
        b = cf.cfdatetime.rt2dt([1, 3], Units("days since 2004-2-28"))
        self.assertTrue((a == b).all())

    def test_Datetime_dt2rt(self):
        units = Units("days since 2004-2-28")
        self.assertEqual(
            cf.cfdatetime.dt2rt(datetime.datetime(2004, 2, 29), None, units),
            numpy.array(1.0),
        )
        self.assertTrue(
            (
                cf.cfdatetime.dt2rt(
                    [
                        datetime.datetime(2004, 2, 29),
                        datetime.datetime(2004, 3, 2),
                    ],
                    None,
                    units,
                )
                == numpy.array([1.0, 3.0])
            ).all()
        )
        units = Units("days since 2004-2-28", "360_day")
        self.assertTrue(
            (
                cf.cfdatetime.dt2rt(
                    [cf.dt(2004, 2, 29), cf.dt(2004, 3, 1)], None, units
                )
                == numpy.array([1.0, 3.0])
            ).all()
        )
        units = Units("seconds since 2004-2-28")
        self.assertEqual(
            cf.cfdatetime.dt2rt(datetime.datetime(2004, 2, 29), None, units),
            numpy.array(86400.0),
        )

    def test_Datetime_Data(self):
        d = cf.Data([1, 2, 3], "days since 2004-02-28")
        self.assertTrue((d < cf.dt(2005, 2, 28)).all())
        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 29)

        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 29, calendar="360_day")

        d = cf.Data([1, 2, 3], "days since 2004-02-28", calendar="360_day")
        self.assertTrue((d < cf.dt(2005, 2, 28)).all())
        self.assertTrue((d < cf.dt(2005, 2, 29)).all())
        self.assertTrue((d < cf.dt(2005, 2, 30)).all())

        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 31)

        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 29, calendar="noleap")

    def test_Datetime_dt_vector(self):
        for v in (2000, [2000], [[2000]], "2000-01-1", ["2000-01-1"]):
            x = cf.dt_vector(v)
            self.assertIsInstance(x, numpy.ndarray)
            self.assertEqual(x[0], cf.dt(2000, 1, 1))

        for v in ([2000, 2001], [[2000], [2001]]):
            x = cf.dt_vector(v)
            self.assertIsInstance(x, numpy.ndarray)
            self.assertEqual(
                x.tolist(), [cf.dt(2000, 1, 1), cf.dt(2001, 1, 1)]
            )

        for v in ([[2000, 1], [2001, 2]], ["2000-01-1", "2001-02-1"]):
            x = cf.dt_vector(v)
            self.assertIsInstance(x, numpy.ndarray)
            self.assertEqual(
                x.tolist(), [cf.dt(2000, 1, 1), cf.dt(2001, 2, 1)]
            )


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
