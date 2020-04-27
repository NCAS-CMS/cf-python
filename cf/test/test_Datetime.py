import datetime
import os
import time
import unittest

import numpy

import cf
# from cfunits import Units
from cf import Units


class DatetimeTest(unittest.TestCase):
    def test_Datetime(self):
        # d = cf.Datetime(2003)
        # d = cf.Datetime(2003, 2)
        # d = cf.Datetime(2003, 2, 30)
        # d = cf.Datetime(2003, 2, 30, 0, 0)
        # d = cf.Datetime(2003, 2, 30, 0, 0, 0)
        # d = cf.Datetime(2003, 4, 5, 12, 30, 15)
        # d = cf.Datetime(
        #     year=2003, month=4, day=5, hour=12, minute=30, second=15)

        d = cf.dt(2003)
        d = cf.dt(2003, 2)
        d = cf.dt(2003, 2, 30, calendar='360_day')
        d = cf.dt(2003, 2, 30, 0, 0, calendar='360_day')
        d = cf.dt(2003, 2, 30, 0, 0, 0, calendar='360_day')
        d = cf.dt(2003, 4, 5, 12, 30, 15)
        d = cf.dt(2003, month=4, day=5, hour=12, minute=30, second=15)

# self.assertTrue(
#     (d.year, d.month, d.day, d.hour, d.minute, d.second) ==
#     (2003, 4, 5, 12, 30, 15)
# )
# self.assertTrue(d.timetuple() == (2003, 4, 5, 12, 30, 15, -1, 1, -1))
# self.assertTrue( d == d)
# self.assertFalse(d >  d)
# self.assertTrue( d >= d)
# self.assertFalse(d <  d)
# self.assertTrue( d <= d)
# self.assertFalse(d != d)
# e = cf.Datetime(2003, 4, 5, 12, 30, 16)
# self.assertFalse(d == e)
# self.assertFalse(d >  e)
# self.assertFalse(d >= e)
# self.assertTrue( d <  e)
# self.assertTrue( d <= e)
# self.assertTrue( d != e)
# e = cf.Datetime(2003, 4, 5, 12, 30, 14)
# self.assertFalse(d == e)
# self.assertTrue( d >  e)
# self.assertTrue( d >= e)
# self.assertFalse(d <  e)
# self.assertFalse(d <= e)
# self.assertTrue( d != e)
#
# d = cf.Datetime(
#     year=2003, month=4, day=5, hour=12, minute=30, second=15,
#     microsecond=12
# )
# self.assertTrue( d == d)
# self.assertFalse(d >  d)
# self.assertTrue( d >= d)
# self.assertFalse(d <  d)
# self.assertTrue( d <= d)
# self.assertFalse(d != d)
# e = cf.Datetime(
#     year=2003, month=4, day=5, hour=12, minute=30, second=15,
#     microsecond=11
# )
# self.assertFalse(e == d)
# self.assertFalse(e >  d)
# self.assertFalse(e >= d)
# self.assertTrue( e <  d)
# self.assertTrue( e <= d)
# self.assertTrue( e != d)
# e = cf.Datetime(
#     year=2003, month=4, day=5, hour=12, minute=30, second=15,
#     microsecond=13
# )
# self.assertFalse(e == d)
# self.assertTrue( e >  d)
# self.assertTrue( e >= d)
# self.assertFalse(e <  d)
# self.assertFalse(e <= d)
# self.assertTrue( e != d)

# def test_Datetime_utcnow(self):
#     d = cf.Datetime.utcnow()

#
# def test_Datetime_copy(self):
#     d = cf.Datetime.utcnow()
#     self.assertTrue(d.equals(d.copy()))

#
# def test_Datetime_equals(self):
#     d = datetime.datetime.utcnow()
#     e = cf.dt(d)
#     d = cf.dt(d)
#     print(repr(d), repr(e))
#     self.assertTrue(e == d)
#     self.assertTrue(e.equals(e))
#     self.assertTrue(e.equals(d))

# def test_Datetime_replace(self):
#     d = cf.Datetime(1999, 4, 5, 12, 30, 15, 987654, calendar='360_day')
#     e = d.replace(1787)
#     self.assertTrue(e.equals(
#         cf.Datetime(1787, 4, 5, 12, 30, 15, 987654, calendar='360_day')))
#     e = d.replace(1787, 12, 3, 6, 23, 45, 2345, 'noleap')
#     self.assertTrue(e.equals(
#         cf.Datetime(1787, 12, 3, 6, 23, 45, 2345, calendar='noleap')))

    def test_Datetime_rt2dt(self):
        self.assertTrue(
            cf.cfdatetime.rt2dt(1, Units('days since 2004-2-28')) ==
            numpy.array(cf.dt(2004, 2, 29, calendar='standard'), dtype='O'))
#        numpy.array(datetime.datetime(2004, 2, 29)))
        self.assertTrue((
            cf.cfdatetime.rt2dt(
                [1, 3], Units('days since 2004-2-28')) ==
            numpy.array(
                [datetime.datetime(2004, 2, 29),
                 datetime.datetime(2004, 3, 2)]
             )
        ).all())
        a = numpy.array(
            [cf.dt(2004, 2, 29, calendar=None),
             cf.dt(2004, 3, 2, calendar='gregorian')],
            dtype='O'
        )
        b = cf.cfdatetime.rt2dt([1, 3], Units('days since 2004-2-28'))
        self.assertTrue((a == b).all())

    def test_Datetime_dt2rt(self):
        units = Units('days since 2004-2-28')
        self.assertTrue(
            cf.cfdatetime.dt2rt(datetime.datetime(2004, 2, 29), None, units) ==
            numpy.array(1.0)
        )
        self.assertTrue((
            cf.cfdatetime.dt2rt(
                [datetime.datetime(2004, 2, 29),
                 datetime.datetime(2004, 3, 2)],
                None, units) ==
            numpy.array([1., 3.])
        ).all())
        units = Units('days since 2004-2-28', '360_day')
        self.assertTrue((
            cf.cfdatetime.dt2rt(
                [cf.dt(2004, 2, 29), cf.dt(2004, 3, 1)], None, units) ==
            numpy.array([1., 3.])
        ).all())
        units = Units('seconds since 2004-2-28')
        self.assertTrue(
            cf.cfdatetime.dt2rt(datetime.datetime(2004, 2, 29), None, units) ==
            numpy.array(86400.0)
        )

    def test_Datetime_Data(self):
        d = cf.Data([1, 2, 3], 'days since 2004-02-28')
        self.assertTrue((d < cf.dt(2005, 2, 28)).all())
        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 29)

        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 29, calendar='360_day')

        d = cf.Data([1, 2, 3], 'days since 2004-02-28', calendar='360_day')
        self.assertTrue((d < cf.dt(2005, 2, 28)).all())
        self.assertTrue((d < cf.dt(2005, 2, 29)).all())
        self.assertTrue((d < cf.dt(2005, 2, 30)).all())

        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 31)

        with self.assertRaises(Exception):
            d < cf.dt(2005, 2, 29, calendar='noleap')


# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
