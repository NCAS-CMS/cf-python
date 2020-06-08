import copy
import datetime
import os
import time
import unittest

import numpy

import cf


class TimeDurationTest(unittest.TestCase):
    def test_TimeDuration(self):
        self.assertTrue(
            cf.TimeDuration(2, 'calendar_years') >
            cf.TimeDuration(1, 'calendar_years')
        )
        self.assertTrue(
            cf.TimeDuration(2, 'calendar_years') <
            cf.TimeDuration(25, 'calendar_months')
        )
        self.assertTrue(
            cf.TimeDuration(2, 'hours') <=
            cf.TimeDuration(1, 'days')
        )
        self.assertTrue(
            cf.TimeDuration(2, 'hours') ==
            cf.TimeDuration(1/12.0, 'days')
        )
        self.assertTrue(
            cf.TimeDuration(2, 'days') ==
            cf.TimeDuration(48, 'hours')
        )
        self.assertTrue(cf.TimeDuration(2, 'days') == cf.Data(2))
        self.assertTrue(cf.TimeDuration(2, 'days') == cf.Data([2.], 'days'))
        self.assertTrue(
            cf.TimeDuration(2, 'days') > cf.Data([[60]], 'seconds'))
        self.assertTrue(cf.TimeDuration(2, 'hours') <= 2)
        self.assertTrue(cf.TimeDuration(0.1, units='seconds') == 0.1)
        self.assertTrue(cf.TimeDuration(2, 'days') != 30.5)
        self.assertTrue(
            cf.TimeDuration(2, 'calendar_years') > numpy.array(1.5))
        self.assertTrue(
            cf.TimeDuration(2, 'calendar_months') < numpy.array([[12]]))

        self.assertFalse(
            cf.TimeDuration(2, 'calendar_years') <=
            cf.TimeDuration(1, 'calendar_years')
        )
        self.assertFalse(
            cf.TimeDuration(2, 'calendar_years') >=
            cf.TimeDuration(25, 'calendar_months')
        )
        self.assertFalse(
            cf.TimeDuration(2, 'hours') > cf.TimeDuration(1, 'days'))
        self.assertFalse(
            cf.TimeDuration(2, 'hours') != cf.TimeDuration(1/12.0, 'days'))
        self.assertFalse(
            cf.TimeDuration(2, 'days') != cf.TimeDuration(48, 'hours'))
        self.assertFalse(cf.TimeDuration(2, 'days') != cf.Data(2))
#        self.assertFalse(cf.TimeDuration(2, 'days') <= cf.Data(1.5, ''))
#        self.assertFalse(cf.TimeDuration(2, 'days') <= cf.Data(1.5, '1'))
#        self.assertFalse(cf.TimeDuration(2, 'days') >= cf.Data(0.03, '100'))
        self.assertFalse(cf.TimeDuration(2, 'days') != cf.Data([2.], 'days'))
        self.assertFalse(
            cf.TimeDuration(2, 'days') <= cf.Data([[60]], 'seconds'))
        self.assertFalse(cf.TimeDuration(2, 'hours') > 2)
        self.assertFalse(cf.TimeDuration(2, 'days') == 30.5)
        self.assertFalse(
            cf.TimeDuration(2, 'calendar_years') <= numpy.array(1.5))
        self.assertFalse(
            cf.TimeDuration(2, 'calendar_months') >= numpy.array([[12]]))

        self.assertTrue(cf.TimeDuration(64, 'calendar_years') + 2 == cf.Y(66))
        self.assertTrue(
            cf.TimeDuration(64, 'calendar_years') - 2.5 == cf.Y(61.5))
        self.assertTrue(
            cf.M(23) + cf.TimeDuration(64, 'calendar_years') == cf.M(791))
        self.assertTrue(
            cf.TimeDuration(64, 'calendar_years') + cf.M(24) == cf.Y(66))

        self.assertTrue(
            cf.TimeDuration(36, 'calendar_months') / 8 == cf.M(4.5))
        self.assertTrue(
            cf.TimeDuration(36, 'calendar_months') // 8 == cf.M(4))

        self.assertTrue(
            cf.TimeDuration(36, 'calendar_months') / numpy.array(8.0) ==
            cf.M(36/8.0)
        )
        self.assertTrue(
            cf.TimeDuration(12, 'calendar_months') * cf.Data([[1.5]]) ==
            cf.Y(1.5)
        )
        self.assertTrue(
            cf.TimeDuration(36, 'calendar_months') // 8.25 == cf.M(4.0))
        self.assertTrue(
            cf.TimeDuration(36, 'calendar_months') % 10 == cf.M(6))

        self.assertTrue(
            cf.TimeDuration(24, 'hours') + cf.TimeDuration(0.5, 'days') ==
            cf.h(36.0)
        )
        self.assertTrue(
            cf.TimeDuration(0.5, 'days') + cf.TimeDuration(24, 'hours') ==
            cf.D(1.5)
        )

        t = cf.TimeDuration(24, 'hours')
        t += 2
        self.assertTrue(t == cf.h(26))
        t -= cf.Data(3, 'hours')
        self.assertTrue(t == cf.h(23))
        t = cf.TimeDuration(24.0, 'hours')
        t += 2
        self.assertTrue(t == cf.h(26))
        self.assertTrue(t - cf.Data(2.5, 'hours') == cf.h(23.5))
        t *= 2
        self.assertTrue(t == cf.h(52.0))
        t -= 1.0
        self.assertTrue(t == cf.h(51))
        t /= 3
        self.assertTrue(t == cf.h(17))
        t += 5.5
        self.assertTrue(t == cf.h(22.5))
        t //= numpy.array(2)
        self.assertTrue(t == cf.h(11.0))
        t *= 10
        self.assertTrue(t == cf.h(110.0))
        t %= 3
        self.assertTrue(t == cf.h(2.0))

        t = cf.TimeDuration(24.5, 'hours')
        self.assertTrue(-t == -24.5)
        self.assertTrue(int(t) == 24)
        self.assertTrue(t/0.5 == 49)
        self.assertTrue(t//2 == 12)
        self.assertTrue(25 - t == 0.5)
        self.assertTrue(2 * t == 49)
        self.assertTrue(2.0 % t == 2, 2.0 % t)

        self.assertTrue(cf.TimeDuration(24, 'hours').isint)
        self.assertTrue(cf.TimeDuration(24.0, 'hours').isint)
        self.assertFalse(t.isint)

        t.Units = 'days'
        self.assertTrue(t.Units == cf.Units('days'))
        t.Units = 'hours'

        self.assertTrue(cf.TimeDuration(12, 'hours').is_day_factor())
        self.assertFalse(cf.TimeDuration(13, 'hours').is_day_factor())
        self.assertFalse(cf.TimeDuration(2, 'days').is_day_factor())

        self.assertTrue(cf.TimeDuration(cf.Data(2, 'days')) == 2)
        self.assertTrue(cf.TimeDuration(cf.Data(48, 'hours')) == 48)
        self.assertTrue(
            cf.TimeDuration(cf.Data(48, 'hours'), units='days') == 2)
        self.assertTrue(cf.TimeDuration(0.1, units='seconds') == 0.1)

        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

        self.assertTrue(t.equivalent(t, verbose=2))
        self.assertTrue(t.equivalent(t.copy(), verbose=2))

        with self.assertRaises(Exception):
            t = cf.TimeDuration(48, 'm')
        with self.assertRaises(Exception):
            t = cf.TimeDuration(cf.Data(48, 'm'))
        with self.assertRaises(Exception):
            t = cf.TimeDuration(cf.Data(48, 'days'), units='m')

        t = t.copy()
        t = copy.deepcopy(t)

        _ = repr(t)
        _ = str(t)

        t //= 2
        t %= 2

    def test_TimeDuration_interval(self):
        self.assertTrue(
            cf.M().interval(cf.dt(1999, 12)) ==
            (cf.dt('1999-12-01 00:00:00', calendar=None),
             cf.dt('2000-01-01 00:00:00', calendar=None))
        )
        self.assertTrue(
            cf.Y(2).interval(cf.dt(2000, 2), end=True) ==
            (cf.dt('1998-02-01 00:00:00', calendar=None),
             cf.dt('2000-02-01 00:00:00', calendar=None))
        )
        self.assertTrue(
            cf.D(30).interval(cf.dt(1983, 12, 1, 6)) ==
            (cf.dt('1983-12-01 06:00:00', calendar=None),
             cf.dt('1983-12-31 06:00:00', calendar=None))
        )
        self.assertTrue(
            cf.D(30).interval(cf.dt(1983, 12, 1, 6), end=True) ==
            (cf.dt('1983-11-01 06:00:00', calendar=None),
             cf.dt('1983-12-01 06:00:00', calendar=None))
        )
        self.assertTrue(
            cf.D(0).interval(cf.dt(1984, 2, 3)) ==
            (cf.dt('1984-02-03 00:00:00', calendar=None),
             cf.dt('1984-02-03 00:00:00', calendar=None))
        )
        self.assertTrue(
            cf.D(5, hour=6).interval(cf.dt(2004, 3, 2), end=True) ==
            (cf.dt('2004-02-26 00:00:00', calendar=None),
             cf.dt('2004-03-02 00:00:00', calendar=None))
        )
        self.assertTrue(
            cf.D(5, hour=6).interval(
                cf.dt(2004, 3, 2, calendar='noleap'), end=True) ==
            (cf.dt('2004-02-25 00:00:00', calendar='noleap'),
             cf.dt('2004-03-02 00:00:00', calendar='noleap'))
        )
        self.assertTrue(
            cf.D(5, hour=6).interval(
                cf.dt(2004, 3, 2, calendar='360_day'), end=True) ==
            (cf.dt('2004-02-27 00:00:00',  calendar='360_day'),
             cf.dt('2004-03-02 00:00:00',  calendar='360_day'))
        )
        self.assertTrue(
            cf.h(19897.5).interval(cf.dt(1984, 2, 3, 0)) ==
            (cf.dt('1984-02-03 00:00:00', calendar=None),
             cf.dt('1986-05-12 01:30:00', calendar=None))
        )
        self.assertTrue(
            cf.h(19897.6).interval(cf.dt(1984, 2, 3, 0), end=True) ==
            (cf.dt('1981-10-26 22:24:00', calendar=None),
             cf.dt('1984-02-03 00:00:00', calendar=None))
        )

    def test_TimeDuration_iso(self):
        self.assertTrue(cf.Y(19).iso == 'P19Y')
        self.assertTrue(cf.M(9).iso == 'P9M')
        self.assertTrue(cf.D(34).iso == 'P34D')
        self.assertTrue(cf.m(16).iso == 'PT16M')
        self.assertTrue(cf.h(19897.546).iso == 'PT19897.546H')
        self.assertTrue(cf.s(1989).iso == 'PT1989S')

    def test_TimeDuration_bounds(self):
        for direction in (True, False):
            for x, y in zip(
                [
                    cf.Y().bounds(cf.dt(1984, 1, 1), direction=direction),
                    cf.Y().bounds(cf.dt(1984, 12, 1), direction=direction),
                    cf.Y().bounds(cf.dt(1984, 12, 3), direction=direction),
                    cf.Y(month=9).bounds(
                        cf.dt(1984, 1, 1), direction=direction),
                    cf.Y(month=9).bounds(
                        cf.dt(1984, 3, 3), direction=direction),
                    cf.Y(month=9).bounds(
                        cf.dt(1984, 9, 20), direction=direction),
                    cf.Y(month=9, day=13).bounds(
                         cf.dt(1984, 12, 12), direction=direction),
                ],
                [
                    (cf.dt('1984-01-01', calendar='standard'),
                     cf.dt('1985-01-01', calendar='standard')),
                    (cf.dt('1984-01-01', calendar='standard'),
                     cf.dt('1985-01-01', calendar='standard')),
                    (cf.dt('1984-01-01', calendar='standard'),
                     cf.dt('1985-01-01', calendar='standard')),
                    (cf.dt('1983-09-01', calendar='standard'),
                     cf.dt('1984-09-01', calendar='standard')),
                    (cf.dt('1983-09-01', calendar='standard'),
                     cf.dt('1984-09-01', calendar='standard')),
                    (cf.dt('1984-09-01', calendar='standard'),
                     cf.dt('1985-09-01', calendar='standard')),
                    (cf.dt('1984-09-13', calendar='standard'),
                     cf.dt('1985-09-13', calendar='standard')),
                ]
            ):
                if direction is False:
                    y = y[::-1]
                self.assertTrue(
                    x == y, "{}!={} direction={}".format(x, y, direction))

            for x, y in zip(
                [
                    cf.M().bounds(cf.dt(1984, 1, 1), direction=direction),
                    cf.M().bounds(cf.dt(1984, 12, 1), direction=direction),
                    cf.M().bounds(cf.dt(1984, 12, 3), direction=direction),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 1), direction=direction),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 3), direction=direction),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 15), direction=direction),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 20), direction=direction),
                ],
                [
                    (cf.dt('1984-01-01', calendar='standard'),
                     cf.dt('1984-02-01', calendar='standard')),
                    (cf.dt('1984-12-01', calendar='standard'),
                     cf.dt('1985-01-01', calendar='standard')),
                    (cf.dt('1984-12-01', calendar='standard'),
                     cf.dt('1985-01-01', calendar='standard')),
                    (cf.dt('1984-11-15', calendar='standard'),
                     cf.dt('1984-12-15', calendar='standard')),
                    (cf.dt('1984-11-15', calendar='standard'),
                     cf.dt('1984-12-15', calendar='standard')),
                    (cf.dt('1984-12-15', calendar='standard'),
                     cf.dt('1985-01-15', calendar='standard')),
                    (cf.dt('1984-12-15', calendar='standard'),
                     cf.dt('1985-01-15', calendar='standard')),
                ]
            ):
                if direction is False:
                    y = y[::-1]
                self.assertTrue(x == y, "{}!={}".format(x, y))

            for x, y in zip(
                [
                    cf.D().bounds(cf.dt(1984, 1, 1), direction=direction),
                    cf.D().bounds(cf.dt(1984, 12, 3), direction=direction),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1), direction=direction),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1, 12), direction=direction),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1, 15), direction=direction),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1, 20), direction=direction),
                ],
                [
                    (cf.dt('1984-01-01', calendar='gregorian'),
                     cf.dt('1984-01-02', calendar='gregorian')),
                    (cf.dt('1984-12-03', calendar='gregorian'),
                     cf.dt('1984-12-04', calendar='gregorian')),
                    (cf.dt('1984-11-30 15:00', calendar='gregorian'),
                     cf.dt('1984-12-01 15:00', calendar='gregorian')),
                    (cf.dt('1984-11-30 15:00', calendar='gregorian'),
                     cf.dt('1984-12-01 15:00', calendar='gregorian')),
                    (cf.dt('1984-12-01 15:00', calendar='gregorian'),
                     cf.dt('1984-12-02 15:00', calendar='gregorian')),
                    (cf.dt('1984-12-01 15:00', calendar='gregorian'),
                     cf.dt('1984-12-02 15:00', calendar='gregorian')),
                ]
            ):
                if direction is False:
                    y = y[::-1]

                self.assertTrue(x == y, "{}!={}".format(x, y))

    def test_TimeDuration_arithmetic(self):
        self.assertTrue(cf.M() + cf.dt(2000, 1, 1) == cf.dt(2000, 2, 1))
        self.assertTrue(cf.M() * 8 == cf.M(8))
        self.assertTrue(cf.M() * 8.5 == cf.M(8.5))
        self.assertTrue(cf.dt(2000, 1, 1) + cf.M() == cf.dt(2000, 2, 1))
        self.assertTrue(cf.dt(2000, 1, 1) - cf.M() == cf.dt(1999, 12, 1))
        self.assertTrue(
            cf.M() + datetime.datetime(2000, 1, 1) == cf.dt(2000, 2, 1))
        self.assertTrue(
            datetime.datetime(2000, 1, 1) + cf.M() == cf.dt(2000, 2, 1))
        self.assertTrue(
            datetime.datetime(2000, 1, 1) - cf.M() == cf.dt(1999, 12, 1))

        d = cf.dt(2000, 1, 1)
        d += cf.M()
        self.assertTrue(d == cf.dt(2000, 2, 1))
        d -= cf.M()
        self.assertTrue(d == cf.dt(2000, 1, 1))

        d = datetime.datetime(2000, 1, 1)
        d += cf.M()
        self.assertTrue(d == cf.dt(2000, 2, 1))
        d -= cf.M()
        self.assertTrue(d == cf.dt(2000, 1, 1))

        self.assertTrue(cf.M() * 8 == cf.M(8))
        self.assertTrue(cf.M() * 8.5 == cf.M(8.5))
        self.assertTrue(cf.M() / 2.0 == cf.M(0.5))
        self.assertTrue(cf.M(8) / 3 == cf.M(8/3))
        self.assertTrue(cf.M(8) // 3 == cf.M(2))

    def test_Timeduration__days_in_month(self):
        self.assertTrue(cf.TimeDuration.days_in_month(1900, 2) == 28)
        self.assertTrue(cf.TimeDuration.days_in_month(1999, 2) == 28)
        self.assertTrue(cf.TimeDuration.days_in_month(2000, 2) == 29)
        self.assertTrue(cf.TimeDuration.days_in_month(2004, 2) == 29)
        self.assertTrue(
            cf.TimeDuration.days_in_month(1900, 2, calendar='360_day') == 30)
        self.assertTrue(
            cf.TimeDuration.days_in_month(1999, 2, calendar='360_day') == 30)
        self.assertTrue(
            cf.TimeDuration.days_in_month(2000, 2, calendar='360_day') == 30)
        self.assertTrue(
            cf.TimeDuration.days_in_month(2004, 2, calendar='360_day') == 30)
        self.assertTrue(
            cf.TimeDuration.days_in_month(1900, 2, calendar='noleap') == 28)
        self.assertTrue(
            cf.TimeDuration.days_in_month(1999, 2, calendar='noleap') == 28)
        self.assertTrue(
            cf.TimeDuration.days_in_month(2000, 2, calendar='noleap') == 28)
        self.assertTrue(
            cf.TimeDuration.days_in_month(2004, 2, calendar='noleap') == 28)
        self.assertTrue(
            cf.TimeDuration.days_in_month(1900, 2, calendar='366_day') == 29)
        self.assertTrue(
            cf.TimeDuration.days_in_month(1999, 2, calendar='366_day') == 29)
        self.assertTrue(
            cf.TimeDuration.days_in_month(2000, 2, calendar='366_day') == 29)
        self.assertTrue(
            cf.TimeDuration.days_in_month(2004, 2, calendar='366_day') == 29)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
