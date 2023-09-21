import copy
import datetime
import faulthandler
import unittest

import numpy as np
from dask.base import tokenize

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class TimeDurationTest(unittest.TestCase):
    def test_TimeDuration(self):
        self.assertGreater(
            cf.TimeDuration(2, "calendar_years"),
            cf.TimeDuration(1, "calendar_years"),
        )
        self.assertLess(
            cf.TimeDuration(2, "calendar_years"),
            cf.TimeDuration(25, "calendar_months"),
        )
        self.assertLessEqual(
            cf.TimeDuration(2, "hours"), cf.TimeDuration(1, "days")
        )
        self.assertEqual(
            cf.TimeDuration(2, "hours"), cf.TimeDuration(1 / 12.0, "days")
        )
        self.assertEqual(
            cf.TimeDuration(2, "days"), cf.TimeDuration(48, "hours")
        )
        self.assertEqual(cf.TimeDuration(2, "days"), cf.Data(2))
        self.assertEqual(cf.TimeDuration(2, "days"), cf.Data([2.0], "days"))
        self.assertGreater(
            cf.TimeDuration(2, "days"), cf.Data([[60]], "seconds")
        )
        self.assertLessEqual(cf.TimeDuration(2, "hours"), 2)
        self.assertEqual(cf.TimeDuration(0.1, units="seconds"), 0.1)
        self.assertNotEqual(cf.TimeDuration(2, "days"), 30.5)
        self.assertGreater(cf.TimeDuration(2, "calendar_years"), np.array(1.5))
        self.assertLess(
            cf.TimeDuration(2, "calendar_months"), np.array([[12]])
        )

        self.assertGreater(
            cf.TimeDuration(2, "calendar_years"),
            cf.TimeDuration(1, "calendar_years"),
        )
        self.assertLessEqual(
            cf.TimeDuration(1, "calendar_years"),
            cf.TimeDuration(2, "calendar_years"),
        )

        self.assertGreaterEqual(
            cf.TimeDuration(25, "calendar_months"),
            cf.TimeDuration(2, "calendar_years"),
        )
        self.assertLess(
            cf.TimeDuration(2, "calendar_years"),
            cf.TimeDuration(25, "calendar_months"),
        )

        self.assertGreaterEqual(
            cf.TimeDuration(1, "days"), cf.TimeDuration(2, "hours")
        )
        self.assertEqual(
            cf.TimeDuration(2, "hours"), cf.TimeDuration(1 / 12.0, "days")
        )
        self.assertEqual(
            cf.TimeDuration(2, "days"), cf.TimeDuration(48, "hours")
        )
        self.assertEqual(cf.TimeDuration(2, "days"), cf.Data(2))
        self.assertEqual(cf.TimeDuration(2, "days"), cf.Data([2.0], "days"))
        self.assertGreater(
            cf.TimeDuration(2, "days"), cf.Data([[60]], "seconds")
        )
        self.assertEqual(cf.TimeDuration(2, "hours"), 2)
        self.assertNotEqual(cf.TimeDuration(2, "days"), 30.5)
        self.assertGreater(cf.TimeDuration(2, "calendar_years"), np.array(1.5))
        self.assertLess(
            cf.TimeDuration(2, "calendar_months"), np.array([[12]])
        )

        self.assertEqual(cf.TimeDuration(64, "calendar_years") + 2, cf.Y(66))
        self.assertEqual(
            cf.TimeDuration(64, "calendar_years") - 2.5, cf.Y(61.5)
        )
        self.assertEqual(
            cf.M(23) + cf.TimeDuration(64, "calendar_years"), cf.M(791)
        )
        self.assertEqual(
            cf.TimeDuration(64, "calendar_years") + cf.M(24), cf.Y(66)
        )

        self.assertEqual(cf.TimeDuration(36, "calendar_months") / 8, cf.M(4.5))
        self.assertEqual(cf.TimeDuration(36, "calendar_months") // 8, cf.M(4))

        self.assertEqual(
            cf.TimeDuration(36, "calendar_months") / np.array(8.0),
            cf.M(36 / 8.0),
        )
        self.assertEqual(
            cf.TimeDuration(12, "calendar_months") * cf.Data([[1.5]]),
            cf.Y(1.5),
        )
        self.assertEqual(
            cf.TimeDuration(36, "calendar_months") // 8.25, cf.M(4.0)
        )
        self.assertEqual(cf.TimeDuration(36, "calendar_months") % 10, cf.M(6))

        self.assertEqual(
            cf.TimeDuration(24, "hours") + cf.TimeDuration(0.5, "days"),
            cf.h(36.0),
        )
        self.assertEqual(
            cf.TimeDuration(0.5, "days") + cf.TimeDuration(24, "hours"),
            cf.D(1.5),
        )

        t = cf.TimeDuration(24, "hours")
        t += 2
        self.assertEqual(t, cf.h(26))
        t -= cf.Data(3, "hours")
        self.assertEqual(t, cf.h(23))
        t = cf.TimeDuration(24.0, "hours")
        t += 2
        self.assertEqual(t, cf.h(26))
        self.assertEqual(t - cf.Data(2.5, "hours"), cf.h(23.5))
        t *= 2
        self.assertEqual(t, cf.h(52.0))
        t -= 1.0
        self.assertEqual(t, cf.h(51))
        t /= 3
        self.assertEqual(t, cf.h(17))
        t += 5.5
        self.assertEqual(t, cf.h(22.5))
        t //= np.array(2)
        self.assertEqual(t, cf.h(11.0))
        t *= 10
        self.assertEqual(t, cf.h(110.0))
        t %= 3
        self.assertEqual(t, cf.h(2.0))

        t = cf.TimeDuration(24.5, "hours")
        self.assertEqual(-t, -24.5)
        self.assertEqual(int(t), 24)
        self.assertEqual(t / 0.5, 49)
        self.assertEqual(t // 2, 12)
        self.assertEqual(25 - t, 0.5)
        self.assertEqual(2 * t, 49)
        self.assertEqual(2.0 % t, 2, 2.0 % t)

        self.assertTrue(cf.TimeDuration(24, "hours").isint)
        self.assertTrue(cf.TimeDuration(24.0, "hours").isint)
        self.assertFalse(t.isint)

        t.Units = "days"
        self.assertEqual(t.Units, cf.Units("days"))
        t.Units = "hours"

        self.assertTrue(cf.TimeDuration(12, "hours").is_day_factor())
        self.assertFalse(cf.TimeDuration(13, "hours").is_day_factor())
        self.assertFalse(cf.TimeDuration(2, "days").is_day_factor())

        self.assertEqual(cf.TimeDuration(cf.Data(2, "days")), 2)
        self.assertEqual(cf.TimeDuration(cf.Data(48, "hours")), 48)
        self.assertEqual(
            cf.TimeDuration(cf.Data(48, "hours"), units="days"), 2
        )
        self.assertEqual(cf.TimeDuration(0.1, units="seconds"), 0.1)

        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

        self.assertTrue(t.equivalent(t, verbose=2))
        self.assertTrue(t.equivalent(t.copy(), verbose=2))

        with self.assertRaises(Exception):
            t = cf.TimeDuration(48, "m")
        with self.assertRaises(Exception):
            t = cf.TimeDuration(cf.Data(48, "m"))
        with self.assertRaises(Exception):
            t = cf.TimeDuration(cf.Data(48, "days"), units="m")

        t = t.copy()
        t = copy.deepcopy(t)

        repr(t)
        str(t)

        t //= 2
        t %= 2

    def test_TimeDuration_interval(self):
        self.assertEqual(
            cf.M().interval(cf.dt(1999, 12)),
            (
                cf.dt("1999-12-01 00:00:00", calendar=None),
                cf.dt("2000-01-01 00:00:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.Y(2).interval(cf.dt(2000, 2), end=True),
            (
                cf.dt("1998-02-01 00:00:00", calendar=None),
                cf.dt("2000-02-01 00:00:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.D(30).interval(cf.dt(1983, 12, 1, 6)),
            (
                cf.dt("1983-12-01 06:00:00", calendar=None),
                cf.dt("1983-12-31 06:00:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.D(30).interval(cf.dt(1983, 12, 1, 6), end=True),
            (
                cf.dt("1983-11-01 06:00:00", calendar=None),
                cf.dt("1983-12-01 06:00:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.D(0).interval(cf.dt(1984, 2, 3)),
            (
                cf.dt("1984-02-03 00:00:00", calendar=None),
                cf.dt("1984-02-03 00:00:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.D(5, hour=6).interval(cf.dt(2004, 3, 2), end=True),
            (
                cf.dt("2004-02-26 00:00:00", calendar=None),
                cf.dt("2004-03-02 00:00:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.D(5, hour=6).interval(
                cf.dt(2004, 3, 2, calendar="noleap"), end=True
            ),
            (
                cf.dt("2004-02-25 00:00:00", calendar="noleap"),
                cf.dt("2004-03-02 00:00:00", calendar="noleap"),
            ),
        )
        self.assertEqual(
            cf.D(5, hour=6).interval(
                cf.dt(2004, 3, 2, calendar="360_day"), end=True
            ),
            (
                cf.dt("2004-02-27 00:00:00", calendar="360_day"),
                cf.dt("2004-03-02 00:00:00", calendar="360_day"),
            ),
        )
        self.assertEqual(
            cf.h(19897.5).interval(cf.dt(1984, 2, 3, 0)),
            (
                cf.dt("1984-02-03 00:00:00", calendar=None),
                cf.dt("1986-05-12 01:30:00", calendar=None),
            ),
        )
        self.assertEqual(
            cf.h(19897.6).interval(cf.dt(1984, 2, 3, 0), end=True),
            (
                cf.dt("1981-10-26 22:24:00", calendar=None),
                cf.dt("1984-02-03 00:00:00", calendar=None),
            ),
        )

    def test_TimeDuration_iso(self):
        self.assertEqual(cf.Y(19).iso, "P19Y")
        self.assertEqual(cf.M(9).iso, "P9M")
        self.assertEqual(cf.D(34).iso, "P34D")
        self.assertEqual(cf.m(16).iso, "PT16M")
        self.assertEqual(cf.h(19897.546).iso, "PT19897.546H")
        self.assertEqual(cf.s(1989).iso, "PT1989S")

    def test_TimeDuration_bounds(self):
        for direction in (True, False):
            for x, y in zip(
                [
                    cf.Y().bounds(cf.dt(1984, 1, 1), direction=direction),
                    cf.Y().bounds(cf.dt(1984, 12, 1), direction=direction),
                    cf.Y().bounds(cf.dt(1984, 12, 3), direction=direction),
                    cf.Y(month=9).bounds(
                        cf.dt(1984, 1, 1), direction=direction
                    ),
                    cf.Y(month=9).bounds(
                        cf.dt(1984, 3, 3), direction=direction
                    ),
                    cf.Y(month=9).bounds(
                        cf.dt(1984, 9, 20), direction=direction
                    ),
                    cf.Y(month=9, day=13).bounds(
                        cf.dt(1984, 12, 12), direction=direction
                    ),
                ],
                [
                    (
                        cf.dt("1984-01-01", calendar="standard"),
                        cf.dt("1985-01-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-01-01", calendar="standard"),
                        cf.dt("1985-01-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-01-01", calendar="standard"),
                        cf.dt("1985-01-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1983-09-01", calendar="standard"),
                        cf.dt("1984-09-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1983-09-01", calendar="standard"),
                        cf.dt("1984-09-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-09-01", calendar="standard"),
                        cf.dt("1985-09-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-09-13", calendar="standard"),
                        cf.dt("1985-09-13", calendar="standard"),
                    ),
                ],
            ):
                if direction is False:
                    y = y[::-1]

                self.assertEqual(x, y, f"{x}!={y} direction={direction}")

            for x, y in zip(
                [
                    cf.M().bounds(cf.dt(1984, 1, 1), direction=direction),
                    cf.M().bounds(cf.dt(1984, 12, 1), direction=direction),
                    cf.M().bounds(cf.dt(1984, 12, 3), direction=direction),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 1), direction=direction
                    ),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 3), direction=direction
                    ),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 15), direction=direction
                    ),
                    cf.M(day=15).bounds(
                        cf.dt(1984, 12, 20), direction=direction
                    ),
                ],
                [
                    (
                        cf.dt("1984-01-01", calendar="standard"),
                        cf.dt("1984-02-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-12-01", calendar="standard"),
                        cf.dt("1985-01-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-12-01", calendar="standard"),
                        cf.dt("1985-01-01", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-11-15", calendar="standard"),
                        cf.dt("1984-12-15", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-11-15", calendar="standard"),
                        cf.dt("1984-12-15", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-12-15", calendar="standard"),
                        cf.dt("1985-01-15", calendar="standard"),
                    ),
                    (
                        cf.dt("1984-12-15", calendar="standard"),
                        cf.dt("1985-01-15", calendar="standard"),
                    ),
                ],
            ):
                if direction is False:
                    y = y[::-1]
                self.assertEqual(x, y, f"{x}!={y}")

            for x, y in zip(
                [
                    cf.D().bounds(cf.dt(1984, 1, 1), direction=direction),
                    cf.D().bounds(cf.dt(1984, 12, 3), direction=direction),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1), direction=direction
                    ),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1, 12), direction=direction
                    ),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1, 15), direction=direction
                    ),
                    cf.D(hour=15).bounds(
                        cf.dt(1984, 12, 1, 20), direction=direction
                    ),
                ],
                [
                    (
                        cf.dt("1984-01-01", calendar="gregorian"),
                        cf.dt("1984-01-02", calendar="gregorian"),
                    ),
                    (
                        cf.dt("1984-12-03", calendar="gregorian"),
                        cf.dt("1984-12-04", calendar="gregorian"),
                    ),
                    (
                        cf.dt("1984-11-30 15:00", calendar="gregorian"),
                        cf.dt("1984-12-01 15:00", calendar="gregorian"),
                    ),
                    (
                        cf.dt("1984-11-30 15:00", calendar="gregorian"),
                        cf.dt("1984-12-01 15:00", calendar="gregorian"),
                    ),
                    (
                        cf.dt("1984-12-01 15:00", calendar="gregorian"),
                        cf.dt("1984-12-02 15:00", calendar="gregorian"),
                    ),
                    (
                        cf.dt("1984-12-01 15:00", calendar="gregorian"),
                        cf.dt("1984-12-02 15:00", calendar="gregorian"),
                    ),
                ],
            ):
                if direction is False:
                    y = y[::-1]

                self.assertEqual(x, y, f"{x}!={y}")

    def test_TimeDuration_arithmetic(self):
        self.assertEqual(cf.M() + cf.dt(2000, 1, 1), cf.dt(2000, 2, 1))
        self.assertEqual(cf.M() * 8, cf.M(8))
        self.assertEqual(cf.M() * 8.5, cf.M(8.5))
        self.assertEqual(cf.dt(2000, 1, 1) + cf.M(), cf.dt(2000, 2, 1))
        self.assertEqual(cf.dt(2000, 1, 1) - cf.M(), cf.dt(1999, 12, 1))
        self.assertEqual(
            cf.M() + datetime.datetime(2000, 1, 1),
            cf.dt(2000, 2, 1, calendar="gregorian"),
        )
        self.assertEqual(
            datetime.datetime(2000, 1, 1) + cf.M(),
            cf.dt(2000, 2, 1, calendar="gregorian"),
        )
        self.assertEqual(
            datetime.datetime(2000, 1, 1) - cf.M(),
            cf.dt(1999, 12, 1, calendar="gregorian"),
        )

        d = cf.dt(2000, 1, 1)
        d += cf.M()
        self.assertEqual(d, cf.dt(2000, 2, 1))
        d -= cf.M()
        self.assertEqual(d, cf.dt(2000, 1, 1))

        d = datetime.datetime(2000, 1, 1)
        d += cf.M()
        self.assertEqual(d, cf.dt(2000, 2, 1, calendar="gregorian"))
        d -= cf.M()
        self.assertEqual(d, cf.dt(2000, 1, 1, calendar="gregorian"))

        self.assertEqual(cf.M() * 8, cf.M(8))
        self.assertEqual(cf.M() * 8.5, cf.M(8.5))
        self.assertEqual(cf.M() / 2.0, cf.M(0.5))
        self.assertEqual(cf.M(8) / 3, cf.M(8 / 3))
        self.assertEqual(cf.M(8) // 3, cf.M(2))

        # Test arithmetic involving Data as well as datetimes:
        da = cf.Data([2], units="days since 2000-01-01")
        dt = cf.TimeDuration(14, "day")
        t0 = da + dt
        t1 = dt + da
        self.assertEqual(t0, cf.dt(2000, 1, 17, calendar="gregorian"))
        self.assertEqual(t0, t1)
        t2 = dt - da
        t3 = da - dt
        self.assertEqual(t2, cf.dt(1999, 12, 20, calendar="gregorian"))
        self.assertEqual(t2, t3)

    def test_Timeduration__days_in_month(self):
        self.assertEqual(cf.TimeDuration.days_in_month(1900, 2), 28)
        self.assertEqual(cf.TimeDuration.days_in_month(1999, 2), 28)
        self.assertEqual(cf.TimeDuration.days_in_month(2000, 2), 29)
        self.assertEqual(cf.TimeDuration.days_in_month(2004, 2), 29)
        self.assertEqual(
            cf.TimeDuration.days_in_month(1900, 2, calendar="360_day"), 30
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(1999, 2, calendar="360_day"), 30
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(2000, 2, calendar="360_day"), 30
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(2004, 2, calendar="360_day"), 30
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(1900, 2, calendar="noleap"), 28
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(1999, 2, calendar="noleap"), 28
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(2000, 2, calendar="noleap"), 28
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(2004, 2, calendar="noleap"), 28
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(1900, 2, calendar="366_day"), 29
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(1999, 2, calendar="366_day"), 29
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(2000, 2, calendar="366_day"), 29
        )
        self.assertEqual(
            cf.TimeDuration.days_in_month(2004, 2, calendar="366_day"), 29
        )

    def test_TimeDuration__dask_tokenize__(self):
        for t in (cf.D(), cf.M(3, day=13, hour=10, minute=40, second=30)):
            self.assertEqual(tokenize(t), tokenize(t.copy()))

        self.assertNotEqual(tokenize(cf.D()), tokenize(cf.h(24)))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
