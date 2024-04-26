import copy
import datetime
import faulthandler
import re
import unittest

import numpy
from dask.base import tokenize

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class QueryTest(unittest.TestCase):
    f = cf.example_field(1)

    def test_Query_contains(self):
        c = self.f.dim("X")
        self.assertTrue(
            (
                (cf.contains(-4.26) == c).array
                == numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
            ).all()
        )
        self.assertTrue(
            (
                (cf.contains(999) == c).array
                == numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
            ).all()
        )

    def test_Query(self):
        q = cf.Query("wi", [2, 5])
        r = cf.Query("le", 67)
        s = q | r
        t = cf.Query("gt", 12, attr="bounds")
        u = s & t
        v = cf.wi(2, 5, open_lower=True)
        w = cf.wi(2, 5, open_upper=True)
        x = cf.wi(2, 5, open_lower=True, open_upper=True)

        repr(q)
        repr(s)
        repr(t)
        repr(u)
        str(q)
        str(s)
        str(t)
        str(u)
        u.dump(display=False)

        # For "wi", check repr. provides correct notation for open/closed-ness
        # of the interval captured.
        self.assertEqual(repr(q), "<CF Query: (wi [2, 5])>")
        self.assertEqual(repr(v), "<CF Query: (wi (2, 5])>")
        self.assertEqual(repr(w), "<CF Query: (wi [2, 5))>")
        self.assertEqual(repr(x), "<CF Query: (wi (2, 5))>")

        u.attr
        u.operator
        q.attr
        q.operator
        q.value
        with self.assertRaises(Exception):
            u.value

        self.assertTrue(u.equals(u.copy(), verbose=2))
        self.assertFalse(u.equals(t, verbose=0))

        self.assertTrue(q.equals(q.copy()))
        self.assertTrue(
            q.equals(cf.wi(2, 5, open_lower=False, open_upper=False))
        )
        self.assertFalse(q.equals(v))
        self.assertFalse(q.equals(w))
        self.assertFalse(q.equals(x))
        self.assertFalse(v.equals(w))
        self.assertFalse(v.equals(x))

        copy.deepcopy(u)

        c = self.f.dimension_coordinate("X")
        self.assertTrue(
            (
                (cf.contains(-4.26) == c).array
                == numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
            ).all()
        )
        self.assertTrue(
            (
                (cf.contains(999) == c).array
                == numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
            ).all()
        )

        cf.cellsize(34)
        cf.cellsize(q)

        cf.celllt(3)
        cf.cellle(3)
        cf.cellge(3)
        cf.cellgt(3)
        cf.cellwi(1, 2)
        cf.cellwo(1, 2)

    def test_Query_object_units(self):
        """Check units are processed correctly in and from queries."""

        equivalent_units = {
            1000: ("m", "km"),
            60: ("s", "minute"),
        }  # keys are conversion factors; only use exact equivalents for test
        for conversion, equivalents in equivalent_units.items():
            s_unit, l_unit = equivalents  # s for smaller, l for larger

            # Case 1: only specify units to one component
            q1 = cf.Query("gt", cf.Data(1, s_unit))
            q2 = cf.Query("ge", 1, units=s_unit)
            # Case 2: specify *equal* units across the two components
            q3 = cf.Query("lt", cf.Data(1, s_unit), units=s_unit)
            # Case 3: specify *equivalent* units across the two components
            q4 = cf.Query("le", cf.Data(1, s_unit), units=l_unit)
            # See also final Case 4, below.

            # A) test processed correctly inside unit itself
            for q in [q1, q2, q3, q4]:
                self.assertIsInstance(q, cf.query.Query)
                # TODO: should q4 should return s_ or l_unit? ATM is s_unit.
                self.assertEqual(q._value.Units._units, s_unit)

            with self.assertRaises(ValueError):
                # Case 4: provide non-equivalent units i.e. non-sensical query
                cf.Query("le", cf.Data(1, "m"), units="s")

            # B) Check units are processed correctly when a Query is evaluated
            q5 = cf.Query("eq", conversion, units=s_unit)
            # Pre-comparison, values should be converted for consistent units
            self.assertTrue(
                q5.evaluate(cf.Data([1, 2], l_unit)).equals(
                    cf.Data([True, False])
                )
            )

    def test_Query_as_where_condition(self):
        """Check queries work correctly as conditions in 'where'
        method."""
        # TODO: extend test; added as-is to capture a specific bug (now fixed)

        s_data = cf.Data([30, 60, 90], "second")

        m_lt_query = cf.lt(1, units="minute")
        s_lt_query = cf.lt(60, units="second")
        m_ge_query = cf.ge(1, units="minute")
        s_ge_query = cf.ge(60, units="second")
        for query_pair in [(m_lt_query, s_lt_query), (m_ge_query, s_ge_query)]:
            m_query, s_query = query_pair

            equal_units_where = s_data.data.where(s_query, 0)
            mixed_units_where = s_data.data.where(m_query, 0)
            self.assertTrue(
                (mixed_units_where.array == equal_units_where.array).all()
            )

            equal_units_where_masked = s_data.data.where(s_query, cf.masked)
            mixed_units_where_masked = s_data.data.where(m_query, cf.masked)
            self.assertEqual(
                mixed_units_where_masked.count(),
                equal_units_where_masked.count(),
            )

    def test_Query_datetime1(self):
        d = cf.Data(
            [[1.0, 5.0], [6, 2]],
            "days since 2000-12-29 21:00:00",
            calendar="standard",
        )

        message = "Diff =" + str(
            (
                d - cf.Data(cf.dt("2001-01-03 21:00:00", calendar="standard"))
            ).array
        )

        self.assertTrue(
            (d == cf.eq(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[False, True], [False, False]]), verbose=2
            ),
            message,
        )
        self.assertTrue(
            (d == cf.ne(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[True, False], [True, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.ge(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[False, True], [True, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.gt(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[False, False], [True, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.le(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[True, True], [False, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.lt(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[True, False], [False, True]])
            ),
            message,
        )
        self.assertTrue(
            (
                d
                == cf.wi(
                    cf.dt("2000-12-31 21:00:00"), cf.dt("2001-01-03 21:00:00")
                )
            ).equals(cf.Data([[False, True], [False, True]])),
            message,
        )
        self.assertTrue(
            (
                d
                == cf.wo(
                    cf.dt("2000-12-31 21:00:00"), cf.dt("2001-01-03 21:00:00")
                )
            ).equals(cf.Data([[True, False], [True, False]])),
            message,
        )
        self.assertTrue(
            (
                d
                == cf.set(
                    [
                        cf.dt("2000-12-31 21:00:00"),
                        cf.dt("2001-01-03 21:00:00"),
                    ]
                )
            ).equals(cf.Data([[False, True], [False, True]])),
            message,
        )

        cf.seasons()
        [
            cf.seasons(n, start)
            for n in [1, 2, 3, 4, 6, 12]
            for start in range(1, 13)
        ]
        with self.assertRaises(Exception):
            cf.seasons(13)
        with self.assertRaises(Exception):
            cf.seasons(start=8.456)

        cf.mam()
        cf.djf()
        cf.jja()
        cf.son()

    def test_Query_year_month_day_hour_minute_second(self):
        d = cf.Data(
            [[1.0, 5.0], [6, 2]],
            "days since 2000-12-29 21:57:57",
            calendar="gregorian",
        )

        self.assertTrue(
            (d == cf.year(2000)).equals(
                cf.Data([[True, False], [False, True]])
            )
        )
        self.assertTrue(
            (d == cf.month(12)).equals(cf.Data([[True, False], [False, True]]))
        )
        self.assertTrue(
            (d == cf.day(3)).equals(cf.Data([[False, True], [False, False]]))
        )
        d = cf.Data([[1.0, 5], [6, 2]], "hours since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.hour(2)).equals(cf.Data([[False, True], [False, False]]))
        )
        d = cf.Data([[1.0, 5], [6, 2]], "minutes since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.minute(2)).equals(
                cf.Data([[False, True], [False, False]])
            )
        )
        d = cf.Data([[1.0, 5], [6, 2]], "seconds since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.second(2)).equals(
                cf.Data([[False, True], [False, False]])
            )
        )
        d = cf.Data([[1.0, 5.0], [6, 2]], "days since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.year(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]])
            )
        )
        self.assertTrue(
            (d == cf.month(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]])
            )
        )
        self.assertTrue(
            (d == cf.day(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]])
            )
        )
        d = cf.Data([[1.0, 5], [6, 2]], "hours since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.hour(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]])
            )
        )
        d = cf.Data([[1.0, 5], [6, 2]], "minutes since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.minute(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]])
            )
        )
        d = cf.Data([[1.0, 5], [6, 2]], "seconds since 2000-12-29 21:57:57")
        self.assertTrue(
            (d == cf.second(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]])
            )
        )

    def test_Query_dteq_dtne_dtge_dtgt_dtle_dtlt(self):
        d = cf.Data([[1.0, 5.0], [6, 2]], "days since 2000-12-29 21:00:00")

        message = "Diff =" + str(
            (
                d - cf.Data(cf.dt("2001-01-03 21:00:00", calendar="standard"))
            ).array
        )

        self.assertTrue(
            (d == cf.eq(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[False, True], [False, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.ne(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[True, False], [True, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.ge(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[False, True], [True, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.gt(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[False, False], [True, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.le(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[True, True], [False, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.lt(cf.dt("2001-01-03 21:00:00"))).equals(
                cf.Data([[True, False], [False, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.eq(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[False, True], [False, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.ne(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[True, False], [True, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.ge(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[False, True], [True, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.gt(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[False, False], [True, False]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.le(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[True, True], [False, True]])
            ),
            message,
        )
        self.assertTrue(
            (d == cf.lt(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[True, False], [False, True]])
            ),
            message,
        )

        d = cf.dt(2002, 6, 16)
        self.assertEqual(cf.eq(cf.dt(2002, 6, 16)), d)
        self.assertNotEqual(cf.eq(cf.dt(1990, 1, 1)), d)
        self.assertNotEqual(cf.eq(cf.dt("2100-1-1")), d)
        self.assertNotEqual(
            cf.eq(cf.dt("2001-1-1")) & cf.eq(cf.dt(2010, 12, 31)), d
        )

        d = cf.dt(2002, 6, 16)
        self.assertEqual(cf.ge(cf.dt(1990, 1, 1)), d)
        self.assertEqual(cf.ge(cf.dt(2002, 6, 16)), d)
        self.assertNotEqual(cf.ge(cf.dt("2100-1-1")), d)
        self.assertNotEqual(
            cf.ge(cf.dt("2001-1-1")) & cf.ge(cf.dt(2010, 12, 31)), d
        )

        d = cf.dt(2002, 6, 16)
        self.assertEqual(cf.gt(cf.dt(1990, 1, 1)), d)
        self.assertNotEqual(cf.gt(cf.dt(2002, 6, 16)), d)
        self.assertNotEqual(cf.gt(cf.dt("2100-1-1")), d)
        self.assertEqual(
            cf.gt(cf.dt("2001-1-1")) & cf.le(cf.dt(2010, 12, 31)), d
        )

        d = cf.dt(2002, 6, 16)
        self.assertEqual(cf.ne(cf.dt(1990, 1, 1)), d)
        self.assertNotEqual(cf.ne(cf.dt(2002, 6, 16)), d)
        self.assertEqual(cf.ne(cf.dt("2100-1-1")), d)
        self.assertEqual(
            cf.ne(cf.dt("2001-1-1")) & cf.ne(cf.dt(2010, 12, 31)), d
        )

        d = cf.dt(2002, 6, 16)
        self.assertNotEqual(cf.le(cf.dt(1990, 1, 1)), d)
        self.assertEqual(cf.le(cf.dt(2002, 6, 16)), d)
        self.assertEqual(cf.le(cf.dt("2100-1-1")), d)
        self.assertNotEqual(
            cf.le(cf.dt("2001-1-1")) & cf.le(cf.dt(2010, 12, 31)), d
        )

        d = cf.dt(2002, 6, 16)
        self.assertNotEqual(cf.lt(cf.dt(1990, 1, 1)), d)
        self.assertNotEqual(cf.lt(cf.dt(2002, 6, 16)), d)
        self.assertEqual(cf.lt(cf.dt("2100-1-1")), d)
        self.assertNotEqual(
            cf.lt(cf.dt("2001-1-1")) & cf.lt(cf.dt(2010, 12, 31)), d
        )

    def test_Query_evaluate(self):
        for x in (5, cf.Data(5, "kg m-2"), cf.Data([5], "kg m-2 s-1")):
            self.assertEqual(x, cf.eq(5))
            self.assertEqual(x, cf.lt(8))
            self.assertEqual(x, cf.le(8))
            self.assertEqual(x, cf.gt(3))
            self.assertEqual(x, cf.ge(3))
            self.assertEqual(x, cf.wi(3, 8))
            self.assertEqual(x, cf.wo(8, 11))
            self.assertEqual(x, cf.set([3, 5, 8]))

            self.assertEqual(cf.eq(5), x)
            self.assertEqual(cf.lt(8), x)
            self.assertEqual(cf.le(8), x)
            self.assertEqual(cf.gt(3), x)
            self.assertEqual(cf.ge(3), x)
            self.assertEqual(cf.wi(3, 8), x)
            self.assertEqual(cf.wo(8, 11), x)
            self.assertEqual(cf.set([3, 5, 8]), x)

            self.assertNotEqual(x, cf.eq(8))
            self.assertNotEqual(x, cf.lt(3))
            self.assertNotEqual(x, cf.le(3))
            self.assertNotEqual(x, cf.gt(8))
            self.assertNotEqual(x, cf.ge(8))
            self.assertNotEqual(x, cf.wi(8, 11))
            self.assertNotEqual(x, cf.wo(3, 8))
            self.assertNotEqual(x, cf.set([3, 8, 11]))

            # Test that the evaluation is commutative i.e. A == B
            # means B == A
            self.assertNotEqual(x, cf.eq(8))
            self.assertNotEqual(x, cf.lt(3))
            self.assertNotEqual(x, cf.le(3))
            self.assertNotEqual(x, cf.gt(8))
            self.assertNotEqual(x, cf.ge(8))
            self.assertNotEqual(x, cf.wi(8, 11))
            self.assertNotEqual(x, cf.wo(3, 8))
            self.assertNotEqual(x, cf.set([3, 8, 11]))

            self.assertNotEqual(cf.eq(8), x)
            self.assertNotEqual(cf.lt(3), x)
            self.assertNotEqual(cf.le(3), x)
            self.assertNotEqual(cf.gt(8), x)
            self.assertNotEqual(cf.ge(8), x)
            self.assertNotEqual(cf.wi(8, 11), x)
            self.assertNotEqual(cf.wo(3, 8), x)
            self.assertNotEqual(cf.set([3, 8, 11]), x)

        c = cf.wi(2, 4)
        c0 = cf.wi(2, 4, open_lower=False)  # equivalent to c, to check default
        c1 = cf.wi(2, 4, open_lower=True)
        c2 = cf.wi(2, 4, open_upper=True)
        c3 = cf.wi(2, 4, open_lower=True, open_upper=True)
        all_c = [c, c0, c1, c2, c3]

        d = cf.wi(6, 8)
        d0 = cf.wi(6, 8, open_lower=False)  # equivalent to d, to check default
        d1 = cf.wi(6, 8, open_lower=True)
        d2 = cf.wi(6, 8, open_upper=True)
        d3 = cf.wi(6, 8, open_lower=True, open_upper=True)
        all_d = [d, d0, d1, d2, d3]

        e = d | c  # interval: [2, 4] | [6, 8]
        e1 = c0 | d1  # interval: [2, 4] | (6, 8]
        e2 = c1 | d2  # interval: (2, 4] | [6, 8)
        e3 = d3 | c3  # interval: (6, 8) | (2, 4)
        all_e = [e, e1, e2, e3]

        for cx in all_c:
            self.assertTrue(cx.evaluate(3))
            self.assertFalse(cx.evaluate(5))

        for dx in all_d:
            self.assertTrue(dx.evaluate(7))
            self.assertFalse(dx.evaluate(9))

        # Test the two open_* keywords for direct (non-compound) queries
        self.assertEqual(c.evaluate(2), c0.evaluate(2))
        self.assertTrue(c0.evaluate(2))
        self.assertFalse(c1.evaluate(2))
        self.assertTrue(c2.evaluate(2))
        self.assertFalse(c3.evaluate(2))
        self.assertEqual(c.evaluate(4), c0.evaluate(4))
        self.assertTrue(c0.evaluate(4))
        self.assertTrue(c1.evaluate(4))
        self.assertFalse(c2.evaluate(4))
        self.assertFalse(c3.evaluate(4))

        for ex in all_e:
            self.assertTrue(e.evaluate(3))
            self.assertTrue(e.evaluate(7))
            self.assertFalse(e.evaluate(5))

        # Test the two open_* keywords for compound queries.
        # Must be careful to capture correct openness/closure of any inner
        # bounds introduced through compound queries, e.g. for 'e' there
        # are internal endpoints at 4 and 6 to behave like in 'c' and 'd'.
        self.assertTrue(e.evaluate(2))
        self.assertTrue(e1.evaluate(2))
        self.assertFalse(e2.evaluate(2))
        self.assertFalse(e3.evaluate(2))
        self.assertTrue(e.evaluate(4))
        self.assertTrue(e1.evaluate(4))
        self.assertTrue(e2.evaluate(4))
        self.assertFalse(e3.evaluate(4))
        self.assertTrue(e.evaluate(6))
        self.assertFalse(e1.evaluate(6))
        self.assertTrue(e2.evaluate(6))
        self.assertFalse(e3.evaluate(6))
        self.assertTrue(e.evaluate(8))
        self.assertTrue(e1.evaluate(8))
        self.assertFalse(e2.evaluate(8))
        self.assertFalse(e3.evaluate(8))

        self.assertEqual(3, c)
        self.assertNotEqual(5, c)

        self.assertEqual(c, 3)
        self.assertNotEqual(c, 5)

        self.assertEqual(3, e)
        self.assertEqual(7, e)
        self.assertNotEqual(5, e)

        self.assertEqual(e, 3)
        self.assertEqual(e, 7)
        self.assertNotEqual(e, 5)

        x = "qwerty"
        self.assertEqual(x, cf.eq("qwerty"))
        self.assertEqual(x, cf.eq(re.compile("^qwerty$")))
        self.assertEqual(x, cf.eq(re.compile("qwe")))
        self.assertEqual(x, cf.eq(re.compile("qwe.*")))
        self.assertEqual(x, cf.eq(re.compile(".*qwe")))
        self.assertEqual(x, cf.eq(re.compile(".*rty")))
        self.assertEqual(x, cf.eq(re.compile(".*rty$")))
        self.assertEqual(x, cf.eq(re.compile("^.*rty$")))
        self.assertEqual(x, cf.eq(re.compile("rty$")))

        self.assertNotEqual(x, cf.eq("QWERTY"))
        self.assertNotEqual(x, cf.eq(re.compile("QWERTY")))
        self.assertNotEqual(x, cf.eq(re.compile("^QWERTY$")))
        self.assertNotEqual(x, cf.eq(re.compile("QWE")))
        self.assertNotEqual(x, cf.eq(re.compile("QWE.*")))
        self.assertNotEqual(x, cf.eq(re.compile(".*QWE")))
        self.assertNotEqual(x, cf.eq(re.compile(".*RTY")))
        self.assertNotEqual(x, cf.eq(re.compile(".*RTY$")))
        self.assertNotEqual(x, cf.eq(re.compile("^.*RTY$")))

    def test_Query_set_condition_units(self):
        q = cf.lt(9)
        q.set_condition_units("km")
        self.assertEqual(q.value.Units, cf.Units("km"))

        with self.assertRaises(ValueError):
            q.set_condition_units("seconds")

        q = cf.lt(9000, units="m")
        q.set_condition_units("km")
        self.assertEqual(q.value.Units, cf.Units("km"))
        self.assertEqual(q.value.array, 9)

        q = cf.lt(9)
        r = cf.ge(3000, units="m")
        s = q & r
        s.set_condition_units("km")
        self.assertEqual(s._compound[0].value.Units, cf.Units("km"))
        self.assertEqual(s._compound[1].value.Units, cf.Units("km"))
        self.assertEqual(s._compound[0].value.array, 9)
        self.assertEqual(s._compound[1].value.array, 3)

        self.assertEqual(r.value.Units, cf.Units("m"))
        self.assertEqual(r.value.array, 3000)
        self.assertEqual(q.value, 9)

    def test_Query_iscontains(self):
        q = cf.contains(9)
        self.assertTrue(q.iscontains())

        r = cf.lt(9)
        self.assertFalse(r.iscontains())
        self.assertFalse((q | r).iscontains())

    def test_Query__and__(self):
        q = cf.gt(9)
        r = cf.lt(11)
        s = q & r
        self.assertTrue(s == 10)
        self.assertFalse(s == 8)
        with self.assertRaises(AttributeError):
            s.value

        r = cf.lt(9)
        s = q & r
        self.assertEqual(s.value, 9)
        self.assertFalse(s == 9)

        q = cf.set(cf.Data([1])) & cf.set(cf.Data([1]))
        self.assertEqual(q.value, cf.Data([1]))

        q = cf.set([1]) & cf.set(cf.Data([1]))
        self.assertEqual(q.value, [1])

        q = cf.set([1, 2]) & cf.set(cf.Data([1]))
        self.assertIsNone(q._value)

        q = cf.set([1]) & cf.set(cf.Data([1, 2]))
        self.assertIsNone(q._value)

        q = cf.set(cf.Data([1, 2, 3])) & cf.set(cf.Data([1, 2]))
        self.assertIsNone(q._value)

    def test_Query__or__(self):
        q = cf.eq(9)
        r = cf.gt(11)
        s = q | r
        self.assertTrue(s == 9)
        self.assertTrue(s == 12)
        self.assertFalse(s == 10)
        with self.assertRaises(AttributeError):
            s.value

        r = cf.gt(9)
        s = q | r
        self.assertEqual(s.value, 9)
        self.assertFalse(s == 8)

        q = cf.set(cf.Data([1])) | cf.set(cf.Data([1]))
        self.assertEqual(q.value, cf.Data([1]))

        q = cf.set([1]) | cf.set(cf.Data([1]))
        self.assertEqual(q.value, [1])

        q = cf.set([1, 2]) | cf.set(cf.Data([1]))
        self.assertIsNone(q._value)

        q = cf.set([1]) | cf.set(cf.Data([1, 2]))
        self.assertIsNone(q._value)

        q = cf.set(cf.Data([1, 2, 3])) | cf.set(cf.Data([1, 2]))
        self.assertIsNone(q._value)

    def test_Query__dask_tokenize__(self):
        for q in (
            cf.eq(9),
            cf.lt(9, "km"),
            cf.wi(2, 5, "km"),
            cf.set([1, 2, 3]),
            cf.set([1, 2, 3], "km"),
            cf.ge(7, attr="year"),
            cf.ge(7, attr="upper_bounds.month", units="K"),
            cf.eq(8) & cf.le(7, "km"),
            cf.wo(2, 5, attr="day") | cf.set(cf.Data([1, 2], "km")),
            cf.eq(8) | cf.lt(9) & cf.ge(10),
            cf.isclose(1, "days", rtol=10, atol=99),
            cf.wi(-5, 5, open_lower=True),
            cf.wi(-5, 5, open_lower=True, open_upper=True),
        ):
            self.assertEqual(tokenize(q), tokenize(q.copy()))

        self.assertEqual(tokenize(cf.eq(9, "km")), tokenize(cf.eq(9, "1000m")))
        self.assertNotEqual(
            tokenize(cf.eq(9, "km")), tokenize(cf.eq(9000, "m"))
        )
        self.assertNotEqual(
            tokenize(cf.isclose(9)), tokenize(cf.isclose(9, rtol=10))
        )

        self.assertNotEqual(
            tokenize(cf.wi(-5, 5, open_lower=True)), tokenize(cf.wi(-5, 5))
        )
        self.assertNotEqual(
            tokenize(cf.wi(-5, 5, open_upper=True)), tokenize(cf.wi(-5, 5))
        )
        self.assertNotEqual(
            tokenize(cf.wi(-5, 5, open_upper=True)),
            tokenize(cf.wi(-5, 5, open_lower=True)),
        )
        self.assertNotEqual(
            tokenize(cf.wi(-5, 5, open_lower=True, open_upper=True)),
            tokenize(cf.wi(-5, 5)),
        )
        self.assertNotEqual(
            tokenize(cf.wi(-5, 5, open_lower=True, open_upper=True)),
            tokenize(cf.wi(-5, 5, open_lower=True)),
        )
        self.assertNotEqual(
            tokenize(cf.wi(-5, 5, open_lower=True, open_upper=True)),
            tokenize(cf.wi(-5, 5, open_upper=True)),
        )
        # Check defaults
        self.assertEqual(
            tokenize(cf.wi(-5, 5, open_lower=False, open_upper=False)),
            tokenize(cf.wi(-5, 5)),
        )

    def test_Query_Units(self):
        self.assertEqual(cf.eq(9).Units, cf.Units())
        self.assertEqual(cf.eq(9, "m s-1").Units, cf.Units("m s-1"))
        self.assertEqual(cf.eq(cf.Data(9, "km")).Units, cf.Units("km"))

        self.assertEqual((cf.eq(9) | cf.gt(9)).Units, cf.Units())
        self.assertEqual((cf.eq(9) | cf.gt(45)).Units, cf.Units())
        self.assertEqual((cf.eq(9, "m") | cf.gt(9, "m")).Units, cf.Units("m"))
        self.assertEqual((cf.eq(9, "m") | cf.gt(45, "m")).Units, cf.Units("m"))
        self.assertEqual((cf.eq(9, "m") | cf.gt(9)).Units, cf.Units("m"))
        self.assertEqual((cf.eq(9) | cf.gt(45, "m")).Units, cf.Units("m"))

        with self.assertRaises(AttributeError):
            (cf.eq(9, "m") | cf.gt(9, "day")).Units

    def test_Query_atol(self):
        self.assertIsNone(cf.eq(9).atol)
        self.assertIsNone(cf.Query("isclose", 9).atol)
        self.assertEqual(cf.Query("isclose", 9, atol=10).atol, 10)

    def test_Query_rtol(self):
        self.assertIsNone(cf.eq(9).rtol)
        self.assertIsNone(cf.Query("isclose", 9).rtol)
        self.assertEqual(cf.Query("isclose", 9, rtol=10).rtol, 10)

    def test_Query_isclose(self):
        q = cf.isclose(9)
        self.assertIsNone(q.atol)
        self.assertIsNone(q.rtol)
        self.assertTrue(9, q)
        self.assertNotEqual(9.000001, q)

        d = cf.Data([9, 9.000001], "m")
        self.assertFalse((d == q).all())

        atol = 0.001
        rtol = 0.0001
        q = cf.isclose(9, atol=atol, rtol=rtol)
        self.assertEqual(q.atol, atol)
        self.assertEqual(q.rtol, rtol)
        self.assertTrue(9, q)
        self.assertTrue(9.000001, q)
        self.assertNotEqual(9.1, q)

        self.assertTrue((d == q).all())

        q = cf.eq(9)
        self.assertIsNone(q.atol)
        self.assertIsNone(q.rtol)

        # Can't set atol and rtol unless operation is 'isclose'
        with self.assertRaises(ValueError):
            cf.Query("eq", 9, atol=atol, rtol=rtol)

    def test_Query_setdefault(self):
        # rtol and atol
        q = cf.isclose(9)
        self.assertIsNone(q.rtol)
        self.assertIsNone(q.atol)

        q.setdefault(rtol=10, atol=99)
        self.assertEqual(q.rtol, 10)
        self.assertEqual(q.atol, 99)

        q.setdefault(rtol=2, atol=3)
        self.assertEqual(q.rtol, 10)
        self.assertEqual(q.atol, 99)

        q = cf.isclose(9, atol=3) | (cf.eq(1) & cf.isclose(4, rtol=2))
        self.assertIsNone(q.rtol)
        self.assertIsNone(q.atol)

        q.setdefault(rtol=10, atol=99)
        self.assertIsNone(q.rtol)
        self.assertIsNone(q.atol)

        c = q._compound[0]
        self.assertEqual(c.rtol, 10)
        self.assertEqual(c.atol, 3)

        c = q._compound[1]._compound[1]
        self.assertEqual(c.rtol, 2)
        self.assertEqual(c.atol, 99)

        q = cf.eq(9)
        q.setdefault(rtol=10, atol=99)
        self.assertIsNone(q.rtol)
        self.assertIsNone(q.atol)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
