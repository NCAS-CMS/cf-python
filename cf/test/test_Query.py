import copy
import datetime
import re
import os
import unittest

import numpy

import cf


class QueryTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')

    def test_Query_contains(self):
        f = cf.read(self.filename)[0]
        c = f.dim('X')
        self.assertTrue(((cf.contains(21.1) == c).array ==
                         numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0], bool)).all())
        self.assertTrue(((cf.contains(999) == c).array ==
                         numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0], bool)).all())

    def test_Query(self):
        f = cf.read(self.filename)[0]

        q = cf.Query('wi', [2, 5])
        r = cf.Query('le', 67)
        s = q | r
        t = cf.Query('gt', 12, attr='bounds')
        u = s & t

        _ = repr(q)
        _ = repr(s)
        _ = repr(t)
        _ = repr(u)
        _ = str(q)
        _ = str(s)
        _ = str(t)
        _ = str(u)
        _ = u.dump(display=False)

        _ = u.attr
        _ = u.operator
        _ = q.attr
        _ = q.operator
        _ = q.value
        with self.assertRaises(Exception):
            _ = u.value

        self.assertTrue(u.equals(u.copy(), verbose=2))
        self.assertFalse(u.equals(t, verbose=0))

        _ = copy.deepcopy(u)

        c = f.dimension_coordinate('X')
        self.assertTrue(((cf.contains(21.1) == c).array ==
                         numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0], bool)).all())
        self.assertTrue(((cf.contains(999) == c).array ==
                         numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0], bool)).all())

        _ = cf.cellsize(34)
        _ = cf.cellsize(q)

        _ = cf.celllt(3)
        _ = cf.cellle(3)
        _ = cf.cellge(3)
        _ = cf.cellgt(3)
        _ = cf.cellwi(1, 2)
        _ = cf.cellwo(1, 2)

    def test_Query_object_units(self):
        """Check units are processed correctly in and from queries."""

        equivalent_units = {
            1000: ('m', 'km'),
            60: ('s', 'minute')
        }  # keys are conversion factors; only use exact equivalents for test
        for conversion, equivalents in equivalent_units.items():
            s_unit, l_unit = equivalents  # s for smaller, l for larger

            # Case 1: only specify units to one component
            q1 = cf.Query('gt', cf.Data(1, s_unit))
            q2 = cf.Query('ge', 1, units=s_unit)
            # Case 2: specify *equal* units across the two components
            q3 = cf.Query('lt', cf.Data(1, s_unit), units=s_unit)
            # Case 3: specify *equivalent* units across the two components
            q4 = cf.Query('le', cf.Data(1, s_unit), units=l_unit)
            # See also final Case 4, below.

            # A) test processed correctly inside unit itself
            for q in [q1, q2, q3, q4]:
                self.assertIsInstance(q, cf.query.Query)
                # TODO: should q4 should return s_ or l_unit? ATM is s_unit.
                self.assertTrue(q._value.Units._units == s_unit)

            with self.assertRaises(ValueError):
                # Case 4: provide non-equivalent units i.e. non-sensical query
                cf.Query('le', cf.Data(1, 'm'), units='s')

            # B) Check units are processed correctly when a Query is evaluated
            q5 = cf.Query('eq', conversion, units=s_unit)
            # Pre-comparison, values should be converted for consistent units
            self.assertTrue(
                q5.evaluate(cf.Data([1, 2], l_unit)).equals(
                    cf.Data([True, False]))
            )

    def test_Query_as_where_condition(self):
        """Check queries work correctly as conditions in 'where' method."""
        # TODO: extend test; added as-is to capture a specific bug (now fixed)

        s_data = cf.Data([30, 60, 90], 'second')

        m_lt_query = cf.lt(1, units="minute")
        s_lt_query = cf.lt(60, units="second")
        m_ge_query = cf.ge(1, units="minute")
        s_ge_query = cf.ge(60, units="second")
        for query_pair in [
                (m_lt_query, s_lt_query), (m_ge_query, s_ge_query)]:
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
                equal_units_where_masked.count()
            )

    def test_Query_datetime1(self):
        d = cf.Data([[1., 5.], [6, 2]], 'days since 2000-12-29 21:00:00',
                    calendar='standard')

        message = 'Diff ='+str((d-cf.Data(cf.dt(
            '2001-01-03 21:00:00', calendar='standard'))).array)

        self.assertTrue(
            (d == cf.eq(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[False, True], [False, False]]), verbose=2),
            message
        )
        self.assertTrue(
            (d == cf.ne(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[True, False], [True, True]])),
            message
        )
        self.assertTrue(
            (d == cf.ge(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[False, True], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.gt(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[False, False], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.le(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[True, True], [False, True]])),
            message
        )
        self.assertTrue(
            (d == cf.lt(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[True, False], [False, True]])),
            message
        )
        self.assertTrue(
            (d == cf.wi(cf.dt('2000-12-31 21:00:00'),
                        cf.dt('2001-01-03 21:00:00'))).equals(
                            cf.Data([[False, True], [False, True]])),
            message
        )
        self.assertTrue(
            (d == cf.wo(cf.dt('2000-12-31 21:00:00'),
                        cf.dt('2001-01-03 21:00:00'))).equals(
                            cf.Data([[True, False], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.set([cf.dt('2000-12-31 21:00:00'),
                          cf.dt('2001-01-03 21:00:00')])).equals(
                              cf.Data([[False, True], [False, True]])),
            message
        )

        _ = cf.seasons()
        [cf.seasons(n, start) for n in [1, 2, 3, 4, 6, 12]
         for start in range(1, 13)]
        with self.assertRaises(Exception):
            cf.seasons(13)
        with self.assertRaises(Exception):
            cf.seasons(start=8.456)

        _ = cf.mam()
        _ = cf.djf()
        _ = cf.jja()
        _ = cf.son()

    def test_Query_year_month_day_hour_minute_second(self):
        d = cf.Data([[1., 5.], [6, 2]], 'days since 2000-12-29 21:57:57',
                    calendar='gregorian')

        self.assertTrue(
            (d == cf.year(2000)).equals(
                cf.Data([[True, False], [False, True]]))
        )
        self.assertTrue(
            (d == cf.month(12)).equals(
                cf.Data([[True, False], [False, True]]))
        )
        self.assertTrue(
            (d == cf.day(3)).equals(cf.Data([[False, True], [False, False]])))
        d = cf.Data([[1., 5], [6, 2]], 'hours since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.hour(2)).equals(cf.Data([[False, True], [False, False]])))
        d = cf.Data([[1., 5], [6, 2]], 'minutes since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.minute(2)).equals(
                cf.Data([[False, True], [False, False]]))
        )
        d = cf.Data([[1., 5], [6, 2]], 'seconds since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.second(2)).equals(
                cf.Data([[False, True], [False, False]]))
        )
        d = cf.Data([[1., 5.], [6, 2]], 'days since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.year(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]]))
        )
        self.assertTrue(
            (d == cf.month(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]]))
        )
        self.assertTrue(
            (d == cf.day(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]]))
        )
        d = cf.Data([[1., 5], [6, 2]], 'hours since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.hour(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]]))
        )
        d = cf.Data([[1., 5], [6, 2]], 'minutes since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.minute(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]]))
        )
        d = cf.Data([[1., 5], [6, 2]], 'seconds since 2000-12-29 21:57:57')
        self.assertTrue(
            (d == cf.second(cf.ne(-1))).equals(
                cf.Data([[True, True], [True, True]]))
        )

    def test_Query_dteq_dtne_dtge_dtgt_dtle_dtlt(self):
        d = cf.Data([[1., 5.], [6, 2]], 'days since 2000-12-29 21:00:00')

        message = 'Diff =' + str((d-cf.Data(
            cf.dt('2001-01-03 21:00:00', calendar='standard'))).array)

        self.assertTrue(
            (d == cf.eq(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[False, True], [False, False]])),
            message
        )
        self.assertTrue(
            (d == cf.ne(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[True, False], [True, True]])),
            message
        )
        self.assertTrue(
            (d == cf.ge(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[False, True], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.gt(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[False, False], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.le(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[True, True], [False, True]])),
            message
        )
        self.assertTrue(
            (d == cf.lt(cf.dt('2001-01-03 21:00:00'))).equals(
                cf.Data([[True, False], [False, True]])),
            message
        )
        self.assertTrue(
            (d == cf.eq(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[False, True], [False, False]])),
            message
        )
        self.assertTrue(
            (d == cf.ne(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[True, False], [True, True]])),
            message
        )
        self.assertTrue(
            (d == cf.ge(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[False, True], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.gt(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[False, False], [True, False]])),
            message
        )
        self.assertTrue(
            (d == cf.le(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[True, True], [False, True]])),
            message
        )
        self.assertTrue(
            (d == cf.lt(cf.dt(2001, 1, 3, 21, 0, 0))).equals(
                cf.Data([[True, False], [False, True]])),
            message
        )

        d = cf.dt(2002, 6, 16)
        self.assertFalse(cf.eq(cf.dt(1990, 1, 1)) == d)
        self.assertTrue(cf.eq(cf.dt(2002, 6, 16)) == d)
        self.assertFalse(cf.eq(cf.dt('2100-1-1')) == d)
        self.assertFalse(
            cf.eq(cf.dt('2001-1-1')) & cf.eq(cf.dt(2010, 12, 31)) == d)

        d = cf.dt(2002, 6, 16)
        self.assertTrue(cf.ge(cf.dt(1990, 1, 1)) == d)
        self.assertTrue(cf.ge(cf.dt(2002, 6, 16)) == d)
        self.assertFalse(cf.ge(cf.dt('2100-1-1')) == d)
        self.assertFalse(
            cf.ge(cf.dt('2001-1-1')) & cf.ge(cf.dt(2010, 12, 31)) == d)

        d = cf.dt(2002, 6, 16)
        self.assertTrue(cf.gt(cf.dt(1990, 1, 1)) == d)
        self.assertFalse(cf.gt(cf.dt(2002, 6, 16)) == d)
        self.assertFalse(cf.gt(cf.dt('2100-1-1')) == d)
        self.assertTrue(
            cf.gt(cf.dt('2001-1-1')) & cf.le(cf.dt(2010, 12, 31)) == d)

        d = cf.dt(2002, 6, 16)
        self.assertTrue(cf.ne(cf.dt(1990, 1, 1)) == d)
        self.assertFalse(cf.ne(cf.dt(2002, 6, 16)) == d)
        self.assertTrue(cf.ne(cf.dt('2100-1-1')) == d)
        self.assertTrue(
            cf.ne(cf.dt('2001-1-1')) & cf.ne(cf.dt(2010, 12, 31)) == d)

        d = cf.dt(2002, 6, 16)
        self.assertFalse(cf.le(cf.dt(1990, 1, 1)) == d)
        self.assertTrue(cf.le(cf.dt(2002, 6, 16)) == d)
        self.assertTrue(cf.le(cf.dt('2100-1-1')) == d)
        self.assertFalse(
            cf.le(cf.dt('2001-1-1')) & cf.le(cf.dt(2010, 12, 31)) == d)

        d = cf.dt(2002, 6, 16)
        self.assertFalse(cf.lt(cf.dt(1990, 1, 1)) == d)
        self.assertFalse(cf.lt(cf.dt(2002, 6, 16)) == d)
        self.assertTrue(cf.lt(cf.dt('2100-1-1')) == d)
        self.assertFalse(
            cf.lt(cf.dt('2001-1-1')) & cf.lt(cf.dt(2010, 12, 31)) == d)

    def test_Query_evaluate(self):
        for x in (5, cf.Data(5, 'kg m-2'), cf.Data([5], 'kg m-2 s-1')):
            self.assertTrue(x == cf.eq(5))
            self.assertTrue(x == cf.lt(8))
            self.assertTrue(x == cf.le(8))
            self.assertTrue(x == cf.gt(3))
            self.assertTrue(x == cf.ge(3))
            self.assertTrue(x == cf.wi(3, 8))
            self.assertTrue(x == cf.wo(8, 11))
            self.assertTrue(x == cf.set([3, 5, 8]))

            self.assertTrue(cf.eq(5) == x)
            self.assertTrue(cf.lt(8) == x)
            self.assertTrue(cf.le(8) == x)
            self.assertTrue(cf.gt(3) == x)
            self.assertTrue(cf.ge(3) == x)
            self.assertTrue(cf.wi(3, 8) == x)
            self.assertTrue(cf.wo(8, 11) == x)
            self.assertTrue(cf.set([3, 5, 8]) == x)

            self.assertFalse(x == cf.eq(8))
            self.assertFalse(x == cf.lt(3))
            self.assertFalse(x == cf.le(3))
            self.assertFalse(x == cf.gt(8))
            self.assertFalse(x == cf.ge(8))
            self.assertFalse(x == cf.wi(8, 11))
            self.assertFalse(x == cf.wo(3, 8))
            self.assertFalse(x == cf.set([3, 8, 11]))

            # Test that the evaluation is commutative i.e. A == B means B == A
            self.assertFalse(x == cf.eq(8) == x)
            self.assertFalse(x == cf.lt(3) == x)
            self.assertFalse(x == cf.le(3) == x)
            self.assertFalse(x == cf.gt(8) == x)
            self.assertFalse(x == cf.ge(8) == x)
            self.assertFalse(x == cf.wi(8, 11) == x)
            self.assertFalse(x == cf.wo(3, 8) == x)
            self.assertFalse(x == cf.set([3, 8, 11]) == x)
        # --- End: for

        c = cf.wi(2, 4)
        d = cf.wi(6, 8)

        e = d | c

        self.assertTrue(c.evaluate(3))
        self.assertFalse(c.evaluate(5))

        self.assertTrue(e.evaluate(3))
        self.assertTrue(e.evaluate(7))
        self.assertFalse(e.evaluate(5))

        self.assertTrue(3 == c)
        self.assertFalse(5 == c)

        self.assertTrue(c == 3)
        self.assertFalse(c == 5)

        self.assertTrue(3 == e)
        self.assertTrue(7 == e)
        self.assertFalse(5 == e)

        self.assertTrue(e == 3)
        self.assertTrue(e == 7)
        self.assertFalse(e == 5)

        x = 'qwerty'
        self.assertTrue(x == cf.eq('qwerty'))
        self.assertTrue(x == cf.eq(re.compile('^qwerty$')))
        self.assertTrue(x == cf.eq(re.compile('qwe')))
        self.assertTrue(x == cf.eq(re.compile('qwe.*')))
        self.assertTrue(x == cf.eq(re.compile('.*qwe')))
        self.assertTrue(x == cf.eq(re.compile('.*rty')))
        self.assertTrue(x == cf.eq(re.compile('.*rty$')))
        self.assertTrue(x == cf.eq(re.compile('^.*rty$')))
        self.assertTrue(x == cf.eq(re.compile('rty$')))

        self.assertTrue(x != cf.eq('QWERTY'))
        self.assertTrue(x != cf.eq(re.compile('QWERTY')))
        self.assertTrue(x != cf.eq(re.compile('^QWERTY$')))
        self.assertTrue(x != cf.eq(re.compile('QWE')))
        self.assertTrue(x != cf.eq(re.compile('QWE.*')))
        self.assertTrue(x != cf.eq(re.compile('.*QWE')))
        self.assertTrue(x != cf.eq(re.compile('.*RTY')))
        self.assertTrue(x != cf.eq(re.compile('.*RTY$')))
        self.assertTrue(x != cf.eq(re.compile('^.*RTY$')))


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
