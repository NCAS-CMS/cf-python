import datetime
import os
import time
import unittest

import numpy

import cf


class DimensionCoordinateTest(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'test_file.nc')

        dim1 = cf.DimensionCoordinate()
        dim1.standard_name = 'latitude'
        a = numpy.array(
            [-30, -23.5, -17.8123, -11.3345, -0.7, -0.2, 0, 0.2, 0.7, 11.30003,
             17.8678678, 23.5, 30]
        )
        dim1.set_data(cf.Data(a, 'degrees_north'))
        bounds = cf.Bounds()
        b = numpy.empty(a.shape + (2,))
        b[:, 0] = a - 0.1
        b[:, 1] = a + 0.1
        bounds.set_data(cf.Data(b))
        dim1.set_bounds(bounds)
        self.dim = dim1

    def test_DimensionCoordinate__repr__str__dump(self):
        f = cf.read(self.filename)[0]
        x = f.dimension_coordinates('X').value()

        _ = repr(x)
        _ = str(x)
        _ = x.dump(display=False)

        self.assertTrue(x.isdimension)

    def test_DimensionCoordinate_convert_reference_time(self):
        d = cf.DimensionCoordinate()
        d.set_data(
            cf.Data([1, 2, 3], 'months since 2004-1-1', calendar='gregorian'))
        self.assertTrue((d.array == [1., 2, 3]).all())

        e = d.copy()
        self.assertIsNone(
            e.convert_reference_time(calendar_months=True, inplace=True))

        f = d.convert_reference_time(calendar_months=True)

        for x in (e, f):
            self.assertTrue((x.array == [31., 60., 91.]).all())
            self.assertTrue((
                x.datetime_array ==
                [cf.dt('2004-02-01 00:00:00', calendar='gregorian'),
                 cf.dt('2004-03-01 00:00:00', calendar='gregorian'),
                 cf.dt('2004-04-01 00:00:00', calendar='gregorian')]
            ).all())

        self.assertTrue((d.array == [1., 2, 3]).all())

        d = cf.DimensionCoordinate()
        d.set_data(
            cf.Data([1, 2, 3], 'months since 2004-1-1', calendar='360_day'))
        e = d.copy()
        self.assertIsNone(
            e.convert_reference_time(calendar_months=True, inplace=True))

        f = d.convert_reference_time(calendar_months=True)

        for x in (e, f):
            self.assertTrue((x.array == [30., 60., 90.]).all())
            self.assertTrue((
                x.datetime_array ==
                [cf.dt('2004-02-01 00:00:00', calendar='360_day'),
                 cf.dt('2004-03-01 00:00:00', calendar='360_day'),
                 cf.dt('2004-04-01 00:00:00', calendar='360_day')]
            ).all())

        self.assertTrue((d.array == [1., 2, 3]).all())

        d = cf.DimensionCoordinate()
        d.set_data(
            cf.Data([1, 2, 3], 'months since 2004-1-1', calendar='noleap'))
        e = d.copy()
        self.assertIsNone(
            e.convert_reference_time(calendar_months=True, inplace=True))

        f = d.convert_reference_time(calendar_months=True)

        for x in (e, f):
            self.assertTrue((x.array == [31., 59., 90.]).all())
            self.assertTrue((
                x.datetime_array ==
                [cf.dt('2004-02-01 00:00:00', calendar='noleap'),
                 cf.dt('2004-03-01 00:00:00', calendar='noleap'),
                 cf.dt('2004-04-01 00:00:00', calendar='noleap')]
            ).all())

        self.assertTrue((d.array == [1., 2, 3]).all())

    def test_DimensionCoordinate_roll(self):
        f = cf.read(self.filename)[0]

        x = f.dimension_coordinates('X').value()
        y = f.dimension_coordinates('Y').value()

        _ = x.roll(0, 3)
        with self.assertRaises(Exception):
            y.roll(0, 3)

        _ = x.roll(0, 3)
        _ = x.roll(-1, 3)
        with self.assertRaises(Exception):
            _ = x.roll(2, 3)

        a = x[0]
        _ = a.roll(0, 3)
        self.assertIsNone(a.roll(0, 3, inplace=True))

        _ = x.roll(0, 0)
        _ = x.roll(0, 3, inplace=True)
        self.assertIsNone(x.roll(0, 0, inplace=True))

        _ = x._centre(360)
        _ = x.flip()._centre(360)

        # Test roll on coordinate without bounds:
        g = f.copy()
        g.dimension_coordinate('X').del_bounds()

        for shift_by in [1, -1, g.shape[2]]:  # vary roll direction and extent
            g_rolled = g.roll('X', shift=shift_by)

            if shift_by == g.shape[2]:  # shift_by equal to the roll axis size
                g_rolled_0 = g.roll('X', shift=0)
                # A roll of the axes size, or 0, should not change the array:
                self.assertTrue((g_rolled.array == g.array).all())
                self.assertTrue((g_rolled.array == g_rolled_0.array).all())

            for index in range(0, 10):  # check all elements are rolled
                self.assertTrue(
                    g_rolled.array[0, index, 0] == g.array[0, index, -shift_by]
                )

    def test_DimensionCoordinate_cellsize(self):
        d = self.dim.copy()

        c = d.cellsize
        self.assertTrue(numpy.allclose(c.array, 0.2))

        self.assertTrue(d.Units.equals(cf.Units('degrees_north')))
        self.assertTrue(d.bounds.Units.equals(cf.Units('degrees_north')))

        d.override_units('km', inplace=True)
        self.assertTrue(d.Units.equals(cf.Units('km')))
        self.assertTrue(d.bounds.Units.equals(cf.Units('km')))

        c = d.cellsize
        self.assertTrue(c.Units.equals(cf.Units('km')))

        d.del_bounds()
        c = d.cellsize
        self.assertTrue(numpy.allclose(c.array, 0))

    def test_DimensionCoordinate_override_units(self):
        d = self.dim.copy()

        self.assertTrue(d.Units.equals(cf.Units('degrees_north')))
        self.assertTrue(d.bounds.Units.equals(cf.Units('degrees_north')))

        d.override_units('km', inplace=True)
        self.assertTrue(d.Units.equals(cf.Units('km')))
        self.assertTrue(d.bounds.Units.equals(cf.Units('km')))

        c = d.cellsize
        self.assertTrue(c.Units.equals(cf.Units('km')))

    def test_DimensionCoordinate_override_calendar(self):
        d = self.dim.copy()

        self.assertTrue(d.Units.equals(cf.Units('degrees_north')))
        self.assertTrue(d.bounds.Units.equals(cf.Units('degrees_north')))

        d.override_units('days since 2000-01-01', inplace=True)
        self.assertTrue(d.Units.equals(cf.Units('days since 2000-01-01')))
        self.assertTrue(
            d.bounds.Units.equals(cf.Units('days since 2000-01-01')))

        d.override_calendar('360_day', inplace=True)
        self.assertTrue(
            d.Units.equals(
                cf.Units('days since 2000-01-01', calendar='360_day')))
        self.assertTrue(
            d.bounds.Units.equals(
                cf.Units('days since 2000-01-01', calendar='360_day')))

        d.override_calendar('365_day', inplace=True)
        self.assertTrue(
            d.Units.equals(
                cf.Units('days since 2000-01-01', calendar='365_day')))
        self.assertTrue(
            d.bounds.Units.equals(
                cf.Units('days since 2000-01-01', calendar='365_day')))

    def test_DimensionCoordinate_bounds(self):
        f = cf.read(self.filename)[0]
        x = f.dimension_coordinates('X').value()

        _ = x.upper_bounds
        _ = x.lower_bounds

        self.assertTrue(x.increasing)

        y = x.flip()
        self.assertTrue(y.decreasing)
        self.assertTrue(y.upper_bounds.equals(x.upper_bounds[::-1]))
        self.assertTrue(y.lower_bounds.equals(x.lower_bounds[::-1]))

        c = x.cellsize
        c = y.cellsize

        y.del_bounds()

        b = y.create_bounds()

    def test_DimensionCoordinate_properties(self):
        f = cf.read(self.filename)[0]
        x = f.dimension_coordinates('X').value()

        x.positive = 'up'
        self.assertTrue(x.positive == 'up')
        del x.positive

        x.axis = 'Z'
        self.assertTrue(x.axis == 'Z')
        del x.axis

        x.axis = 'T'
        self.assertTrue(x.ndim == 1)

    def test_DimensionCoordinate_insert_dimension(self):
        f = cf.read(self.filename)[0]
        x = f.dimension_coordinates('X').value()

        self.assertTrue(x.shape == (9,))
        self.assertTrue(x.bounds.shape == (9, 2))

        y = x.insert_dimension(0)
        self.assertTrue(y.shape == (1, 9))
        self.assertTrue(y.bounds.shape == (1, 9, 2), y.bounds.shape)

        x.insert_dimension(-1, inplace=True)
        self.assertTrue(x.shape == (9, 1))
        self.assertTrue(x.bounds.shape == (9, 1, 2), x.bounds.shape)

    def test_DimensionCoordinate_binary_operation(self):
        f = cf.read(self.filename)[0]
        x = f.dimension_coordinates('X').value()

        d = x.array
        b = x.bounds.array
        d2 = numpy.expand_dims(d, -1)

        # --------------------------------------------------------
        # Out-of-place addition
        # --------------------------------------------------------
        c = x + 2
        self.assertTrue((c.array == d + 2).all())
        self.assertTrue((c.bounds.array == b + 2).all())

        c = x + x
        self.assertTrue((c.array == d + d).all())
        self.assertTrue((c.bounds.array == b + d2).all())

        c = x + 2
        self.assertTrue((c.array == d + 2).all())
        self.assertTrue((c.bounds.array == b + 2).all())

        self.assertTrue((x.array == d).all())
        self.assertTrue((x.bounds.array == b).all())

        # --------------------------------------------------------
        # In-place addition
        # --------------------------------------------------------
        x += 2
        self.assertTrue((x.array == d + 2).all())
        self.assertTrue((x.bounds.array == b + 2).all())

        x += x
        self.assertTrue((x.array == (d+2) * 2).all())
        self.assertTrue((x.bounds.array == b+2 + d2+2).all())

        x += 2
        self.assertTrue((x.array == (d+2)*2 + 2).all())
        self.assertTrue((x.bounds.array == b+2 + d2+2 + 2).all())

        # --------------------------------------------------------
        # Out-of-place addition (no bounds)
        # --------------------------------------------------------
        f = cf.read(self.filename)[0]
        x = f.dimension_coordinates('X').value()
        x.del_bounds()

        self.assertFalse(x.has_bounds())

        d = x.array

        c = x + 2
        self.assertTrue((c.array == d + 2).all())

        c = x + x
        self.assertTrue((c.array == d + d).all())

        c = x + 2
        self.assertTrue((c.array == d + 2).all())

        self.assertTrue((x.array == d).all())

        # --------------------------------------------------------
        # In-place addition (no bounds)
        # --------------------------------------------------------
        x += 2
        self.assertTrue((x.array == d + 2).all())

        x += x
        self.assertTrue((x.array == (d+2) * 2).all())

        x += 2
        self.assertTrue((x.array == (d+2)*2 + 2).all())


# --- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
