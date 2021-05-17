import datetime
import faulthandler
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DimensionCoordinateTest(unittest.TestCase):
    f = cf.example_field(1)

    dim = cf.DimensionCoordinate()
    dim.standard_name = "latitude"
    a = numpy.array(
        [
            -30,
            -23.5,
            -17.8123,
            -11.3345,
            -0.7,
            -0.2,
            0,
            0.2,
            0.7,
            11.30003,
            17.8678678,
            23.5,
            30,
        ]
    )
    dim.set_data(cf.Data(a, "degrees_north"))
    bounds = cf.Bounds()
    b = numpy.empty(a.shape + (2,))
    b[:, 0] = a - 0.1
    b[:, 1] = a + 0.1
    bounds.set_data(cf.Data(b))
    dim.set_bounds(bounds)

    def test_DimensionCoordinate__repr__str__dump(self):
        x = self.f.dimension_coordinate("X")

        repr(x)
        str(x)
        x.dump(display=False)

    def test_DimensionCoordinate_convert_reference_time(self):
        d = cf.DimensionCoordinate()
        d.set_data(
            cf.Data([1, 2, 3], "months since 2004-1-1", calendar="gregorian")
        )
        self.assertTrue((d.array == [1.0, 2, 3]).all())

        e = d.copy()
        self.assertIsNone(
            e.convert_reference_time(calendar_months=True, inplace=True)
        )

        f = d.convert_reference_time(calendar_months=True)

        for x in (e, f):
            self.assertTrue((x.array == [31.0, 60.0, 91.0]).all())
            self.assertTrue(
                (
                    x.datetime_array
                    == [
                        cf.dt("2004-02-01 00:00:00", calendar="gregorian"),
                        cf.dt("2004-03-01 00:00:00", calendar="gregorian"),
                        cf.dt("2004-04-01 00:00:00", calendar="gregorian"),
                    ]
                ).all()
            )

        self.assertTrue((d.array == [1.0, 2, 3]).all())

        d = cf.DimensionCoordinate()
        d.set_data(
            cf.Data([1, 2, 3], "months since 2004-1-1", calendar="360_day")
        )
        e = d.copy()
        self.assertIsNone(
            e.convert_reference_time(calendar_months=True, inplace=True)
        )

        f = d.convert_reference_time(calendar_months=True)

        for x in (e, f):
            self.assertTrue((x.array == [30.0, 60.0, 90.0]).all())
            self.assertTrue(
                (
                    x.datetime_array
                    == [
                        cf.dt("2004-02-01 00:00:00", calendar="360_day"),
                        cf.dt("2004-03-01 00:00:00", calendar="360_day"),
                        cf.dt("2004-04-01 00:00:00", calendar="360_day"),
                    ]
                ).all()
            )

        self.assertTrue((d.array == [1.0, 2, 3]).all())

        d = cf.DimensionCoordinate()
        d.set_data(
            cf.Data([1, 2, 3], "months since 2004-1-1", calendar="noleap")
        )
        e = d.copy()
        self.assertIsNone(
            e.convert_reference_time(calendar_months=True, inplace=True)
        )

        f = d.convert_reference_time(calendar_months=True)

        for x in (e, f):
            self.assertTrue((x.array == [31.0, 59.0, 90.0]).all())
            self.assertTrue(
                (
                    x.datetime_array
                    == [
                        cf.dt("2004-02-01 00:00:00", calendar="noleap"),
                        cf.dt("2004-03-01 00:00:00", calendar="noleap"),
                        cf.dt("2004-04-01 00:00:00", calendar="noleap"),
                    ]
                ).all()
            )

        self.assertTrue((d.array == [1.0, 2, 3]).all())

    def test_DimensionCoordinate_roll(self):
        x = self.f.dimension_coordinate("X").copy()
        y = self.f.dimension_coordinate("Y")

        x.roll(0, 3)
        with self.assertRaises(Exception):
            y.roll(0, 3)

        x.roll(0, 3)
        x.roll(-1, 3)
        with self.assertRaises(Exception):
            x.roll(2, 3)

        a = x[0]
        a.roll(0, 3)
        self.assertIsNone(a.roll(0, 3, inplace=True))

        x.roll(0, 0)
        x.roll(0, 3, inplace=True)
        self.assertIsNone(x.roll(0, 0, inplace=True))

        x._centre(360)
        x.flip()._centre(360)

        # Test roll on coordinate without bounds:
        g = self.f.copy()
        g.dimension_coordinate("X").del_bounds()

        for shift_by in [1, -1, g.shape[2]]:  # vary roll direction and extent
            g_rolled = g.roll("X", shift=shift_by)

            if shift_by == g.shape[2]:  # shift_by equal to the roll axis size
                g_rolled_0 = g.roll("X", shift=0)
                # A roll of the axes size, or 0, should not change the array:
                self.assertTrue((g_rolled.array == g.array).all())
                self.assertTrue((g_rolled.array == g_rolled_0.array).all())

            for index in range(0, 10):  # check all elements are rolled
                self.assertEqual(
                    g_rolled.array[0, index, 0], g.array[0, index, -shift_by]
                )

    def test_DimensionCoordinate_cellsize(self):
        d = self.dim.copy()

        c = d.cellsize
        self.assertTrue(numpy.allclose(c.array, 0.2))

        self.assertTrue(d.Units.equals(cf.Units("degrees_north")))
        self.assertTrue(d.bounds.Units.equals(cf.Units("degrees_north")))

        d.override_units("km", inplace=True)
        self.assertTrue(d.Units.equals(cf.Units("km")))
        self.assertTrue(d.bounds.Units.equals(cf.Units("km")))

        c = d.cellsize
        self.assertTrue(c.Units.equals(cf.Units("km")))

        d.del_bounds()
        c = d.cellsize
        self.assertTrue(numpy.allclose(c.array, 0))

    def test_DimensionCoordinate_override_units(self):
        d = self.dim.copy()

        self.assertTrue(d.Units.equals(cf.Units("degrees_north")))
        self.assertTrue(d.bounds.Units.equals(cf.Units("degrees_north")))

        d.override_units("km", inplace=True)
        self.assertTrue(d.Units.equals(cf.Units("km")))
        self.assertTrue(d.bounds.Units.equals(cf.Units("km")))

        c = d.cellsize
        self.assertTrue(c.Units.equals(cf.Units("km")))

    def test_DimensionCoordinate_override_calendar(self):
        d = self.dim.copy()

        self.assertTrue(d.Units.equals(cf.Units("degrees_north")))
        self.assertTrue(d.bounds.Units.equals(cf.Units("degrees_north")))

        d.override_units("days since 2000-01-01", inplace=True)
        self.assertTrue(d.Units.equals(cf.Units("days since 2000-01-01")))
        self.assertTrue(
            d.bounds.Units.equals(cf.Units("days since 2000-01-01"))
        )

        d.override_calendar("360_day", inplace=True)
        self.assertTrue(
            d.Units.equals(
                cf.Units("days since 2000-01-01", calendar="360_day")
            )
        )
        self.assertTrue(
            d.bounds.Units.equals(
                cf.Units("days since 2000-01-01", calendar="360_day")
            )
        )

        d.override_calendar("365_day", inplace=True)
        self.assertTrue(
            d.Units.equals(
                cf.Units("days since 2000-01-01", calendar="365_day")
            )
        )
        self.assertTrue(
            d.bounds.Units.equals(
                cf.Units("days since 2000-01-01", calendar="365_day")
            )
        )

    def test_DimensionCoordinate_bounds(self):
        x = self.f.dimension_coordinate("X")

        x.upper_bounds
        x.lower_bounds

        self.assertTrue(x.increasing)

        y = x.flip()
        self.assertTrue(y.decreasing)
        self.assertTrue(y.upper_bounds.equals(x.upper_bounds[::-1]))
        self.assertTrue(y.lower_bounds.equals(x.lower_bounds[::-1]))

        x.cellsize
        y.cellsize

        y.del_bounds()

        y.create_bounds()

    def test_DimensionCoordinate_properties(self):
        x = self.f.dimension_coordinate("X").copy()

        x.positive = "up"
        self.assertEqual(x.positive, "up")
        del x.positive

        x.axis = "Z"
        self.assertEqual(x.axis, "Z")
        del x.axis

        x.axis = "T"
        self.assertEqual(x.ndim, 1)

    def test_DimensionCoordinate_insert_dimension(self):
        x = self.f.dimension_coordinate("X").copy()

        self.assertEqual(x.shape, (9,))
        self.assertEqual(x.bounds.shape, (9, 2))

        y = x.insert_dimension(0)
        self.assertEqual(y.shape, (1, 9))
        self.assertEqual(y.bounds.shape, (1, 9, 2), y.bounds.shape)

        x.insert_dimension(-1, inplace=True)
        self.assertEqual(x.shape, (9, 1))
        self.assertEqual(x.bounds.shape, (9, 1, 2), x.bounds.shape)

    def test_DimensionCoordinate_unary_operation(self):
        d = self.dim

        self.assertLess(d.minimum(), 0)
        self.assertLess(d.bounds.minimum(), 0)

        d = abs(d)
        self.assertGreaterEqual(d.minimum(), 0, d.array)
        self.assertGreaterEqual(d.bounds.minimum(), 0, d.bounds.array)

        d = -d
        self.assertLess(d.minimum(), 0)
        self.assertLess(d.bounds.minimum(), 0)

        d = +d
        self.assertLess(d.minimum(), 0)
        self.assertLess(d.bounds.minimum(), 0)

        d.dtype = int
        d.bounds.dtype = int
        d = ~d

    def test_DimensionCoordinate_binary_operation(self):
        dim = self.dim

        c = dim.array
        b = dim.bounds.array
        c2 = numpy.expand_dims(c, -1)

        x = dim.copy()
        y = dim.copy()

        old = cf.bounds_combination_mode()

        # ------------------------------------------------------------
        # Out-of-place addition
        # ------------------------------------------------------------
        for value in ("AND", "NONE"):
            cf.bounds_combination_mode(value)
            z = x + 2
            self.assertTrue((z.array == c + 2).all())
            self.assertFalse(z.has_bounds())

        for value in ("OR", "XOR"):
            cf.bounds_combination_mode(value)
            z = x + 2
            self.assertTrue((z.array == c + 2).all())
            self.assertTrue((z.bounds.array == b + 2).all())

        for value in ("AND", "OR"):
            cf.bounds_combination_mode(value)
            z = x + y
            self.assertTrue((z.array == c + c).all())
            self.assertTrue((z.bounds.array == b + b).all())

        for value in ("XOR", "NONE"):
            cf.bounds_combination_mode(value)
            z = x + y
            self.assertTrue((z.array == c + c).all())
            self.assertFalse(z.has_bounds())

        x.del_bounds()

        for value in ("AND", "XOR", "OR", "NONE"):
            cf.bounds_combination_mode(value)
            z = x + 2
            self.assertTrue((z.array == c + 2).all())
            self.assertFalse(z.has_bounds())

        for value in ("AND", "NONE"):
            cf.bounds_combination_mode(value)
            z = x + y
            self.assertTrue((z.array == c + c).all())
            self.assertFalse(z.has_bounds())

        for value in ("OR", "XOR"):
            cf.bounds_combination_mode(value)
            z = x + y
            self.assertTrue((z.array == c + c).all())
            self.assertTrue((z.bounds.array == c2 + b).all())

        # ------------------------------------------------------------
        # In-place addition
        # ------------------------------------------------------------
        for value in ("AND", "NONE"):
            cf.bounds_combination_mode(value)
            x = dim.copy()
            x += 2
            self.assertTrue((x.array == c + 2).all())
            self.assertFalse(x.has_bounds())

        for value in ("OR", "XOR"):
            cf.bounds_combination_mode(value)
            x = dim.copy()
            x += 2
            self.assertTrue((x.array == c + 2).all())
            self.assertTrue((x.bounds.array == b + 2).all())

        for value in ("AND", "OR"):
            cf.bounds_combination_mode(value)
            x = dim.copy()
            x += y
            self.assertTrue((x.array == c + c).all())
            self.assertTrue((x.bounds.array == b + b).all())

        for value in ("XOR", "NONE"):
            cf.bounds_combination_mode(value)
            x = dim.copy()
            x += y
            self.assertTrue((x.array == c + c).all())
            self.assertFalse(x.has_bounds())

        for value in ("XOR", "OR"):
            cf.bounds_combination_mode(value)
            x = dim.copy()
            x.del_bounds()
            x += y
            self.assertTrue((x.array == c + c).all())
            self.assertTrue((x.bounds.array == c2 + b).all())

        for value in ("AND", "NONE"):
            cf.bounds_combination_mode(value)
            x = dim.copy()
            x.del_bounds()
            x += y
            self.assertTrue((x.array == c + c).all())
            self.assertFalse(x.has_bounds())

        # ------------------------------------------------------------
        # Reset constant
        # ------------------------------------------------------------
        cf.bounds_combination_mode(old)

    def test_DimensionCoordinate_set_data(self):
        x = cf.DimensionCoordinate()

        y = x.set_data(cf.Data([1, 2, 3]))
        self.assertIsNone(y)
        self.assertTrue(x.has_data())

        # Test inplace
        x.del_data()
        y = x.set_data(cf.Data([1, 2, 3]), inplace=False)
        self.assertIsInstance(y, cf.DimensionCoordinate)
        self.assertFalse(x.has_data())
        self.assertTrue(y.has_data())

        # Exceptions should be raised for 0-d and N-d (N>=2) data
        with self.assertRaises(Exception):
            y = x.set_data(cf.Data([[1, 2, 3]]))

        with self.assertRaises(Exception):
            y = x.set_data(cf.Data(1))

    def test_DimensionCoordinate__setitem__(self):
        d = self.dim.copy()

        d.array
        d.bounds.array

        d[...] = 999
        self.assertTrue(d.bounds.equals(self.dim.bounds, verbose=3))

        d = self.dim.copy()
        e = self.dim.copy()
        d[...] = -e
        self.assertTrue(d.data.equals(-e.data, verbose=3))
        self.assertTrue(d.bounds.equals(-e.bounds, verbose=3))

        d = self.dim.copy()
        e = self.dim.copy()
        e.del_bounds()
        d[...] = -e
        self.assertTrue(d.data.equals(-e.data, verbose=3))
        self.assertTrue(d.bounds.equals(self.dim.bounds, verbose=3))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
