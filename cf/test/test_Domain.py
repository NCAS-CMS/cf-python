import datetime
import re
import unittest

import numpy as np

import cf


class DomainTest(unittest.TestCase):
    d = cf.example_field(1).domain

    def setUp(self):
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_Domain__repr__str__dump(self):
        d = self.d

        _ = repr(d)
        _ = str(d)
        self.assertIsInstance(d.dump(display=False), str)

    def test_Domain__init__(self):
        cf.Domain(source="qwerty")

    def test_Domain_equals(self):
        d = self.d
        e = d.copy()

        self.assertTrue(d.equals(d, verbose=3))
        self.assertTrue(d.equals(e, verbose=3))
        self.assertTrue(e.equals(d, verbose=3))

    def test_Domain_flip(self):
        f = self.d.copy()

        kwargs = {
            axis: slice(None, None, -1) for axis in f.domain_axes(todict=True)
        }
        g = f.subspace(**kwargs)

        h = f.flip()
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip(["X", "Z", "Y"])
        self.assertTrue(h.equals(g, verbose=1))

        h = f.flip((re.compile("^atmos"), "grid_latitude", "grid_longitude"))
        self.assertTrue(h.equals(g, verbose=1))

        g = f.subspace(grid_longitude=slice(None, None, -1))
        self.assertIsNone(f.flip("X", inplace=True))
        self.assertTrue(f.equals(g, verbose=1))

    def test_Domain_properties(self):
        d = cf.Domain()

        d.set_property("long_name", "qwerty")

        self.assertEqual(d.properties(), {"long_name": "qwerty"})
        self.assertEqual(d.get_property("long_name"), "qwerty")
        self.assertEqual(d.del_property("long_name"), "qwerty")
        self.assertIsNone(d.get_property("long_name", None))
        self.assertIsNone(d.del_property("long_name", None))

        d.set_property("long_name", "qwerty")
        self.assertEqual(d.clear_properties(), {"long_name": "qwerty"})

        d.set_properties({"long_name": "qwerty"})
        d.set_properties({"foo": "bar"})
        self.assertEqual(d.properties(), {"long_name": "qwerty", "foo": "bar"})

    def test_Domain_creation_commands(self):
        for f in cf.example_fields():
            _ = f.domain.creation_commands()

        f = cf.example_field(1).domain

        for rd in (False, True):
            _ = f.creation_commands(representative_data=rd)

        for indent in (0, 4):
            _ = f.creation_commands(indent=indent)

        for s in (False, True):
            _ = f.creation_commands(string=s)

        for ns in ("cf", ""):
            _ = f.creation_commands(namespace=ns)

    def test_Domain_subspace(self):
        f = self.d.copy()

        x = f.dimension_coordinate("X")
        x[...] = np.arange(0, 360, 40)
        x.set_bounds(x.create_bounds())
        f.cyclic("X", iscyclic=True, period=360)

        f0 = f.copy()

        # wi (increasing)
        g = f.subspace(grid_longitude=cf.wi(50, 130))
        self.assertEqual(g.size, 20)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [80, 120]).all())

        g = f.subspace(grid_longitude=cf.wi(-90, 50))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-80, -40, 0, 40]).all())

        g = f.subspace(grid_longitude=cf.wi(310, 450))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        g = f.subspace(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        g = f.subspace(grid_longitude=cf.wi(310 + 720, 450 + 720))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        g = f.subspace(grid_longitude=cf.wi(-90, 370))
        self.assertEqual(g.size, 90)
        x = g.construct("grid_longitude").array
        self.assertTrue(
            (x == [-80, -40, 0, 40, 80, 120, 160, 200, 240.0]).all()
        )

        with self.assertRaises(ValueError):
            f.indices(grid_longitude=cf.wi(90, 100))

        # wi (decreasing)
        f.flip("X", inplace=True)

        g = f.subspace(grid_longitude=cf.wi(50, 130))
        self.assertEqual(g.size, 20)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [80, 120][::-1]).all())

        g = f.subspace(grid_longitude=cf.wi(-90, 50))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-80, -40, 0, 40][::-1]).all())

        g = f.subspace(grid_longitude=cf.wi(310, 450))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        g = f.subspace(grid_longitude=cf.wi(310 - 1080, 450 - 1080))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        g = f.subspace(grid_longitude=cf.wi(310 + 720, 450 + 720))
        self.assertEqual(g.size, 40)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        with self.assertRaises(ValueError):
            f.indices(grid_longitude=cf.wi(90, 100))

        # wo
        f = f0.copy()

        g = f.subspace(grid_longitude=cf.wo(50, 130))
        self.assertEqual(g.size, 70)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40]).all())

        with self.assertRaises(ValueError):
            f.indices(grid_longitude=cf.wo(-90, 370))

        # set
        g = f.subspace(grid_longitude=cf.set([320, 40, 80, 99999]))
        self.assertEqual(g.size, 30)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [40, 80, 320]).all())

        g = f.subspace(grid_longitude=cf.lt(90))
        self.assertEqual(g.size, 30)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [0, 40, 80]).all())

        g = f.subspace(grid_longitude=cf.gt(90))
        self.assertEqual(g.size, 60)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [120, 160, 200, 240, 280, 320]).all())

        g = f.subspace(grid_longitude=cf.le(80))
        self.assertEqual(g.size, 30)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [0, 40, 80]).all())

        g = f.subspace(grid_longitude=cf.ge(80))
        self.assertEqual(g.size, 70)
        x = g.construct("grid_longitude").array
        self.assertTrue((x == [80, 120, 160, 200, 240, 280, 320]).all())

        # 2-d
        lon = f.auxiliary_coordinate("X")
        lon.data[...] = np.arange(60, 150).reshape(9, 10)

        lat = f.auxiliary_coordinate("Y")
        lat.data[...] = np.arange(-45, 45).reshape(10, 9)

        for mode in ("compress", "envelope"):
            g = f.subspace(mode, longitude=cf.wi(92, 134))
            size = 50
            self.assertEqual(g.size, size)

        for mode in (("compress",), ("envelope",)):
            g = f.subspace(*mode, longitude=cf.wi(72, 83) | cf.gt(118))
            if mode == ("envelope",):
                size = 80
            else:
                size = 60

            self.assertEqual(g.size, size, mode)

        for mode in ("compress", "envelope"):
            g = f.subspace(mode, grid_longitude=cf.contains(23.2))
            size = 10
            self.assertEqual(g.size, size)
            self.assertEqual(g.construct("grid_longitude").array, 40)

        for mode in ("compress", "envelope"):
            g = f.subspace(mode, grid_latitude=cf.contains(1))
            size = 9
            self.assertEqual(g.size, size)
            self.assertEqual(g.construct("grid_latitude").array, 0.88)

        for mode in ("compress", "envelope"):
            g = f.subspace(mode, longitude=cf.contains(83))
            size = 1
            self.assertEqual(g.size, size)

            self.assertEqual(g.construct("longitude").array, 83)

        # Calls that should fail
        with self.assertRaises(Exception):
            f.subspace(grid_longitude=cf.gt(23), X=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.subspace(grid_longitude=cf.gt(23), longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.subspace(grid_latitude=cf.contains(-23.2))

    def test_Domain_transpose(self):
        f = cf.example_field(1)
        d = f.domain

        axes = [re.compile("^atmos"), "grid_latitude", "X"]

        g = f.transpose(axes, constructs=True)
        e = d.transpose(axes + ["T"])
        self.assertTrue(e.equals(g.domain))

        self.assertIsNone(e.transpose(axes + ["T"], inplace=True))

        with self.assertRaises(ValueError):
            d.transpose(["X", "Y"])

        with self.assertRaises(ValueError):
            d.transpose(["X", "Y", 1])

        with self.assertRaises(ValueError):
            d.transpose([2, 1])

        with self.assertRaises(ValueError):
            d.transpose(["Y", "Z"])

        with self.assertRaises(ValueError):
            d.transpose(["Y", "Y", "Z"])

        with self.assertRaises(ValueError):
            d.transpose(["Y", "X", "Z", "Y"])

        with self.assertRaises(ValueError):
            d.transpose(["Y", "X", "Z", 1])

    def test_Domain_size(self):
        self.assertEqual(self.d.size, 90)

    def test_Domain_create_regular(self):
        domain = cf.Domain.create_regular((-180, 180, 1), (-90, 90, 1))
        self.assertIsInstance(domain, cf.Domain)

        # Invalid inputs
        with self.assertRaises(ValueError):
            cf.Domain.create_regular((-180, 180, 1, 2), (-90, 90, 1))

        with self.assertRaises(ValueError):
            cf.Domain.create_regular((-180, 180, 1), (-90, 90, 1, 2))

        # Test dx and dy as divisors of the range
        domain = cf.Domain.create_regular((-180, 180, 60), (-90, 90, 45))
        self.assertIsNotNone(domain)

        x_bounds = np.linspace(-180, 180, 7)
        y_bounds = np.linspace(-90, 90, 5)

        x_points = (x_bounds[:-1] + x_bounds[1:]) / 2
        y_points = (y_bounds[:-1] + y_bounds[1:]) / 2

        longitude = domain.construct("longitude")
        latitude = domain.construct("latitude")

        self.assertTrue(np.allclose(longitude.array, x_points))
        self.assertTrue(np.allclose(latitude.array, y_points))

        # Test if range difference in x_range is greater than 360
        with self.assertRaises(ValueError):
            cf.Domain.create_regular((-180, 190, 1), (-90, 90, 1))

        # Test for y_range out of bounds
        with self.assertRaises(ValueError):
            cf.Domain.create_regular((-180, 180, 1), (-91, 90, 1))
        with self.assertRaises(ValueError):
            cf.Domain.create_regular((-180, 180, 1), (-90, 91, 1))

        # Test for decreasing coordinates range
        with self.assertRaises(ValueError):
            cf.Domain.create_regular((180, -180, 1), (-90, 90, 1))
        with self.assertRaises(ValueError):
            cf.Domain.create_regular((-180, 180, 1), (90, -90, 1))

        # Test cyclicity
        d = cf.Domain.create_regular((-180, 180, 1), (90, -90, -1))
        axis = d.domain_axis("X", key=True)
        self.assertEqual(d.cyclic().pop(), axis)

        # Test with bounds=False
        domain_no_bounds = cf.Domain.create_regular(
            (-180, 180, 1), (-90, 90, 1), bounds=False
        )
        self.assertIsInstance(domain_no_bounds, cf.Domain)

        x_points_no_bounds = np.arange(-180, 181, 1)
        y_points_no_bounds = np.arange(-90, 91, 1)

        longitude_no_bounds = domain_no_bounds.construct("longitude")
        latitude_no_bounds = domain_no_bounds.construct("latitude")

        self.assertTrue(
            np.allclose(longitude_no_bounds.array, x_points_no_bounds)
        )
        self.assertTrue(
            np.allclose(latitude_no_bounds.array, y_points_no_bounds)
        )

        # Test for the given specific domain
        ymin, ymax, dy = 45.0, 90.0, 0.0083333
        xmin, xmax, dx = 250.0, 360.0, 0.0083333

        domain_specific = cf.Domain.create_regular(
            (xmin, xmax, dx), (ymin, ymax, dy)
        )
        self.assertIsInstance(domain_specific, cf.Domain)

        x_bounds_specific = np.arange(xmin, xmax + dx, dx)
        y_bounds_specific = np.arange(ymin, ymax + dy, dy)

        x_points_specific = (
            x_bounds_specific[:-1] + x_bounds_specific[1:]
        ) / 2
        y_points_specific = (
            y_bounds_specific[:-1] + y_bounds_specific[1:]
        ) / 2

        longitude_specific = domain_specific.construct("longitude")
        latitude_specific = domain_specific.construct("latitude")

        self.assertTrue(
            np.allclose(longitude_specific.array - x_points_specific, 0)
        )
        self.assertTrue(
            np.allclose(latitude_specific.array - y_points_specific, 0)
        )

    def test_Domain_del_construct(self):
        """Test the `del_construct` Domain method."""
        # Test a domain without cyclic axes. These are equivalent tests to
        # those in the cfdm test suite, to check behaviour is the same in cf.
        d = self.d.copy()

        self.assertIsInstance(
            d.del_construct("dimensioncoordinate1"), cf.DimensionCoordinate
        )
        self.assertIsInstance(
            d.del_construct("auxiliarycoordinate1"), cf.AuxiliaryCoordinate
        )
        with self.assertRaises(ValueError):
            d.del_construct("auxiliarycoordinate1")

        self.assertIsNone(
            d.del_construct("auxiliarycoordinate1", default=None)
        )

        self.assertIsInstance(d.del_construct("measure:area"), cf.CellMeasure)

        # NOTE: this test will fail presently because of a bug which means
        # that Field.domain doesn't inherit the cyclic() axes of the
        # corresponding Field (see Issue #762) which will be fixed shortly.
        #
        # Test a domain with cyclic axes, to ensure the cyclic() set is
        # updated accordingly if a cyclic axes is the one removed.
        e = cf.example_field(2).domain  # this has a cyclic axes 'domainaxis2'
        # To delete a cyclic axes, must first delete this dimension coordinate
        # because 'domainaxis2' spans it.
        self.assertIsInstance(
            e.del_construct("dimensioncoordinate2"), cf.DimensionCoordinate
        )
        self.assertEqual(e.cyclic(), set(("domainaxis2",)))
        self.assertIsInstance(e.del_construct("domainaxis2"), cf.DomainAxis)
        self.assertEqual(e.cyclic(), set())

    def test_Domain_cyclic_iscyclic(self):
        """Test the `cyclic` and `iscyclic` Domain methods."""
        # A field and its domain should have the same cyclic() output.
        f1 = cf.example_field(1)  # no cyclic axes
        d1 = f1.domain
        f2 = cf.example_field(2)  # one cyclic axis, 'domainaxis2' ('X')
        d2 = f2.domain

        # Getting
        self.assertEqual(d1.cyclic(), f1.cyclic())
        self.assertEqual(d1.cyclic(), set())
        self.assertFalse(d1.iscyclic("X"))
        self.assertFalse(d1.iscyclic("Y"))
        self.assertFalse(d1.iscyclic("Z"))
        self.assertFalse(d1.iscyclic("T"))
        self.assertEqual(d2.cyclic(), f2.cyclic())
        self.assertEqual(d2.cyclic(), set(("domainaxis2",)))
        self.assertTrue(d2.iscyclic("X"))
        self.assertFalse(d2.iscyclic("Y"))
        self.assertFalse(d2.iscyclic("Z"))
        self.assertFalse(d2.iscyclic("T"))

        # Setting
        self.assertEqual(d2.cyclic("X", iscyclic=False), set(("domainaxis2",)))
        self.assertEqual(d2.cyclic(), set())
        self.assertEqual(d2.cyclic("X", period=360), set())
        self.assertEqual(d2.cyclic(), set(("domainaxis2",)))
        self.assertTrue(d2.iscyclic("X"))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
