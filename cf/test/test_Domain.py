import datetime
import re
import unittest

import numpy

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
        x[...] = numpy.arange(0, 360, 40)
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
        lon.data[...] = numpy.arange(60, 150).reshape(9, 10)

        lat = f.auxiliary_coordinate("Y")
        lat.data[...] = numpy.arange(-45, 45).reshape(10, 9)

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


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
