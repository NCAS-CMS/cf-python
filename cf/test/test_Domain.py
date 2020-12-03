import datetime
import inspect
import os
import unittest

import cf


class DomainTest(unittest.TestCase):
    d = cf.example_field(1)

    def setUp(self):
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL('DISABLE')
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')
        self.test_only = []

    def test_Domain__repr__str__dump(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = self.d

        _ = repr(d)
        _ = str(d)
        self.assertIsInstance(d.dump(display=False), str)

    def test_Domain__init__(self):
        d = cf.Domain(source='qwerty')

    def test_Domain_equals(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        d = self.d
        e = d.copy()

        self.assertTrue(d.equals(d, verbose=3))
        self.assertTrue(d.equals(e, verbose=3))
        self.assertTrue(e.equals(d, verbose=3))

    def test_Domain_properties(self):
        d = cf.Domain()

        d.set_property('long_name', 'qwerty')

        self.assertEqual(d.properties(), {'long_name': 'qwerty'})
        self.assertEqual(d.get_property('long_name'), 'qwerty')
        self.assertEqual(d.del_property('long_name'), 'qwerty')
        self.assertIsNone(d.get_property('long_name', None))
        self.assertIsNonec(d.del_property('long_name', None))

        d.set_property('long_name', 'qwerty')
        self.assertEqual(d.clear_properties(), {'long_name': 'qwerty'})

        d.set_properties({'long_name': 'qwerty'})
        d.set_properties({'foo': 'bar'})
        self.assertEqual(d.properties(),
                         {'long_name': 'qwerty', 'foo': 'bar'})

    def test_Domain_indices(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = self.d

        x = f.dimension_coordinate('X')
        a = x.varray
        a[...] = numpy.arange(0, 360, 40)
        x.set_bounds(x.create_bounds())
        f.cyclic('X', iscyclic=True, period=360)

        f0 = f.copy()

        # wi (increasing)
        indices = f.indices(grid_longitude=cf.wi(50, 130))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 2), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [80, 120]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-80, -40, 0, 40]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))

        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310-1080, 450-1080))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(310+720, 450+720))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 370))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue(
            (x == [-80, -40, 0, 40,  80, 120, 160, 200, 240.]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices('full', grid_longitude=cf.wi(310, 450))
        self.assertTrue(indices[0], 'mask')
        self.assertTrue(
            (
                indices[1][0].array == [[[False, False, False,
                                          True, True, True, True, True,
                                          False]]]
            ).all()
        )
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)

        x = g.dimension_coordinate('X').array
        self.assertEqual(x.shape, (9,), x.shape)

        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x)

        a = array.copy()
        a[..., 3:8] = numpy.ma.masked

        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        indices = f.indices('full', grid_longitude=cf.wi(70, 200))
        self.assertTrue(indices[0], 'mask')
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertEqual(x.shape, (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320]).all(), x)
        a = array.copy()
        a[..., [0, 1, 6, 7, 8]] = numpy.ma.masked
        self.assertTrue(cf.functions._numpy_allclose(g.array, a), g.array)

        # wi (decreasing)
        f.flip('X', inplace=True)

        indices = f.indices(grid_longitude=cf.wi(50, 130))
        self.assertTrue(indices[0], 'mask')
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 2), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [80, 120][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(-90, 50))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-80, -40, 0, 40][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310-1080, 450-1080))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        indices = f.indices(grid_longitude=cf.wi(310+720, 450+720))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 4), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-40, 0, 40, 80][::-1]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wi(90, 100))

        indices = f.indices('full', grid_longitude=cf.wi(310, 450))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertEqual(x.shape, (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x)

        indices = f.indices('full', grid_longitude=cf.wi(70, 200))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertEqual(x.shape, (9,), x.shape)
        self.assertTrue(
            (x == [0, 40, 80, 120, 160, 200, 240, 280, 320][::-1]).all(), x)

        # wo
        f = f0.copy()

        indices = f.indices(grid_longitude=cf.wo(50, 130))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 7), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [-200, -160, -120, -80, -40, 0, 40]).all())

        with self.assertRaises(IndexError):
            f.indices(grid_longitude=cf.wo(-90, 370))

        # set
        indices = f.indices(grid_longitude=cf.set([320, 40, 80, 99999]))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [40, 80, 320]).all())

        indices = f.indices(grid_longitude=cf.lt(90))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.gt(90))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 6), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [120, 160, 200, 240, 280, 320]).all())

        indices = f.indices(grid_longitude=cf.le(80))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 3), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [0, 40, 80]).all())

        indices = f.indices(grid_longitude=cf.ge(80))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 7), g.shape)
        x = g.dimension_coordinate('X').array
        self.assertTrue((x == [80, 120, 160, 200, 240, 280, 320]).all())

        # 2-d
        lon = f.construct('longitude').array
        lon = numpy.transpose(lon)
        lon = numpy.expand_dims(lon, 0)

        lat = f.construct('latitude').array
        lat = numpy.expand_dims(lat, 0)

        array = numpy.ma.where((lon >= 92) & (lon <= 134),
                               f.array,
                               numpy.ma.masked)

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, longitude=cf.wi(92, 134))
            g = f[indices]
            if mode == 'full':
                shape = (1, 10, 9)
                array2 = array
            elif mode == 'envelope':
                shape = (1, 10, 5)
                array2 = array[..., 3:8]
            else:
                shape = (1, 10, 5)
                array2 = array[..., 3:8]

            self.assertEqual(g.shape, shape, str(g.shape)+'!='+str(shape))
            self.assertTrue(
                cf.functions._numpy_allclose(array2, g.array), g.array)

        array = numpy.ma.where(((lon >= 72) & (lon <= 83)) | (lon >= 118),
                               f.array,
                               numpy.ma.masked)

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, longitude=cf.wi(72, 83) | cf.gt(118))
            g = f[indices]
            if mode == 'full':
                shape = (1, 10, 9)
            elif mode == 'envelope':
                shape = (1, 10, 8)
            else:
                shape = (1, 10, 6)

            self.assertEqual(g.shape, shape, str(g.shape)+'!='+str(shape))

        indices = f.indices('full',
                            longitude=cf.wi(92, 134),
                            latitude=cf.wi(-26, -20) | cf.ge(30))
        g = f[indices]
        self.assertEqual(g.shape, (1, 10, 9), g.shape)
        array = numpy.ma.where(
            (((lon >= 92) & (lon <= 134)) &
             (((lat >= -26) & (lat <= -20)) | (lat >= 30))),
            f.array,
            numpy.ma.masked
        )
        self.assertTrue(cf.functions._numpy_allclose(array, g.array), g.array)

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, grid_longitude=cf.contains(23.2))
            g = f[indices]
            if mode == 'full':
                shape = f.shape
            else:
                shape = (1, 10, 1)

            self.assertEqual(g.shape, shape, g.shape)

            if mode != 'full':
                self.assertEqual(
                    g.construct('grid_longitude').array, 40)  # TODO
        # --- End: for

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, grid_latitude=cf.contains(3))
            g = f[indices]
            if mode == 'full':
                shape = f.shape
            else:
                shape = (1, 1, 9)

            self.assertEqual(g.shape, shape, g.shape)

            if mode != 'full':
                self.assertEqual(g.construct('grid_latitude').array, 3)
        # --- End: for

        for mode in ('', 'compress', 'full', 'envelope'):
            indices = f.indices(mode, longitude=cf.contains(83))
            g = f[indices]
            if mode == 'full':
                shape = f.shape
            else:
                shape = (1, 1, 1)

            self.assertEqual(g.shape, shape, g.shape)

            if mode != 'full':
                self.assertEqual(g.construct('longitude').array, 83)
        # --- End: for

        # Calls that should fail
        with self.assertRaises(Exception):
            f.indices(grid_longitudecf.gt(23), grid_longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.indices(grid_longitude=cf.gt(23), longitude=cf.wi(92, 134))
        with self.assertRaises(Exception):
            f.indices(grid_latitude=cf.contains(-23.2))

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print('')
    unittest.main(verbosity=2)
