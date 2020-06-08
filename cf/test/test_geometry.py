from __future__ import print_function
import datetime
import inspect
import os
import tempfile
import unittest

from distutils.version import LooseVersion

import numpy
import netCDF4

import cf

VN = cf.CF()


class DSGTest(unittest.TestCase):
    def setUp(self):
        self.geometry_1_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'geometry_1.nc')
        self.geometry_2_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'geometry_2.nc')
        self.geometry_3_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'geometry_3.nc')
        self.geometry_4_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'geometry_4.nc')
        self.geometry_interior_ring_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'geometry_interior_ring.nc')
        self.geometry_interior_ring_file_2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'geometry_interior_ring_2.nc')

        (fd, self.tempfilename) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_', dir='.')
        os.close(fd)
#        self.tempfilename = 'delme.nc'

        self.test_only = []
#        self.test_only = ['test_node_count']
#        self.test_only = ['test_geometry_interior_ring']

    def tearDown(self):
        os.remove(self.tempfilename)

    def test_node_count(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_1_file, verbose=0)

        self.assertTrue(len(f) == 2, 'f = '+repr(f))
        for g in f:
            self.assertTrue(g.equals(g.copy(), verbose=2))
            self.assertTrue(len(g.auxiliary_coordinates) == 2)

        g = f[0]
        for axis in ('X', 'Y'):
            coord = g.construct('axis='+axis)
            self.assertTrue(coord.has_node_count(), 'axis='+axis)
            self.assertFalse(coord.has_part_node_count(), 'axis='+axis)
            self.assertFalse(coord.has_interior_ring(), 'axis='+axis)

        cf.write(f, self.tempfilename, Conventions='CF-'+VN, verbose=0)

        f2 = cf.read(self.tempfilename, verbose=0)
        self.assertTrue(len(f2) == 2, 'f2 = '+repr(f2))
        for a, b in zip(f, f2):
            self.assertTrue(a.equals(b, verbose=2))

        # Setting of node count properties
        coord = f[0].construct('axis=X')
        nc = coord.get_node_count()
        cf.write(f, self.tempfilename)
        nc.set_property('long_name', 'Node counts')
        cf.write(f, self.tempfilename)
        nc.nc_set_variable('new_var_name_X')
        cf.write(f, self.tempfilename)

        # Node count access
        c = g.construct('longitude').copy()
        self.assertTrue(c.has_node_count())
        n = c.del_node_count()
        self.assertFalse(c.has_node_count())
        self.assertIsNone(c.get_node_count(None))
        self.assertIsNone(c.del_node_count(None))
        c.set_node_count(n)
        self.assertTrue(c.has_node_count())
        self.assertTrue(c.get_node_count(None).equals(n, verbose=2))
        self.assertTrue(c.del_node_count(None).equals(n, verbose=2))
        self.assertFalse(c.has_node_count())

    def test_geometry_2(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_2_file, verbose=0)

        self.assertTrue(len(f) == 2, 'f = '+repr(f))

        for g in f:
            self.assertTrue(g.equals(g.copy(), verbose=2))
            self.assertTrue(len(g.auxiliary_coordinates) == 3)

        g = f[0]
        for axis in ('X', 'Y', 'Z'):
            coord = g.construct('axis='+axis)
            self.assertTrue(coord.has_node_count(), 'axis=' + axis)
            self.assertFalse(coord.has_part_node_count(), 'axis=' + axis)
            self.assertFalse(coord.has_interior_ring(), 'axis=' + axis)

        cf.write(f, self.tempfilename, Conventions='CF-'+VN, verbose=0)

        f2 = cf.read(self.tempfilename, verbose=0)

        self.assertTrue(len(f2) == 2, 'f2 = '+repr(f2))

        for a, b in zip(f, f2):
            self.assertTrue(a.equals(b, verbose=2))

        # Setting of node count properties
        coord = f[0].construct('axis=X')
        nc = coord.get_node_count()
        cf.write(f, self.tempfilename)
        nc.set_property('long_name', 'Node counts')
        cf.write(f, self.tempfilename, verbose=0)
        nc.nc_set_variable('new_var_name')
        cf.write(f, self.tempfilename, verbose=0)

    def test_geometry_3(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_3_file, verbose=0)

        self.assertTrue(len(f) == 2, 'f = '+repr(f))

        for g in f:
            self.assertTrue(g.equals(g.copy(), verbose=2))
            self.assertTrue(len(g.auxiliary_coordinates) == 3)

        g = f[0]
        for axis in ('X', 'Y', 'Z'):
            coord = g.construct('axis='+axis)
            self.assertFalse(coord.has_node_count(), 'axis='+axis)
            self.assertFalse(coord.has_part_node_count(), 'axis='+axis)
            self.assertFalse(coord.has_interior_ring(), 'axis='+axis)

        cf.write(f, self.tempfilename, Conventions='CF-'+VN, verbose=0)

        f2 = cf.read(self.tempfilename, verbose=0)

        self.assertTrue(len(f2) == 2, 'f2 = '+repr(f2))

        for a, b in zip(f, f2):
            self.assertTrue(a.equals(b, verbose=2))

    def test_geometry_4(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_4_file, verbose=0)

        self.assertTrue(len(f) == 2, 'f = '+repr(f))

        for g in f:
            self.assertTrue(g.equals(g.copy(), verbose=2))
            self.assertTrue(len(g.auxiliary_coordinates) == 3)

        for axis in ('X', 'Y'):
            coord = g.construct('axis='+axis)
            self.assertTrue(coord.has_node_count(), 'axis='+axis)
            self.assertFalse(coord.has_part_node_count(), 'axis='+axis)
            self.assertFalse(coord.has_interior_ring(), 'axis='+axis)

        cf.write(f, self.tempfilename, Conventions='CF-'+VN, verbose=0)

        f2 = cf.read(self.tempfilename, verbose=0)

        self.assertTrue(len(f2) == 2, 'f2 = '+repr(f2))

        for a, b in zip(f, f2):
            self.assertTrue(a.equals(b, verbose=2))

        # Setting of node count properties
        coord = f[0].construct('axis=X')
        nc = coord.get_node_count()
        cf.write(f, self.tempfilename)
        nc.set_property('long_name', 'Node counts')
        cf.write(f, self.tempfilename, verbose=0)
        nc.nc_set_variable('new_var_name')
        cf.write(f, self.tempfilename, verbose=0)

    def test_geometry_interior_ring(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        for geometry_file in (self.geometry_interior_ring_file,
                              self.geometry_interior_ring_file_2):
            f = cf.read(geometry_file, verbose=0)

            self.assertTrue(len(f) == 2, 'f = '+repr(f))

            for g in f:
                self.assertTrue(g.equals(g.copy(), verbose=2))
                self.assertTrue(len(g.auxiliary_coordinates) == 4)

            g = f[0]
            for axis in ('X', 'Y'):
                coord = g.construct('axis='+axis)
                self.assertTrue(coord.has_node_count(), 'axis='+axis)
                self.assertTrue(coord.has_part_node_count(), 'axis='+axis)
                self.assertTrue(coord.has_interior_ring(), 'axis='+axis)

            cf.write(f, self.tempfilename, Conventions='CF-'+VN, verbose=0)

            f2 = cf.read(self.tempfilename, verbose=0)

            self.assertTrue(len(f2) == 2, 'f2 = '+repr(f2))

            for a, b in zip(f, f2):
                self.assertTrue(a.equals(b, verbose=2))

            # Interior ring component
            c = g.construct('longitude')

            self.assertTrue(c.interior_ring.equals(
                g.construct('longitude').get_interior_ring()))
            self.assertTrue(c.interior_ring.data.ndim == c.data.ndim + 1)
            self.assertTrue(c.interior_ring.data.shape[0] == c.data.shape[0])

            _ = g.dump(display=False)

            d = c.insert_dimension(0)
            self.assertTrue(d.data.shape == (1,) + c.data.shape)
            self.assertTrue(
                d.interior_ring.data.shape ==
                (1,) + c.interior_ring.data.shape
            )

            e = d.squeeze(0)
            self.assertTrue(e.data.shape == c.data.shape)
            self.assertTrue(
                e.interior_ring.data.shape == c.interior_ring.data.shape)

            t = d.transpose()
            self.assertTrue(
                t.data.shape == d.data.shape[::-1],
                (t.data.shape, c.data.shape[::-1])
            )
            self.assertTrue(
                t.interior_ring.data.shape ==
                d.interior_ring.data.shape[-2::-1] +
                (d.interior_ring.data.shape[-1],)
            )

            # Subspacing
            g = g[1, ...]
            c = g.construct('longitude')

            self.assertTrue(
                c.interior_ring.data.shape[0] == 1, c.interior_ring.data.shape)
            self.assertTrue(c.interior_ring.data.ndim == c.data.ndim + 1)
            self.assertTrue(c.interior_ring.data.shape[0] == c.data.shape[0])

            # Setting of node count properties
            coord = f[0].construct('axis=Y')
            nc = coord.get_node_count()
            nc.set_property('long_name', 'Node counts')
            cf.write(f, self.tempfilename)

            nc.nc_set_variable('new_var_name')
            cf.write(f, self.tempfilename)

            # Setting of part node count properties
            coord = f[0].construct('axis=X')
            pnc = coord.get_part_node_count()
            pnc.set_property('long_name', 'Part node counts')
            cf.write(f, self.tempfilename)

            pnc.nc_set_variable('new_var_name')
            cf.write(f, self.tempfilename)

            pnc.nc_set_dimension('new_dim_name')
            cf.write(f, self.tempfilename)

    def test_geometry_interior_ring_roll(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_interior_ring_file, verbose=0)[0]

        g = f.roll(0, 1)
        self.assertFalse(f.equals(g))
        h = g.roll(0, 1)
        self.assertTrue(f.equals(h))

        for r in (-4, -2, 0, 2, 4):
            h = f.roll(0, r)
            self.assertTrue(f.equals(h))

        for r in (-3, -1, 1, 3):
            h = f.roll(0, r)
            self.assertFalse(f.equals(h))

    def test_geometry_interior_ring_flip(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_interior_ring_file, verbose=0)[0]

        g = f.flip(0)
        self.assertFalse(f.equals(g))
        h = g.flip(0)
        self.assertTrue(f.equals(h))

    def test_geometry_interior_ring_flatten(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_interior_ring_file, verbose=0)[0]

        for i in (0, 1):
            self.assertTrue(f.equals(f.flatten(i), verbose=1))

    def test_geometry_interior_ring_close(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_interior_ring_file, verbose=0)[0]

        self.assertIsNone(f.close())

    def test_geometry_interior_ring_files(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.geometry_interior_ring_file, verbose=0)[0]

        self.assertTrue(isinstance(f.get_filenames(), set))

# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
