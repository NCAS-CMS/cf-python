import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf

tmpfile = tempfile.mkstemp(".nc")[1]
tmpfile2 = tempfile.mkstemp(".nca")[1]
tmpfiles = [tmpfile, tmpfile2, "delme.nc", "delme.nca"]


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class generalTest(unittest.TestCase):
    def setUp(self):
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
        )
        self.f = cf.read(filename)[0]

    def test_GENERAL(self):
        g = self.f.squeeze()
        f = self.f.copy()

        c = cf.set([0, 3, 4, 5])

        f == c

        # +, -, *, /, **
        h = g.copy()
        h **= 2
        h **= 0.5
        h.standard_name = g.standard_name
        self.assertTrue(g.data.allclose(h.data), repr(g.array - h.array))
        h *= 10
        h /= 10.0
        self.assertTrue(g.data.allclose(h.data), repr(g.array - h.array))
        h += 1
        h -= 1
        self.assertTrue(g.data.allclose(h.data), repr(g.array - h.array))
        h = h * 10
        h = h / 10.0
        self.assertTrue(g.data.allclose(h.data), repr(g.array - h.array))
        h = h + 1
        h = h - 1
        self.assertTrue(g.data.allclose(h.data), repr(g.array - h.array))

        # flip, expand_dims, squeeze and remove_axes
        h = g.copy()
        h.flip((1, 0), inplace=True)
        h.flip((1, 0), inplace=True)
        h.flip(0, inplace=True)
        h.flip(1, inplace=True)
        h.flip([0, 1], inplace=True)
        self.assertTrue(g.equals(h, verbose=2))

        # Access the field's data as a numpy array
        g.array
        g.construct("latitude").array
        g.construct("longitude").array

        # Subspace the field
        g[..., 2:5].array
        g[9::-4, ...].array
        h = g[(slice(None, None, -1),) * g.ndim]
        h = h[(slice(None, None, -1),) * h.ndim]
        self.assertTrue(g.equals(h, verbose=2))

        # Indices for a subspace defined by coordinates
        f.indices()
        f.indices(grid_latitude=cf.lt(5), grid_longitude=27)
        f.indices(
            grid_latitude=cf.lt(5),
            grid_longitude=27,
            atmosphere_hybrid_height_coordinate=1.5,
        )

        # Subspace the field
        g.subspace(
            grid_latitude=cf.lt(5),
            grid_longitude=27,
            atmosphere_hybrid_height_coordinate=1.5,
        )

        # Create list of fields
        fl = cf.FieldList([g, g, g, g])

        # Write a list of fields to disk
        cf.write((f, fl), tmpfile)
        cf.write(fl, tmpfile)

        # Read a list of fields from disk
        fl = cf.read(tmpfile, squeeze=True)
        for f in fl:
            try:
                del f.history
            except AttributeError:
                pass

        # Access the last field in the list
        x = fl[-1]

        # Access the data of the last field in the list
        x = fl[-1].array

        # Modify the last field in the list
        fl[-1] *= -1
        x = fl[-1].array

        # Changing units
        fl[-1].units = "mm.s-1"
        x = fl[-1].array

        # Combine fields not in place
        g = fl[-1] - fl[-1]
        x = g.array

        # Combine field with a size 1 Data object
        g += cf.Data([[[[[1.5]]]]], "cm.s-1")
        x = g.array

        # Setting of (un)masked elements with where()
        g[::2, 1::2] = numpy.ma.masked
        g.where(True, 99)
        g.where(g.mask, 2)

        g[slice(None, None, 2), slice(1, None, 2)] = cf.masked
        g.where(g.mask, [[-1]])
        g.where(True, cf.Data(0, None))

        h = g[:3, :4]
        h.where(True, -1)
        h[0, 2] = 2
        h.transpose([1, 0], inplace=True)

        h.flip([1, 0], inplace=True)

        g[slice(None, 3), slice(None, 4)] = h

        h = g[:3, :4]
        h[...] = -1
        h[0, 2] = 2
        g[slice(None, 3), slice(None, 4)] = h

        # Iterate through array values
        for x in f.data.flat():
            pass

        f.transpose(inplace=True)
        f.flip(inplace=True)

        cf.write(f, "delme.nc")
        f = cf.read("delme.nc")[0]

        b = f[:, 0:6, :]
        c = f[:, 6:, :]
        cf.aggregate([b, c], verbose=2)[0]


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
