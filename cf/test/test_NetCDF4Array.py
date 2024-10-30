import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

import numpy as np
from dask.base import tokenize

faulthandler.enable()  # to debug seg faults and timeouts

import cf

# REVIEW: h5: `test_NetCDF4Array.py`: renamed 'NetCDFArray' to 'NetCDF4Array'
n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_NetCDF4Array.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(tmpfile1,) = tmpfiles


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class NetCDF4ArrayTest(unittest.TestCase):
    n = cf.NetCDF4Array(
        filename="filename.nc",
        address="x",
        shape=(5, 8),
        dtype=np.dtype(float),
    )

    # REVIEW: h5: `test_NetCDF4Array`: renamed 'NetCDFArray' to 'NetCDF4Array'
    def test_NetCDF4Array_del_file_location(self):
        a = cf.NetCDF4Array(("/data1/file1", "/data2/file2"), ("tas1", "tas2"))
        b = a.del_file_location("/data1")
        self.assertIsNot(b, a)
        self.assertEqual(b.get_filenames(), ("/data2/file2",))
        self.assertEqual(b.get_addresses(), ("tas2",))

        a = cf.NetCDF4Array(
            ("/data1/file1", "/data2/file1", "/data2/file2"),
            ("tas1", "tas1", "tas2"),
        )
        b = a.del_file_location("/data2")
        self.assertEqual(b.get_filenames(), ("/data1/file1",))
        self.assertEqual(b.get_addresses(), ("tas1",))

        # Can't be left with no files
        self.assertEqual(b.file_locations(), ("/data1",))
        with self.assertRaises(ValueError):
            b.del_file_location("/data1/")

    # REVIEW: h5: `test_NetCDF4Array`: renamed 'NetCDFArray' to 'NetCDF4Array'
    def test_NetCDF4Array_file_locations(self):
        a = cf.NetCDF4Array("/data1/file1")
        self.assertEqual(a.file_locations(), ("/data1",))

        a = cf.NetCDF4Array(("/data1/file1", "/data2/file2"))
        self.assertEqual(a.file_locations(), ("/data1", "/data2"))

        a = cf.NetCDF4Array(("/data1/file1", "/data2/file2", "/data1/file2"))
        self.assertEqual(a.file_locations(), ("/data1", "/data2", "/data1"))

    # REVIEW: h5: `test_NetCDF4Array`: renamed 'NetCDFArray' to 'NetCDF4Array'
    def test_NetCDF4Array_add_file_location(self):
        a = cf.NetCDF4Array("/data1/file1", "tas")
        b = a.add_file_location("/home/user")
        self.assertIsNot(b, a)
        self.assertEqual(
            b.get_filenames(), ("/data1/file1", "/home/user/file1")
        )
        self.assertEqual(b.get_addresses(), ("tas", "tas"))

        a = cf.NetCDF4Array(("/data1/file1", "/data2/file2"), ("tas1", "tas2"))
        b = a.add_file_location("/home/user")
        self.assertEqual(
            b.get_filenames(),
            (
                "/data1/file1",
                "/data2/file2",
                "/home/user/file1",
                "/home/user/file2",
            ),
        )
        self.assertEqual(b.get_addresses(), ("tas1", "tas2", "tas1", "tas2"))

        a = cf.NetCDF4Array(("/data1/file1", "/data2/file1"), ("tas1", "tas2"))
        b = a.add_file_location("/home/user")
        self.assertEqual(
            b.get_filenames(),
            ("/data1/file1", "/data2/file1", "/home/user/file1"),
        )
        self.assertEqual(b.get_addresses(), ("tas1", "tas2", "tas1"))

        a = cf.NetCDF4Array(("/data1/file1", "/data2/file1"), ("tas1", "tas2"))
        b = a.add_file_location("/data1/")
        self.assertEqual(b.get_filenames(), a.get_filenames())
        self.assertEqual(b.get_addresses(), a.get_addresses())

    # REVIEW: h5: `test_NetCDF4Array`: renamed 'NetCDFArray' to 'NetCDF4Array'
    def test_NetCDF4Array__dask_tokenize__(self):
        a = cf.NetCDF4Array("/data1/file1", "tas", shape=(12, 2), mask=False)
        self.assertEqual(tokenize(a), tokenize(a.copy()))

        b = cf.NetCDF4Array("/home/file2", "tas", shape=(12, 2))
        self.assertNotEqual(tokenize(a), tokenize(b))

    # REVIEW: h5: `test_NetCDF4Array`: renamed 'NetCDFArray' to 'NetCDF4Array'
    def test_NetCDF4Array_multiple_files(self):
        f = cf.example_field(0)
        cf.write(f, tmpfile1)

        # Create instance with non-existent file
        n = cf.NetCDF4Array(
            filename=os.path.join("/bad/location", os.path.basename(tmpfile1)),
            address=f.nc_get_variable(),
            shape=f.shape,
            dtype=f.dtype,
        )
        # Add file that exists
        n = n.add_file_location(os.path.dirname(tmpfile1))

        self.assertEqual(len(n.get_filenames()), 2)
        self.assertTrue((n[...] == f.array).all())

    # REVIEW: getitem: `test_NetCDF4Array`: test `NetCDF4Array.shape`
    def test_NetCDF4Array_shape(self):
        shape = (12, 73, 96)
        a = cf.NetCDF4Array("/home/file2", "tas", shape=shape)
        self.assertEqual(a.shape, shape)
        self.assertEqual(a.original_shape, shape)
        a = a[::2]
        self.assertEqual(a.shape, (shape[0] // 2,) + shape[1:])
        self.assertEqual(a.original_shape, shape)

    # REVIEW: getitem: `test_NetCDF4Array`: test `NetCDF4Array.index`
    def test_NetCDF4Array_index(self):
        shape = (12, 73, 96)
        a = cf.NetCDF4Array("/home/file2", "tas", shape=shape)
        self.assertEqual(
            a.index(),
            (
                slice(
                    None,
                ),
            )
            * len(shape),
        )
        a = a[8:7:-1, 10:19:3, [15, 1, 4, 12]]
        a = a[[0], [True, False, True], ::-2]
        self.assertEqual(a.shape, (1, 2, 2))
        self.assertEqual(
            a.index(),
            (slice(8, 9, None), slice(10, 17, 6), slice(12, -1, -11)),
        )

        index = a.index(conform=False)
        self.assertTrue((index[0] == [8]).all())
        self.assertTrue((index[1] == [10, 16]).all())
        self.assertTrue((index[2] == [12, 1]).all())


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
