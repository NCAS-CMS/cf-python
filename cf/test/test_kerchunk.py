import datetime
import faulthandler
import json
import os
import unittest

import fsspec

faulthandler.enable()  # to debug seg faults and timeouts


import cf

warnings = False


kerchunk_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "example_field_0.kerchunk"
)

fs = fsspec.filesystem("reference", fo=kerchunk_file)
kerchunk_mapper = fs.get_mapper()


class read_writeTest(unittest.TestCase):
    """Test the reading and writing of field constructs from/to disk."""

    netcdf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "example_field_0.nc"
    )
    kerchunk = kerchunk_mapper

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows: cf.LOG_LEVEL('DEBUG')
        #
        # < ... test code ... >
        # cf.log_level('DISABLE')

    def test_kerchunk_read(self):
        """Test cf.read with Kerchunk."""
        f = cf.read(self.netcdf)[0]

        k = cf.read(self.kerchunk, dask_chunks=3)
        self.assertEqual(len(k), 1)
        self.assertTrue(k[0].equals(f))
        self.assertGreater(k[0].data.npartitions, 1)

        k = cf.read([self.kerchunk, self.kerchunk], dask_chunks=3)
        self.assertEqual(len(k), 2)
        self.assertTrue(k[0].equals(k[-1]))

        k = cf.read([self.kerchunk, self.kerchunk, self.netcdf], dask_chunks=3)
        self.assertEqual(len(k), 3)
        self.assertTrue(k[0].equals(k[-1]))
        self.assertTrue(k[1].equals(k[-1]))

    def test_kerchunk_original_filenames(self):
        """Test original_filenames with Kerchunk."""
        k = cf.read(self.kerchunk)[0]
        self.assertEqual(k.get_original_filenames(), set())

    def test_read_dict(self):
        """Test cf.read with an Kerchunk dictionary."""
        with open(kerchunk_file, "r") as fh:
            d = json.load(fh)

        with self.assertRaises(ValueError):
            cf.read(d)

        fs = fsspec.filesystem("reference", fo=d)
        kerchunk = fs.get_mapper()
        self.assertEqual(len(cf.read(kerchunk)), 1)

    def test_read_bytes(self):
        """Test cf.read with an Kerchunk dictionary."""
        with open(kerchunk_file, "r") as fh:
            d = json.load(fh)

        b = json.dumps(d).encode("utf-8")
        with self.assertRaises(ValueError):
            cf.read(b)

        d = json.loads(b)
        fs = fsspec.filesystem("reference", fo=d)
        kerchunk = fs.get_mapper()
        self.assertEqual(len(cf.read(kerchunk)), 1)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
