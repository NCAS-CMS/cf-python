import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

ok = True
try:
    import iris  # noqa: F401
    import ncdata  # noqa: F401
    import xarray  # noqa: F401
except ImportError:
    ok = False

import cf


class irisTest(unittest.TestCase):
    """Unit test for converting to and from iris."""

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.log_level("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.log_level('DISABLE')

    @unittest.skipIf(not ok, "requires iris, xarray, and ncdata")
    def test_to_iris(self):
        """Test cf.to_iris."""
        fields = cf.example_fields()

        # Write each field to a different xarray dataset
        for f in fields:
            c = cf.to_iris(f)
            self.assertIsInstance(c, iris.cube.CubeList)
            self.assertIn("Conventions", c[0].attributes)
            str(c)

        # All fields
        c = cf.to_iris(fields)
        self.assertIsInstance(c, iris.cube.CubeList)
        str(c)

    @unittest.skipIf(not ok, "requires iris, xarray, and ncdata")
    def test_to_iris_from_disk(self):
        """Test cf.to_iris on datasets read from disk."""
        for dataset in (
            "example_field_0.nc",
            "example_field_0.zarr2",
            "example_field_0.zarr3",
            "gathered.nc",
            "DSG_timeSeries_contiguous.nc",
            "DSG_timeSeries_indexed.nc",
            "DSG_timeSeriesProfile_indexed_contiguous.nc",
            "parent.nc",
            "external.nc",
            "external_missing.nc",
            "combined.nc",
            "geometry_1.nc",
            "geometry_2.nc",
            "geometry_3.nc",
            "geometry_interior_ring.nc",
            "geometry_interior_ring_2.nc",
            "string_char.nc",
            "subsampled_2.nc",
            "ugrid_1.nc",
            "ugrid_2.nc",
            "ugrid_3.nc",
            "test_file.nc",
        ):
            f = cf.read(dataset)
            c = cf.to_iris(f)
            self.assertIsInstance(c, iris.cube.CubeList)
            str(c)

    @unittest.skipIf(not ok, "requires iris, xarray, and ncdata")
    def test_from_iris(self):
        """Test cf.from_iris."""
        # These fields round trip OK!
        f = cf.example_fields(0, 2, 5, 7, 12, 13)
        ds = cf.to_iris(f)
        g = cf.from_iris(ds)
        self.assertEqual(len(f), len(g))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
