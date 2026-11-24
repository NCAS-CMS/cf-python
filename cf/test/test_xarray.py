import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import xarray as xr

import cf


class xarrayTest(unittest.TestCase):
    """Unit test for converting to xarray."""

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

    def test_Field_to_xarray(self):
        """Test cf.to_xarray."""
        fields = cf.example_fields()

        # Write each field to a different xarray dataset
        for f in fields:
            ds = cf.to_xarray(f)
            self.assertIsInstance(ds, xr.Dataset)
            self.assertIn("Conventions", ds.attrs)
            str(ds)

        # Write all fields to one xarray dataset
        ds = cf.to_xarray(fields)
        self.assertIsInstance(ds, xr.Dataset)
        str(ds)

    def test_Field_to_xarray_from_disk(self):
        """Test cf.to_xarray on datasets read from disk."""
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
            "geometry_4.nc",
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
            ds = cf.to_xarray(f)
            self.assertIsInstance(ds, xr.Dataset)
            str(ds)

    def test_Field_to_xarray_groups(self):
        """Test cf.to_xarray with groups."""
        f = cf.example_field(0)
        g = f.copy()

        ds = cf.to_xarray(f)
        self.assertIsInstance(ds, xr.Dataset)

        f.nc_set_variable("/forecast/model/q2")
        ds = cf.to_xarray(f)
        self.assertIsInstance(ds, xr.DataTree)
        self.assertIn("q2", ds["/forecast/model"])
        str(ds)

        # group=True
        ds = cf.to_xarray([f, g])
        self.assertIsInstance(ds, xr.DataTree)
        str(ds)

        self.assertIn("q", ds)
        self.assertIn("q2", ds["/forecast/model"])

        # group=False
        ds = cf.to_xarray(f, group=False)
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("q2", ds)
        str(ds)

        ds = cf.to_xarray([f, g], group=False)
        self.assertIsInstance(ds, xr.Dataset)
        str(ds)

        self.assertIn("q", ds)
        self.assertIn("q2", ds)

    def test_write_xarray_external(self):
        """Test cf.write with xarray and external file."""
        f = cf.example_field(0)
        with self.assertRaises(ValueError):
            cf.write(f, fmt="XARRAY", external="file.nc")

    def test_from_xarray(self):
        """Test cf.from_xarray."""
        # These fields round trip OK!
        f = cf.example_fields(0, 2, 5, 7, 12, 13)
        ds = cf.to_xarray(f)
        g = cf.from_xarray(ds)
        self.assertEqual(len(f), len(g))
        self.assertIsInstance(g, cf.FieldList)

        f = sorted(f, key=lambda x: x.get_property("standard_name"))
        g = sorted(g, key=lambda x: x.get_property("standard_name"))
        self.assertEqual(
            [field1.equals(field0) for field0, field1 in zip(f, g)],
            [True] * len(f),
        )

    def test_to_xarray_domain(self):
        """Test cf.to_xarray with domain."""
        domains = [f.domain for f in cf.example_fields()]

        # Write each domain to a different xarray dataset
        for d in domains:
            ds = cf.to_xarray(d)
            self.assertIsInstance(ds, xr.Dataset)
            str(ds)

        # Write all domains to one xarray dataset
        ds = cf.to_xarray(domains)
        self.assertIsInstance(ds, xr.Dataset)
        str(ds)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
