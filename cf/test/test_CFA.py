import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest
from pathlib import PurePath

import netCDF4

faulthandler.enable()  # to debug seg faults and timeouts

from cfdm.read_write.netcdf.netcdfwrite import AggregationError

import cf

n_tmpfiles = 5
tmpfiles = [
    tempfile.mkstemp("_test_CFA.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(
    tmpfile1,
    tmpfile2,
    nc_file,
    cfa_file,
    cfa_file2,
) = tmpfiles


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class CFATest(unittest.TestCase):
    """Unit test for aggregation variables."""

    netcdf3_fmts = [
        "NETCDF3_CLASSIC",
        "NETCDF3_64BIT",
        "NETCDF3_64BIT_OFFSET",
        "NETCDF3_64BIT_DATA",
    ]
    netcdf4_fmts = ["NETCDF4", "NETCDF4_CLASSIC"]
    netcdf_fmts = netcdf3_fmts + netcdf4_fmts

    aggregation_value = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "aggregation_value.nc"
    )

    def test_CFA_fmt(self):
        """Test the cf.read 'fmt' keyword with cfa."""
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1, cfa_write="field")[0]

        for fmt in self.netcdf_fmts:
            cf.write(f, cfa_file, fmt=fmt, cfa="field")
            g = cf.read(cfa_file)
            self.assertEqual(len(g), 1)
            self.assertTrue(f.equals(g[0]))

    def test_CFA_multiple_fragments(self):
        """Test aggregation variables with more than one fragment."""
        f = cf.example_field(0)

        cf.write(f[:2], tmpfile1)
        cf.write(f[2:], tmpfile2)

        a = cf.read(tmpfile1, cfa_write="field")[0]
        b = cf.read(tmpfile2, cfa_write="field")[0]
        a = cf.Field.concatenate([a, b], axis=0)

        cf.write(a, nc_file)
        cf.write(a, cfa_file, cfa="field")

        n = cf.read(nc_file)
        c = cf.read(cfa_file)
        self.assertEqual(len(n), 1)
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0].equals(f))
        self.assertTrue(n[0].equals(c[0]))

    def test_CFA_strict(self):
        """Test 'strict' option to the cf.write 'cfa' keyword."""
        f = cf.example_field(0)

        # By default, can't write in-memory arrays as aggregation
        # variables
        with self.assertRaises(AggregationError):
            cf.write(f, cfa_file, cfa="field")

        # The previous line should have deleted the output file
        self.assertFalse(os.path.exists(cfa_file))

        cf.write(f, nc_file, cfa={"constructs": "field", "strict": False})
        g = cf.read(nc_file, cfa_write="field")
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0].equals(f))

        cf.write(g, cfa_file, cfa={"constructs": "field", "strict": True})
        g = cf.read(cfa_file)
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0].equals(f))

    def test_CFA_uri_0(self):
        """Test aggregation 'uri' option to cf.write."""
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1, cfa_write="field")[0]

        absuri_filename = PurePath(os.path.abspath(tmpfile1)).as_uri()
        reluri_filename = os.path.basename(tmpfile1)

        for uri, filename in zip(
            ("absolute", "relative"), (absuri_filename, reluri_filename)
        ):
            cf.write(
                f,
                cfa_file,
                cfa={"constructs": "field", "uri": uri},
            )

            nc = netCDF4.Dataset(cfa_file, "r")
            fragment_uris = nc.variables["fragment_uris"]
            self.assertEqual(fragment_uris[...], filename)
            nc.close()

            g = cf.read(cfa_file)
            self.assertEqual(len(g), 1)
            g = g[0]
            self.assertTrue(f.equals(g))
            self.assertEqual(
                g.data.get_filenames(normalise=False), set([filename])
            )

    def test_CFA_uri_1(self):
        """Test aggregation 'uri=default' option to cf.write."""
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1, cfa_write="field")[0]

        absuri_filename = PurePath(os.path.abspath(tmpfile1)).as_uri()
        reluri_filename = os.path.basename(tmpfile1)

        for uri, filename in zip(
            ("absolute", "relative"), (absuri_filename, reluri_filename)
        ):
            cf.write(
                f,
                cfa_file,
                cfa={"constructs": "field", "uri": uri},
            )

            g = cf.read(cfa_file)[0]
            cf.write(
                g,
                cfa_file2,
                cfa="field",
            )

            nc = netCDF4.Dataset(cfa_file2, "r")
            fragment_uris = nc.variables["fragment_uris"]
            self.assertEqual(fragment_uris[...], filename)
            nc.close()

    def test_CFA_constructs(self):
        """Test aggregation 'constructs' option to cf.write."""
        f = cf.example_field(1)
        f.del_construct("time")
        f.del_construct("long_name=Grid latitude name")
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1, cfa_write="all")[0]

        # No constructs
        cf.write(f, tmpfile2, cfa={"constructs": []})
        nc = netCDF4.Dataset(tmpfile2, "r")
        for var in nc.variables.values():
            attrs = var.ncattrs()
            self.assertNotIn("aggregated_dimensions", attrs)
            self.assertNotIn("aggregated_data", attrs)

        nc.close()

        # Field construct
        cf.write(f, tmpfile2, cfa={"constructs": "field"})
        nc = netCDF4.Dataset(tmpfile2, "r")
        for ncvar, var in nc.variables.items():
            attrs = var.ncattrs()
            if ncvar in ("ta",):
                self.assertFalse(var.ndim)
                self.assertIn("aggregated_dimensions", attrs)
                self.assertIn("aggregated_data", attrs)
            else:
                self.assertNotIn("aggregated_dimensions", attrs)
                self.assertNotIn("aggregated_data", attrs)

        nc.close()

        # Dimension construct
        for constructs in (
            "dimension_coordinate",
            ["dimension_coordinate"],
            {"dimension_coordinate": None},
            {"dimension_coordinate": 1},
        ):
            cf.write(f, tmpfile2, cfa={"constructs": constructs})
            nc = netCDF4.Dataset(tmpfile2, "r")
            for ncvar, var in nc.variables.items():
                attrs = var.ncattrs()
                if ncvar in (
                    "x",
                    "x_bnds",
                    "y",
                    "y_bnds",
                    "atmosphere_hybrid_height_coordinate",
                    "atmosphere_hybrid_height_coordinate_bounds",
                ):
                    self.assertFalse(var.ndim)
                    self.assertIn("aggregated_dimensions", attrs)
                    self.assertIn("aggregated_data", attrs)
                else:
                    self.assertNotIn("aggregated_dimensions", attrs)
                    self.assertNotIn("aggregated_data", attrs)

            nc.close()

        # Dimension and auxiliary constructs
        for constructs in (
            ["dimension_coordinate", "auxiliary_coordinate"],
            {"dimension_coordinate": None, "auxiliary_coordinate": 2},
        ):
            cf.write(f, tmpfile2, cfa={"constructs": constructs})
            nc = netCDF4.Dataset(tmpfile2, "r")
            for ncvar, var in nc.variables.items():
                attrs = var.ncattrs()
                if ncvar in (
                    "x",
                    "x_bnds",
                    "y",
                    "y_bnds",
                    "atmosphere_hybrid_height_coordinate",
                    "atmosphere_hybrid_height_coordinate_bounds",
                    "latitude_1",
                    "longitude_1",
                ):
                    self.assertFalse(var.ndim)
                    self.assertIn("aggregated_dimensions", attrs)
                    self.assertIn("aggregated_data", attrs)
                else:
                    self.assertNotIn("aggregated_dimensions", attrs)
                    self.assertNotIn("aggregated_data", attrs)

            nc.close()

    def test_CFA_scalar(self):
        """Test scalar aggregation variable."""
        f = cf.example_field(0)
        f = f[0, 0].squeeze()
        cf.write(f, tmpfile1)
        g = cf.read(tmpfile1, cfa_write="field")[0]
        cf.write(g, cfa_file, cfa="field")
        h = cf.read(cfa_file)[0]
        self.assertTrue(h.equals(f))

    def test_CFA_unique_value(self):
        """Test the unique value fragment array variable."""
        write = True
        for aggregation_value_file in (self.aggregation_value, cfa_file):
            f = cf.read(aggregation_value_file, cfa_write="all")
            self.assertEqual(len(f), 1)
            f = f[0]
            fa = f.field_ancillary()
            self.assertEqual(fa.shape, (12,))
            self.assertEqual(fa.data.chunks, ((3, 9),))
            self.assertEqual(
                fa.data.nc_get_aggregation_fragment_type(), "unique_value"
            )
            self.assertEqual(
                fa.data.nc_get_aggregated_data(),
                {
                    "map": "fragment_map_uid",
                    "unique_values": "fragment_value_uid",
                },
            )

            nc = netCDF4.Dataset(aggregation_value_file, "r")
            fragment_value_uid = nc.variables["fragment_value_uid"][...]
            nc.close()

            self.assertTrue((fa[:3].array == fragment_value_uid[0]).all())
            self.assertTrue((fa[3:].array == fragment_value_uid[1]).all())

            if write:
                cf.write(f, cfa_file)
                write = False

    def test_CFA_cfa(self):
        """Test the cf.write 'cfa' keyword."""
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1, cfa_write="field")[0]
        cf.write(f, tmpfile2, cfa="field")
        g = cf.read(tmpfile2, cfa_write="field")[0]

        # Default of cfa="auto" - check that aggregation variable
        # gets written
        cf.write(g, cfa_file)
        nc = netCDF4.Dataset(cfa_file, "r")
        self.assertIsNotNone(
            getattr(nc.variables["q"], "aggregated_data", None)
        )
        nc.close()

        cf.write(g, cfa_file, cfa={"constructs": {"auto": 2}})
        nc = netCDF4.Dataset(cfa_file, "r")

        self.assertIsNotNone(
            getattr(nc.variables["q"], "aggregated_data", None)
        )
        nc.close()

        cf.write(
            g,
            cfa_file,
            cfa={
                "constructs": ["auto", "dimension_coordinate"],
                "strict": False,
            },
        )
        nc = netCDF4.Dataset(cfa_file, "r")
        for ncvar in ("q", "lat", "lon"):
            self.assertIsNotNone(
                getattr(nc.variables[ncvar], "aggregated_data", None)
            )

        nc.close()

        # Check bad values of cfa
        for cfa in (False, True, (), []):
            with self.assertRaises(ValueError):
                cf.write(g, cfa_file, cfa=cfa)

    def test_CFA_subspace(self):
        """Test the writing subspaces of aggregations."""
        f = cf.example_field(0)

        cf.write(f[:2], tmpfile1)
        cf.write(f[2:], tmpfile2)

        a = cf.read(tmpfile1, cfa_write="field")[0]
        b = cf.read(tmpfile2, cfa_write="field")[0]
        c = cf.Field.concatenate([a, b], axis=0)

        cf.write(c, cfa_file, cfa="field")

        f = cf.read(cfa_file, cfa_write="field")[0]
        cf.write(f[:2], cfa_file2, cfa="field")
        g = cf.read(cfa_file2)[0]
        self.assertTrue(g.equals(a))

        cf.write(f[2:], cfa_file2, cfa="field")
        g = cf.read(cfa_file2)[0]
        self.assertTrue(g.equals(b))

        # Can't straddle Dask chunks
        with self.assertRaises(AggregationError):
            cf.write(f[1:3], cfa_file2, cfa="field")


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
