#import atexit
#import datetime
#import faulthandler
#import os
#import tempfile
#import unittest
#from pathlib import PurePath
#
#import netCDF4
#
#faulthandler.enable()  # to debug seg faults and timeouts
#
#import cf
#
#n_tmpfiles = 5
#tmpfiles = [
#    tempfile.mkstemp("_test_CFA.nc", dir=os.getcwd())[1]
#    for i in range(n_tmpfiles)
#]
#(
#    tmpfile1,
#    tmpfile2,
#    nc_file,
#    cfa_file,
#    cfa_file2,
#) = tmpfiles
#
#
#def _remove_tmpfiles():
#    """Try to remove defined temporary files by deleting their paths."""
#    for f in tmpfiles:
#        try:
#            os.remove(f)
#        except OSError:
#            pass
#
#
#atexit.register(_remove_tmpfiles)
#
#
#class CFATest(unittest.TestCase):
#    """Unit test for aggregation variables."""
#
#    netcdf3_fmts = [
#        "NETCDF3_CLASSIC",
#        "NETCDF3_64BIT",
#        "NETCDF3_64BIT_OFFSET",
#        "NETCDF3_64BIT_DATA",
#    ]
#    netcdf4_fmts = ["NETCDF4", "NETCDF4_CLASSIC"]
#    netcdf_fmts = netcdf3_fmts + netcdf4_fmts
#
#    aggregation_value = os.path.join(
#        os.path.dirname(os.path.abspath(__file__)), "aggregation_value.nc"
#    )
#
#    def test_CFA_fmt(self):
#        """Test the cf.read 'fmt' keyword with cfa."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        for fmt in self.netcdf_fmts:
#            cf.write(f, cfa_file, fmt=fmt, cfa="field")
#            g = cf.read(cfa_file)
#            self.assertEqual(len(g), 1)
#            self.assertTrue(f.equals(g[0]))
#
#    def test_CFA_multiple_fragments(self):
#        """Test aggregation variables with more than one fragment."""
#        f = cf.example_field(0)
#
#        cf.write(f[:2], tmpfile1)
#        cf.write(f[2:], tmpfile2)
#
#        a = cf.read(tmpfile1)[0]
#        b = cf.read(tmpfile2)[0]
#        a = cf.Field.concatenate([a, b], axis=0)
#
#        cf.write(a, nc_file)
#        cf.write(a, cfa_file, cfa="field")
#
#        n = cf.read(nc_file)
#        c = cf.read(cfa_file)
#        self.assertEqual(len(n), 1)
#        self.assertEqual(len(c), 1)
#        self.assertTrue(c[0].equals(f))
#        self.assertTrue(n[0].equals(c[0]))
#
#    def test_CFA_strict(self):
#        """Test 'strict' option to the cf.write 'cfa' keyword."""
#        f = cf.example_field(0)
#
#        # By default, can't write in-memory arrays as aggregation
#        # variables
#        with self.assertRaises(ValueError):
#            cf.write(f, cfa_file, cfa="field")
#
#        # The previous line should have deleted the output file
#        self.assertFalse(os.path.exists(cfa_file))
#
#        cf.write(f, nc_file, cfa={"constructs": "field", "strict": False})
#        g = cf.read(nc_file)
#        self.assertEqual(len(g), 1)
#        self.assertTrue(g[0].equals(f))
#
#        cf.write(g, cfa_file, cfa={"constructs": "field", "strict": True})
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        self.assertTrue(g[0].equals(f))
#
#    def test_CFA_substitutions_0(self):
#        """Test aggregation substitution URI substitutions (0)."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        cwd = os.getcwd()
#        f.data.nc_update_aggregation_substitutions({"base": cwd})
#
#        cf.write(
#            f,
#            cfa_file,
#            cfa={"constructs": "field", "uri": "absolute"},
#        )
#
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            f"${{base}}: {cwd}",
#        )
#        self.assertEqual(
#            cfa_location[...], f"${{base}}/{os.path.basename(tmpfile1)}"
#        )
#        nc.close()
#
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        self.assertTrue(f.equals(g[0]))
#
#    def test_CFA_substitutions_1(self):
#        """Test aggregation substitution URI substitutions (1)."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        cwd = os.getcwd()
#        for base in ("base", "${base}"):
#            cf.write(
#                f,
#                cfa_file,
#                cfa={
#                    "constructs": "field",
#                    "uri": "absolute",
#                    "substitutions": {base: cwd},
#                },
#            )
#
#            nc = netCDF4.Dataset(cfa_file, "r")
#            cfa_location = nc.variables["cfa_location"]
#            self.assertEqual(
#                cfa_location.getncattr("substitutions"),
#                f"${{base}}: {cwd}",
#            )
#            self.assertEqual(
#                cfa_location[...],
#                f"${{base}}/{os.path.basename(tmpfile1)}",
#            )
#            nc.close()
#
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        self.assertTrue(f.equals(g[0]))
#
#    def test_CFA_substitutions_2(self):
#        """Test aggregation substitution URI substitutions (2)."""
#        # TODOCFA: delete
#        tmpfile1 = "tmpfile1.nc"
#
#        f = cf.example_field(0)
#
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        cwd = os.getcwd()
#        basename = os.path.basename(tmpfile1)
#
#        # TODOCFA: delete
#        cfa_file = "cfa_file.nc"
#
#        f.data.nc_clear_aggregation_substitutions()
#        f.data.nc_update_aggregation_substitutions({"base": f"{cwd}"})
#        cf.write(
#            f,
#            cfa_file,
#            cfa={
#                "constructs": "field",
#                "uri": "absolute",
#                "substitutions": {"base2": "/bad/location"},
#            },
#        )
#
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            f"${{base2}}: /bad/location ${{base}}: {cwd}",
#        )
#        self.assertEqual(cfa_location[...], f"file://${{base}}/{basename}")
#        nc.close()
#
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        self.assertTrue(f.equals(g[0]))
#
#        f.data.nc_clear_aggregation_substitutions()
#        f.data.nc_update_aggregation_substitutions({"base": "/bad/location"})
#
#        cf.write(
#            f,
#            cfa_file,
#            cfa={
#                "constructs": "field",
#                "uri": "absolute",
#                "substitutions": {"base": cwd},
#            },
#        )
#
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            f"${{base}}: {cwd}",
#        )
#        self.assertEqual(cfa_location[...], f"file://${{base}}/{basename}")
#        nc.close()
#
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        self.assertTrue(f.equals(g[0]))
#
#        f.data.nc_clear_aggregation_substitutions()
#        f.data.nc_update_aggregation_substitutions({"base2": "/bad/location"})
#
#        cf.write(
#            f,
#            cfa_file,
#            cfa={
#                "constructs": "field",
#                "uri": "absolute",
#                "substitutions": {"base": cwd},
#            },
#        )
#
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            f"${{base2}}: /bad/location ${{base}}: {cwd}",
#        )
#        self.assertEqual(
#            cfa_location[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
#        )
#        nc.close()
#
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        g = g[0]
#        self.assertTrue(f.equals(g))
#
#        self.assertEqual(
#            g.data.get_filenames(normalise=False),
#            set((f"file://${{base}}/{os.path.basename(tmpfile1)}",)),
#        )
#        g.data.nc_update_aggregation_substitutions({"base": "/new/location"})
#        self.assertEqual(
#            g.data.nc_aggregation_substitutions(),
#            {"${base2}": "/bad/location", "${base}": "/new/location"},
#        )
#        self.assertEqual(
#            g.data.get_filenames(normalise=False),
#            set((f"file://${{base}}/{os.path.basename(tmpfile1)}",)),
#        )
#
#        # TODOCFA: delete
#        cfa_file2 = "cfa_file2.nc"
#        cf.write(
#            g,
#            cfa_file2,
#            cfa={
#                "constructs": "field",
#                "uri": "absolute",
#            },
#        )
#        nc = netCDF4.Dataset(cfa_file2, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            "${base2}: /bad/location ${base}: /new/location",
#        )
#        self.assertEqual(
#            cfa_location[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
#        )
#        nc.close()
#
#    def test_CFA_substitutions_3(self):
#        """Test aggregation substitution URI substitutions (2)."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        cwd = os.getcwd()
#        basename = os.path.basename(tmpfile1)
#
#        f.data.nc_clear_aggregation_substitutions()
#        f.data.nc_update_aggregation_substitutions({"base": f"{cwd}/"})
#        cf.write(
#            f,
#            cfa_file,
#            cfa={
#                "constructs": "field",
#                "uri": "absolute",
#            },
#        )
#
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"), f"${{base}}: {cwd}/"
#        )
#        self.assertEqual(cfa_location[...], f"${{base}}{basename}")
#        nc.close()
#
#        # TODOCFA: delete
#        cfa_file2 = "cfa_file2.nc"
#
#        g = cf.read(cfa_file)[0]
#        self.assertTrue(f.equals(g))
#        cf.write(
#            g,
#            cfa_file2,
#            cfa={
#                "constructs": "field",
#                "uri": "absolute",
#            },
#        )
#
#    def test_CFA_substitutions_4(self):
#        """Test aggregation substitution URI substitutions (2)."""
#        f = cf.example_field(0)
#
#        # TODOCFA: delete
#        tmpfile1 = "tmpfile1.nc"
#
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        cwd = os.getcwd()
#        basename = os.path.basename(tmpfile1)
#
#        # TODOCFA: delete
#        cfa_file = "cfa_file.nc"
#
#        replacement = f"{cwd}/"
#        f.data.nc_clear_aggregation_substitutions()
#        f.data.nc_update_aggregation_substitutions({"base": replacement})
#        cf.write(
#            f,
#            cfa_file,
#            cfa={
#                "constructs": "field",
#                "uri": "relative",
#            },
#        )
#
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            f"${{base}}: {replacement}",
#        )
#        self.assertEqual(cfa_location[...], basename)
#        nc.close()
#
#        cf.write(
#            f,
#            cfa_file,
#            cfa={"constructs": "field", "uri": "absolute"},
#        )
#        nc = netCDF4.Dataset(cfa_file, "r")
#        cfa_location = nc.variables["cfa_location"]
#        self.assertEqual(
#            cfa_location.getncattr("substitutions"),
#            f"${{base}}: {replacement}",
#        )
#        self.assertEqual(cfa_location[...], f"file://${{base}}{basename}")
#        nc.close()
#
#    def test_CFA_uri(self):
#        """Test aggregation 'uri' option to cf.write."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        for uri, filename in zip(
#            ("absolute", "relative"),
#            (
#                PurePath(os.path.abspath(tmpfile1)).as_uri(),
#                os.path.basename(tmpfile1),
#            ),
#        ):
#            cf.write(
#                f,
#                cfa_file,
#                cfa={"constructs": "field", "uri": uri},
#            )
#
#            nc = netCDF4.Dataset(cfa_file, "r")
#            cfa_location = nc.variables["cfa_location"]
#            self.assertEqual(cfa_location[...], filename)
#            nc.close()
#
#            g = cf.read(cfa_file)
#            self.assertEqual(len(g), 1)
#            self.assertTrue(f.equals(g[0]))
#
#    def test_CFA_constructs(self):
#        """Test aggregation 'constructs' option to cf.write."""
#        f = cf.example_field(1)
#        f.del_construct("time")
#        f.del_construct("long_name=Grid latitude name")
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#
#        # No constructs
#        cf.write(f, tmpfile2, cfa={"constructs": []})
#        nc = netCDF4.Dataset(tmpfile2, "r")
#        for var in nc.variables.values():
#            attrs = var.ncattrs()
#            self.assertNotIn("aggregated_dimensions", attrs)
#            self.assertNotIn("aggregated_data", attrs)
#
#        nc.close()
#
#        # Field construct
#        cf.write(f, tmpfile2, cfa={"constructs": "field"})
#        nc = netCDF4.Dataset(tmpfile2, "r")
#        for ncvar, var in nc.variables.items():
#            attrs = var.ncattrs()
#            if ncvar in ("ta",):
#                self.assertFalse(var.ndim)
#                self.assertIn("aggregated_dimensions", attrs)
#                self.assertIn("aggregated_data", attrs)
#            else:
#                self.assertNotIn("aggregated_dimensions", attrs)
#                self.assertNotIn("aggregated_data", attrs)
#
#        nc.close()
#
#        # Dimension construct
#        for constructs in (
#            "dimension_coordinate",
#            ["dimension_coordinate"],
#            {"dimension_coordinate": None},
#            {"dimension_coordinate": 1},
#        ):
#            cf.write(f, tmpfile2, cfa={"constructs": constructs})
#            nc = netCDF4.Dataset(tmpfile2, "r")
#            for ncvar, var in nc.variables.items():
#                attrs = var.ncattrs()
#                if ncvar in (
#                    "x",
#                    "x_bnds",
#                    "y",
#                    "y_bnds",
#                    "atmosphere_hybrid_height_coordinate",
#                    "atmosphere_hybrid_height_coordinate_bounds",
#                ):
#                    self.assertFalse(var.ndim)
#                    self.assertIn("aggregated_dimensions", attrs)
#                    self.assertIn("aggregated_data", attrs)
#                else:
#                    self.assertNotIn("aggregated_dimensions", attrs)
#                    self.assertNotIn("aggregated_data", attrs)
#
#            nc.close()
#
#        # Dimension and auxiliary constructs
#        for constructs in (
#            ["dimension_coordinate", "auxiliary_coordinate"],
#            {"dimension_coordinate": None, "auxiliary_coordinate": 2},
#        ):
#            cf.write(f, tmpfile2, cfa={"constructs": constructs})
#            nc = netCDF4.Dataset(tmpfile2, "r")
#            for ncvar, var in nc.variables.items():
#                attrs = var.ncattrs()
#                if ncvar in (
#                    "x",
#                    "x_bnds",
#                    "y",
#                    "y_bnds",
#                    "atmosphere_hybrid_height_coordinate",
#                    "atmosphere_hybrid_height_coordinate_bounds",
#                    "latitude_1",
#                    "longitude_1",
#                ):
#                    self.assertFalse(var.ndim)
#                    self.assertIn("aggregated_dimensions", attrs)
#                    self.assertIn("aggregated_data", attrs)
#                else:
#                    self.assertNotIn("aggregated_dimensions", attrs)
#                    self.assertNotIn("aggregated_data", attrs)
#
#            nc.close()
#
#    def test_CFA_multiple_files(self):
#        """Test storing multiple locations for the same fragment."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#        f.add_file_directory("/new/path")
#
#        cf.write(f, cfa_file, cfa="field")
#        g = cf.read(cfa_file)
#        self.assertEqual(len(g), 1)
#        g = g[0]
#        self.assertTrue(f.equals(g))
#
#        self.assertEqual(len(g.data.get_filenames()), 2)
#        self.assertEqual(len(g.get_filenames()), 3)
#
#    def test_CFA_unlimited_dimension(self):
#        """Test aggregation files with unlimited dimensions."""
#        # Aggregated dimensions cannot be unlimited
#        f = cf.example_field(0)
#        axis = f.domain_axis("longitude")
#        axis.nc_set_unlimited(True)
#        cf.write(f, tmpfile1)
#        g = cf.read(tmpfile1)
#        with self.assertRaises(ValueError):
#            cf.write(g, cfa_file, cfa="field")
#
#    def test_CFA_scalar(self):
#        """Test scalar aggregation variable."""
#        f = cf.example_field(0)
#        f = f[0, 0].squeeze()
#        cf.write(f, tmpfile1)
#        g = cf.read(tmpfile1)[0]
#        cf.write(g, cfa_file, cfa="field")
#        h = cf.read(cfa_file)[0]
#        self.assertTrue(h.equals(f))
#
#    def test_CFA_value(self):
#        """Test the value fragment array variable."""
#        write = True
#        for aggregation_value_file in (self.aggregation_value, cfa_file):
#            f = cf.read(aggregation_value_file)
#            self.assertEqual(len(f), 1)
#            f = f[0]
#            fa = f.field_ancillary()
#            self.assertEqual(fa.shape, (12,))
#            self.assertEqual(fa.data.chunks, ((3, 9),))
#            self.assertEqual(
#                fa.data.nc_get_aggregation_fragment_type(), "value"
#            )
#            self.assertEqual(
#                fa.data.nc_get_aggregated_data(),
#                {"shape": "fragment_shape_uid", "value": "fragment_value_uid"},
#            )
#
#            nc = netCDF4.Dataset(aggregation_value_file, "r")
#            fragment_value_uid = nc.variables["fragment_value_uid"][...]
#            nc.close()
#
#            self.assertTrue((fa[:3].array == fragment_value_uid[0]).all())
#            self.assertTrue((fa[3:].array == fragment_value_uid[1]).all())
#
#            if write:
#                cf.write(f, cfa_file)
#                write = False
#
#    def test_CFA_cfa(self):
#        """Test the cf.write 'cfa' keyword."""
#        f = cf.example_field(0)
#        cf.write(f, tmpfile1)
#        f = cf.read(tmpfile1)[0]
#        cf.write(f, tmpfile2, cfa="field")
#        g = cf.read(tmpfile2)[0]
#
#        # Default of cfa="auto" - check that aggregation variable
#        # gets written
#        cf.write(g, cfa_file)
#        nc = netCDF4.Dataset(cfa_file, "r")
#        self.assertIsNotNone(
#            getattr(nc.variables["q"], "aggregated_data", None)
#        )
#        nc.close()
#
#        cf.write(g, cfa_file, cfa={"constructs": {"auto": 2}})
#        nc = netCDF4.Dataset(cfa_file, "r")
#        self.assertIsNotNone(
#            getattr(nc.variables["q"], "aggregated_data", None)
#        )
#        nc.close()
#
#        cf.write(
#            g,
#            cfa_file,
#            cfa={
#                "constructs": ["auto", "dimension_coordinate"],
#                "strict": False,
#            },
#        )
#        nc = netCDF4.Dataset(cfa_file, "r")
#        for ncvar in ("q", "lat", "lon"):
#            self.assertIsNotNone(
#                getattr(nc.variables[ncvar], "aggregated_data", None)
#            )
#
#        nc.close()
#
#        # Check bad values of cfa
#        for cfa in (False, True, (), []):
#            with self.assertRaises(ValueError):
#                cf.write(g, cfa_file, cfa=cfa)
#
#
#if __name__ == "__main__":
#    print("Run date:", datetime.datetime.now())
#    cf.environment()
#    print()
#    unittest.main(verbosity=2)
#
## n_tmpfiles = 5
## tmpfiles = [
##    tempfile.mkstemp("_test_CFA.nc", dir=os.getcwd())[1]
##    for i in range(n_tmpfiles)
## ]
## (
##    tmpfile1,
##    tmpfile2,
##    tmpfile3,
##    tmpfile4,
##    tmpfile5,
## ) = tmpfiles
##
##
## def _remove_tmpfiles():
##    """Try to remove defined temporary files by deleting their paths."""
##    for f in tmpfiles:
##        try:
##            os.remove(f)
##        except OSError:
##            pass
##
##
## atexit.register(_remove_tmpfiles)
##
##
## class CFATest(unittest.TestCase):
##    netcdf3_fmts = [
##        "NETCDF3_CLASSIC",
##        "NETCDF3_64BIT",
##        "NETCDF3_64BIT_OFFSET",
##        "NETCDF3_64BIT_DATA",
##    ]
##    netcdf4_fmts = ["NETCDF4", "NETCDF4_CLASSIC"]
##    netcdf_fmts = netcdf3_fmts + netcdf4_fmts
##
##    def test_CFA_fmt(self):
##        """Test the cf.read 'fmt' and 'cfa' keywords."""
##        f = cf.example_field(0)
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##
##        for fmt in self.netcdf_fmts:
##            cf.write(f, tmpfile2, fmt=fmt, cfa=True)
##            g = cf.read(tmpfile2)
##            self.assertEqual(len(g), 1)
##            self.assertTrue(f.equals(g[0]))
##
##    def test_CFA_multiple_fragments(self):
##        """Test CFA with more than one fragment."""
##        f = cf.example_field(0)
##
##        cf.write(f[:2], tmpfile1)
##        cf.write(f[2:], tmpfile2)
##
##        a = cf.read([tmpfile1, tmpfile2])
##        self.assertEqual(len(a), 1)
##        a = a[0]
##
##        nc_file = tmpfile3
##        cfa_file = tmpfile4
##        cf.write(a, nc_file)
##        cf.write(a, cfa_file, cfa=True)
##
##        n = cf.read(nc_file)
##        c = cf.read(cfa_file)
##        self.assertEqual(len(n), 1)
##        self.assertEqual(len(c), 1)
##        self.assertTrue(c[0].equals(f))
##        self.assertTrue(n[0].equals(c[0]))
##
##    def test_CFA_strict(self):
##        """Test CFA 'strict' option to the cfa.write 'cfa' keyword."""
##        f = cf.example_field(0)
##
##        # By default, can't write as CF-netCDF those variables
##        # selected for CFA treatment, but which aren't suitable.
##        with self.assertRaises(ValueError):
##            cf.write(f, tmpfile1, cfa=True)
##
##        # The previous line should have deleted the output file
##        self.assertFalse(os.path.exists(tmpfile1))
##
##        cf.write(f, tmpfile1, cfa={"strict": False})
##        g = cf.read(tmpfile1)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(g[0].equals(f))
##
##        cf.write(g, tmpfile2, cfa={"strict": True})
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(g[0].equals(f))
##
##    def test_CFA_field_ancillaries(self):
##        """Test creation of field ancillaries from non-standard CFA terms."""
##        f = cf.example_field(0)
##        self.assertFalse(f.field_ancillaries())
##
##        a = f[:2]
##        b = f[2:]
##        a.set_property("foo", "bar_a")
##        b.set_property("foo", "bar_b")
##        cf.write(a, tmpfile1)
##        cf.write(b, tmpfile2)
##
##        c = cf.read(
##            [tmpfile1, tmpfile2], aggregate={"field_ancillaries": "foo"}
##        )
##        self.assertEqual(len(c), 1)
##        c = c[0]
##        self.assertEqual(len(c.field_ancillaries()), 1)
##        anc = c.field_ancillary()
##        self.assertTrue(anc.data.cfa_get_term())
##        self.assertFalse(anc.data.cfa_get_write())
##
##        cf.write(c, tmpfile3, cfa=False)
##        c2 = cf.read(tmpfile3)
##        self.assertEqual(len(c2), 1)
##        self.assertFalse(c2[0].field_ancillaries())
##
##        cf.write(c, tmpfile4, cfa=True)
##        d = cf.read(tmpfile4)
##        self.assertEqual(len(d), 1)
##        d = d[0]
##
##        self.assertEqual(len(d.field_ancillaries()), 1)
##        anc = d.field_ancillary()
##        self.assertTrue(anc.data.cfa_get_term())
##        self.assertFalse(anc.data.cfa_get_write())
##        self.assertTrue(d.equals(c))
##
##        cf.write(d, tmpfile5, cfa=False)
##        e = cf.read(tmpfile5)
##        self.assertEqual(len(e), 1)
##        self.assertFalse(e[0].field_ancillaries())
##
##        cf.write(d, tmpfile5, cfa=True)
##        e = cf.read(tmpfile5)
##        self.assertEqual(len(e), 1)
##        self.assertTrue(e[0].equals(d))
##
##    def test_CFA_substitutions_0(self):
##        """Test CFA substitution URI substitutions (0)."""
##        f = cf.example_field(0)
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##
##        cwd = os.getcwd()
##
##        f.data.cfa_update_file_substitutions({"base": cwd})
##
##        cf.write(
##            f,
##            tmpfile2,
##            cfa={"absolute_paths": True},
##        )
##
##        nc = netCDF4.Dataset(tmpfile2, "r")
##        cfa_file = nc.variables["cfa_file"]
##        self.assertEqual(
##            cfa_file.getncattr("substitutions"),
##            f"${{base}}: {cwd}",
##        )
##        self.assertEqual(
##            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
##        )
##        nc.close()
##
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(f.equals(g[0]))
##
##    def test_CFA_substitutions_1(self):
##        """Test CFA substitution URI substitutions (1)."""
##        f = cf.example_field(0)
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##
##        cwd = os.getcwd()
##        for base in ("base", "${base}"):
##            cf.write(
##                f,
##                tmpfile2,
##                cfa={"absolute_paths": True, "substitutions": {base: cwd}},
##            )
##
##            nc = netCDF4.Dataset(tmpfile2, "r")
##            cfa_file = nc.variables["cfa_file"]
##            self.assertEqual(
##                cfa_file.getncattr("substitutions"),
##                f"${{base}}: {cwd}",
##            )
##            self.assertEqual(
##                cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
##            )
##            nc.close()
##
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(f.equals(g[0]))
##
##    def test_CFA_substitutions_2(self):
##        """Test CFA substitution URI substitutions (2)."""
##        f = cf.example_field(0)
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##
##        cwd = os.getcwd()
##
##        f.data.cfa_clear_file_substitutions()
##        f.data.cfa_update_file_substitutions({"base": cwd})
##
##        cf.write(
##            f,
##            tmpfile2,
##            cfa={
##                "absolute_paths": True,
##                "substitutions": {"base2": "/bad/location"},
##            },
##        )
##
##        nc = netCDF4.Dataset(tmpfile2, "r")
##        cfa_file = nc.variables["cfa_file"]
##        self.assertEqual(
##            cfa_file.getncattr("substitutions"),
##            f"${{base2}}: /bad/location ${{base}}: {cwd}",
##        )
##        self.assertEqual(
##            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
##        )
##        nc.close()
##
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(f.equals(g[0]))
##
##        f.data.cfa_clear_file_substitutions()
##        f.data.cfa_update_file_substitutions({"base": "/bad/location"})
##
##        cf.write(
##            f,
##            tmpfile2,
##            cfa={"absolute_paths": True, "substitutions": {"base": cwd}},
##        )
##
##        nc = netCDF4.Dataset(tmpfile2, "r")
##        cfa_file = nc.variables["cfa_file"]
##        self.assertEqual(
##            cfa_file.getncattr("substitutions"),
##            f"${{base}}: {cwd}",
##        )
##        self.assertEqual(
##            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
##        )
##        nc.close()
##
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(f.equals(g[0]))
##
##        f.data.cfa_clear_file_substitutions()
##        f.data.cfa_update_file_substitutions({"base2": "/bad/location"})
##
##        cf.write(
##            f,
##            tmpfile2,
##            cfa={"absolute_paths": True, "substitutions": {"base": cwd}},
##        )
##
##        nc = netCDF4.Dataset(tmpfile2, "r")
##        cfa_file = nc.variables["cfa_file"]
##        self.assertEqual(
##            cfa_file.getncattr("substitutions"),
##            f"${{base2}}: /bad/location ${{base}}: {cwd}",
##        )
##        self.assertEqual(
##            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
##        )
##        nc.close()
##
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(f.equals(g[0]))
##
##    def test_CFA_absolute_paths(self):
##        """Test CFA 'absolute_paths' option to the cfa.write 'cfa' keyword."""
##        f = cf.example_field(0)
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##
##        for absolute_paths, filename in zip(
##            (True, False),
##            (
##                PurePath(os.path.abspath(tmpfile1)).as_uri(),
##                os.path.basename(tmpfile1),
##            ),
##        ):
##            cf.write(f, tmpfile2, cfa={"absolute_paths": absolute_paths})
##
##            nc = netCDF4.Dataset(tmpfile2, "r")
##            cfa_file = nc.variables["cfa_file"]
##            self.assertEqual(cfa_file[...], filename)
##            nc.close()
##
##            g = cf.read(tmpfile2)
##            self.assertEqual(len(g), 1)
##            self.assertTrue(f.equals(g[0]))
##
##    def test_CFA_constructs(self):
##        """Test choice of constructs to write as CFA-netCDF variables."""
##        f = cf.example_field(1)
##        f.del_construct("T")
##        f.del_construct("long_name=Grid latitude name")
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##
##        # No constructs
##        cf.write(f, tmpfile2, cfa={"constructs": []})
##        nc = netCDF4.Dataset(tmpfile2, "r")
##        for var in nc.variables.values():
##            attrs = var.ncattrs()
##            self.assertNotIn("aggregated_dimensions", attrs)
##            self.assertNotIn("aggregated_data", attrs)
##
##        nc.close()
##
##        # Field construct
##        cf.write(f, tmpfile2, cfa={"constructs": "field"})
##        nc = netCDF4.Dataset(tmpfile2, "r")
##        for ncvar, var in nc.variables.items():
##            attrs = var.ncattrs()
##            if ncvar in ("ta",):
##                self.assertFalse(var.ndim)
##                self.assertIn("aggregated_dimensions", attrs)
##                self.assertIn("aggregated_data", attrs)
##            else:
##                self.assertNotIn("aggregated_dimensions", attrs)
##                self.assertNotIn("aggregated_data", attrs)
##
##        nc.close()
##
##        # Dimension construct
##        for constructs in (
##            "dimension_coordinate",
##            ["dimension_coordinate"],
##            {"dimension_coordinate": None},
##            {"dimension_coordinate": 1},
##            {"dimension_coordinate": cf.eq(1)},
##        ):
##            cf.write(f, tmpfile2, cfa={"constructs": constructs})
##            nc = netCDF4.Dataset(tmpfile2, "r")
##            for ncvar, var in nc.variables.items():
##                attrs = var.ncattrs()
##                if ncvar in (
##                    "x",
##                    "x_bnds",
##                    "y",
##                    "y_bnds",
##                    "atmosphere_hybrid_height_coordinate",
##                    "atmosphere_hybrid_height_coordinate_bounds",
##                ):
##                    self.assertFalse(var.ndim)
##                    self.assertIn("aggregated_dimensions", attrs)
##                    self.assertIn("aggregated_data", attrs)
##                else:
##                    self.assertNotIn("aggregated_dimensions", attrs)
##                    self.assertNotIn("aggregated_data", attrs)
##
##            nc.close()
##
##        # Dimension and auxiliary constructs
##        for constructs in (
##            ["dimension_coordinate", "auxiliary_coordinate"],
##            {"dimension_coordinate": None, "auxiliary_coordinate": cf.ge(2)},
##        ):
##            cf.write(f, tmpfile2, cfa={"constructs": constructs})
##            nc = netCDF4.Dataset(tmpfile2, "r")
##            for ncvar, var in nc.variables.items():
##                attrs = var.ncattrs()
##                if ncvar in (
##                    "x",
##                    "x_bnds",
##                    "y",
##                    "y_bnds",
##                    "atmosphere_hybrid_height_coordinate",
##                    "atmosphere_hybrid_height_coordinate_bounds",
##                    "latitude_1",
##                    "longitude_1",
##                ):
##                    self.assertFalse(var.ndim)
##                    self.assertIn("aggregated_dimensions", attrs)
##                    self.assertIn("aggregated_data", attrs)
##                else:
##                    self.assertNotIn("aggregated_dimensions", attrs)
##                    self.assertNotIn("aggregated_data", attrs)
##
##            nc.close()
##
##    def test_CFA_PP(self):
##        """Test writing CFA-netCDF with PP format fragments."""
##        f = cf.read("file1.pp")[0]
##        cf.write(f, tmpfile1, cfa=True)
##
##        # Check that only the fields have been aggregated
##        nc = netCDF4.Dataset(tmpfile1, "r")
##        for ncvar, var in nc.variables.items():
##            attrs = var.ncattrs()
##            if ncvar in ("UM_m01s15i201_vn405",):
##                self.assertFalse(var.ndim)
##                self.assertIn("aggregated_dimensions", attrs)
##                self.assertIn("aggregated_data", attrs)
##            else:
##                self.assertNotIn("aggregated_dimensions", attrs)
##                self.assertNotIn("aggregated_data", attrs)
##
##        nc.close()
##
##        g = cf.read(tmpfile1)
##        self.assertEqual(len(g), 1)
##        self.assertTrue(f.equals(g[0]))
##
##    def test_CFA_multiple_files(self):
##        """Test storing multiple CFA frgament locations."""
##        f = cf.example_field(0)
##        cf.write(f, tmpfile1)
##        f = cf.read(tmpfile1)[0]
##        f.add_file_location("/new/location")
##
##        cf.write(f, tmpfile2, cfa=True)
##        g = cf.read(tmpfile2)
##        self.assertEqual(len(g), 1)
##        g = g[0]
##        self.assertTrue(f.equals(g))
##
##        self.assertEqual(len(g.data.get_filenames()), 2)
##        self.assertEqual(len(g.get_filenames()), 3)
##
##    def test_CFA_unlimited_dimension(self):
##        """Test CFA with unlimited dimensions"""
##        # Create a CFA file from a field that has an unlimited
##        # dimension and no metadata constructs spanning that dimension
##        f = cf.example_field(0)
##        d = f.domain_axis("X")
##        d.nc_set_unlimited(True)
##        f.del_construct("X")
##        cf.write(f, tmpfile1)
##        g = cf.read(tmpfile1)
##        cf.write(g, tmpfile2, cfa=True)
##
##        # Check that the CFA file can be read
##        h = cf.read(tmpfile2)
##        self.assertEqual(len(h), 1)
##
##
## if __name__ == "__main__":
##    print("Run date:", datetime.datetime.now())
##    cf.environment()
##    print()
##    unittest.main(verbosity=2)
