import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest
from pathlib import PurePath

import netCDF4

faulthandler.enable()  # to debug seg faults and timeouts

import cf

n_tmpfiles = 5
tmpfiles = [
    tempfile.mkstemp("_test_CFA.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(
    tmpfile1,
    tmpfile2,
    tmpfile3,
    tmpfile4,
    tmpfile5,
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
    netcdf3_fmts = [
        "NETCDF3_CLASSIC",
        "NETCDF3_64BIT",
        "NETCDF3_64BIT_OFFSET",
        "NETCDF3_64BIT_DATA",
    ]
    netcdf4_fmts = ["NETCDF4", "NETCDF4_CLASSIC"]
    netcdf_fmts = netcdf3_fmts + netcdf4_fmts

    def test_CFA_fmt(self):
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1)[0]

        for fmt in self.netcdf_fmts:
            cf.write(f, tmpfile2, fmt=fmt, cfa=True)
            g = cf.read(tmpfile2)
            self.assertEqual(len(g), 1)
            self.assertTrue(f.equals(g[0]))

    def test_CFA_multiple_fragments(self):
        f = cf.example_field(0)

        cf.write(f[:2], tmpfile1)
        cf.write(f[2:], tmpfile2)

        a = cf.read([tmpfile1, tmpfile2])
        self.assertEqual(len(a), 1)
        a = a[0]

        nc_file = tmpfile3
        cfa_file = tmpfile4
        cf.write(a, nc_file)
        cf.write(a, cfa_file, cfa=True)

        n = cf.read(nc_file)
        c = cf.read(cfa_file)
        self.assertEqual(len(n), 1)
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0].equals(f))
        self.assertTrue(n[0].equals(c[0]))

    def test_CFA_strict(self):
        f = cf.example_field(0)

        # By default, can't write as CF-netCDF those variables
        # selected for CFA treatment, but which aren't suitable.
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1, cfa=True)

        # The previous line should have deleted the output file
        self.assertFalse(os.path.exists(tmpfile1))

        cf.write(f, tmpfile1, cfa={"strict": False})
        g = cf.read(tmpfile1)
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0].equals(f))

        cf.write(g, tmpfile2, cfa={"strict": True})
        g = cf.read(tmpfile2)
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0].equals(f))

    def test_CFA_field_ancillaries(self):
        f = cf.example_field(0)
        self.assertFalse(f.field_ancillaries())

        a = f[:2]
        b = f[2:]
        a.set_property("foo", "bar_a")
        b.set_property("foo", "bar_b")
        cf.write(a, tmpfile1)
        cf.write(b, tmpfile2)

        c = cf.read(
            [tmpfile1, tmpfile2], aggregate={"field_ancillaries": "foo"}
        )
        self.assertEqual(len(c), 1)
        c = c[0]
        self.assertEqual(len(c.field_ancillaries()), 1)
        anc = c.field_ancillary()
        self.assertTrue(anc.data.cfa_get_term())
        self.assertFalse(anc.data.cfa_get_write())

        cf.write(c, tmpfile3, cfa=False)
        c2 = cf.read(tmpfile3)
        self.assertEqual(len(c2), 1)
        self.assertFalse(c2[0].field_ancillaries())

        cf.write(c, tmpfile4, cfa=True)
        d = cf.read(tmpfile4)
        self.assertEqual(len(d), 1)
        d = d[0]

        self.assertEqual(len(d.field_ancillaries()), 1)
        anc = d.field_ancillary()
        self.assertTrue(anc.data.cfa_get_term())
        self.assertFalse(anc.data.cfa_get_write())
        self.assertTrue(d.equals(c))

        cf.write(d, tmpfile5, cfa=False)
        e = cf.read(tmpfile5)
        self.assertEqual(len(e), 1)
        self.assertFalse(e[0].field_ancillaries())

        cf.write(d, tmpfile5, cfa=True)
        e = cf.read(tmpfile5)
        self.assertEqual(len(e), 1)
        self.assertTrue(e[0].equals(d))

    def test_CFA_substitutions_0(self):
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1)[0]

        cwd = os.getcwd()

        f.data.cfa_set_file_substitutions({"base": cwd})

        cf.write(
            f,
            tmpfile2,
            cfa={"absolute_paths": True},
        )

        nc = netCDF4.Dataset(tmpfile2, "r")
        cfa_file = nc.variables["cfa_file"]
        self.assertEqual(
            cfa_file.getncattr("substitutions"),
            f"${{base}}: {cwd}",
        )
        self.assertEqual(
            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
        )
        nc.close()

        g = cf.read(tmpfile2)
        self.assertEqual(len(g), 1)
        self.assertTrue(f.equals(g[0]))

    def test_CFA_substitutions_1(self):
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1)[0]

        cwd = os.getcwd()
        for base in ("base", "${base}"):
            cf.write(
                f,
                tmpfile2,
                cfa={"absolute_paths": True, "substitutions": {base: cwd}},
            )

            nc = netCDF4.Dataset(tmpfile2, "r")
            cfa_file = nc.variables["cfa_file"]
            self.assertEqual(
                cfa_file.getncattr("substitutions"),
                f"${{base}}: {cwd}",
            )
            self.assertEqual(
                cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
            )
            nc.close()

        g = cf.read(tmpfile2)
        self.assertEqual(len(g), 1)
        self.assertTrue(f.equals(g[0]))

    def test_CFA_substitutions_2(self):
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1)[0]

        cwd = os.getcwd()

        # RRRRRRRRRRRRRRRRRRR
        f.data.cfa_clear_file_substitutions()
        f.data.cfa_set_file_substitutions({"base": cwd})

        cf.write(
            f,
            tmpfile2,
            cfa={
                "absolute_paths": True,
                "substitutions": {"base2": "/bad/location"},
            },
        )

        nc = netCDF4.Dataset(tmpfile2, "r")
        cfa_file = nc.variables["cfa_file"]
        self.assertEqual(
            cfa_file.getncattr("substitutions"),
            f"${{base2}}: /bad/location ${{base}}: {cwd}",
        )
        self.assertEqual(
            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
        )
        nc.close()

        g = cf.read(tmpfile2)
        self.assertEqual(len(g), 1)
        self.assertTrue(f.equals(g[0]))

        # RRRRRRRRRRRRRRRRRRR
        f.data.cfa_clear_file_substitutions()
        f.data.cfa_set_file_substitutions({"base": "/bad/location"})

        cf.write(
            f,
            tmpfile2,
            cfa={"absolute_paths": True, "substitutions": {"base": cwd}},
        )

        nc = netCDF4.Dataset(tmpfile2, "r")
        cfa_file = nc.variables["cfa_file"]
        self.assertEqual(
            cfa_file.getncattr("substitutions"),
            f"${{base}}: {cwd}",
        )
        self.assertEqual(
            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
        )
        nc.close()

        g = cf.read(tmpfile2)
        self.assertEqual(len(g), 1)
        self.assertTrue(f.equals(g[0]))

        # RRRRRRRRRRRRRRRR
        f.data.cfa_clear_file_substitutions()
        f.data.cfa_set_file_substitutions({"base2": "/bad/location"})

        cf.write(
            f,
            tmpfile2,
            cfa={"absolute_paths": True, "substitutions": {"base": cwd}},
        )

        nc = netCDF4.Dataset(tmpfile2, "r")
        cfa_file = nc.variables["cfa_file"]
        self.assertEqual(
            cfa_file.getncattr("substitutions"),
            f"${{base2}}: /bad/location ${{base}}: {cwd}",
        )
        self.assertEqual(
            cfa_file[...], f"file://${{base}}/{os.path.basename(tmpfile1)}"
        )
        nc.close()

        g = cf.read(tmpfile2)
        self.assertEqual(len(g), 1)
        self.assertTrue(f.equals(g[0]))

    def test_CFA_absolute_paths(self):
        f = cf.example_field(0)
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1)[0]

        for absolute_paths, filename in zip(
            (True, False),
            (
                PurePath(os.path.abspath(tmpfile1)).as_uri(),
                os.path.basename(tmpfile1),
            ),
        ):
            cf.write(f, tmpfile2, cfa={"absolute_paths": absolute_paths})

            nc = netCDF4.Dataset(tmpfile2, "r")
            cfa_file = nc.variables["cfa_file"]
            self.assertEqual(cfa_file[...], filename)
            nc.close()

            g = cf.read(tmpfile2)
            self.assertEqual(len(g), 1)
            self.assertTrue(f.equals(g[0]))

    def test_CFA_constructs(self):
        f = cf.example_field(1)
        f.del_construct("T")
        f.del_construct("long_name=Grid latitude name")
        cf.write(f, tmpfile1)
        f = cf.read(tmpfile1)[0]

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
            {"dimension_coordinate": cf.eq(1)},
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
            {"dimension_coordinate": None, "auxiliary_coordinate": cf.ge(2)},
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

    def test_CFA_PP(self):
        f = cf.read("file1.pp")[0]
        cf.write(f, tmpfile1, cfa=True)

        nc = netCDF4.Dataset(tmpfile1, "r")
        for ncvar, var in nc.variables.items():
            attrs = var.ncattrs()
            if ncvar in ("UM_m01s15i201_vn405",):
                self.assertFalse(var.ndim)
                self.assertIn("aggregated_dimensions", attrs)
                self.assertIn("aggregated_data", attrs)
            else:
                self.assertNotIn("aggregated_dimensions", attrs)
                self.assertNotIn("aggregated_data", attrs)

        nc.close()

        g = cf.read(tmpfile1)
        self.assertEqual(len(g), 1)
        self.assertTrue(f.equals(g[0]))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
