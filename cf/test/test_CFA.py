import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

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
            g = g[0]

            self.assertTrue(f.equals(g))

    def test_CFA_general(self):
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

        n = n[0]
        c = c[0]
        self.assertTrue(c.equals(f))
        self.assertTrue(c.equals(n))

    def test_CFA_strict(self):
        f = cf.example_field(0)

        # By default, can't write as CF-netCDF those variables
        # selected for CFA treatment, but which aren't suitable.
        with self.assertRaises(ValueError):
            cf.write(f, tmpfile1, cfa=True)

        # The previous line should have deleted the file
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
        import cf

        f = cf.example_field(0)
        self.assertFalse(f.field_ancillaries())

        tmpfile1 = "delme1.nc"
        tmpfile2 = "delme2.nc"
        tmpfile3 = "delme3.nc"
        tmpfile4 = "delme4.nc"

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
        e = e[0]
        self.assertTrue(e.equals(d))

    def test_CFA_PP(self):
        pass


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
