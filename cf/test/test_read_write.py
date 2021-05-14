import atexit
import datetime
import faulthandler
import inspect
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy

faulthandler.enable()  # to debug seg faults and timeouts

import cf


n_tmpfiles = 8
tmpfiles = [
    tempfile.mkstemp("_test_read_write.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(
    tmpfile,
    tmpfileh,
    tmpfileh2,
    tmpfilec,
    tmpfilec2,
    tmpfile0,
    tmpfile1,
    tmpfile2,
) = tmpfiles


def _remove_tmpfiles():
    """TODO."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class read_writeTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )

    broken_bounds = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "broken_bounds.cdl"
    )

    string_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "string_char.nc"
    )

    chunk_sizes = (100000, 300)

    f0 = cf.example_field(0)
    f1 = cf.example_field(1)

    netcdf3_fmts = [
        "NETCDF3_CLASSIC",
        "NETCDF3_64BIT",
        "NETCDF3_64BIT_OFFSET",
        "NETCDF3_64BIT_DATA",
    ]
    netcdf4_fmts = [
        "NETCDF4",
        "NETCDF4_CLASSIC",
    ]
    netcdf_fmts = netcdf3_fmts + netcdf4_fmts

    def test_write_filename(self):
        f = self.f0
        a = f.array

        cf.write(f, tmpfile)
        g = cf.read(tmpfile)

        with self.assertRaises(Exception):
            cf.write(g, tmpfile)

        self.assertTrue((a == g[0].array).all())

    def test_read_mask(self):
        f = self.f0.copy()

        N = f.size

        f.data[1, 1] = cf.masked
        f.data[2, 2] = cf.masked

        f.del_property("_FillValue", None)
        f.del_property("missing_value", None)

        cf.write(f, tmpfile)

        g = cf.read(tmpfile)[0]
        self.assertEqual(numpy.ma.count(g.data.array), N - 2)

        g = cf.read(tmpfile, mask=False)[0]
        self.assertEqual(numpy.ma.count(g.data.array), N)

        g.apply_masking(inplace=True)
        self.assertEqual(numpy.ma.count(g.data.array), N - 2)

        f.set_property("_FillValue", 999)
        f.set_property("missing_value", -111)
        cf.write(f, tmpfile)

        g = cf.read(tmpfile)[0]
        self.assertEqual(numpy.ma.count(g.data.array), N - 2)

        g = cf.read(tmpfile, mask=False)[0]
        self.assertEqual(numpy.ma.count(g.data.array), N)

        g.apply_masking(inplace=True)
        self.assertEqual(numpy.ma.count(g.data.array), N - 2)

    def test_read_directory(self):
        pwd = os.getcwd() + "/"

        dir = "dir_" + inspect.stack()[0][3]

        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        except Exception:
            raise ValueError(f"Can not mkdir {pwd}{dir}")

        f = "test_file2.nc"
        try:
            os.symlink(pwd + f, pwd + dir + "/" + f)
        except FileExistsError:
            pass

        subdir = dir + "/subdir"
        try:
            os.mkdir(subdir)
        except FileExistsError:
            pass
        except Exception:
            raise ValueError(f"Can not mkdir {pwd}{subdir}")

        for f in ("test_file3.nc", "test_file.nc"):
            try:
                os.symlink(pwd + f, pwd + subdir + "/" + f)
            except FileExistsError:
                pass

        f = cf.read(dir, aggregate=False)
        self.assertEqual(len(f), 1, f)

        f = cf.read(dir, recursive=True, aggregate=False)
        self.assertEqual(len(f), 3)

        f = cf.read([dir, subdir], aggregate=False)
        self.assertEqual(len(f), 3)

        f = cf.read([subdir, dir], aggregate=False)
        self.assertEqual(len(f), 3)

        f = cf.read([dir, subdir], recursive=True, aggregate=False)
        self.assertEqual(len(f), 5)

        f = cf.read(subdir, aggregate=False)
        self.assertEqual(len(f), 2)

        f = cf.read(subdir, recursive=True, aggregate=False)
        self.assertEqual(len(f), 2)

        shutil.rmtree(dir)

    def test_read_select(self):
        # select on field list
        f = cf.read(self.filename, select="eastward_wind")
        g = cf.read(self.filename)
        self.assertTrue(f.equals(g, verbose=2), "Bad read with select keyword")

    def test_read_squeeze(self):
        # select on field list
        cf.read(self.filename, squeeze=True)
        cf.read(self.filename, unsqueeze=True)
        with self.assertRaises(Exception):
            cf.read(self.filename, unsqueeze=True, squeeze=True)

    def test_read_aggregate(self):
        cf.read(self.filename, aggregate=True)
        cf.read(self.filename, aggregate=False)
        cf.read(self.filename, aggregate={})

    def test_read_extra(self):
        # Test field keyword of cf.read
        filename = self.filename

        f = cf.read(filename)
        self.assertEqual(len(f), 1, "\n" + str(f))

        f = cf.read(filename, extra=["auxiliary_coordinate"])
        self.assertEqual(len(f), 4, "\n" + str(f))

        f = cf.read(filename, extra="cell_measure")
        self.assertEqual(len(f), 2, "\n" + str(f))

        f = cf.read(filename, extra=["field_ancillary"])
        self.assertEqual(len(f), 5, "\n" + str(f))

        f = cf.read(filename, extra="domain_ancillary", verbose=0)
        self.assertEqual(len(f), 4, "\n" + str(f))

        f = cf.read(
            filename, extra=["field_ancillary", "auxiliary_coordinate"]
        )
        self.assertEqual(len(f), 8, "\n" + str(f))

        self.assertEqual(
            len(
                cf.read(
                    filename,
                    extra=["domain_ancillary", "auxiliary_coordinate"],
                )
            ),
            7,
        )
        f = cf.read(
            filename,
            extra=["domain_ancillary", "cell_measure", "auxiliary_coordinate"],
        )
        self.assertEqual(len(f), 8, "\n" + str(f))

        f = cf.read(
            filename,
            extra=(
                "field_ancillary",
                "dimension_coordinate",
                "cell_measure",
                "auxiliary_coordinate",
                "domain_ancillary",
            ),
        )
        self.assertEqual(len(f), 15, "\n" + str(f))

    def test_read_write_format(self):
        cf.write(self.f1, tmpfile)

        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                for fmt in self.netcdf3_fmts + ["CFA"]:
                    f = cf.read(tmpfile)[0]

                    cf.write(f, tmpfile2, fmt=fmt)
                    g = cf.read(tmpfile2, verbose=0)
                    self.assertEqual(len(g), 1)
                    g = g[0]

                    self.assertTrue(
                        f.equals(g, verbose=1),
                        f"Bad read/write of format {fmt!r}",
                    )

    def test_read_write_netCDF4_compress_shuffle(self):
        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                f = cf.read(self.filename)[0]
                for fmt in ("NETCDF4", "NETCDF4_CLASSIC", "CFA4"):
                    cf.write(
                        f,
                        tmpfile,
                        fmt=fmt,
                        compress=1,
                        shuffle=True,
                    )
                    g = cf.read(tmpfile)[0]
                    self.assertTrue(
                        f.equals(g, verbose=2),
                        f"Bad read/write with lossless compression: {fmt}",
                    )

    def test_write_datatype(self):
        for chunksize in self.chunk_sizes:
            with cf.chunksize(chunksize):
                f = cf.read(self.filename)[0]
                self.assertEqual(f.dtype, numpy.dtype(float))
                cf.write(
                    f,
                    tmpfile,
                    fmt="NETCDF4",
                    datatype={numpy.dtype(float): numpy.dtype("float32")},
                )
                g = cf.read(tmpfile)[0]
                self.assertEqual(
                    g.dtype,
                    numpy.dtype("float32"),
                    "datatype read in is " + str(g.dtype),
                )

        # Keyword single
        f = cf.read(self.filename)[0]
        self.assertEqual(f.dtype, numpy.dtype(float))
        cf.write(f, tmpfile, fmt="NETCDF4", single=True)
        g = cf.read(tmpfile)[0]
        self.assertEqual(
            g.dtype,
            numpy.dtype("float32"),
            "datatype read in is " + str(g.dtype),
        )

        # Keyword double
        f = g
        self.assertEqual(f.dtype, numpy.dtype("float32"))
        cf.write(f, tmpfile2, fmt="NETCDF4", double=True)
        g = cf.read(tmpfile2)[0]
        self.assertEqual(
            g.dtype, numpy.dtype(float), "datatype read in is " + str(g.dtype)
        )

        for single in (True, False):
            for double in (True, False):
                with self.assertRaises(Exception):
                    cf.write(g, double=double, single=single)

        datatype = {numpy.dtype(float): numpy.dtype("float32")}
        with self.assertRaises(Exception):
            cf.write(g, datatype=datatype, single=True)

        with self.assertRaises(Exception):
            cf.write(g, datatype=datatype, double=True)

    def test_write_reference_datetime(self):
        for reference_datetime in ("1751-2-3", "1492-12-30"):
            cf.write(self.f0, tmpfile, reference_datetime=reference_datetime)

            g = cf.read(tmpfile)[0]

            t = g.dimension_coordinate("T")
            self.assertEqual(
                t.Units,
                cf.Units("days since " + reference_datetime),
                f"Units written were {t.Units.reftime!r} not "
                f"{reference_datetime!r}",
            )

    def test_read_write_unlimited(self):
        for fmt in ("NETCDF4", "NETCDF3_CLASSIC"):
            f = self.f1.copy()
            domain_axes = f.domain_axes()

            domain_axes["domainaxis0"].nc_set_unlimited(True)
            cf.write(f, tmpfile, fmt=fmt)

            f = cf.read(tmpfile)[0]
            domain_axes = f.domain_axes()
            self.assertTrue(domain_axes["domainaxis0"].nc_is_unlimited())

        fmt = "NETCDF4"
        f = self.f1.copy()
        domain_axes = f.domain_axes()
        domain_axes["domainaxis0"].nc_set_unlimited(True)
        domain_axes["domainaxis2"].nc_set_unlimited(True)
        cf.write(f, tmpfile, fmt=fmt)

        f = cf.read(tmpfile)[0]
        domain_axes = f.domain_axes()
        self.assertTrue(domain_axes["domainaxis0"].nc_is_unlimited())
        self.assertTrue(domain_axes["domainaxis2"].nc_is_unlimited())

    def test_read_pp(self):
        p = cf.read("wgdos_packed.pp")[0]
        p0 = cf.read(
            "wgdos_packed.pp",
            um={
                "fmt": "PP",
                "endian": "big",
                "word_size": 4,
                "version": 4.5,
                "height_at_top_of_model": 23423.65,
            },
        )[0]

        self.assertTrue(p.equals(p0, verbose=2))

    def test_read_CDL(self):
        subprocess.run(
            " ".join(["ncdump", self.filename, ">", tmpfile]),
            shell=True,
            check=True,
        )

        # For the cases of '-h' and '-c', i.e. only header info or coordinates,
        # notably no data, take two cases each: one where there is sufficient
        # info from the metadata to map to fields, and one where there isn't:
        #     1. Sufficient metadata, so should be read-in successfully
        subprocess.run(
            " ".join(["ncdump", "-h", self.filename, ">", tmpfileh]),
            shell=True,
            check=True,
        )
        subprocess.run(
            " ".join(["ncdump", "-c", self.filename, ">", tmpfilec]),
            shell=True,
            check=True,
        )

        #     2. Insufficient metadata, so should error with a message as such
        geometry_1_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "geometry_1.nc"
        )
        subprocess.run(
            " ".join(["ncdump", "-h", geometry_1_file, ">", tmpfileh2]),
            shell=True,
            check=True,
        )
        subprocess.run(
            " ".join(["ncdump", "-c", geometry_1_file, ">", tmpfilec2]),
            shell=True,
            check=True,
        )

        f0 = cf.read(self.filename)[0]

        # Case (1) as above, so read in and check the fields are as should be
        f = cf.read(tmpfile)[0]
        cf.read(tmpfileh)[0]
        c = cf.read(tmpfilec)[0]

        # Case (2) as above, so the right error should be raised on read
        with self.assertRaises(ValueError):
            cf.read(tmpfileh2)[0]

        with self.assertRaises(ValueError):
            cf.read(tmpfilec2)[0]

        self.assertTrue(f0.equals(f, verbose=2))

        self.assertTrue(
            f.construct("grid_latitude").equals(
                c.construct("grid_latitude"), verbose=2
            )
        )
        self.assertTrue(
            f0.construct("grid_latitude").equals(
                c.construct("grid_latitude"), verbose=2
            )
        )

        with self.assertRaises(Exception):
            cf.read("test_read_write.py")

    def test_read_write_string(self):
        f = cf.read(self.string_filename)

        n = int(len(f) / 2)

        for i in range(n):
            j = i + n
            self.assertTrue(
                f[i].data.equals(f[j].data, verbose=1),
                "{!r} {!r}".format(f[i], f[j]),
            )
            self.assertTrue(
                f[j].data.equals(f[i].data, verbose=1),
                "{!r} {!r}".format(f[j], f[i]),
            )

        for string0 in (True, False):
            for fmt0 in ("NETCDF4", "NETCDF3_CLASSIC"):
                cf.write(f, tmpfile0, fmt=fmt0, string=string0)

                for string1 in (True, False):
                    for fmt1 in ("NETCDF4", "NETCDF3_CLASSIC"):
                        cf.write(f, tmpfile1, fmt=fmt1, string=string1)

                        for i, j in zip(cf.read(tmpfile1), cf.read(tmpfile0)):
                            self.assertTrue(i.equals(j, verbose=1))

    def test_read_broken_bounds(self):
        f = cf.read(self.broken_bounds, verbose=0)
        self.assertEqual(len(f), 2)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
