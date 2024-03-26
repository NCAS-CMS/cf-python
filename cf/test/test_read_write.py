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
    """Try to remove defined temporary files by deleting their paths."""
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
    netcdf4_fmts = ["NETCDF4", "NETCDF4_CLASSIC"]
    netcdf_fmts = netcdf3_fmts + netcdf4_fmts

    def test_write_filename(self):
        f = self.f0
        a = f.array

        cf.write(f, tmpfile)
        g = cf.read(tmpfile)

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
        f = cf.read(tmpfile)[0]

        # TODO: reinstate "CFA" at version > 3.14
        for fmt in self.netcdf_fmts:  # + ["CFA"]:
            cf.write(f, tmpfile2, fmt=fmt)
            g = cf.read(tmpfile2, verbose=0)
            self.assertEqual(len(g), 1)
            g = g[0]

            self.assertTrue(
                f.equals(g, verbose=1),
                f"Bad read/write of format {fmt!r}",
            )

    def test_write_netcdf_mode(self):
        """Test the `mode` parameter to `write`, notably append mode."""
        g = cf.read(self.filename)  # note 'g' has one field

        # Test special case #1: attempt to append fields with groups
        # (other than 'root') which should be forbidden. Using fmt="NETCDF4"
        # since it is the only format where groups are allowed.
        #
        # Note: this is not the most natural test to do first, but putting
        # it before the rest reduces spurious seg faults for me, so...
        g[0].nc_set_variable_groups(["forecast", "model"])
        cf.write(g, tmpfile, fmt="NETCDF4", mode="w")  # 1. overwrite to wipe
        f = cf.read(tmpfile)
        with self.assertRaises(ValueError):
            cf.write(g[0], tmpfile, fmt="NETCDF4", mode="a")

        # Test special case #2: attempt to append fields with contradictory
        # featureType to the original file:
        g[0].nc_clear_variable_groups()
        g[0].nc_set_global_attribute("featureType", "profile")
        cf.write(
            g,
            tmpfile,
            fmt="NETCDF4",
            mode="w",
            global_attributes=("featureType", "profile"),
        )  # 1. overwrite to wipe
        h = cf.example_field(3)
        h.nc_set_global_attribute("featureType", "timeSeries")
        with self.assertRaises(ValueError):
            cf.write(h, tmpfile, fmt="NETCDF4", mode="a")
        # Now remove featureType attribute for subsquent tests:
        g_attrs = g[0].nc_clear_global_attributes()
        del g_attrs["featureType"]
        g[0].nc_set_global_attributes(g_attrs)

        # Set a non-trivial (i.e. not only 'Conventions') global attribute to
        # make the global attribute testing more robust:
        add_global_attr = ["remark", "A global comment."]
        original_global_attrs = g[0].nc_global_attributes()
        original_global_attrs[add_global_attr[0]] = None  # -> None on fields
        g[0].nc_set_global_attribute(*add_global_attr)

        # First test a bad mode value:
        with self.assertRaises(ValueError):
            cf.write(g[0], tmpfile, mode="g")

        g_copy = g.copy()

        for fmt in self.netcdf_fmts:  # test over all netCDF 3 and 4 formats
            # Other tests cover write as default mode (i.e. test with no mode
            # argument); here test explicit provision of 'w' as argument:
            cf.write(
                g,
                tmpfile,
                fmt=fmt,
                mode="w",
                global_attributes=add_global_attr,
            )
            f = cf.read(tmpfile)

            new_length = 1  # since 1 == len(g)
            self.assertEqual(len(f), new_length)
            # Ignore as 'remark' should be 'None' on the field as tested below
            self.assertTrue(f[0].equals(g[0], ignore_properties=["remark"]))
            self.assertEqual(
                f[0].nc_global_attributes(), original_global_attrs
            )

            # Main aspect of this test: testing the append mode ('a'): now
            # append all other example fields, to check a diverse variety.
            for ex_field_n, ex_field in enumerate(cf.example_fields()):
                # Note: after Issue #141, this skip can be removed.
                if ex_field_n == 1:
                    continue

                # Skip since "RuntimeError: Can't create variable in
                # NETCDF4_CLASSIC file from (2)  (NetCDF: Attempting netcdf-4
                # operation on strict nc3 netcdf-4 file)" i.e. not possible.
                if fmt == "NETCDF4_CLASSIC" and ex_field_n in (6, 7):
                    continue

                print(
                    "TODOUGRID: excluding example fields 8, 9, 10 until writing UGRID is enabled"
                )
                if ex_field_n in (8, 9, 10):
                    continue

                cf.write(ex_field, tmpfile, fmt=fmt, mode="a")
                f = cf.read(tmpfile)

                if ex_field_n == 5:  # another special case
                    # The n=2 and n=5 example fields for cf-python aggregate
                    # down to one field, e.g. for b as n=2 and c as n=5:
                    #   >>> c.equals(b, verbose=-1)
                    #   Data: Different shapes: (118, 5, 8) != (36, 5, 8)
                    #   Field: Different data
                    #   False
                    #   >>> a = cf.aggregate([b, c])
                    #   >>> a
                    #   [<CF Field: air_potential_temperature(
                    #    time(154), latitude(5), longitude(8)) K>]
                    #
                    # therefore need to check FL length hasn't changed and
                    # (further below) that n=2,5 aggregated field is present.
                    pass  # i.e. new_length should remain the same as before
                else:
                    new_length += 1  # should be exactly one more field now
                self.assertEqual(len(f), new_length)

                if ex_field_n == 5:
                    ex_n2_and_n5_aggregated = cf.aggregate(
                        [cf.example_field(2), cf.example_field(5)]
                    )[0]
                    self.assertTrue(
                        any(
                            [
                                ex_n2_and_n5_aggregated.equals(
                                    file_field,
                                    ignore_properties=[
                                        "comment",
                                        "featureType",
                                        "remark",
                                    ],
                                )
                                for file_field in f
                            ]
                        )
                    )
                else:
                    # Can't guarantee order of fields created during append op.
                    # so check new field is *somewhere* in read-in fieldlist
                    self.assertTrue(
                        any(
                            [
                                ex_field.equals(
                                    file_field,
                                    ignore_properties=[
                                        "comment",
                                        "featureType",
                                        "remark",
                                    ],
                                )
                                for file_field in f
                            ]
                        )
                    )
                for file_field in f:
                    self.assertEqual(
                        file_field.nc_global_attributes(),
                        original_global_attrs,
                    )

            # Now do the same test, but appending all of the example fields in
            # one operation rather than one at a time, to check that it works.
            cf.write(g, tmpfile, fmt=fmt, mode="w")  # 1. overwrite to wipe
            print(
                "TODOUGRID: excluding example fields 8, 9, 10 until writing UGRID is enabled"
            )
            append_ex_fields = cf.example_fields(0, 1, 2, 3, 4, 5, 6, 7)
            del append_ex_fields[1]  # note: can remove after Issue #141 closed
            if fmt in "NETCDF4_CLASSIC":
                # Remove n=6 and =7 for reasons as given above (del => minus 1)
                append_ex_fields = append_ex_fields[:5]

            # Equals len(append_ex_fields), + 1 [for original 'g'] and -1 [for
            # field n=5 which aggregates to one with n=2] => + 1 - 1 = + 0:
            overall_length = len(append_ex_fields)
            cf.write(
                append_ex_fields, tmpfile, fmt=fmt, mode="a"
            )  # 2. now append
            f = cf.read(tmpfile)
            self.assertEqual(len(f), overall_length)

            # Also test the mode="r+" alias for mode="a".
            cf.write(g, tmpfile, fmt=fmt, mode="w")  # 1. overwrite to wipe
            cf.write(
                append_ex_fields, tmpfile, fmt=fmt, mode="r+"
            )  # 2. now append
            f = cf.read(tmpfile)
            self.assertEqual(len(f), overall_length)

            # The appended fields themselves are now known to be correct,
            # but we also need to check that any coordinates that are
            # equal across different fields have been shared in the
            # source netCDF, rather than written in separately.
            #
            # Note that the coordinates that are shared across the set of
            # all example fields plus the field 'g' from the contents of
            # the original file (self.filename) are as follows:
            #
            # 1. Example fields n=0 and n=1 share:
            #    <DimensionCoordinate: time(1) days since 2018-12-01 >
            # 2. Example fields n=0, n=2 and n=5 share:
            #    <DimensionCoordinate: latitude(5) degrees_north> and
            #    <DimensionCoordinate: longitude(8) degrees_east>
            # 3. Example fields n=2 and n=5 share:
            #    <DimensionCoordinate: air_pressure(1) hPa>
            # 4. The original file field ('g') and example field n=1 share:
            #    <AuxiliaryCoordinate: latitude(10, 9) degrees_N>,
            #    <AuxiliaryCoordinate: longitude(9, 10) degrees_E>,
            #    <Dimension...: atmosphere_hybrid_height_coordinate(1) >,
            #    <DimensionCoordinate: grid_latitude(10) degrees>,
            #    <DimensionCoordinate: grid_longitude(9) degrees> and
            #    <DimensionCoordinate: time(1) days since 2018-12-01 >
            #
            # Therefore we check all of those coordinates for singularity,
            # i.e. the same underlying netCDF variables, in turn.

            # But first, since the order of the fields appended isn't
            # guaranteed, we must find the mapping of the example fields to
            # their position in the read-in FieldList.
            f = cf.read(tmpfile)
            # Element at index N gives position of example field n=N in file
            file_field_order = []
            for ex_field in cf.example_fields():
                position = [
                    f.index(file_field)
                    for file_field in f
                    if ex_field.equals(
                        file_field,
                        ignore_properties=["comment", "featureType", "remark"],
                    )
                ]
                if not position:
                    position = [None]  # to record skipped example fields
                file_field_order.append(position[0])

            equal_coors = {
                ((0, "dimensioncoordinate2"), (1, "dimensioncoordinate3")),
                ((0, "dimensioncoordinate0"), (2, "dimensioncoordinate1")),
                ((0, "dimensioncoordinate1"), (2, "dimensioncoordinate2")),
                ((0, "dimensioncoordinate0"), (5, "dimensioncoordinate1")),
                ((0, "dimensioncoordinate1"), (5, "dimensioncoordinate2")),
                ((2, "dimensioncoordinate3"), (5, "dimensioncoordinate3")),
            }
            for coor_1, coor_2 in equal_coors:
                ex_field_1_position, c_1 = coor_1
                ex_field_2_position, c_2 = coor_2
                # Now map the appropriate example field to the file FieldList
                f_1 = file_field_order[ex_field_1_position]
                f_2 = file_field_order[ex_field_2_position]
                # None for fields skipped in test, distinguish from falsy 0
                if f_1 is None or f_2 is None:
                    continue
                self.assertEqual(
                    f[f_1]
                    .constructs()
                    .filter_by_identity(c_1)
                    .value()
                    .nc_get_variable(),
                    f[f_2]
                    .constructs()
                    .filter_by_identity(c_2)
                    .value()
                    .nc_get_variable(),
                )

            # Note: after Issue #141, the block below should be un-commented.
            #
            # The original file field 'g' must be at the remaining position:
            # rem_position = list(set(
            #     range(len(f))).difference(set(file_field_order)))[0]
            # # In the final cases, it is easier to remove the one differing
            # # coordinate to get the equal coordinates that should be shared:
            # original_field_coors = dict(f[rem_position].coordinates())
            # ex_field_1_coors = dict(f[file_field_order[1]].coordinates())
            # for orig_coor, ex_1_coor in zip(
            #         original_field_coors.values(), ex_field_1_coors.values()):
            #     # The 'auxiliarycoordinate2' construct differs for both, so
            #     # skip that but otherwise the two fields have the same coors:
            #     if orig_coor.identity == "auxiliarycoordinate2":
            #         continue
            #     self.assertEqual(
            #         orig_coor.nc_get_variable(),
            #         ex_1_coor.nc_get_variable(),
            #     )

            # Check behaviour when append identical fields, as an edge case:
            cf.write(g, tmpfile, fmt=fmt, mode="w")  # 1. overwrite to wipe
            cf.write(g_copy, tmpfile, fmt=fmt, mode="a")  # 2. now append
            f = cf.read(tmpfile)
            self.assertEqual(len(f), 2 * len(g))
            self.assertTrue(
                any(
                    [
                        file_field.equals(g[0], ignore_properties=["remark"])
                        for file_field in f
                    ]
                )
            )
            self.assertEqual(
                f[0].nc_global_attributes(), original_global_attrs
            )

    def test_read_write_netCDF4_compress_shuffle(self):
        f = cf.read(self.filename)[0]
        # TODODASK: reinstate "CFA4" at version > 3.14
        for fmt in ("NETCDF4", "NETCDF4_CLASSIC"):  # , "CFA4"):
            cf.write(f, tmpfile, fmt=fmt, compress=1, shuffle=True)
            g = cf.read(tmpfile)[0]
            self.assertTrue(
                f.equals(g, verbose=2),
                f"Bad read/write with lossless compression: {fmt}",
            )

    def test_write_datatype(self):
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

    def test_read_cdl_string(self):
        """Test the `cdl_string` keyword of the `read` function."""
        # Test CDL in full, header-only and coordinate-only type:
        tempfile_to_option_mapping = {
            tmpfile: None,
            tmpfileh: "-h",
            tmpfilec: "-c",
        }

        for tempf, option in tempfile_to_option_mapping.items():
            # Set up the CDL string to test...
            command_to_run = ["ncdump", self.filename, ">", tempf]
            if option:
                command_to_run.insert(1, option)
            subprocess.run(" ".join(command_to_run), shell=True, check=True)
            with open(tempf, "r") as file:
                cdl_string_1 = file.read()

            # ... and now test it as an individual string input
            f_from_str = cf.read(cdl_string_1, cdl_string=True)
            f_from_file = cf.read(tempf)  # len 1 so only one field to check
            self.assertEqual(len(f_from_str), len(f_from_file))
            self.assertEqual(f_from_str[0], f_from_file[0])

            # ... and test further by inputting it in duplicate as a sequence
            f_from_str = cf.read([cdl_string_1, cdl_string_1], cdl_string=True)
            f_from_file = cf.read(tempf)  # len 1 so only one field to check
            self.assertEqual(len(f_from_str), 2 * len(f_from_file))
            self.assertEqual(f_from_str[0], f_from_file[0])
            self.assertEqual(f_from_str[1], f_from_file[0])

            # Check compatibility with the `fmt` kwarg.
            f0 = cf.read(cdl_string_1, cdl_string=True, fmt="CDL")  # fine
            self.assertEqual(len(f0), len(f_from_file))
            self.assertEqual(f0[0], f_from_file[0])
            # If the 'fmt' and 'cdl_string' values contradict each other,
            # alert the user to this. Note that the default fmt is None but
            # it then gets interpreted as NETCDF, so default fmt is fine and
            # it is tested in f_from_str above where fmt is not set.
            with self.assertRaises(ValueError):
                f0 = cf.read(cdl_string_1, cdl_string=True, fmt="NETCDF")

        # If the user forgets the cdl_string=True argument they will
        # accidentally attempt to create a file with a very long name of
        # the CDL string, which will in most, if not all, cases result in
        # an "OSError: [Errno 36] File name too long" error:
        with self.assertRaises(OSError):
            cf.read(cdl_string_1)

    def test_read_write_string(self):
        f = cf.read(self.string_filename)

        n = int(len(f) / 2)

        for i in range(n):
            j = i + n
            self.assertTrue(
                f[i].data.equals(f[j].data, verbose=1), f"{f[i]!r} {f[j]!r}"
            )
            self.assertTrue(
                f[j].data.equals(f[i].data, verbose=1), f"{f[j]!r} {f[i]!r}"
            )

        # Note: Don't loop round all netCDF formats for better
        #       performance. Just one netCDF3 and one netCDF4 format
        #       is sufficient to test the functionality

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

    def test_write_coordinates(self):
        f = cf.example_field(0)

        cf.write(f, tmpfile, coordinates=True)
        g = cf.read(tmpfile)

        self.assertEqual(len(g), 1)
        self.assertTrue(g[0].equals(f))

    def test_read_write_domain(self):
        f = cf.read(self.filename)[0]
        d = f.domain

        # 1 domain
        cf.write(d, tmpfile)
        e = cf.read(tmpfile)
        self.assertIsInstance(e, cf.FieldList)
        self.assertTrue(len(e), 10)

        e = cf.read(tmpfile, domain=True, verbose=1)
        self.assertEqual(len(e), 1)
        self.assertIsInstance(e, cf.DomainList)
        e = e[0]
        self.assertIsInstance(e, cf.Domain)
        self.assertTrue(e.equals(e.copy(), verbose=3))
        self.assertTrue(d.equals(e, verbose=3))
        self.assertTrue(e.equals(d, verbose=3))

        # 1 field and 1 domain
        cf.write([f, d], tmpfile)
        g = cf.read(tmpfile)
        self.assertTrue(len(g), 1)
        g = g[0]
        self.assertIsInstance(g, cf.Field)
        self.assertTrue(g.equals(f, verbose=3))

        e = cf.read(tmpfile, domain=True, verbose=1)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertIsInstance(e, cf.Domain)

        # 1 field and 2 domains
        cf.write([f, d, d], tmpfile)
        g = cf.read(tmpfile)
        self.assertTrue(len(g), 1)
        g = g[0]
        self.assertIsInstance(g, cf.Field)
        self.assertTrue(g.equals(f, verbose=3))

        e = cf.read(tmpfile, domain=True, verbose=1)
        self.assertEqual(len(e), 2)
        self.assertIsInstance(e[0], cf.Domain)
        self.assertIsInstance(e[1], cf.Domain)
        self.assertTrue(e[0].equals(e[1]))

    def test_read_chunks(self):
        f = cf.example_field(0)
        f.construct("latitude").axis = "Y"
        cf.write(f, tmpfile)

        f = cf.read(tmpfile, chunks={})[0]
        self.assertEqual(f.data.chunks, ((5,), (8,)))

        f = cf.read(tmpfile, chunks=-1)[0]
        self.assertEqual(f.data.chunks, ((5,), (8,)))

        f = cf.read(tmpfile, chunks=None)[0]
        self.assertEqual(f.data.chunks, ((5,), (8,)))

        f = cf.read(tmpfile, chunks={"foo": 2, "bar": 3})[0]
        self.assertEqual(f.data.chunks, ((5,), (8,)))

        with cf.chunksize("200GB"):
            f = cf.read(tmpfile)[0]
            self.assertEqual(f.data.chunks, ((5,), (8,)))

        with cf.chunksize("150B"):
            f = cf.read(tmpfile)[0]
            self.assertEqual(f.data.chunks, ((4, 1), (4, 4)))

        f = cf.read(tmpfile, chunks="150B")[0]
        self.assertEqual(f.data.chunks, ((4, 1), (4, 4)))

        f = cf.read(tmpfile, chunks=3)[0]
        self.assertEqual(f.data.chunks, ((3, 2), (3, 3, 2)))

        y = f.construct("Y")
        self.assertEqual(y.data.chunks, ((3, 2),))

        f = cf.read(tmpfile, chunks={"ncdim%lon": 3})[0]
        self.assertEqual(f.data.chunks, ((5,), (3, 3, 2)))

        f = cf.read(tmpfile, chunks={"longitude": 5, "Y": "150B"})[0]
        self.assertEqual(f.data.chunks, ((3, 2), (5, 3)))

        y = f.construct("Y")
        self.assertEqual(y.data.chunks, ((5,),))

    def test_write_omit_data(self):
        """Test the `omit_data` parameter to `write`."""
        f = cf.example_field(1)
        cf.write(f, tmpfile)

        cf.write(f, tmpfile, omit_data="all")
        g = cf.read(tmpfile)
        self.assertEqual(len(g), 1)
        g = g[0]

        # Check that the data are missing
        self.assertFalse(g.array.count())
        self.assertFalse(g.construct("grid_latitude").array.count())

        # Check that a dump works
        g.dump(display=False)

        cf.write(f, tmpfile, omit_data=("field", "dimension_coordinate"))
        g = cf.read(tmpfile)[0]

        # Check that only the field and dimension coordinate data are
        # missing
        self.assertFalse(g.array.count())
        self.assertFalse(g.construct("grid_latitude").array.count())
        self.assertTrue(g.construct("latitude").array.count())

        cf.write(f, tmpfile, omit_data="field")
        g = cf.read(tmpfile)[0]

        # Check that only the field data are missing
        self.assertFalse(g.array.count())
        self.assertTrue(g.construct("grid_latitude").array.count())

    @unittest.skipUnless(
        False, "URL TEST: UNRELIABLE FLAKEY URL DESTINATION. TODO REPLACE URL"
    )
    def test_read_url(self):
        """Test reading urls."""
        for scheme in ("http", "https"):
            remote = f"{scheme}://psl.noaa.gov/thredds/dodsC/Datasets/cru/crutem5/Monthlies/air.mon.anom.nobs.nc"
            # Check that cf can access it
            f = cf.read(remote)
            self.assertEqual(len(f), 1)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
