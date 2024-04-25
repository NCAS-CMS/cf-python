import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import numpy as np

import cf

n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_regrid.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(tmpfile,) = tmpfiles


def _remove_tmpfiles():
    """Try to remove defined temporary files by deleting their paths."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


# ESMF renamed its Python module to `esmpy` at ESMF version 8.4.0. Allow
# either for now for backwards compatibility.
esmpy_imported = False
try:
    import esmpy

    esmpy_imported = True
except ImportError:
    try:
        # Take the new name to use in preference to the old one.
        import ESMF as esmpy

        esmpy_imported = True
    except ImportError:
        pass

all_methods = (
    "linear",
    "conservative",
    "conservative_2nd",
    "nearest_dtos",
    "nearest_stod",
    "patch",
)


# Set numerical comparison tolerances
atol = 1e-12
rtol = 0


def esmpy_regrid_1d(method, src, dst, **kwargs):
    """Helper function that regrids one dimension of Field data using
    pure esmpy.

    Used to verify `cf.Field.regridc`

    :Returns:

        Regridded numpy masked array.

    """
    esmpy_regrid = cf.regrid.regrid(
        "Cartesian",
        src,
        dst,
        method,
        return_esmpy_regrid_operator=True,
        **kwargs
    )

    src = src.transpose(["Y", "X", "T"])

    ndbounds = src.shape[1:3]

    src_field = esmpy.Field(
        esmpy_regrid.srcfield.grid, "src", ndbounds=ndbounds
    )
    dst_field = esmpy.Field(
        esmpy_regrid.dstfield.grid, "dst", ndbounds=ndbounds
    )

    fill_value = 1e20
    array = src.array
    array = np.expand_dims(array, 1)

    src_field.data[...] = np.ma.MaskedArray(array, copy=False).filled(
        fill_value
    )
    dst_field.data[...] = fill_value

    esmpy_regrid(src_field, dst_field, zero_region=esmpy.Region.SELECT)

    out = dst_field.data[:, 0, :, :]

    return np.ma.MaskedArray(out.copy(), mask=(out == fill_value))


def esmpy_regrid_Nd(coord_sys, method, src, dst, **kwargs):
    """Helper function that regrids Field data using pure esmpy.

    Used to verify `cf.Field.regrids` and `cf.Field.regridc`

    :Returns:

        Regridded numpy masked array.

    """
    esmpy_regrid = cf.regrid.regrid(
        coord_sys,
        src,
        dst,
        method,
        return_esmpy_regrid_operator=True,
        **kwargs
    )

    src = src.transpose(["X", "Y", "T"]).squeeze()

    src_field = esmpy.Field(esmpy_regrid.srcfield.grid, "src")
    dst_field = esmpy.Field(esmpy_regrid.dstfield.grid, "dst")

    fill_value = 1e20
    src_field.data[...] = np.ma.MaskedArray(src.array, copy=False).filled(
        fill_value
    )

    dst_field.data[...] = fill_value

    esmpy_regrid(src_field, dst_field, zero_region=esmpy.Region.SELECT)

    return np.ma.MaskedArray(
        dst_field.data.copy(),
        mask=(dst_field.data[...] == fill_value),
    )


class RegridTest(unittest.TestCase):
    # Get the test source and destination fields
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
    )
    dst_src = cf.read(filename)
    dst = dst_src[0]
    src = dst_src[1]

    filename_xyz = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_xyz.nc"
    )

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_2d_field(self):
        """2-d regridding with Field destination grid."""
        self.assertFalse(cf.regrid_logging())

        dst = self.dst.copy()
        src0 = self.src.copy()

        src0.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        # Mask some destination grid points
        dst[2:25, 0, 2:35] = cf.masked

        # Loop round spherical and Cartesian coordinate systems
        for coord_sys, regrid_func, kwargs in zip(
            ("spherical", "Cartesian"),
            ("regrids", "regridc"),
            ({}, {"axes": ["Y", "X"]}),
        ):
            # Loop over whether or not to use the destination grid
            # masked points
            for use_dst_mask in (False, True):
                src = src0.copy()

                # ----------------------------------------------------
                # No source grid masked elements
                # ----------------------------------------------------
                for method in all_methods:
                    if use_dst_mask and method == "nearest_dtos":
                        # TODO: This test does not pass, but it seems to
                        #       be problem with the esmpy "truth", rather
                        #       than cf
                        continue

                    x = getattr(src, regrid_func)(
                        dst, method=method, use_dst_mask=use_dst_mask, **kwargs
                    )
                    x.transpose(["X", "Y", "T"], inplace=True)
                    x = x.array
                    for t in (0, 1):
                        y = esmpy_regrid_Nd(
                            coord_sys,
                            method,
                            src.subspace(T=[t]),
                            dst,
                            use_dst_mask=use_dst_mask,
                            **kwargs
                        )
                        a = x[..., t]

                        if isinstance(a, np.ma.MaskedArray):
                            self.assertTrue((y.mask == a.mask).all())
                        else:
                            self.assertFalse(y.mask.any())

                        self.assertTrue(
                            np.allclose(y, a, atol=atol, rtol=rtol)
                        )

                # ----------------------------------------------------
                # Mask the souce grid with the same mask over all 2-d
                # regridding slices
                # ----------------------------------------------------
                src[slice(2, 10, 1), slice(1, 10, 1), :] = cf.masked

                for method in all_methods:
                    if use_dst_mask and method == "nearest_dtos":
                        # TODO: This test does not pass, but it seems to
                        #       be problem with the esmpy "truth", rather
                        #       than cf
                        continue

                    x = getattr(src, regrid_func)(
                        dst, method=method, use_dst_mask=use_dst_mask, **kwargs
                    )
                    x.transpose(["X", "Y", "T"], inplace=True)
                    x = x.array
                    for t in (0, 1):
                        y = esmpy_regrid_Nd(
                            coord_sys,
                            method,
                            src.subspace(T=[t]),
                            dst,
                            use_dst_mask=use_dst_mask,
                            **kwargs
                        )
                        a = x[..., t]

                        if isinstance(a, np.ma.MaskedArray):
                            self.assertTrue((y.mask == a.mask).all())

                        else:
                            self.assertFalse(y.mask.any())

                        self.assertTrue(
                            np.allclose(y, a, atol=atol, rtol=rtol)
                        )

                # ----------------------------------------------------
                # Now make the source mask vary over different 2-d
                # regridding slices
                # ----------------------------------------------------
                src[slice(11, 19, 1), slice(11, 20, 1), 1] = cf.masked

                for method in (
                    "linear",
                    "conservative",
                    "nearest_dtos",
                ):
                    if use_dst_mask and method == "nearest_dtos":
                        # TODO: This test does not pass, but it seems to
                        #       be problem with the esmpy "truth", rather
                        #       than cf
                        continue

                    x = getattr(src, regrid_func)(
                        dst, method=method, use_dst_mask=use_dst_mask, **kwargs
                    )
                    x.transpose(["X", "Y", "T"], inplace=True)
                    x = x.array
                    for t in (0, 1):
                        y = esmpy_regrid_Nd(
                            coord_sys,
                            method,
                            src.subspace(T=[t]),
                            dst,
                            use_dst_mask=use_dst_mask,
                            **kwargs
                        )
                        a = x[..., t]

                        if isinstance(a, np.ma.MaskedArray):
                            self.assertTrue((y.mask == a.mask).all())
                        else:
                            self.assertFalse(y.mask.any())

                        self.assertTrue(
                            np.allclose(y, a, atol=atol, rtol=rtol)
                        )

        # Can't compute the 2-d regrid of the following methods when
        # the source grid mask varies over different regridding slices
        # (which it does coming out of the previous for loop)
        for method in (
            "conservative_2nd",
            "nearest_stod",
            "patch",
        ):
            with self.assertRaises(ValueError):
                src.regrids(dst, method=method).array

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_coords(self):
        """Spherical regridding with coords destination grid."""
        dst = self.dst.copy()
        src = self.src.copy()

        src.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        # Truth = destination grid defined by a field
        d0 = src.regrids(dst, method="conservative")

        # Sequence of 1-d dimension coordinates
        x = dst.coord("X")
        y = dst.coord("Y")

        d1 = src.regrids([x, y], method="conservative")
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

        # Regrid operator defined by 1-d dimension coordinates
        r = src.regrids([x, y], method="conservative", return_operator=True)
        d1 = src.regrids(r)
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

        # Sequence of 2-d auxiliary coordinates
        x_bounds = x.bounds.array
        y_bounds = y.bounds.array

        lat = np.empty((y.size, x.size))
        lat[...] = y.array.reshape(y.size, 1)
        lon = np.empty((y.size, x.size))
        lon[...] = x.array

        lon_bounds = np.empty(lon.shape + (4,))
        lon_bounds[..., [0, 3]] = x_bounds[:, 0].reshape(1, x.size, 1)
        lon_bounds[..., [1, 2]] = x_bounds[:, 1].reshape(1, x.size, 1)

        lat_bounds = np.empty(lat.shape + (4,))
        lat_bounds[..., [0, 1]] = y_bounds[:, 0].reshape(y.size, 1, 1)
        lat_bounds[..., [2, 3]] = y_bounds[:, 1].reshape(y.size, 1, 1)

        lon_2d_coord = cf.AuxiliaryCoordinate(
            data=cf.Data(lon, units=x.Units), bounds=cf.Bounds(data=lon_bounds)
        )
        lat_2d_coord = cf.AuxiliaryCoordinate(
            data=cf.Data(lat, units=y.Units), bounds=cf.Bounds(data=lat_bounds)
        )

        d1 = src.regrids(
            [lon_2d_coord, lat_2d_coord],
            dst_axes={"X": 1, "Y": 0},
            method="conservative",
            dst_cyclic=True,
        )
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

        # Regrid operator defined by 2-d auxiliary coordinates
        r = src.regrids(
            [lon_2d_coord, lat_2d_coord],
            dst_axes={"X": 1, "Y": 0},
            method="conservative",
            dst_cyclic=True,
            return_operator=True,
        )

        d1 = src.regrids(r)
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regridc_2d_coords(self):
        """2-d Cartesian regridding with coords destination grid."""
        dst = self.dst.copy()
        src = self.src.copy()

        src.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        axes = ["Y", "X"]

        # 1-d dimension coordinates
        x = dst.coord("X")
        y = dst.coord("Y")

        # Truth = destination grid defined by a field
        d0 = src.regridc(dst, method="conservative", axes=axes)

        # Sequence of 1-d dimension coordinates
        d1 = src.regridc([y, x], method="conservative", axes=axes)
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

        # Regrid operator defined by 1-d dimension coordinates
        r = src.regridc(
            [y, x], method="conservative", axes=axes, return_operator=True
        )
        d1 = src.regridc(r)
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

    def test_Field_regrids_bad_dst(self):
        """Disallowed destination grid types raise an exception."""
        with self.assertRaises(TypeError):
            self.src.regrids(999, method="conservative")

        with self.assertRaises(ValueError):
            self.src.regrids([], method="conservative")

        with self.assertRaises(ValueError):
            self.src.regrids("foobar", method="conservative")

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_domain(self):
        """Spherical regridding with Domain destination grid."""
        dst = self.dst
        src = self.src

        # Truth = destination grid defined by a field
        d0 = src.regrids(dst, method="conservative")

        d1 = src.regrids(dst.domain, method="conservative")
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

        # Regrid operator defined by domain
        r = src.regrids(
            dst.domain,
            method="conservative",
            return_operator=True,
        )

        d1 = src.regrids(r)
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regridc_domain(self):
        """Spherical regridding with Domain destination grid."""
        dst = self.dst
        src = self.src

        axes = ["Y", "X"]

        # Truth = destination grid defined by a field
        d0 = src.regridc(dst, method="conservative", axes=axes)

        d1 = src.regridc(dst.domain, method="conservative", axes=axes)
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

        # Regrid operator defined by domain
        r = src.regridc(
            dst.domain,
            method="conservative",
            axes=axes,
            return_operator=True,
        )
        d1 = src.regridc(r)
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_field_operator(self):
        """Spherical regridding with operator destination grid."""
        dst = self.dst
        src = self.src

        d0 = src.regrids(dst, method="conservative")

        # Regrid operator defined by field
        r = src.regrids(
            dst,
            method="conservative",
            return_operator=True,
        )

        d1 = src.regrids(r)
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

        # Check coordinates
        d1 = src.regrids(r, check_coordinates=True)

        # Check regridded domain
        for coord in ("X", "Y"):
            # The coords for regridded axes should be the same as the
            # destination grid
            self.assertTrue(d1.coord(coord).equals(d0.coord(coord)))

        for coord in ("T", "Z"):
            # The coords for non-regridded axes should be the same as
            # the source grid
            self.assertTrue(d1.coord(coord).equals(src.coord(coord)))

        # Regrid operator does not match source grid
        with self.assertRaises(ValueError):
            dst.regrids(r)

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_non_coordinates(self):
        """Check setting of non-coordinate metadata."""
        dst = cf.example_field(1)
        src = self.src

        d0 = src.regrids(dst, method="linear")

        # Coordinate references
        self.assertTrue(len(d0.coordinate_references(todict=True)), 1)

        ref = "grid_mapping_name:rotated_latitude_longitude"
        self.assertTrue(
            d0.coordinate_reference(ref).equivalent(
                dst.coordinate_reference(ref)
            )
        )

        # Now swap source and destination fields
        src, dst = dst.copy(), src.copy()
        d1 = src.regrids(dst, method="linear")

        # Coordinate references
        self.assertTrue(len(d1.coordinate_references(todict=True)), 1)

        ref = d1.coordinate_reference(
            "standard_name:atmosphere_hybrid_height_coordinate", default=None
        )
        self.assertIsNotNone(ref)

        # Domain ancillaries
        self.assertEqual(
            len(ref.coordinate_conversion.domain_ancillaries()), 3
        )

        orog = d1.domain_ancillary("surface_altitude", default=None)
        self.assertIsNotNone(orog)
        self.assertEqual(
            orog.shape, (dst.domain_axis("Y").size, dst.domain_axis("X").size)
        )

        # Field ancillaries
        self.assertFalse(d1.field_ancillaries())

        # Cell measures
        self.assertFalse(d1.cell_measures())

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regridc_3d_field(self):
        """3-d Cartesian regridding with Field destination grid."""
        methods = list(all_methods)
        methods.remove("conservative_2nd")

        dst = self.dst.copy()
        src0 = self.src.copy()

        src0.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        # Mask some destination grid points
        dst[2:25, 0, 2:35] = cf.masked

        axes = ["T", "Y", "X"]

        # Loop over whether or not to use the destination grid masked
        # points
        for use_dst_mask in (False, True):
            src = src0.copy()

            # ----------------------------------------------------
            # No source grid masked elements
            # ----------------------------------------------------
            for method in methods:
                # if method in('conservative_2nd', 'patch'):
                #     # These methods aren't meant to work for (what is
                #     # effectively) 1-d regridding
                #     continue

                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the esmpy "truth", rather
                    #       than cf
                    continue

                x = src.regridc(
                    dst, method=method, use_dst_mask=use_dst_mask, axes=axes
                )
                x.transpose(["X", "Y", "T"], inplace=True)
                a = x.array

                y = esmpy_regrid_Nd(
                    "Cartesian",
                    method,
                    src,
                    dst,
                    use_dst_mask=use_dst_mask,
                    axes=axes,
                )

                if isinstance(a, np.ma.MaskedArray):
                    self.assertTrue((y.mask == a.mask).all())
                else:
                    self.assertFalse(y.mask.any())

                self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

            # --------------------------------------------------------
            # Mask the souce grid
            # --------------------------------------------------------
            src[slice(2, 10, 1), slice(1, 10, 1), 0] = cf.masked

            for method in methods:
                # if method in('conservative_2nd', 'patch'):
                #    # These methods aren't meant to work for (what is
                #    # effectively) 1-d regridding
                #    continue

                if method in ("nearest_dtos"):
                    continue

                x = src.regridc(
                    dst, method=method, use_dst_mask=use_dst_mask, axes=axes
                )
                x.transpose(["X", "Y", "T"], inplace=True)
                a = x.array

                y = esmpy_regrid_Nd(
                    "Cartesian",
                    method,
                    src,
                    dst,
                    use_dst_mask=use_dst_mask,
                    axes=axes,
                )

                if isinstance(a, np.ma.MaskedArray):
                    self.assertTrue((y.mask == a.mask).all())
                else:
                    self.assertFalse(y.mask.any())

                self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

        # These methods aren't meant to work for 3-d regridding
        for method in ("conservative_2nd",):
            with self.assertRaises(ValueError):
                src.regridc(dst, method=method, axes=axes)

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regridc_1d_field(self):
        """1-d Cartesian regridding with Field destination grid."""
        methods = list(all_methods)
        methods.remove("conservative_2nd")
        methods.remove("patch")

        dst = self.dst.copy()
        src0 = self.src.copy()

        src0.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        # Mask some destination grid points
        dst[2:25, 0, 2:35] = cf.masked

        axes = ["Y"]

        # Loop over whether or not to use the destination grid masked
        # points
        for use_dst_mask in (False, True):
            src = src0.copy()

            # ----------------------------------------------------
            # No source grid masked elements
            # ----------------------------------------------------
            for method in methods:
                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the esmpy "truth", rather
                    #       than cf
                    continue

                x = src.regridc(
                    dst, method=method, use_dst_mask=use_dst_mask, axes=axes
                )
                x.transpose(["Y", "X", "T"], inplace=True)
                a = x.array

                y = esmpy_regrid_1d(
                    method, src, dst, use_dst_mask=use_dst_mask, axes=axes
                )

                if isinstance(a, np.ma.MaskedArray):
                    self.assertTrue((y.mask == a.mask).all())
                else:
                    self.assertFalse(y.mask.any())

                self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

            # --------------------------------------------------------
            # Mask the souce grid
            # --------------------------------------------------------
            src[:, slice(1, 10, 1), :] = cf.masked

            for method in methods:
                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the esmpy "truth", rather
                    #       than cf
                    continue

                x = src.regridc(
                    dst, method=method, use_dst_mask=use_dst_mask, axes=axes
                )
                x.transpose(["Y", "X", "T"], inplace=True)
                a = x.array

                y = esmpy_regrid_1d(
                    method,
                    src,
                    dst,
                    use_dst_mask=use_dst_mask,
                    axes=axes,
                )

                if isinstance(a, np.ma.MaskedArray):
                    self.assertTrue((y.mask == a.mask).all())
                else:
                    self.assertFalse(y.mask.any())

                self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

        # These methods aren't meant to work for (what is effectively)
        # 1-d regridding
        for method in ("conservative_2nd", "patch"):
            with self.assertRaises(ValueError):
                src.regridc(dst, method=method, axes=axes)

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regridc_1d_coordinates_z(self):
        """1-d Z Cartesian regridding with coordinates destination grid."""
        src = cf.read(self.filename_xyz)[0]
        dst = cf.DimensionCoordinate(
            data=cf.Data([800, 705, 632, 510, 320.0], "hPa")
        )
        d = src.regridc([dst], method="linear", axes="Z", z="Z", ln_z=True)
        z = d.dimension_coordinate("Z")
        self.assertTrue(z.data.equals(dst.data))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_chunks(self):
        """Regridding of chunked axes"""
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
        )
        dst, src = cf.read(filename, chunks={"latitude": 20, "longitude": 30})
        self.assertEqual(src.data.numblocks, (1, 2, 2))
        self.assertEqual(dst.data.numblocks, (1, 4, 4))

        d0 = src.regrids(dst, method="linear")
        self.assertEqual(d0.data.numblocks, (1, 1, 1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_weights_file(self):
        """Regridding creation/use of weights file"""
        dst = self.dst
        src = self.src

        try:
            os.remove(tmpfile)
        except OSError:
            pass

        r = src.regrids(
            dst, method="linear", return_operator=True, weights_file=tmpfile
        )
        self.assertTrue(os.path.isfile(tmpfile))
        self.assertIsNone(r.weights_file)

        r = src.regrids(
            dst, method="linear", return_operator=True, weights_file=tmpfile
        )
        self.assertEqual(r.weights_file, tmpfile)

        # Can't provide weights_file when dst is a RegridOperator
        with self.assertRaises(ValueError):
            self.assertEqual(
                src.regrids(r, method="linear", weights_file=tmpfile)
            )

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_return_esmpy_regrid_operator(self):
        """esmpy regrid operator returns esmpy.Regrid in regrids and regridc"""
        dst = self.dst
        src = self.src

        opers = src.regrids(
            dst, method="conservative", return_esmpy_regrid_operator=True
        )
        operc = src.regridc(
            dst,
            axes=["Y", "X"],
            method="conservative",
            return_esmpy_regrid_operator=True,
        )

        self.assertIsInstance(opers, esmpy.api.regrid.Regrid)
        self.assertIsInstance(operc, esmpy.api.regrid.Regrid)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
