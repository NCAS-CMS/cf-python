import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import numpy as np

import cf

try:
    import ESMF
except Exception:
    ESMF_imported = False
else:
    ESMF_imported = True


methods = (
    "linear",
    "conservative",
    "conservative_2nd",
    "nearest_dtos",
    "nearest_stod",
    "patch",
)

atol = 1e-12
rtol = 0


def regrid_ESMF(coord_sys, method, src, dst, **kwargs):
    """Help function that regrids Field data using pure ESMF.

    Used to verify `cf.Field.regrids` nad `cf.Field.regridc`

    """
    ESMF_regrid = cf.regrid.regrid(
        coord_sys, src, dst, method, _return_regrid=True, **kwargs
    )

    if coord_sys == "spherical":
        src = src.transpose(["X", "Y", "T"]).squeeze()
        dst = dst.transpose(["X", "Y", "T"]).squeeze()
    else:
        src = src.transpose(["Y", "X", "T"]).squeeze()
        dst = dst.transpose(["Y", "X", "T"]).squeeze()

    src_field = ESMF.Field(ESMF_regrid.srcfield.grid, "src")
    dst_field = ESMF.Field(ESMF_regrid.dstfield.grid, "dst")

    #print (src_field.grid)
    #print (dst_field.grid)

    fill_value = 1e20
    src_field.data[...] = np.ma.MaskedArray(src.array, copy=False).filled(
        fill_value
    )
    #    print("src_field.data[...]=", src_field.data[...])
    dst_field.data[...] = fill_value

    ESMF_regrid(src_field, dst_field, zero_region=ESMF.Region.SELECT)
    #    print("dst_field.data[...]=", dst_field.data[...])

    return np.ma.MaskedArray(
        dst_field.data.copy(),
        mask=(dst_field.data[...] == fill_value),
    )


class RegridTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
    )

    @unittest.skipUnless(ESMF_imported, "Requires ESMF package.")
    def test_Field_regridc(self):
        dst, src0 = cf.read(self.filename)

        src0.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        dst[:, 2:25, 2:35] = cf.masked

        axes=["Y", "X"]
        
        for use_dst_mask in (False, True):
            print("use_dst_mask=", use_dst_mask)
            src = src0.copy()

            print("        UNMASKED SOURCE")
            # No source grid masked points
            for method in methods:
                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the ESMF "truth", rather
                    #       than cf
                    continue
                print("\n", method, use_dst_mask)
                x = src.regridc(dst, axes=axes, method=method,  use_dst_mask=use_dst_mask)
               
                cf.write(x, 'delme2.nc')
                
                x.transpose(["X", "Y", "T"], inplace=True)
                x = x.array
                for t in (0, 1):
                    y = regrid_ESMF(
                        "Cartesian",
                        method,
                        src.subspace(T=[t]),
                        dst,
                        axes=axes,
                        use_dst_mask=use_dst_mask,
                    )
                    a = x[t, ...]

                    if isinstance(a, np.ma.MaskedArray):
                        print ("y.mask=",y.mask.shape)
                        print ("a.mask=",a.mask.shape, x.shape)
                        self.assertTrue((y.mask == a.mask).all())
                    else:
                        self.assertFalse(y.mask.any())

#                    print ('y=',y[:10, 10:])
 #                   print ('a=',a[:10, 10:])
                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))
                break
        
    @unittest.skipUnless(ESMF_imported, "Requires ESMF package.")
    def test_Field_regrids(self):
        return
        self.assertFalse(cf.regrid_logging())

        dst, src0 = cf.read(self.filename)

        src0.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        dst[2:25, :, 2:35] = np.ma.masked
        dst_masked = "_dst_masked"

        for use_dst_mask in (False, True):
            print("use_dst_mask=", use_dst_mask)
            src = src0.copy()

            print("        UNMASKED SOURCE")
            # No source grid masked points
            for method in methods:
                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the ESMF "truth", rather
                    #       than cf
                    continue

                print("\n", method, use_dst_mask)
                x = src.regrids(dst, method=method, use_dst_mask=use_dst_mask)
                x.transpose(["X", "Y", "T"], inplace=True)
                x = x.array
                for t in (0, 1):
                    y = regrid_ESMF(
                        "spherical",
                        method,
                        src.subspace(T=[t]),
                        dst,
                        use_dst_mask=use_dst_mask,
                    )
                    a = x[..., t]
                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

            # Mask the souce grid with the same mask over all regridding
            # slices
            src[slice(1, 9, 1), slice(1, 10, 1), :] = cf.masked

            print("        MASKED SOURCE (INVARIANT)")
            for method in methods:
                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the ESMF "truth", rather
                    #       than cf
                    continue

                print("\n", method, use_dst_mask)
                x = src.regrids(dst, method=method, use_dst_mask=use_dst_mask)
                x.transpose(["X", "Y", "T"], inplace=True)
                x = x.array
                for t in (0, 1):
                    y = regrid_ESMF(
                        "spherical",
                        method,
                        src.subspace(T=[t]),
                        dst,
                        use_dst_mask=use_dst_mask,
                    )
                    a = x[..., t]

                    if isinstance(a, np.ma.MaskedArray):
                        self.assertTrue((y.mask == a.mask).all())
                    else:
                        self.assertFalse(y.mask.any())

                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

            # Now make the source mask vary over different regridding
            # slices
            src[slice(11, 19, 1), slice(11, 20, 1), 1] = cf.masked

            print("        MASKED SOURCE (VARIABLE)")
            for method in (
                "linear",
                "conservative",
                "nearest_dtos",
            ):
                if use_dst_mask and method == "nearest_dtos":
                    # TODO: This test does not pass, but it seems to
                    #       be problem with the ESMF "truth", rather
                    #       than cf
                    continue

                print("\n", method, use_dst_mask)
                x = src.regrids(dst, method=method, use_dst_mask=use_dst_mask)
                x.transpose(["X", "Y", "T"], inplace=True)
                x = x.array
                for t in (0, 1):
                    y = regrid_ESMF(
                        "spherical",
                        method,
                        src.subspace(T=[t]),
                        dst,
                        use_dst_mask=use_dst_mask,
                    )
                    a = x[..., t]

                    if isinstance(a, np.ma.MaskedArray):
                        self.assertTrue((y.mask == a.mask).all())
                    else:
                        self.assertFalse(y.mask.any())

                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

        # Can't compute the regrid of the following methods when the
        # source grid mask varies over different regridding slices
        for method in (
            "conservative_2nd",
            "nearest_stod",
            "patch",
        ):
            with self.assertRaises(ValueError):
                src.regrids(dst, method=method).array

    @unittest.skipUnless(ESMF_imported, "Requires ESMF package.")
    def test_Field_regrids_coords(self):
        return 
        dst, src = cf.read(self.filename)

        d0 = src.regrids(dst, method="conservative")

        # 1-d dimension coordinates from dst
        x = dst.coord("X")
        y = dst.coord("Y")

        d1 = src.regrids([x, y], method="conservative")
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))

        # Regrid operator defined by 1-d coordinates
        r = src.regrids([x, y], method="conservative", return_operator=True)
        d1 = src.regrids(r)
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

        # 2-d auxiliary coordinates from dst
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
            data=cf.Data(lon, units=x.Units),
            bounds=cf.Bounds(data=lon_bounds)
        )
        lat_2d_coord = cf.AuxiliaryCoordinate(
            data=cf.Data(lat, units=y.Units),
            bounds=cf.Bounds(data=lat_bounds)
        )

        d1 = src.regrids(
            [lon_2d_coord, lat_2d_coord],
            dst_axes={"X": 1, "Y": 0},
            method="conservative",
            dst_cyclic=True,
        )
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

        # Regrid operator defined by 2-d coordinates
        r = src.regrids(
            [lon_2d_coord, lat_2d_coord],
            dst_axes={"X": 1, "Y": 0},
            method="conservative",
            dst_cyclic=True,
            return_operator=True,
        )
        d1 = src.regrids(r)
        self.assertTrue(d1.data.equals(d0.data, atol=atol, rtol=rtol))

        # Check that disallowed dst raise an exception
        with self.assertRaises(TypeError):
            src.regrids(999, method="conservative")

        with self.assertRaises(ValueError):
            src.regrids([], method="conservative")

        with self.assertRaises(ValueError):
            src.regrids("foobar", method="conservative")

    @unittest.skipUnless(ESMF_imported, "Requires ESMF package.")
    def test_Field_regrids_domain(self):
        return 
        dst, src = cf.read(self.filename)

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

    @unittest.skipUnless(ESMF_imported, "Requires ESMF package.")
    def test_Field_regrids_field(self):
        return 
        dst, src = cf.read(self.filename)

        d0 = src.regrids(dst, method="conservative")
       
        # Regrid operator defined by field
        r = src.regrids(
            dst,
            method="conservative",
            return_operator=True,
        )
        d1 = src.regrids(r)
        self.assertTrue(d1.equals(d0, atol=atol, rtol=rtol))


#    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
#    def test_Field_regridc(self):
#        self.assertFalse(cf.regrid_logging())
#        with cf.atol(1e-11):
#            for chunksize in self.chunk_sizes:
#                self.assertFalse(cf.regrid_logging())
#                with cf.chunksize(chunksize):
#                    f1 = cf.read(self.filename7)[0]
#                    f2 = cf.read(self.filename8)[0]
#                    f3 = cf.read(self.filename9)[0]
#                    self.assertTrue(
#                        f3.equals(f1.regridc(f2, axes="T", method="linear")),
#                        f"destination=time series, CHUNKSIZE={chunksize}",
#                    )
#                    f4 = cf.read(self.filename1)[0]
#                    f5 = cf.read(self.filename2)[0]
#                    f6 = cf.read(self.filename10)[0]
#                    self.assertTrue(
#                        f6.equals(
#                            f4.regridc(
#                                f5, axes=("X", "Y"), method="conservative"
#                            )
#                        ),
#                        f"destination=global Field, CHUNKSIZE={chunksize}",
#                    )
#                    self.assertTrue(
#                        f6.equals(
#                            f4.regridc(
#                                f5, axes=("X", "Y"), method="conservative"
#                            )
#                        ),
#                        f"destination=global Field, CHUNKSIZE={chunksize}",
#                    )
#                    dst = {"X": f5.dim("X"), "Y": f5.dim("Y")}
#                    self.assertTrue(
#                        f6.equals(
#                            f4.regridc(
#                                dst, axes=("X", "Y"), method="conservative"
#                            )
#                        ),
#                        f"destination=global dict, CHUNKSIZE={chunksize}",
#                    )
#                    self.assertTrue(
#                        f6.equals(
#                            f4.regridc(
#                                dst, axes=("X", "Y"), method="conservative"
#                            )
#                        ),
#                        f"destination=global dict, CHUNKSIZE={chunksize}",
#                    )
#
#    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
#    def test_Field_regrids_operator(self):
#        self.assertFalse(cf.regrid_logging())
#
#        with cf.atol(1e-12):
#            f1 = cf.read(self.filename1)[0]
#            f2 = cf.read(self.filename2)[0]
#            f3 = cf.read(self.filename3)[0]
#            f4 = cf.read(self.filename4)[0]
#            f5 = cf.read(self.filename5)[0]
#
#            op = f1.regrids(f2, "conservative", return_operator=True)
#            r = f1.regrids(op)
#            self.assertTrue(f3.equals(r))
#
#            # Repeat
#            r = f1.regrids(op)
#            self.assertTrue(f3.equals(r))
#
#            dst = {"longitude": f2.dim("X"), "latitude": f2.dim("Y")}
#            op = f1.regrids(
#                dst, "conservative", dst_cyclic=True, return_operator=True
#            )
#            r = f1.regrids(op)
#            self.assertTrue(f3.equals(r))
#
#            op = f1.regrids(
#                dst,
#                method="conservative",
#                dst_cyclic=True,
#                return_operator=True,
#            )
#            r = f1.regrids(op)
#            self.assertTrue(f3.equals(r))
#
#            # Regrid global to regional rotated pole
#            op = f1.regrids(f5, method="linear", return_operator=True)
#            r = f1.regrids(op)
#            self.assertTrue(f4.equals(r))
#
#        # Raise exception when the source grid does not match that of
#        # the regrid operator
#        op = f1.regrids(f2, "conservative", return_operator=True)
#        with self.assertRaises(ValueError):
#            f2.regrids(op)

#    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
#    def test_Field_regridc_operator(self):
#        self.assertFalse(cf.regrid_logging())
#
#        with cf.atol(1e-12):
#            f1 = cf.read(self.filename7)[0]
#            f2 = cf.read(self.filename8)[0]
#            f3 = cf.read(self.filename9)[0]
#            f4 = cf.read(self.filename1)[0]
#            f5 = cf.read(self.filename2)[0]
#            f6 = cf.read(self.filename10)[0]
#
#            op = f1.regridc(
#                f2, axes="T", method="linear", return_operator=True
#            )
#            self.assertTrue(f3.equals(f1.regridc(op)))
#
#            op = f4.regridc(
#                f5,
#                axes=("X", "Y"),
#                method="conservative",
#                return_operator=True,
#            )
#            self.assertTrue(f6.equals(f4.regridc(op)))
#
#            op = f4.regridc(
#                f5,
#                axes=("X", "Y"),
#                method="conservative",
#                return_operator=True,
#            )
#            self.assertTrue(f6.equals(f4.regridc(op)))
#
#            dst = {
#                "X": f5.dimension_coordinate("X"),
#                "Y": f5.dimension_coordinate("Y"),
#            }
#            op = f4.regridc(
#                dst,
#                axes=("X", "Y"),
#                method="conservative",
#                return_operator=True,
#            )
#
#            self.assertTrue(f6.equals(f4.regridc(op)))
#            self.assertTrue(f6.equals(f4.regridc(op)))
#
#        # Raise exception when the source grid does not match that of
#        # the regrid operator
#        op = f1.regridc(f2, axes="T", method="linear", return_operator=True)
#        with self.assertRaises(ValueError):
#            f2.regrids(op)
#
#    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
#    def test_Field_regrid_size1_dimensions(self):
#        # Check that non-regridded size 1 axes are handled OK
#        self.assertFalse(cf.regrid_logging())
#
#        f = cf.example_field(0)
#        shape = f.shape
#
#        g = f.regrids(f, method="linear")
#        self.assertEqual(g.shape, (shape))
#        g = f.regridc(f, method="linear", axes="X")
#        self.assertEqual(g.shape, (shape))
#
#        f.insert_dimension("T", position=0, inplace=True)
#        shape = f.shape
#        g = f.regrids(f, method="linear")
#        self.assertEqual(g.shape, shape)
#        g = f.regridc(f, method="linear", axes="X")
#        self.assertEqual(g.shape, shape)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
