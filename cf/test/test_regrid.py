import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf

import numpy as np

try:
    import ESMF
except Exception:
    ESMF_imported = False
else:
    ESMF_imported = True

def regrid_ESMF(coord_sys,  method, src, dst,**kwargs):
    """TODO"""    
    ESMF_regrid = cf.regrid.utils.regrid(coord_sys, src, dst, method,
                                         _return_regrid=True,
                                         **kwargs)

    if coord_sys == "spherical":
        src = src.transpose(['X', 'Y', 'T']).squeeze()
        dst = dst.transpose(['X', 'Y', 'T']).squeeze()
    else:
        pass
    print ('src.array=', src.array)

    src_field = ESMF.Field(ESMF_regrid.srcfield.grid, "src")
    dst_field = ESMF.Field(ESMF_regrid.dstfield.grid, "dst")

    print (src_field.grid)

    fill_value = 1e20
    src_field.data[...] = np.ma.MaskedArray(src.array, copy=False).filled(
         fill_value
    )
    print ('src_field.data[...]=',src_field.data[...])
    dst_field.data[...] = fill_value

    ESMF_regrid(src_field, dst_field, zero_region=ESMF.Region.SELECT)
    print ('dst_field.data[...]=',dst_field.data[...])
 
    return np.ma.MaskedArray(
        dst_field.data.copy(),
        mask=(dst_field.data[...] == fill_value),
    )

class RegridTest(unittest.TestCase):
    filename =  os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
    )
    
    @unittest.skipUnless(ESMF_imported, "Requires ESMF package.")
    def test_Field_regrids(self):
        self.assertFalse(cf.regrid_logging())

        dst, src = cf.read(self.filename)

        src.transpose(['X', 'Y', 'T'], inplace=True)
        dst.transpose(['Y', 'T', 'X'], inplace=True)

#        dst[dst.indices(Y=cf.wi(30, 60))] = cf.masked
#        dst[dst.indices(X=cf.wi(30, 60))] = cf.masked
#
#        src[src.indices(Y=cf.wi(45, 75))] = cf.masked
#        src[src.indices(X=cf.wi(45, 75))] = cf.masked
        
#        src[src.indices(T=[0], Y=cf.wi(45, 75))] = cf.masked
#        src[src.indices(T=[0], X=cf.wi(45, 75))] = cf.masked
#
#        src[src.indices(T=[1], Y=cf.wi(15, 45))] = cf.masked
#        src[src.indices(T=[1], X=cf.wi(15, 45))] = cf.masked
#        
#        with cf.atol(1e-12):

##        for method in cf.regrid.utils.ESMF_methods:
##            x = src.regrids(dst, method=method)
##            x.transpose(['X', 'Y', 'T'], inplace=True)
##            for t in (0, 1):
##                y = regrid_ESMF("spherical", method, src.subspace(T=[t]), dst)
##                # print (method,t,(y - a).max())
##                a = x.subspace(T=[t]).squeeze().array
##                self.assertTrue(np.allclose(y, a, atol=1e-12, rtol=0))
            
#        src[src.indices(T=slice(1, 2), Y=cf.wi(15, 45))] = cf.masked
#        src[src.indices(T=slice(1, 2), X=cf.wi(15, 45))] = cf.masked

#        src[slice(2, 6, 1), slice(17, 23, 1), :] = cf.masked
        print (src)
        src[slice(0, 3, 1), slice(0, 3, 1), :] = cf.masked
        
        print ('        MASKED')
        for method in cf.regrid.utils.ESMF_methods:
            print (method)
            x = src.regrids(dst, method=method)
            x.transpose(['X', 'Y', 'T'], inplace=True)
            for t in (0, 1):
                print(t)
                y = regrid_ESMF("spherical", method, src.subspace(T=[t]), dst)
                a = x.subspace(T=[t]).squeeze().array
                                
                print ((y - a).max())
                print (a, y)
                self.assertTrue((y.mask == a.mask).all())
                self.assertTrue(np.allclose(y, a, atol=1e-12, rtol=0))
            
    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regridc(self):
        self.assertFalse(cf.regrid_logging())
        with cf.atol(1e-11):
            for chunksize in self.chunk_sizes:
                self.assertFalse(cf.regrid_logging())
                with cf.chunksize(chunksize):
                    f1 = cf.read(self.filename7)[0]
                    f2 = cf.read(self.filename8)[0]
                    f3 = cf.read(self.filename9)[0]
                    self.assertTrue(
                        f3.equals(f1.regridc(f2, axes="T", method="linear")),
                        f"destination=time series, CHUNKSIZE={chunksize}",
                    )
                    f4 = cf.read(self.filename1)[0]
                    f5 = cf.read(self.filename2)[0]
                    f6 = cf.read(self.filename10)[0]
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                f5, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        f"destination=global Field, CHUNKSIZE={chunksize}",
                    )
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                f5, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        f"destination=global Field, CHUNKSIZE={chunksize}",
                    )
                    dst = {"X": f5.dim("X"), "Y": f5.dim("Y")}
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                dst, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        f"destination=global dict, CHUNKSIZE={chunksize}",
                    )
                    self.assertTrue(
                        f6.equals(
                            f4.regridc(
                                dst, axes=("X", "Y"), method="conservative"
                            )
                        ),
                        f"destination=global dict, CHUNKSIZE={chunksize}",
                    )

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regrids_operator(self):
        self.assertFalse(cf.regrid_logging())

        with cf.atol(1e-12):
            f1 = cf.read(self.filename1)[0]
            f2 = cf.read(self.filename2)[0]
            f3 = cf.read(self.filename3)[0]
            f4 = cf.read(self.filename4)[0]
            f5 = cf.read(self.filename5)[0]

            op = f1.regrids(f2, "conservative", return_operator=True)
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            # Repeat
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            dst = {"longitude": f2.dim("X"), "latitude": f2.dim("Y")}
            op = f1.regrids(
                dst, "conservative", dst_cyclic=True, return_operator=True
            )
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            op = f1.regrids(
                dst,
                method="conservative",
                dst_cyclic=True,
                return_operator=True,
            )
            r = f1.regrids(op)
            self.assertTrue(f3.equals(r))

            # Regrid global to regional rotated pole
            op = f1.regrids(f5, method="linear", return_operator=True)
            r = f1.regrids(op)
            self.assertTrue(f4.equals(r))

        # Raise exception when the source grid does not match that of
        # the regrid operator
        op = f1.regrids(f2, "conservative", return_operator=True)
        with self.assertRaises(ValueError):
            f2.regrids(op)

    @unittest.skipUnless(cf._found_ESMF, "Requires ESMF package.")
    def test_Field_regridc_operator(self):
        self.assertFalse(cf.regrid_logging())

        with cf.atol(1e-12):
            f1 = cf.read(self.filename7)[0]
            f2 = cf.read(self.filename8)[0]
            f3 = cf.read(self.filename9)[0]
            f4 = cf.read(self.filename1)[0]
            f5 = cf.read(self.filename2)[0]
            f6 = cf.read(self.filename10)[0]

            op = f1.regridc(
                f2, axes="T", method="linear", return_operator=True
            )
            self.assertTrue(f3.equals(f1.regridc(op)))

            op = f4.regridc(
                f5,
                axes=("X", "Y"),
                method="conservative",
                return_operator=True,
            )
            self.assertTrue(f6.equals(f4.regridc(op)))

            op = f4.regridc(
                f5,
                axes=("X", "Y"),
                method="conservative",
                return_operator=True,
            )
            self.assertTrue(f6.equals(f4.regridc(op)))

            dst = {
                "X": f5.dimension_coordinate("X"),
                "Y": f5.dimension_coordinate("Y"),
            }
            op = f4.regridc(
                dst,
                axes=("X", "Y"),
                method="conservative",
                return_operator=True,
            )

            self.assertTrue(f6.equals(f4.regridc(op)))
            self.assertTrue(f6.equals(f4.regridc(op)))

        # Raise exception when the source grid does not match that of
        # the regrid operator
        op = f1.regridc(f2, axes="T", method="linear", return_operator=True)
        with self.assertRaises(ValueError):
            f2.regrids(op)

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
