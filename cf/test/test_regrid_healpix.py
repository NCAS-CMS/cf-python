import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import numpy as np

import cf

esmpy_imported = True
try:
    import esmpy  # noqa: F401
except ImportError:
    esmpy_imported = False


all_methods = (
    "linear",
    "conservative",
    "conservative_2nd",
    "nearest_dtos",
    "nearest_stod",
    "patch",
)


# Set numerical comparison tolerances
atol = 0
rtol = 0


class RegridMeshTest(unittest.TestCase):
    # Get the test source and destination fields
    src_mesh_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_1.nc"
    )
    dst_mesh_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_2.nc"
    )
    src_mesh = cf.read(src_mesh_file)[0]
    dst_mesh = cf.read(dst_mesh_file)[0]

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.log_level("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        # cfdm.log_level('DEBUG')
        # < ... test code ... >
        # cfdm.log_level('DISABLE')

    @unittest.skipUnless(esmpy_imported, "Requires esmpy package.")
    def test_Field_regrid_mesh_to_healpix(self):
        # Check that UGRID -> healpix is the same as UGRID -> UGRUD
        self.assertFalse(cf.regrid_logging())

        dst = cf.Domain.create_healpix(3)  # 768 cells
        dst_ugrid = dst.healpix_to_ugrid()
        src = self.src_mesh.copy()

        for src_masked in (False, True):
            if src_masked:
                src = src.copy()
                src[100:200] = cf.masked

            # Loop over whether or not to use the destination grid
            # masked points
            for method in all_methods:
                x = src.regrids(dst, method=method)
                y = src.regrids(dst_ugrid, method=method)
                a = x.array
                b = y.array
                self.assertTrue(np.allclose(b, a, atol=atol, rtol=rtol))

                if np.ma.isMA(a):
                    self.assertTrue((b.mask == a.mask).all())

                # Check that the result is a HEALPix grid
                self.assertTrue(cf.healpix.healpix_info(x))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy package.")
    def test_Field_regrid_healpix_to_mesh(self):
        # Check that healpix -> UGRID is the same as UGRID -> UGRUD
        self.assertFalse(cf.regrid_logging())

        src = cf.Field(source=cf.Domain.create_healpix(3))  # 768 cells
        src.set_data(np.arange(768))

        src_ugrid = src.healpix_to_ugrid()

        dst = self.dst_mesh.copy()

        for src_masked in (False, True):
            if src_masked:
                src[100:200] = cf.masked
                src_ugrid[100:200] = cf.masked

            # Loop over whether or not to use the destination grid
            # masked points
            for method in all_methods:
                x = src.regrids(dst, method=method)
                y = src_ugrid.regrids(dst, method=method)
                a = x.array
                b = y.array
                self.assertTrue(np.allclose(b, a, atol=atol, rtol=rtol))

                if np.ma.isMA(a):
                    self.assertTrue((b.mask == a.mask).all())


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
