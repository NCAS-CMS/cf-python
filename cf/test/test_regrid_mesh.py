import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import numpy as np

import cf

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
atol = 2e-12
rtol = 0

meshloc = {
    "face": esmpy.MeshLoc.ELEMENT,
    "node": esmpy.MeshLoc.NODE,
}


def esmpy_regrid(coord_sys, method, src, dst, **kwargs):
    """Helper function that regrids one dimension of Field data using
    pure esmpy.

    Used to verify `cf.Field.regridc`

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

    src_meshloc = None
    dst_meshloc = None

    domain_topology = src.domain_topology(default=None)
    if domain_topology is not None:
        src_meshloc = meshloc[domain_topology.get_cell()]

    domain_topology = dst.domain_topology(default=None)
    if domain_topology is not None:
        dst_meshloc = meshloc[domain_topology.get_cell()]

    src_field = esmpy.Field(
        esmpy_regrid.srcfield.grid, meshloc=src_meshloc, name="src"
    )
    dst_field = esmpy.Field(
        esmpy_regrid.dstfield.grid, meshloc=dst_meshloc, name="dst"
    )

    fill_value = 1e20
    array = np.squeeze(src.array)
    if array.shape != src_field.data.shape:
        array = array.transpose()

    src_field.data[...] = np.ma.MaskedArray(array, copy=False).filled(
        fill_value
    )
    dst_field.data[...] = fill_value

    esmpy_regrid(src_field, dst_field, zero_region=esmpy.Region.SELECT)

    out = dst_field.data

    return np.ma.MaskedArray(out.copy(), mask=(out == fill_value))


class RegridMeshTest(unittest.TestCase):
    # Get the test source and destination fields
    src_mesh_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_1.nc"
    )
    dst_mesh_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_2.nc"
    )
    grid_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
    )

    src_mesh = cf.read(src_mesh_file)[0]
    dst_mesh = cf.read(dst_mesh_file)[0]
    grid = cf.read(grid_file)[0]

    def setUp(self):
        """Preparations called immediately before each test method."""
        # Disable log messages to silence expected warnings
        cf.log_level("DISABLE")
        # Note: to enable all messages for given methods, lines or calls (those
        # without a 'verbose' option to do the same) e.g. to debug them, wrap
        # them (for methods, start-to-end internally) as follows:
        # cfdm.log_level('DEBUG')
        # < ... test code ... >
        # cfdm.log_level('DISABLE')

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_mesh_to_mesh(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.dst_mesh.copy()
        src = self.src_mesh.copy()

        # Mask some destination grid points
        dst[0, 2:35] = cf.masked

        coord_sys = "spherical"

        for src_masked in (False, True):
            if src_masked:
                src = src.copy()
                src[100:200] = cf.masked

            # Loop over whether or not to use the destination grid
            # masked points
            for use_dst_mask in (False, True):
                for method in all_methods:
                    x = src.regrids(
                        dst, method=method, use_dst_mask=use_dst_mask
                    )
                    a = x.array

                    y = esmpy_regrid(
                        coord_sys,
                        method,
                        src,
                        dst,
                        use_dst_mask=use_dst_mask,
                    )

                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

                    if isinstance(a, np.ma.MaskedArray):
                        self.assertTrue((y.mask == a.mask).all())
                    else:
                        self.assertFalse(y.mask.any())

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_mesh_to_grid(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.grid.copy()
        src = self.src_mesh.copy()

        # Mask some destination grid points
        dst[0, 30, 2:35] = cf.masked

        coord_sys = "spherical"

        for src_masked in (False, True):
            if src_masked:
                src = src.copy()
                src[100:200] = cf.masked

            # Loop over whether or not to use the destination grid
            # masked points
            for use_dst_mask in (False, True):
                for method in all_methods:
                    if method == "nearest_dtos":
                        continue

                    x = src.regrids(
                        dst, method=method, use_dst_mask=use_dst_mask
                    )
                    a = x.array
                    a = a.transpose()

                    y = esmpy_regrid(
                        coord_sys,
                        method,
                        src,
                        dst,
                        use_dst_mask=use_dst_mask,
                    )

                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

                    if isinstance(a, np.ma.MaskedArray):
                        self.assertTrue((y.mask == a.mask).all())
                    else:
                        self.assertFalse(y.mask.any())

        # nearest_dtos doesn't work at the moment
        with self.assertRaises(ValueError):
            src.regrids(dst, method="nearest_dtos")

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_grid_to_mesh(self):
        self.assertFalse(cf.regrid_logging())

        src = self.grid.copy()
        dst = self.src_mesh.copy()

        # Mask some destination grid points
        dst[100:300] = cf.masked

        coord_sys = "spherical"

        for src_masked in (False, True):
            if src_masked:
                src = src.copy()
                src[0, 30:34, 10:80] = cf.masked

            # Loop over whether or not to use the destination grid
            # masked points
            for use_dst_mask in (False, True):
                for method in all_methods:
                    if method == "nearest_dtos":
                        continue

                    x = src.regrids(
                        dst, method=method, use_dst_mask=use_dst_mask
                    )
                    a = x.array
                    a = np.squeeze(a)

                    y = esmpy_regrid(
                        coord_sys,
                        method,
                        src,
                        dst,
                        use_dst_mask=use_dst_mask,
                    )

                    self.assertTrue(np.allclose(y, a, atol=atol, rtol=rtol))

                    if isinstance(a, np.ma.MaskedArray):
                        self.assertTrue((y.mask == a.mask).all())
                    else:
                        self.assertFalse(y.mask.any())

        # nearest_dtos doesn't work at the moment
        with self.assertRaises(ValueError):
            src.regrids(dst, method="nearest_dtos")

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_mesh_cartesian(self):
        self.assertFalse(cf.regrid_logging())

        # Cartesian regridding involving meshes is not currently
        # supported
        src = self.src_mesh
        dst = self.dst_mesh
        with self.assertRaises(ValueError):
            src.regridc(dst, method="linear")

        dst = self.grid
        with self.assertRaises(ValueError):
            src.regridc(dst, method="linear")

        src = self.grid
        dst = self.dst_mesh
        with self.assertRaises(ValueError):
            src.regridc(dst, method="linear")


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
