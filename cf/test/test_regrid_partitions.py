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


valid_methods = (
    "linear",
    "conservative",
    "nearest_stod",
    "patch",
)
invalid_methods = ("conservative_2nd", "nearest_dtos")


class RegridPartitionsTest(unittest.TestCase):
    # Get the test source and destination fields
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid.nc"
    )
    src_mesh_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_1.nc"
    )
    dst_mesh_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ugrid_global_2.nc"
    )
    filename_xyz = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "regrid_xyz.nc"
    )
    dst_featureType_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dsg_trajectory.nc"
    )

    dst_src_grid = cf.read(filename)
    dst_grid = dst_src_grid[0]
    src_grid = dst_src_grid[1]
    src_mesh = cf.read(src_mesh_file)[0]
    dst_mesh = cf.read(dst_mesh_file)[0]
    dst_featureType = cf.read(dst_featureType_file)[0]
    src_grid_xyz = cf.read(filename_xyz)[0]

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_partitions_2d_grid_to_grid(self):
        self.assertFalse(cf.regrid_logging())

        src = cf.example_field(0)

        dst = src.copy()
        with cf.bounds_combination_mode("XOR"):
            x = dst.dimension_coordinate("X")
            x += 1

        # Mask some destination grid points
        dst[2:3, 4] = cf.masked

        # Loop round spherical and Cartesian coordinate systems
        for coord_sys, regrid_func, kwargs in zip(
            ("spherical", "Cartesian"),
            ("regrids", "regridc"),
            ({}, {"axes": ["Y", "X"]}),
        ):
            kwargs["return_operator"] = True
            # Loop over whether or not to use the destination grid
            # masked points
            for use_dst_mask in (False, True):
                kwargs["use_dst_mask"] = use_dst_mask
                for method in valid_methods:
                    kwargs["method"] = method
                    r0 = getattr(src, regrid_func)(dst, **kwargs)
                    for n in (2, 3, "maximum"):
                        r1 = getattr(src, regrid_func)(
                            dst, dst_grid_partitions=n, **kwargs
                        )
                        self.assertTrue(r0.equal_weights(r1))
                        self.assertTrue(r0.equal_dst_mask(r1))

        # ------------------------------------------------------------
        # Destination grid defined by 2-d lats and lons
        # ------------------------------------------------------------
        x = dst.coord("X")
        y = dst.coord("Y")
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
        dst = [lon_2d_coord, lat_2d_coord]

        kwargs = {
            "return_operator": True,
            "dst_axes": {"X": 1, "Y": 0},
            "dst_cyclic": True,
        }
        for method in valid_methods:
            kwargs["method"] = method
            r0 = src.regrids(dst, **kwargs)
            for n in (2, 3, "maximum"):
                r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                self.assertTrue(r0.equal_weights(r1))
                self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_3d_grid_to_grid(self):
        self.assertFalse(cf.regrid_logging())

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrid_partitions_cannot_partition(self):
        src = self.src_grid
        dst = self.dst_grid
        for n in (2, "maximum"):
            # Can't partition for particular regrid methods
            for method in invalid_methods:
                with self.assertRaises(ValueError):
                    src.regrids(dst, method=method, dst_grid_partitions=n)

            # Can't partition when return_esmpy_regrid_operator=True
            with self.assertRaises(ValueError):
                src.regrids(
                    dst,
                    method="linear",
                    dst_grid_partitions=n,
                    return_esmpy_regrid_operator=True,
                )

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regridc_partitions_1d_3d_grid_to_grid(self):
        src = self.src_grid.copy()
        dst = self.dst_grid.copy()

        src.transpose(["X", "Y", "T"], inplace=True)
        dst.transpose(["Y", "T", "X"], inplace=True)

        # Mask some destination grid points
        dst[2:25, 0, 2:35] = cf.masked

        for axes in (["T", "Y", "X"], ["Y"]):
            kwargs = {"axes": axes, "return_operator": True}
            for use_dst_mask in (False, True):
                kwargs["use_dst_mask"] = use_dst_mask
                for method in valid_methods:
                    if method in ("patch",) and len(axes) == 1:
                        # 'patch' regridding is not available for 1-d
                        # regridding
                        continue

                    kwargs["method"] = method
                    r0 = src.regridc(dst, **kwargs)
                    for n in (2,):
                        r1 = src.regridc(dst, dst_grid_partitions=n, **kwargs)
                        self.assertTrue(r0.equal_weights(r1))
                        self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_mesh_to_mesh(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.dst_mesh.copy()
        src = self.src_mesh.copy()

        # Mask some destination grid points
        dst[0, 2:35] = cf.masked

        kwargs = {"return_operator": True}
        for use_dst_mask in (False, True):
            kwargs["use_dst_mask"] = use_dst_mask
            for method in valid_methods:
                kwargs["method"] = method
                r0 = src.regrids(dst, **kwargs)
                for n in (2,):
                    r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                    self.assertTrue(r0.equal_weights(r1))
                    self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_mesh_to_grid(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.dst_grid.copy()
        src = self.src_mesh.copy()

        # Mask some destination grid points
        dst[0, 30, 2:35] = cf.masked

        kwargs = {"return_operator": True}
        for src_masked in (False, True):
            for use_dst_mask in (False, True):
                kwargs["use_dst_mask"] = use_dst_mask
                for method in valid_methods:
                    kwargs["method"] = method
                    r0 = src.regrids(dst, **kwargs)
                    for n in (2,):
                        r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                        self.assertTrue(r0.equal_weights(r1))
                        self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_grid_to_mesh(self):
        self.assertFalse(cf.regrid_logging())

        src = self.src_grid.copy()
        dst = self.src_mesh.copy()

        # Mask some destination grid points
        dst[100:300] = cf.masked

        kwargs = {"return_operator": True}
        for src_masked in (False, True):
            for use_dst_mask in (False, True):
                kwargs["use_dst_mask"] = use_dst_mask
                for method in valid_methods:
                    kwargs["method"] = method
                    r0 = src.regrids(dst, **kwargs)
                    for n in (2,):
                        r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                        self.assertTrue(r0.equal_weights(r1))
                        self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_grid_to_featureType_3d(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.dst_featureType.copy()
        src = self.src_grid_xyz.copy()

        # Mask some destination grid points
        dst[20:25] = cf.masked

        kwargs = {"return_operator": True, "z": "air_pressure", "ln_z": True}
        for use_dst_mask in (False, True):
            kwargs["use_dst_mask"] = use_dst_mask
            for method in valid_methods:
                if method == "conservative":
                    # Can't do conservative regridding to a
                    # destination DSG featureType
                    continue

                kwargs["method"] = method
                r0 = src.regrids(dst, **kwargs)
                for n in (2,):
                    r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                    self.assertTrue(r0.equal_weights(r1))
                    self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_grid_to_featureType_2d(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.dst_featureType.copy()
        src = self.src_grid_xyz.copy()
        src = src[0, 0, :, :]

        # Mask some destination grid points
        dst[20:25] = cf.masked

        kwargs = {"return_operator": True}
        for use_dst_mask in (False, True):
            kwargs["use_dst_mask"] = use_dst_mask
            for method in valid_methods:
                if method == "conservative":
                    # Can't do conservative regridding to a
                    # destination DSG featureType
                    continue

                kwargs["method"] = method
                r0 = src.regrids(dst, **kwargs)
                for n in (2,):
                    r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                    self.assertTrue(r0.equal_weights(r1))
                    self.assertTrue(r0.equal_dst_mask(r1))

    @unittest.skipUnless(esmpy_imported, "Requires esmpy/ESMF package.")
    def test_Field_regrids_partitions_mesh_to_featureType_2d(self):
        self.assertFalse(cf.regrid_logging())

        dst = self.dst_featureType.copy()
        src = self.src_mesh.copy()

        # Mask some destination grid points
        dst[20:25] = cf.masked

        kwargs = {"return_operator": True}
        for use_dst_mask in (False, True):
            kwargs["use_dst_mask"] = use_dst_mask
            for method in valid_methods:
                if method == "conservative":
                    # Can't do conservative regridding to a
                    # destination DSG featureType
                    continue

                kwargs["method"] = method
                r0 = src.regrids(dst, **kwargs)
                for n in (2,):
                    r1 = src.regrids(dst, dst_grid_partitions=n, **kwargs)
                    self.assertTrue(r0.equal_weights(r1))
                    self.assertTrue(r0.equal_dst_mask(r1))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
