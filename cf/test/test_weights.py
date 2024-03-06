import datetime
import unittest

import numpy as np

import cf

# A radius greater than 1. Used since weights based on the unit
# sphere and non-spheres are tested separately.
r = 2
radius = cf.Data(r, "m")

# --------------------------------------------------------------------
# Spherical polygon geometry with duplicated first/last node.
#
# The cells have areas pi/2 and pi
# The cells have line lengths 3pi/2 and 5pi/2
# --------------------------------------------------------------------
gps = cf.example_field(6)
gps.del_construct("auxiliarycoordinate3")
gps.del_construct("grid_mapping_name:latitude_longitude")

lon = cf.AuxiliaryCoordinate()
lon.standard_name = "longitude"
bounds = cf.Data(
    [[315, 45, 45, 315, 999, 999, 999], [90, 90, 0, 45, 45, 135, 90]],
    "degrees_east",
    mask_value=999,
).reshape(2, 1, 7)
lon.set_bounds(cf.Bounds(data=bounds))
lon.set_geometry("polygon")

lat = cf.AuxiliaryCoordinate()
lat.standard_name = "latitude"
bounds = cf.Data(
    [[0, 0, 90, 0, 999, 999, 999], [0, 90, 0, 0, -90, 0, 0]],
    "degrees_north",
    mask_value=999,
).reshape(2, 1, 7)
lat.set_bounds(cf.Bounds(data=bounds))
lat.set_geometry("polygon")

gps.del_construct("longitude")
gps.del_construct("latitude")
gps.set_construct(lon, axes="domainaxis0", copy=False)
gps.set_construct(lat, axes="domainaxis0", copy=False)

# --------------------------------------------------------------------
# Plane polygon geometry with interior ring and without duplicated
# first/last node
#
# The cells have areas 3 and 8
# The cells have line lengths 9 and 13
# --------------------------------------------------------------------
gppi = gps.copy()
lon = gppi.auxiliary_coordinate("X")
lat = gppi.auxiliary_coordinate("Y")

lon.override_units("m", inplace=True)
lon.standard_name = "projection_x_coordinate"
bounds = cf.Data(
    [
        [
            [2, 2, 0, 0, 999, 999, 999, 999],
            [0.5, 1.5, 1.5, 0.5, 999, 999, 999, 999],
        ],
        [
            [2, 2, 0, 0, 1, 1, 3, 3],
            [999, 999, 999, 999, 999, 999, 999, 999],
        ],
    ],
    "m",
    mask_value=999,
).reshape(2, 2, 8)
lon.set_bounds(cf.Bounds(data=bounds))
lon.set_interior_ring(cf.InteriorRing(data=[[0, 1], [0, 0]]))

lat.override_units("m", inplace=True)
lat.standard_name = "projection_y_coordinate"
bounds = cf.Data(
    [
        [
            [-1, 1, 1, -1, 999, 999, 999, 999],
            [0.5, 0.5, -0.5, -0.5, 999, 999, 999, 999],
        ],
        [
            [-1, 1, 1, -1, -1, -3, -3, -1],
            [999, 999, 999, 999, 999, 999, 999, 999],
        ],
    ],
    "m",
    mask_value=999,
).reshape(2, 2, 8)
lat.set_bounds(cf.Bounds(data=bounds))
lat.set_interior_ring(cf.InteriorRing(data=[[0, 1], [0, 0]]))


class WeightsTest(unittest.TestCase):
    def test_weights_polygon_area_geometry(self):
        # Spherical polygon geometry weights with duplicated first/last
        # node
        f = gps.copy()
        lon = f.auxiliary_coordinate("X")
        lat = f.auxiliary_coordinate("Y")

        # Surface area of unit sphere
        sphere_area = 4 * np.pi
        correct_weights = np.array([sphere_area / 8, sphere_area / 4])

        # Spherical polygon geometry weights with duplicated first/last
        # node
        w = gps.weights("X", great_circle=True)
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("1"))

        w = gps.weights("area", great_circle=True, measure=True, radius=radius)
        self.assertTrue((w.array == (r**2) * correct_weights).all())
        self.assertEqual(w.Units, cf.Units("m2"))

        # Spherical polygon geometry weights without duplicated
        # first/last node
        bounds = cf.Data(
            [[315, 45, 45, 999, 999, 999], [90, 90, 0, 45, 45, 135]],
            "degrees_east",
            mask_value=999,
        ).reshape(2, 1, 6)
        lon.set_bounds(cf.Bounds(data=bounds))

        bounds = cf.Data(
            [[0, 0, 90, 999, 999, 999], [0, 90, 0, 0, -90, 0]],
            "degrees_north",
            mask_value=999,
        ).reshape(2, 1, 6)
        lat.set_bounds(cf.Bounds(data=bounds))

        w = f.weights("X", great_circle=True)
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("1"))

        w = f.weights("area", great_circle=True, measure=True, radius=radius)
        self.assertTrue((w.array == (r**2) * correct_weights).all())
        self.assertEqual(w.Units, cf.Units("m2"))

        # Plane polygon geometry with no duplicated first/last nodes,
        # and an interior ring
        correct_weights = np.array([3, 8])
        w = gppi.weights("area")
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("1"))

        w = gppi.weights("area", measure=True)
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("m2"))

    def test_weights_polygon_area_ugrid(self):
        f = cf.example_field(8)
        f = f[..., [0, 2]]

        # Surface area of unit sphere
        sphere_area = 4 * np.pi
        correct_weights = np.array([sphere_area / 8, sphere_area / 4])

        # Spherical polygon weights
        lon = f.auxiliary_coordinate("X")
        lon.del_data()
        bounds = cf.Data(
            [[315, 45, 45, 999, 999, 999], [90, 90, 0, 45, 45, 135]],
            "degrees_east",
            mask_value=999,
        ).reshape(2, 6)
        lon.set_bounds(cf.Bounds(data=bounds))

        lat = f.auxiliary_coordinate("Y")
        bounds = cf.Data(
            [[0, 0, 90, 999, 999, 999], [0, 90, 0, 0, -90, 0]],
            "degrees_north",
            mask_value=999,
        ).reshape(2, 6)
        lat.set_bounds(cf.Bounds(data=bounds))

        w = f.weights("X", great_circle=True)
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("1"))

        w = f.weights("area", great_circle=True, measure=True, radius=radius)
        self.assertTrue((w.array == (r**2) * correct_weights).all())
        self.assertEqual(w.Units, cf.Units("m2"))

        # Plane polygon weights
        lon.override_units("m", inplace=True)
        lon.standard_name = "projection_x_coordinate"
        bounds = cf.Data(
            [[2, 2, 0, 0, 999, 999, 999, 999], [2, 2, 0, 0, 1, 1, 3, 3]],
            "m",
            mask_value=999,
        ).reshape(2, 8)
        lon.set_bounds(cf.Bounds(data=bounds))

        lat.override_units("m", inplace=True)
        lat.standard_name = "projection_y_coordinate"
        bounds = cf.Data(
            [
                [-1, 1, 1, -1, 999, 999, 999, 999],
                [-1, 1, 1, -1, -1, -3, -3, -1],
            ],
            "m",
            mask_value=999,
        ).reshape(2, 8)
        lat.set_bounds(cf.Bounds(data=bounds))

        correct_weights = np.array([4, 8])
        w = f.weights("area")
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("1"))

        w = f.weights("area", measure=True)
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("m2"))

    def test_weights_line_length_geometry(self):
        # Spherical line geometry
        gls = gps.copy()
        lon = gls.auxiliary_coordinate("X")
        lat = gls.auxiliary_coordinate("Y")
        lon.set_geometry("line")
        lat.set_geometry("line")

        # Circumference of unit sphere
        cirumference = 2 * np.pi
        correct_weights = np.array(
            [3 * cirumference / 4, 5 * cirumference / 4]
        )

        w = gls.weights("X", great_circle=True)
        self.assertTrue((w.array == correct_weights).all())
        self.assertEqual(w.Units, cf.Units("1"))

        w = gls.weights("X", great_circle=True, measure=True, radius=radius)
        self.assertTrue((w.array == r * correct_weights).all())
        self.assertEqual(w.Units, cf.Units("m"))

        # Plane line geometry with multiple parts
        gppm = gppi.copy()
        lon = gppm.auxiliary_coordinate("X")
        lat = gppm.auxiliary_coordinate("Y")
        lon.set_geometry("line")
        lat.set_geometry("line")
        lon.del_interior_ring()
        lat.del_interior_ring()

        correct_weights = np.array([9, 13])
        w = gppm.weights("X")
        self.assertTrue((w.array == correct_weights).all())

    def test_weights_line_area_ugrid(self):
        f = cf.example_field(9)
        f = f[..., 0:3]
        lon = f.auxiliary_coordinate("X")
        lat = f.auxiliary_coordinate("Y")

        lon.del_data()
        bounds = cf.Data(
            [[315, 45], [45, 45], [45, 315]],
            "degrees_east",
            mask_value=999,
        )
        lon.set_bounds(cf.Bounds(data=bounds))

        lat.del_data()
        bounds = cf.Data(
            [[0, 0], [0, 90], [90, 0]],
            "degrees_north",
            mask_value=999,
        )
        lat.set_bounds(cf.Bounds(data=bounds))

        # Circumference of unit sphere
        cirumference = 2 * np.pi
        correct_weights = np.array([cirumference / 4] * 3)

        # Spherical line weights
        w = f.weights("X", great_circle=True)
        self.assertTrue(
            np.isclose(w, correct_weights, rtol=0, atol=9e-15).all()
        )
        self.assertEqual(w.Units, cf.Units("1"))

        w = f.weights("X", great_circle=True, measure=True, radius=radius)
        self.assertTrue(
            np.isclose(w, r * correct_weights, rtol=0, atol=9e-15).all()
        )
        self.assertEqual(w.Units, cf.Units("m"))

    def test_weights_cell_measures_coordinates(self):
        f = cf.example_field(0)

        areas1 = f.cell_area()
        areas2 = areas1.copy()
        areas2[...] = -1

        c = cf.CellMeasure(source=areas2)
        c.set_measure("area")
        f.set_construct(c)

        w = f.weights(True, measure=True)
        self.assertTrue(w.data.allclose(areas2.data))

        w = f.weights(True, measure=True, cell_measures=False)
        self.assertTrue(w.data.allclose(areas1.data))

        w = f.weights(True, measure=True, coordinates=False)
        self.assertTrue(w.data.allclose(areas2.data))

        with self.assertRaises(ValueError):
            w = f.weights(True, cell_measures=False, coordinates=False)

        w = f.weights("area", measure=True)
        self.assertTrue(w.data.allclose(areas2.data))

        w = f.weights("area", measure=True, cell_measures=False)
        self.assertTrue(w.data.allclose(areas1.data))

        w = f.weights("area", measure=True, coordinates=False)
        self.assertTrue(w.data.allclose(areas2.data))

        with self.assertRaises(ValueError):
            w = f.weights("area", cell_measures=False, coordinates=False)

    def test_weights_exceptions(self):
        f = cf.example_field(0)
        f.coordinate("X").del_bounds()
        f.coordinate("Y").del_bounds()

        with self.assertRaisesRegex(
            ValueError, "Can't create weights: Unable to find cell areas"
        ):
            f.weights("area")


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print("")
    unittest.main(verbosity=2)
