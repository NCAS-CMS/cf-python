import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import netCDF4

import cf

n_tmpfiles = 8
tmpfiles = [
    tempfile.mkstemp("_test_groups.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
(
    ungrouped_file1,
    ungrouped_file2,
    ungrouped_file3,
    ungrouped_file4,
    grouped_file1,
    grouped_file2,
    grouped_file3,
    grouped_file4,
) = tmpfiles


def _remove_tmpfiles():
    """Remove temporary files created during tests."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class GroupsTest(unittest.TestCase):
    def setUp(self):
        # Disable log messages to silence expected warnings
        cf.LOG_LEVEL("DISABLE")
        # Note: to enable all messages for given methods, lines or
        # calls (those without a 'verbose' option to do the same)
        # e.g. to debug them, wrap them (for methods, start-to-end
        # internally) as follows:
        #
        # cf.LOG_LEVEL('DEBUG')
        # < ... test code ... >
        # cf.LOG_LEVEL('DISABLE')

    def test_groups(self):
        f = cf.example_field(1)

        ungrouped_file = ungrouped_file1
        grouped_file = grouped_file1

        # Add a second grid mapping
        datum = cf.Datum(parameters={"earth_radius": 7000000})
        conversion = cf.CoordinateConversion(
            parameters={"grid_mapping_name": "latitude_longitude"}
        )

        grid = cf.CoordinateReference(
            coordinate_conversion=conversion,
            datum=datum,
            coordinates=["auxiliarycoordinate0", "auxiliarycoordinate1"],
        )

        f.set_construct(grid)

        grid0 = f.construct("grid_mapping_name:rotated_latitude_longitude")
        grid0.del_coordinate("auxiliarycoordinate0")
        grid0.del_coordinate("auxiliarycoordinate1")

        cf.write(f, ungrouped_file)
        g = cf.read(ungrouped_file, verbose=1)
        self.assertEqual(len(g), 1)
        g = g[0]
        self.assertTrue(f.equals(g, verbose=2))

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(["forecast", "model"])
        cf.write(g, grouped_file)

        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.nc_get_variable(),
            nc.groups["forecast"].groups["model"].variables,
        )
        nc.close()

        h = cf.read(grouped_file, verbose=1)
        self.assertEqual(len(h), 1)
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move constructs one by one to the /forecast group. The order
        # in which we do this matters!
        # ------------------------------------------------------------
        for name in (
            "longitude",  # Auxiliary coordinate
            "latitude",  # Auxiliary coordinate
            "long_name=Grid latitude name",  # Auxiliary coordinate
            "measure:area",  # Cell measure
            "surface_altitude",  # Domain ancillary
            "air_temperature standard_error",  # Field ancillary
            "grid_mapping_name:rotated_latitude_longitude",
            "time",  # Dimension coordinate
            "grid_latitude",  # Dimension coordinate
        ):
            g.construct(name).nc_set_variable_groups(["forecast"])
            cf.write(g, grouped_file, verbose=1)

            # Check that the variable is in the right group
            nc = netCDF4.Dataset(grouped_file, "r")
            self.assertIn(
                f.construct(name).nc_get_variable(),
                nc.groups["forecast"].variables,
            )
            nc.close()

            # Check that the field construct hasn't changed
            h = cf.read(grouped_file, verbose=1)
            self.assertEqual(len(h), 1, repr(h))
            self.assertTrue(f.equals(h[0], verbose=2), name)

        # ------------------------------------------------------------
        # Move bounds to the /forecast group
        # ------------------------------------------------------------
        name = "grid_latitude"
        g.construct(name).bounds.nc_set_variable_groups(["forecast"])
        cf.write(g, grouped_file)

        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.construct(name).bounds.nc_get_variable(),
            nc.groups["forecast"].variables,
        )
        nc.close()

        h = cf.read(grouped_file, verbose="WARNING")
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

    def test_groups_geometry(self):
        f = cf.example_field(6)

        ungrouped_file = ungrouped_file2
        grouped_file = grouped_file2

        cf.write(f, ungrouped_file)
        g = cf.read(ungrouped_file, verbose=1)
        self.assertEqual(len(g), 1)
        g = g[0]

        self.assertTrue(f.equals(g, verbose=3))

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(["forecast", "model"])
        cf.write(g, grouped_file)

        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.nc_get_variable(),
            nc.groups["forecast"].groups["model"].variables,
        )
        nc.close()

        h = cf.read(grouped_file)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=3))

        # ------------------------------------------------------------
        # Move the geometry container to the /forecast group
        # ------------------------------------------------------------
        g.nc_set_geometry_variable_groups(["forecast"])
        cf.write(g, grouped_file)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.nc_get_geometry_variable(), nc.groups["forecast"].variables
        )
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(grouped_file)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move a node coordinate variable to the /forecast group
        # ------------------------------------------------------------
        g.construct("longitude").bounds.nc_set_variable_groups(["forecast"])
        cf.write(g, grouped_file)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.construct("longitude").bounds.nc_get_variable(),
            nc.groups["forecast"].variables,
        )
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(grouped_file)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move a node count variable to the /forecast group
        # ------------------------------------------------------------
        ncvar = g.construct("longitude").get_node_count().nc_get_variable()
        g.nc_set_component_variable_groups("node_count", ["forecast"])

        cf.write(g, grouped_file)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(ncvar, nc.groups["forecast"].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(grouped_file, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move a part node count variable to the /forecast group
        # ------------------------------------------------------------
        ncvar = (
            g.construct("longitude").get_part_node_count().nc_get_variable()
        )
        g.nc_set_component_variable_groups("part_node_count", ["forecast"])

        cf.write(g, grouped_file)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(ncvar, nc.groups["forecast"].variables)
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(grouped_file)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

        # ------------------------------------------------------------
        # Move interior ring variable to the /forecast group
        # ------------------------------------------------------------
        g.nc_set_component_variable("interior_ring", "interior_ring")
        g.nc_set_component_variable_groups("interior_ring", ["forecast"])

        cf.write(g, grouped_file)

        # Check that the variable is in the right group
        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.construct("longitude").get_interior_ring().nc_get_variable(),
            nc.groups["forecast"].variables,
        )
        nc.close()

        # Check that the field construct hasn't changed
        h = cf.read(grouped_file, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

    def test_groups_compression(self):
        f = cf.example_field(4)

        ungrouped_file = ungrouped_file3
        grouped_file = grouped_file3

        f.compress("indexed_contiguous", inplace=True)
        f.data.get_count().nc_set_variable("count")
        f.data.get_index().nc_set_variable("index")
        cf.write(f, ungrouped_file, verbose=1)
        g = cf.read(ungrouped_file)[0]
        self.assertTrue(f.equals(g, verbose=2))

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(["forecast", "model"])

        # ------------------------------------------------------------
        # Move the count variable to the /forecast group
        # ------------------------------------------------------------
        g.data.get_count().nc_set_variable_groups(["forecast"])

        # ------------------------------------------------------------
        # Move the index variable to the /forecast group
        # ------------------------------------------------------------
        g.data.get_index().nc_set_variable_groups(["forecast"])

        # ------------------------------------------------------------
        # Move the coordinates that span the element dimension to the
        # /forecast group
        # ------------------------------------------------------------
        name = "altitude"
        g.construct(name).nc_set_variable_groups(["forecast"])

        # ------------------------------------------------------------
        # Move the sample dimension to the /forecast group
        # ------------------------------------------------------------
        g.data.get_count().nc_set_sample_dimension_groups(["forecast"])

        cf.write(g, grouped_file, verbose=1)

        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.nc_get_variable(),
            nc.groups["forecast"].groups["model"].variables,
        )
        self.assertIn(
            f.data.get_count().nc_get_variable(),
            nc.groups["forecast"].variables,
        )
        self.assertIn(
            f.data.get_index().nc_get_variable(),
            nc.groups["forecast"].variables,
        )
        self.assertIn(
            f.construct("altitude").nc_get_variable(),
            nc.groups["forecast"].variables,
        )
        nc.close()

        h = cf.read(grouped_file, verbose=1)
        self.assertEqual(len(h), 1, repr(h))
        self.assertTrue(f.equals(h[0], verbose=2))

    def test_groups_dimension(self):
        f = cf.example_field(0)

        ungrouped_file = ungrouped_file4
        grouped_file = grouped_file4

        cf.write(f, ungrouped_file)
        g = cf.read(ungrouped_file, verbose=1)
        self.assertEqual(len(g), 1)
        g = g[0]
        self.assertTrue(f.equals(g, verbose=3))

        # ------------------------------------------------------------
        # Move the field construct to the /forecast/model group
        # ------------------------------------------------------------
        g.nc_set_variable_groups(["forecast", "model"])

        # ------------------------------------------------------------
        # Move all data constructs to the /forecast group
        # ------------------------------------------------------------
        for construct in g.constructs.filter_by_data().values():
            construct.nc_set_variable_groups(["forecast"])

        # ------------------------------------------------------------
        # Move all coordinate bounds constructs to the /forecast group
        # ------------------------------------------------------------
        for construct in g.coordinates().values():
            try:
                construct.bounds.nc_set_variable_groups(["forecast"])
            except ValueError:
                pass

        cf.write(g, grouped_file, verbose=1)

        nc = netCDF4.Dataset(grouped_file, "r")
        self.assertIn(
            f.nc_get_variable(),
            nc.groups["forecast"].groups["model"].variables,
        )

        for key, construct in g.constructs.filter_by_data().items():
            self.assertIn(
                f.constructs[key].nc_get_variable(),
                nc.groups["forecast"].variables,
            )

        nc.close()

        h = cf.read(grouped_file, verbose=1)
        self.assertEqual(len(h), 1)
        h = h[0]
        self.assertTrue(f.equals(h, verbose=3))

        # ------------------------------------------------------------
        # Move all the lat dimension to the /forecast group
        # ------------------------------------------------------------
        key = g.domain_axis_key("latitude")
        domain_axis = g.constructs[key]
        domain_axis.nc_set_dimension_groups(["forecast"])

        cf.write(g, grouped_file, verbose=1)

        h = cf.read(grouped_file, verbose=1)
        self.assertEqual(len(h), 1)
        h = h[0]
        self.assertTrue(f.equals(h, verbose=3))


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
