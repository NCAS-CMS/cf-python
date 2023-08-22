import datetime
import faulthandler
import unittest

import numpy as np

import cf

faulthandler.enable()  # to debug seg faults and timeouts

pyproj_imported = False
try:
    import pyproj  # noqa: F401

    pyproj_imported = True
except ImportError:
    pass


# These are those of the above which have required positional arguments
all_concrete_grid_mappings_req_args = {
    "AlbersEqualArea": {"standard_parallel": 0.0},
    "Geostationary": {"perspective_point_height": 1000},
    "VerticalPerspective": {"perspective_point_height": 1000},
    "LambertConformalConic": {"standard_parallel": 0.0},
    "RotatedLatitudeLongitude": {
        "grid_north_pole_latitude": 0.0,
        "grid_north_pole_longitude": 0.0,
    },
}


class GridMappingsTest(unittest.TestCase):
    """TODO."""

    # Of the example fields, only 1, 6 and 7 have any coordinate references
    # with a coordinate conversion, hence use these to test, plus 0 as an
    # example case of a field without a coordinate reference at all.
    f = cf.example_fields()
    f0 = f[0]  # No coordinate reference
    f1 = f[1]  # 2, with grid mappings of [None, 'rotated_latitude_longitude']
    f6 = f[6]  # 1, with grid mapping of ['latitude_longitude']
    f7 = f[7]  # 1, with grid mapping of ['rotated_latitude_longitude']
    f_with_gm = (f1, f6, f7)
    f_wth_gm_mapping = {
        "f1": cf.RotatedLatitudeLongitude,
        "f6": cf.LatitudeLongitude,
        "f7": cf.RotatedLatitudeLongitude,
    }

    # From a custom netCDF file with Oblique Mercator GM
    # TODO generate this .nc via create_test_files.py and un-commit
    # forced commit of the (data-free / header-only) netCDF file.
    f_om = cf.read("oblique_mercator.nc")

    # Create some coordinate references with different GMs to test on:
    cr_aea = cf.CoordinateReference(
        coordinates=["coordA", "coordB", "coordC"],
        coordinate_conversion=cf.CoordinateConversion(
            parameters={
                "grid_mapping_name": "albers_conical_equal_area",
                "standard_parallel": [10, 10],
                "longitude_of_projection_origin": 45.0,
                "false_easting": -1000,
                "false_northing": 500,
            }
        ),
    )
    cr_aea_actual_proj_string = (
        "+proj=aea +lat_1=10. +lat_2=10. +lon_0=45.0 +x_0=-1000. +y_0=-500."
    )

    cr_om = cf.CoordinateReference(
        coordinates=["coordA", "coordB"],
        coordinate_conversion=cf.CoordinateConversion(
            parameters={
                "grid_mapping_name": "oblique_mercator",
                "latitude_of_projection_origin": -22.0,
                "longitude_of_projection_origin": -59.0,
                "false_easting": -12500.0,
                "false_northing": -12500.0,
                "azimuth_of_central_line": 89.999999,
                "scale_factor_at_projection_origin": 1.0,
                "inverse_flattening": 0.0,
                "semi_major_axis": 6371229.0,
            }
        ),
    )
    cr_om_actual_proj_string = (
        "+proj=omerc +lat_0=-22.00 +alpha=89.999999 +lonc=-59.00 "
        "+x_0=-12500. +y_0=-12500. +ellps=sphere +a=6371229. +b=6371229. "
        "+units=m +no_defs"
    )

    # @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping__repr__str__(self):
        """TODO."""
        for cls in cf._all_concrete_grid_mappings:
            if cls.__name__ not in all_concrete_grid_mappings_req_args:
                g = cls()
            else:
                example_minimal_args = all_concrete_grid_mappings_req_args[
                    cls.__name__
                ]
                g = cls(*example_minimal_args)
            repr(g)
            str(g)

    def test_grid_mapping__get_cf_grid_mapping_from_name(self):
        """TODO."""
        for gm_name, cf_gm_class in {
            "vertical_perspective": cf.VerticalPerspective,
            "oblique_mercator": cf.ObliqueMercator,
            "albers_conical_equal_area": cf.AlbersEqualArea,
            "lambert_conformal_conic": cf.LambertConformalConic,
            "some_unsupported_name": None,
        }.items():
            self.assertEqual(
                cf._get_cf_grid_mapping_from_name(gm_name), cf_gm_class
            )

    def test_grid_mapping_convert_proj_angular_data_to_cf(self):
        """Test the 'convert_proj_angular_data_to_cf' function."""
        for input_with_correct_output in [
            # Check float value and no suffix
            (("45.0", None), cf.Data(45.0, units="degrees")),
            (("45.0", "lat"), cf.Data(45.0, units="degrees_north")),
            (("45.0", "lon"), cf.Data(45.0, units="degrees_east")),
            # Check integer and no suffix
            (("100", None), cf.Data(100, units="degrees")),
            (("100", "lat"), cf.Data(100, units="degrees_north")),
            (("100", "lon"), cf.Data(100, units="degrees_east")),
            # Check >360 degrees (over a full revolution) and  "r" suffix
            ((f"{3.0 * np.pi}r", None), cf.Data(3.0 * np.pi, units="radians")),
            (
                (f"{3.0 * np.pi}r", "lat"),
                cf.Data(540.0, units="degrees_north"),
            ),
            ((f"{3.0 * np.pi}r", "lon"), cf.Data(540.0, units="degrees_east")),
            # Check "R" suffix
            ((f"{0.5 * np.pi}R", None), cf.Data(0.5 * np.pi, units="radians")),
            ((f"{0.5 * np.pi}R", "lat"), cf.Data(90.0, units="degrees_north")),
            ((f"{0.5 * np.pi}R", "lon"), cf.Data(90.0, units="degrees_east")),
            # Check integer value and "d" suffix
            (("10d", None), cf.Data(10, units="degrees")),
            (("10d", "lat"), cf.Data(10, units="degrees_north")),
            (("10d", "lon"), cf.Data(10, units="degrees_east")),
            # Check >180 float and "D" suffix
            (("200.123D", None), cf.Data(200.123, units="degrees")),
            (("200.123D", "lat"), cf.Data(200.123, units="degrees_north")),
            (("200.123D", "lon"), cf.Data(200.123, units="degrees_east")),
            # Check negative numeric value and "°" suffix
            (("-70.5°", None), cf.Data(-70.5, units="degrees")),
            (("-70.5°", "lat"), cf.Data(-70.5, units="degrees_north")),
            (("-70.5°", "lon"), cf.Data(-70.5, units="degrees_east")),
            # Check zero and lack of digits after point edge cases
            (("0", None), cf.Data(0, units="degrees")),
            (("0.0", "lat"), cf.Data(0.0, units="degrees_north")),
            (("-0.", "lon"), cf.Data(0.0, units="degrees_east")),
        ]:
            _input, correct_output = input_with_correct_output
            d = cf.convert_proj_angular_data_to_cf(*_input)
            self.assertTrue(d.equals(correct_output, verbose=2))

    def test_grid_mapping_convert_cf_angular_data_to_proj(self):
        """Test the 'convert_cf_angular_data_to_proj' function."""
        for input_with_correct_output in [
            # Check float and basic lat/lon-context free degree unit
            (cf.Data(45.0, units="degrees"), "45.0"),
            # Check integer and various units possible for lat/lon
            (cf.Data(45, units="degrees_north"), "45"),
            (cf.Data(45, units="degrees_N"), "45"),
            (cf.Data(45, units="degreeN"), "45"),
            (cf.Data(45, units="degrees_east"), "45"),
            (cf.Data(45, units="degrees_E"), "45"),
            (cf.Data(45, units="degreeE"), "45"),
        ]:
            _input, correct_output = input_with_correct_output
            p = cf.convert_cf_angular_data_to_proj(_input)
            self.assertEqual(p, correct_output)

        # Note that 'convert_cf_angular_data_to_proj' and
        # 'convert_proj_angular_data_to_cf' are not strict inverse
        # functions, since the former will convert to the *simplest*
        # way to specify the PROJ input, namely with no suffix for
        # degrees(_X) units and the 'R' suffix for radians, whereas
        # the input might have 'D' or 'r' etc. instead.
        #
        # However, we check that inputs that are expected to be
        # undone to their original form by operation of both
        # functions, namely those with this 'simplest' PROJ form,
        # do indeed get re-generated with operation of both:
        for p in ("10", "-10", "0", "1R", "0.2R", "0.1R"):
            p2 = cf.convert_cf_angular_data_to_proj(
                cf.convert_proj_angular_data_to_cf(p)
            )
            self.assertEqual(p, p2)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
