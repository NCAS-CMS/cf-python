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


_all_abstract_grid_mappings = (
    cf.GridMapping,
    cf.AzimuthalGridMapping,
    cf.ConicGridMapping,
    cf.CylindricalGridMapping,
    cf.LatLonGridMapping,
    cf.PerspectiveGridMapping,
)
_all_concrete_grid_mappings = (
    cf.AlbersEqualArea,
    cf.AzimuthalEquidistant,
    cf.Geostationary,
    cf.LambertAzimuthalEqualArea,
    cf.LambertConformalConic,
    cf.LambertCylindricalEqualArea,
    cf.LatitudeLongitude,
    cf.Mercator,
    cf.ObliqueMercator,
    cf.Orthographic,
    cf.PolarStereographic,
    cf.RotatedLatitudeLongitude,
    cf.Sinusoidal,
    cf.Stereographic,
    cf.TransverseMercator,
    cf.VerticalPerspective,
)


# These are those of the above which have required positional arguments
all_grid_mappings_required_args = {
    "AlbersEqualArea": {"standard_parallel": (0.0, None)},
    "ConicGridMapping": {"standard_parallel": (0.0, None)},
    "Geostationary": {"perspective_point_height": 1000},
    "LambertConformalConic": {"standard_parallel": (1.0, 1.0)},
    "PerspectiveGridMapping": {"perspective_point_height": 1000},
    "RotatedLatitudeLongitude": {
        "grid_north_pole_latitude": 0.0,
        "grid_north_pole_longitude": 0.0,
    },
    "VerticalPerspective": {"perspective_point_height": 1000},
}


class GridMappingsTest(unittest.TestCase):
    """Unit test for the GridMapping class and any derived classes."""

    # Of the example fields, only 1, 6 and 7 have any coordinate references
    # with a coordinate conversion, hence use these to test, plus 0 as an
    # example case of a field without a coordinate reference at all.
    f = cf.example_fields()
    # TODOGM, could do with a new example field or two with a grid mapping
    # other than those two below, for diversity and testing on etc.
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

    @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping__init__(self):
        """Test GridMapping object initiation."""
        for cls in _all_concrete_grid_mappings:
            if cls.__name__ not in all_grid_mappings_required_args:
                g = cls()
                g.grid_mapping_name
            else:
                example_minimal_args = all_grid_mappings_required_args[
                    cls.__name__
                ]
                g = cls(**example_minimal_args)
                g.grid_mapping_name

        for cls in _all_abstract_grid_mappings:
            if cls.__name__ not in all_grid_mappings_required_args:
                g = cls()
                self.assertEqual(g.grid_mapping_name, None)
            else:
                example_minimal_args = all_grid_mappings_required_args[
                    cls.__name__
                ]
                g = cls(**example_minimal_args)
                self.assertEqual(g.grid_mapping_name, None)

    @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping__repr__str__(self):
        """Test all means of GridMapping inspection."""
        for cls in _all_concrete_grid_mappings:
            if cls.__name__ not in all_grid_mappings_required_args:
                g = cls()
            else:
                example_minimal_args = all_grid_mappings_required_args[
                    cls.__name__
                ]
                g = cls(**example_minimal_args)
            repr(g)
            str(g)

        g1 = cf.Mercator()
        self.assertEqual(repr(g1), "<CF CylindricalGridMapping: Mercator>")
        self.assertEqual(
            str(g1), "<CF CylindricalGridMapping: Mercator +proj=merc>"
        )

        g2 = cf.Orthographic()
        self.assertEqual(repr(g2), "<CF AzimuthalGridMapping: Orthographic>")
        self.assertEqual(
            str(g2), "<CF AzimuthalGridMapping: Orthographic +proj=ortho>"
        )

        g3 = cf.Sinusoidal()
        self.assertEqual(repr(g3), "<CF GridMapping: Sinusoidal>")
        self.assertEqual(str(g3), "<CF GridMapping: Sinusoidal +proj=sinu>")

        g4 = cf.Stereographic()
        self.assertEqual(repr(g4), "<CF AzimuthalGridMapping: Stereographic>")
        self.assertEqual(
            str(g4), "<CF AzimuthalGridMapping: Stereographic +proj=stere>"
        )

    @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping_is_latlon_gm(self):
        """Test the 'is_latlon_gm' method on all GridMappings."""
        # In this one case we expect True...
        # TODOGM: what about cf.RotatedLatitudeLongitude?
        g = cf.LatitudeLongitude
        self.assertTrue(g.is_latlon_gm())  # check on class
        self.assertTrue(g().is_latlon_gm())  # check on instance

        # ...and expect False for all other GridMappings
        for cls in _all_concrete_grid_mappings:
            if not issubclass(cls, cf.LatitudeLongitude):
                self.assertFalse(cls.is_latlon_gm())

    @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping_map_parameter_validation(self):
        """Test the validation of map parameters to Grid Mapping classes."""
        g1 = cf.Mercator(
            false_easting=10.0,
            false_northing=cf.Data(-20, units="cm"),
            standard_parallel=(None, 50),
            longitude_of_projection_origin=cf.Data(
                -40.0, units="degrees_east"
            ),
            scale_factor_at_projection_origin=3.0,
            prime_meridian_name="brussels",
        )
        self.assertEqual(g1.false_easting, cf.Data(10.0, "m"))
        self.assertEqual(g1.false_northing, cf.Data(-0.2, "m"))
        self.assertEqual(
            g1.standard_parallel, (None, cf.Data(50, "degrees_north"))
        )
        self.assertEqual(
            g1.longitude_of_projection_origin, cf.Data(-40.0, "degrees_east")
        )
        self.assertEqual(g1.scale_factor_at_projection_origin, cf.Data(3.0, 1))
        self.assertEqual(g1.prime_meridian_name, "brussels")

        # TODOGM extend this test a lot with testing like the above and
        # with systematic coverage over valid inputs

    @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping__get_cf_grid_mapping_from_name(self):
        """Test the '_get_cf_grid_mapping_from_name' function."""
        for gm_name, cf_gm_class in {
            "vertical_perspective": cf.VerticalPerspective,
            "oblique_mercator": cf.ObliqueMercator,
            "albers_conical_equal_area": cf.AlbersEqualArea,
            "lambert_conformal_conic": cf.LambertConformalConic,
            "some_unsupported_name": None,
        }.items():
            pass  # TODO UPDATE with class
            # self.assertEqual(
            #     cf._get_cf_grid_mapping_from_name(gm_name), cf_gm_class
            # )

    def test_grid_mapping_convert_proj_angular_data_to_cf(self):
        """Test the 'convert_proj_angular_data_to_cf' function."""

        # Check representative valid inputs
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
        # Check representative valid inputs
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
            (cf.Data(45, units="degree"), "45"),
            # Check negative
            (cf.Data(-0.1, units="degrees"), "-0.1"),
            (cf.Data(-10, units="degrees"), "-10"),
            # Check zero case
            (cf.Data(0, units="degrees_north"), "0"),
            (cf.Data(0.0, units="degrees_north"), "0.0"),
            # Check radians units cases and >180
            (cf.Data(190, units="radians"), "190R"),
            (cf.Data(190.0, units="radians"), "190.0R"),
            # Check flot with superfluous 0
            (cf.Data(120.100, units="degrees"), "120.1"),
        ]:
            _input, correct_output = input_with_correct_output
            p = cf.convert_cf_angular_data_to_proj(_input)
            self.assertEqual(p, correct_output)

        # Check representative invalid inputs error correctly
        for bad_input in [
            cf.Data([1, 2, 3]),  # not singular (size 1)
            cf.Data(45),  # no units
            cf.Data(45, "m"),  # non-angular units
            cf.Data(2, "elephants"),  # bad/non-CF units
        ]:
            with self.assertRaises(ValueError):
                cf.convert_cf_angular_data_to_proj(bad_input)
        with self.assertRaises(TypeError):
            # Non-numeric value
            cf.convert_cf_angular_data_to_proj(cf.Data("N", "radians"))

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
        for p in ("10", "-10", "10.11", "0", "1R", "0.2R", "-0.1R", "0R"):
            p2 = cf.convert_cf_angular_data_to_proj(
                cf.convert_proj_angular_data_to_cf(p)
            )
            self.assertEqual(p, p2)

            # With a lat or lon 'context'. Only non-radians inputs will
            # be re-generated since degrees_X gets converted back to the
            # default degrees, so skip those in these test cases.
            if not p.endswith("R"):
                p3 = cf.convert_cf_angular_data_to_proj(
                    cf.convert_proj_angular_data_to_cf(p, context="lat")
                )
                self.assertEqual(p, p3)

                p4 = cf.convert_cf_angular_data_to_proj(
                    cf.convert_proj_angular_data_to_cf(p, context="lon")
                )
                self.assertEqual(p, p4)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
