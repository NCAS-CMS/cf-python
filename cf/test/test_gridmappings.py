import datetime
import faulthandler
import unittest

# import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf

pyproj_imported = False
try:
    import pyproj

    pyproj_imported = True
except ImportError:
    pass


all_abstract_grid_mappings = (
    cf.GridMapping,
    cf.AzimuthalGridMapping,
    cf.ConicGridMapping,
    cf.CylindricalGridMapping,
    cf.LatLonGridMapping,
    cf.PerspectiveGridMapping,
)
# Representing all Grid Mappings repsented by the CF Conventions (APpendix F)
all_concrete_grid_mappings = (
    cf.AlbersEqualArea,
    cf.AzimuthalEquidistant,
    cf.Geostationary,
    cf.LambertAzimuthalEqualArea,
    cf.LambertConformalConic,
    cf.LambertCylindricalEqualArea,
    cf.Mercator,
    cf.ObliqueMercator,
    cf.Orthographic,
    cf.PolarStereographic,
    cf.RotatedLatitudeLongitude,
    cf.LatitudeLongitude,
    cf.Sinusoidal,
    cf.Stereographic,
    cf.TransverseMercator,
    cf.VerticalPerspective,
)
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

    # @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
    def test_grid_mapping__repr__str__(self):
        """TODO."""
        for cls in all_concrete_grid_mappings:
            if cls.__name__ not in all_concrete_grid_mappings_req_args:
                g = cls()
            else:
                example_minimal_args = all_concrete_grid_mappings_req_args[
                    cls.__name__
                ]
                g = cls(*example_minimal_args)
            repr(g)
            str(g)

    def test_grid_mapping_find_gm_class(self):
        """TODO."""
        for f in self.f_with_gm:
            crefs = f.coordinate_references().values()
            for cref in crefs:
                gm = cref.coordinate_conversion.get_parameter(
                    "grid_mapping_name", default=None
                )
                # TODO test that matches with GM class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
