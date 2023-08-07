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
    f = cf.example_field(1)

    @unittest.skipUnless(pyproj_imported, "Requires pyproj package.")
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


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
