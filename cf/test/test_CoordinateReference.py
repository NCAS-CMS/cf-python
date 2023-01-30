import atexit
import datetime
import faulthandler
import os
import tempfile
import unittest

import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf

n_tmpfiles = 1
tmpfiles = [
    tempfile.mkstemp("_test_CoordinateReference.nc", dir=os.getcwd())[1]
    for i in range(n_tmpfiles)
]
[tempfile] = tmpfiles


def _remove_tmpfiles():
    """Remove temporary files created during tests."""
    for f in tmpfiles:
        try:
            os.remove(f)
        except OSError:
            pass


atexit.register(_remove_tmpfiles)


class CoordinateReferenceTest(unittest.TestCase):
    f = cf.example_field(1)

    datum = cf.Datum(parameters={"earth_radius": 6371007})

    # Create a vertical grid mapping coordinate reference
    vconversion = cf.CoordinateConversion(
        parameters={"standard_name": "atmosphere_hybrid_height_coordinate"},
        domain_ancillaries={
            "a": "auxiliarycoordinate0",
            "b": "auxiliarycoordinate1",
            "orog": "domainancillary0",
        },
    )

    vcr = cf.CoordinateReference(
        coordinates=("coord1",), datum=datum, coordinate_conversion=vconversion
    )

    # Create a horizontal grid mapping coordinate reference
    hconversion = cf.CoordinateConversion(
        parameters={
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 38.0,
            "grid_north_pole_longitude": 190.0,
        }
    )

    hcr = cf.CoordinateReference(
        coordinate_conversion=hconversion,
        datum=datum,
        coordinates=["x", "y", "lat", "lon"],
    )

    def test_CoordinateReference__repr__str__dump(self):
        coordinate_conversion = cf.CoordinateConversion(
            parameters={
                "standard_name": "atmosphere_hybrid_height_coordinate"
            },
            domain_ancillaries={"a": "aux0", "b": "aux1", "orog": "orog"},
        )

        datum = cf.Datum(parameters={"earth_radius": 23423423423.34})

        # Create a vertical grid mapping coordinate reference
        t = cf.CoordinateReference(
            coordinates=("coord1",),
            coordinate_conversion=coordinate_conversion,
            datum=datum,
        )

        repr(t)
        str(t)
        t.dump(display=False)

        self.assertFalse(t.has_bounds())

        repr(datum)
        str(datum)

        repr(coordinate_conversion)
        str(coordinate_conversion)

    def test_CoordinateReference_equals(self):
        # Create a vertical grid mapping coordinate reference
        t = cf.CoordinateReference(
            coordinates=("coord1",),
            coordinate_conversion=cf.CoordinateConversion(
                parameters={
                    "standard_name": "atmosphere_hybrid_height_coordinate"
                },
                domain_ancillaries={"a": "aux0", "b": "aux1", "orog": "orog"},
            ),
        )
        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

        # Create a horizontal grid mapping coordinate reference
        t = cf.CoordinateReference(
            coordinates=["coord1", "fred", "coord3"],
            coordinate_conversion=cf.CoordinateConversion(
                parameters={
                    "grid_mapping_name": "rotated_latitude_longitude",
                    "grid_north_pole_latitude": 38.0,
                    "grid_north_pole_longitude": 190.0,
                }
            ),
        )
        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

        datum = cf.Datum(parameters={"earth_radius": 6371007})
        conversion = cf.CoordinateConversion(
            parameters={
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": 38.0,
                "grid_north_pole_longitude": 190.0,
            }
        )

        t = cf.CoordinateReference(
            coordinate_conversion=conversion,
            datum=datum,
            coordinates=["x", "y", "lat", "lon"],
        )

        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

        # Create a horizontal grid mapping coordinate reference
        t = cf.CoordinateReference(
            coordinates=["coord1", "fred", "coord3"],
            coordinate_conversion=cf.CoordinateConversion(
                parameters={
                    "grid_mapping_name": "albers_conical_equal_area",
                    "standard_parallel": [-30, 10],
                    "longitude_of_projection_origin": 34.8,
                    "false_easting": -20000,
                    "false_northing": -30000,
                }
            ),
        )
        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

        # Create a horizontal grid mapping coordinate reference
        t = cf.CoordinateReference(
            coordinates=["coord1", "fred", "coord3"],
            coordinate_conversion=cf.CoordinateConversion(
                parameters={
                    "grid_mapping_name": "albers_conical_equal_area",
                    "standard_parallel": cf.Data([-30, 10]),
                    "longitude_of_projection_origin": 34.8,
                    "false_easting": -20000,
                    "false_northing": -30000,
                }
            ),
        )
        self.assertTrue(t.equals(t, verbose=2))
        self.assertTrue(t.equals(t.copy(), verbose=2))

    def test_CoordinateReference_default_value(self):
        f = self.f.copy()

        self.assertEqual(cf.CoordinateReference.default_value("qwerty"), 0.0)
        self.assertEqual(
            cf.CoordinateReference.default_value("earth_depth"), 0.0
        )

        cr = f.construct("standard_name:atmosphere_hybrid_height_coordinate")
        self.assertEqual(cr.default_value("qwerty"), 0.0)
        self.assertEqual(cr.default_value("earth_depth"), 0.0)

    def test_CoordinateReference_canonical_units(self):
        f = self.f.copy()

        self.assertIsNone(cf.CoordinateReference.canonical_units("qwerty"))
        self.assertEqual(
            cf.CoordinateReference.canonical_units("earth_radius"),
            cf.Units("m"),
        )

        cr = f.construct("standard_name:atmosphere_hybrid_height_coordinate")
        self.assertIsNone(cr.canonical_units("qwerty"))
        self.assertEqual(cr.canonical_units("earth_radius"), cf.Units("m"))

    def test_CoordinateReference_match(self):
        self.assertTrue(self.vcr.match())
        self.assertTrue(
            self.vcr.match("standard_name:atmosphere_hybrid_height_coordinate")
        )
        self.assertTrue(self.vcr.match("atmosphere_hybrid_height_coordinate"))
        self.assertTrue(
            self.vcr.match("atmosphere_hybrid_height_coordinate", "qwerty")
        )

        self.assertTrue(self.hcr.match())
        self.assertTrue(
            self.hcr.match("grid_mapping_name:rotated_latitude_longitude")
        )
        self.assertTrue(self.hcr.match("rotated_latitude_longitude"))
        self.assertTrue(
            self.hcr.match(
                "grid_mapping_name:rotated_latitude_longitude", "qwerty"
            )
        )

    def test_CoordinateReference_get__getitem__(self):
        self.assertEqual(
            self.vcr["earth_radius"], self.datum.get_parameter("earth_radius")
        )
        self.assertTrue(
            self.vcr["standard_name"],
            self.vconversion.get_parameter("standard_name"),
        )
        self.assertEqual(
            self.vcr.get("earth_radius"),
            self.datum.get_parameter("earth_radius"),
        )
        self.assertIsNone(self.vcr.get("orog"))
        self.assertEqual(self.vcr.get("orog", "qwerty"), "qwerty")
        self.assertIsNone(self.vcr.get("qwerty"))
        self.assertEqual(
            self.vcr["standard_name"],
            self.vconversion.get_parameter("standard_name"),
        )
        with self.assertRaises(Exception):
            self.vcr["orog"]

        self.assertEqual(
            self.hcr["earth_radius"], self.datum.get_parameter("earth_radius")
        )
        self.assertEqual(
            self.hcr["grid_north_pole_latitude"],
            self.hconversion.get_parameter("grid_north_pole_latitude"),
        )
        self.assertEqual(
            self.hcr["grid_mapping_name"],
            self.hconversion.get_parameter("grid_mapping_name"),
        )
        self.assertEqual(
            self.hcr.get("earth_radius"),
            self.datum.get_parameter("earth_radius"),
        )
        self.assertEqual(
            self.hcr.get("grid_north_pole_latitude", "qwerty"),
            self.hconversion.get_parameter("grid_north_pole_latitude"),
        )
        self.assertIsNone(self.hcr.get("qwerty"))
        self.assertEqual(self.hcr.get("qwerty", 12), 12)
        with self.assertRaises(Exception):
            self.hcr["qwerty"]

    def test_CoordinateReference_structural_signature(self):
        c = self.hcr.copy()

        self.assertIsInstance(c.structural_signature(), tuple)

        c.datum.set_parameter("test", [23])
        s = c.structural_signature()
        self.assertEqual(s[2], ("datum:test", (23.0,), None))

        c.datum.set_parameter("test", [23, 45])
        s = c.structural_signature()
        self.assertEqual(s[2], ("datum:test", (23.0, 45.0), None))

        c.datum.set_parameter("test", [[23, 45]])
        s = c.structural_signature()
        self.assertEqual(s[2], ("datum:test", ((23.0, 45.0),), None))

        c.datum.set_parameter("test", np.array([[23, 45], [67, 89]]))
        s = c.structural_signature()
        self.assertEqual(
            s[2], ("datum:test", ((23.0, 45.0), (67.0, 89.0)), None)
        )


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
