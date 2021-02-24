import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class FieldAncillaryTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )
    #    f = cf.read(filename)[0]

    def test_FieldAncillary(self):
        f = cf.FieldAncillary()

        _ = repr(f)
        _ = str(f)
        _ = f.dump(display=False)

    def test_FieldAncillary_source(self):
        f = cf.read(self.filename)[0]

        a = f.auxiliary_coordinates("latitude").value()
        cf.FieldAncillary(source=a)

    def test_FieldAncillary_properties(self):
        f = cf.read(self.filename)[0]
        x = f.domain_ancillaries("ncvar%a").value()

        x.set_property("long_name", "qwerty")

        self.assertEqual(x.get_property("long_name"), "qwerty")
        self.assertEqual(x.del_property("long_name"), "qwerty")
        self.assertIsNone(x.get_property("long_name", None))
        self.assertIsNone(x.del_property("long_name", None))

    def test_FieldAncillary_insert_dimension(self):
        f = cf.read(self.filename)[0]
        d = f.dimension_coordinates("grid_longitude").value()
        x = cf.FieldAncillary(source=d)

        self.assertEqual(x.shape, (9,))

        y = x.insert_dimension(0)
        self.assertEqual(y.shape, (1, 9))

        x.insert_dimension(-1, inplace=True)
        self.assertEqual(x.shape, (9, 1))

    def test_FieldAncillary_transpose(self):
        f = cf.read(self.filename)[0]
        a = f.auxiliary_coordinates("longitude").value()
        x = cf.FieldAncillary(source=a)

        self.assertEqual(x.shape, (9, 10))

        y = x.transpose()
        self.assertEqual(y.shape, (10, 9))

        x.transpose([1, 0], inplace=True)
        self.assertEqual(x.shape, (10, 9))

    def test_FieldAncillary_squeeze(self):
        f = cf.read(self.filename)[0]
        a = f.auxiliary_coordinates("longitude").value()
        x = cf.FieldAncillary(source=a)

        x.insert_dimension(1, inplace=True)
        x.insert_dimension(0, inplace=True)

        self.assertEqual(x.shape, (1, 9, 1, 10))

        y = x.squeeze()
        self.assertEqual(y.shape, (9, 10))

        x.squeeze(2, inplace=True)
        self.assertEqual(x.shape, (1, 9, 10))


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
