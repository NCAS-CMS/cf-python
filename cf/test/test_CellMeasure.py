import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class CellMeasureTest(unittest.TestCase):
    f = cf.example_field(1)

    def test_CellMeasure__repr__str__dump(self):
        x = self.f.cell_measure("measure:area")

        repr(x)
        str(x)
        x.dump(display=False)

    def test_CellMeasure_measure(self):
        x = self.f.cell_measure("measure:area").copy()

        self.assertEqual(x.measure, "area")
        del x.measure
        self.assertIsNone(getattr(x, "measure", None))
        x.measure = "qwerty"
        self.assertEqual(x.measure, "qwerty")

    def test_CellMeasure_identity(self):
        x = self.f.cell_measure("measure:area").copy()

        self.assertEqual(x.identity(), "measure:area")
        del x.measure
        self.assertEqual(x.identity(), "ncvar%cell_measure", x.identity())
        x.nc_del_variable()
        self.assertEqual(x.identity(), "")


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
