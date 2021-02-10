import datetime
import faulthandler
import os
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class CellMeasureTest(unittest.TestCase):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_file.nc"
    )
    #    f = cf.read(filename)[0]

    def test_CellMeasure__repr__str__dump(self):
        f = cf.read(self.filename)[0]
        x = f.cell_measures("measure:area").value()

        _ = repr(x)
        _ = str(x)
        _ = x.dump(display=False)

    def test_CellMeasure_measure(self):
        f = cf.read(self.filename)[0]
        x = f.cell_measures("measure:area").value()

        self.assertEqual(x.measure, "area")
        del x.measure
        self.assertIsNone(getattr(x, "measure", None))
        x.measure = "qwerty"
        self.assertEqual(x.measure, "qwerty")

    def test_CellMeasure_identity(self):
        f = cf.read(self.filename)[0]
        x = f.cell_measures("measure:area").value()

        self.assertEqual(x.identity(), "measure:area")
        del x.measure
        self.assertEqual(x.identity(), "ncvar%cell_measure", x.identity())
        x.nc_del_variable()
        self.assertEqual(x.identity(), "")


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
