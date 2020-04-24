import datetime
import os
import time
import unittest

import numpy

import cf


class CellMeasureTest(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'test_file.nc')

    def test_CellMeasure__repr__str__dump(self):
        f = cf.read(self.filename)[0]
        x = f.cell_measures('measure:area').value()

        _ = repr(x)
        _ = str(x)
        _ = x.dump(display=False)

        self.assertTrue(x.ismeasure)

    def test_CellMeasure_measure(self):
        f = cf.read(self.filename)[0]
        x = f.cell_measures('measure:area').value()

        self.assertTrue(x.measure == 'area')
        del x.measure
        self.assertIsNone(getattr(x, 'measure', None))
        x.measure = 'qwerty'
        self.assertTrue(x.measure == 'qwerty')

    def test_CellMeasure_identity(self):
        f = cf.read(self.filename)[0]
        x = f.cell_measures('measure:area').value()

        self.assertTrue(x.identity() == 'measure:area')
        del x.measure
        self.assertTrue(x.identity() == 'ncvar%cell_measure', x.identity())
        x.nc_del_variable()
        self.assertTrue(x.identity() == '')


# --- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
