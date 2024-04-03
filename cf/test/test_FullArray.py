import datetime
import faulthandler
import unittest

import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class FullArrayTest(unittest.TestCase):
    def test_FullValue_inspection(self):
        full = 9
        f = cf.FullArray(full, np.dtype(int), shape=(2, 3, 4))
        self.assertEqual(f.get_full_value(), full)
        self.assertEqual(f.shape, (2, 3, 4))
        self.assertEqual(f.dtype, np.dtype(int))
        self.assertIsNone(f.set_full_value(10))
        self.assertEqual(f.get_full_value(), 10)

    def test_FullValue_array(self):
        full = 9
        f = cf.FullArray(full, np.dtype(int), shape=(2, 3, 4))
        self.assertTrue((f.array == np.full(f.shape, full)).all())

        f = f[0, [True, False, True], ::3]
        self.assertTrue((f.array == np.full((2, 1), full)).all())

    def test_FullValue_masked_array(self):
        full = np.ma.masked
        f = cf.FullArray(full, np.dtype(int), shape=(2, 3))

        a = np.ma.masked_all(f.shape, dtype=np.dtype(int))
        array = f.array
        self.assertEqual(array.dtype, a.dtype)
        self.assertTrue(
            (np.ma.getmaskarray(array) == np.ma.getmaskarray(a)).all()
        )

    def test_FullValue_get_array(self):
        full = 9
        f = cf.FullArray(full, np.dtype(int), shape=(2, 3))
        f = f[0, 1]
        self.assertEqual(f.shape, ())

        array = f._get_array(index=Ellipsis)
        self.assertTrue((array == np.full((2, 3), full)).all())


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
