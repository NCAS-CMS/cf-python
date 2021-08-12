import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DomainAxisTest(unittest.TestCase):
    def test_DomainAxis__repr__str(self):
        x = cf.DomainAxis(size=56)
        x.nc_set_dimension("tas")

        repr(x)
        str(x)

    def test_DomainAxis(self):
        x = cf.DomainAxis(size=111)
        x.nc_set_dimension("tas")

        self.assertEqual(x.size, 111)
        del x.size
        self.assertIsNone(getattr(x, "size", None))
        x.size = 56
        self.assertEqual(x.size, 56)
        self.assertEqual(x, 56)

        x += 1
        self.assertEqual(x.size, 57)
        x -= 1
        self.assertEqual(x.size, 56)
        y = x + 1
        self.assertEqual(y.size, 57)
        y = x - 1
        self.assertEqual(y.size, 55)
        y = 1 + x
        self.assertEqual(y.size, 57)

        self.assertEqual(int(x), 56)

        self.assertGreater(x, 1)
        self.assertLess(x, 100)
        self.assertGreaterEqual(x, 1)
        self.assertLessEqual(x, 100)
        self.assertNotEqual(x, 100)

        hash(x)


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
