import datetime
import os
import time
import unittest

import cf


class DomainAxisTest(unittest.TestCase):
    def test_DomainAxis__repr__str(self):
        x = cf.DomainAxis(size=56)
        x.nc_set_dimension('tas')

        _ = repr(x)
        _ = str(x)

    def test_DomainAxis(self):
        x = cf.DomainAxis(size=111)
        x.nc_set_dimension('tas')

        self.assertTrue(x.size == 111)
        del x.size
        self.assertIsNone(getattr(x, 'size', None))
        x.size = 56
        self.assertTrue(x.size == 56)

        self.assertTrue(x == 56)

        x += 1
        self.assertTrue(x.size == 57)
        x -= 1
        self.assertTrue(x.size == 56)
        y = x + 1
        self.assertTrue(y.size == 57)
        y = x - 1
        self.assertTrue(y.size == 55)
        y = 1 + x
        self.assertTrue(y.size == 57)

        self.assertTrue(int(x) == 56)

        self.assertTrue(x > 1)
        self.assertTrue(x < 100)
        self.assertTrue(x >= 1)
        self.assertTrue(x <= 100)
        self.assertTrue(x != 100)

        _ = hash(x)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
