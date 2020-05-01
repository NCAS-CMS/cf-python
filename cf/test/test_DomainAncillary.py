import datetime
import os
import time
import unittest

import numpy

import cf


class DomainAncillaryTest(unittest.TestCase):
    def test_DomainAncillary(self):
        f = cf.DomainAncillary()
        self.assertTrue(f.isdomainancillary)

        _ = repr(f)
        _ = str(f)
        _ = f.dump(display=False)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
