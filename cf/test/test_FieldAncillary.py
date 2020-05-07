import datetime
import os
import time
import unittest

import cf


class FieldAncillaryTest(unittest.TestCase):
    def test_FieldAncillary(self):
        f = cf.FieldAncillary()
        self.assertTrue(f.isfieldancillary)

        _ = repr(f)
        _ = str(f)
        _ = f.dump(display=False)


# --- End: class

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    cf.environment()
    print()
    unittest.main(verbosity=2)
