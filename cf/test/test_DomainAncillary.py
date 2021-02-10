import datetime
import faulthandler
import unittest

faulthandler.enable()  # to debug seg faults and timeouts

import cf


class DomainAncillaryTest(unittest.TestCase):
    def test_DomainAncillary(self):
        f = cf.DomainAncillary()

        _ = repr(f)
        _ = str(f)
        _ = f.dump(display=False)


# --- End: class


if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
